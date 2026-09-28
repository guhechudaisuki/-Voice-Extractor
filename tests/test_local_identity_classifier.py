from __future__ import annotations

import sys
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extractor.local_identity_classifier import audit_candidates, window_starts
from extractor.types import CandidateSentence
from extractor.pipeline import ExtractionPipeline, PipelineOptions


class LocalClassifierAuditTests(unittest.TestCase):
    def test_tail_window_covers_non_multiple_duration_without_padding(self):
        self.assertEqual(window_starts(19001), [0, 4000, 8000, 11001])
        self.assertEqual(window_starts(7999), [])
        self.assertEqual(window_starts(8000), [0])

    def audit(self, scores, duration=2.0, wave=None):
        candidate = CandidateSentence(0.0, duration, "target words", accepted=True)
        accepted, rejected = [candidate], []
        values = iter(scores)
        audit_candidates(
            accepted, rejected, torch.ones(int(duration * 16000)) * 0.1 if wave is None else wave,
            lambda _wave: next(values), -2.0,
        )
        return candidate, accepted, rejected

    def test_positive_mean_cannot_hide_two_negative_windows(self):
        candidate, accepted, rejected = self.audit([10, 10, -5, -4, 10, 10, 10])
        self.assertEqual(accepted, [])
        self.assertEqual(rejected, [candidate])
        self.assertFalse(candidate.accepted)

    def test_vetoed_clip_with_long_clean_run_returns_trim_proposal(self):
        candidate, accepted, rejected = self.audit(
            [10, 10, 10, -5, -6, 10, 10, 10, 10, 10, 10], duration=3.0,
        )
        self.assertEqual(accepted, [])
        self.assertEqual(rejected, [candidate])
        proposal = candidate.diagnostics["classifier_trim_proposal"]
        self.assertEqual(proposal["span"], [1.25, 3.0])
        self.assertEqual(proposal["windows"], 6)

    def test_short_clean_run_does_not_propose_a_trim(self):
        candidate, accepted, rejected = self.audit(
            [10, 10, -5, -6, 10, 10, 10], duration=2.0,
        )
        self.assertEqual(accepted, [])
        self.assertNotIn("classifier_trim_proposal", candidate.diagnostics)

    def test_single_negative_window_does_not_veto_a_whole_sentence(self):
        candidate, accepted, rejected = self.audit([3, 3, -5, 3, 3, 3, 3])
        self.assertEqual(accepted, [candidate])
        self.assertEqual(rejected, [])

    def test_silence_is_not_negative_identity_evidence(self):
        candidate, accepted, rejected = self.audit([], wave=torch.zeros(32000))
        self.assertEqual(accepted, [candidate])
        self.assertEqual(rejected, [])

    def test_missing_window_breaks_negative_chain(self):
        candidate, accepted, rejected = self.audit([3, -5, float("nan"), -5, 3, 3, 3])
        self.assertEqual(accepted, [candidate])
        self.assertEqual(rejected, [])

    def test_short_final_tail_is_actually_scored(self):
        candidate, accepted, rejected = self.audit([3, 3, -4, -5], duration=1.19)
        self.assertEqual(accepted, [])
        self.assertEqual(rejected, [candidate])
        self.assertAlmostEqual(candidate.diagnostics["local_classifier_audit"]["windows"][-1]["end"], 1.19)

    def test_checkpoint_for_different_reference_target_is_not_loaded(self):
        import io

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "models" / "anime_t3" / "classifier_v2.pt"
            checkpoint.parent.mkdir(parents=True)
            (root / "model" / "speaker" / "wavlm-base-plus-sv").mkdir(parents=True)
            buffer = io.BytesIO()
            torch.save({"calibration_status": "passed", "reference_sha256": ["other-target"]}, buffer)
            checkpoint.write_bytes(buffer.getvalue())
            pipeline = ExtractionPipeline(PipelineOptions(), device="cpu")
            pipeline._reference_sha256 = ["current-target"]
            with patch("extractor.pipeline.ASSET_ROOT", root):
                self.assertIsNone(pipeline._ensure_tail_scorer())
            self.assertIsNone(pipeline._local_identity_reject_threshold)


if __name__ == "__main__":
    unittest.main()
