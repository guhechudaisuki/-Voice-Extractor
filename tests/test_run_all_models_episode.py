"""The three external models control review proposals, not training acceptance."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.run_all_models_episode import (  # noqa: E402
    _transcribe_new, choose_model_proposals, eligible_proposal, export_review_clip,
)
from extractor.types import CandidateSentence  # noqa: E402


class FullModelReviewPolicyTests(unittest.TestCase):
    def test_review_export_reads_exact_uvr_stem_span(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stem = root / "stem.wav"
            samples = 0.2 * np.sin(np.linspace(0, 200, 32000)).astype(np.float32)
            sf.write(stem, samples, 16000)
            row = {"start": 0.5, "end": 1.5, "base_accepted": False}
            record = export_review_clip(stem, root / "preview", 1, row, "こんにちは")
            output, rate = sf.read(root / "preview" / record["audio"])
            self.assertEqual(rate, 16000)
            self.assertEqual(len(output), 16000)
            self.assertGreater(np.corrcoef(output, samples[8000:24000])[0, 1], 0.99)
            self.assertEqual((root / "preview" / record["text_file"]).read_text(encoding="utf-8"),
                             "こんにちは")

    def test_stt_segments_map_by_span_index_not_overlap(self):
        spans = [{"start": 1.0, "end": 2.0}, {"start": 1.9, "end": 3.0}]
        first = CandidateSentence(1.9, 2.0, "前の句")
        second = CandidateSentence(1.9, 2.1, "次の句")
        first.diagnostics["transcription_span_index"] = 0
        second.diagnostics["transcription_span_index"] = 1
        with patch("evaluation.run_all_models_episode.WhisperSegmenter") as segmenter:
            segmenter.return_value.transcribe_spans.return_value = [first, second]
            self.assertEqual(_transcribe_new(Path("unused.wav"), spans, "cpu"),
                             ["前の句", "次の句"])

    def test_empty_model_proposal_set_has_standard_json_policy(self):
        import json

        selected, policy = choose_model_proposals([])
        self.assertEqual(selected, [])
        self.assertIsNone(policy["voicefilter_episode_candidate_median_db"])
        self.assertIsNone(policy["spexplus_episode_candidate_median_db"])
        json.dumps(policy, allow_nan=False)

    def test_hard_rejections_never_become_model_proposals(self):
        accepted = [{"start": 5.0, "end": 7.0}]
        row = {"start": 1.0, "end": 3.0, "accepted": False,
               "reject_reason": "声纹匹配不足", "singing_score": 0, "overlap_score": 0}
        self.assertTrue(eligible_proposal(row, accepted, 1.2))
        for reason in ("检测到有人唱歌", "检测到多人同时发声", "更接近排除角色 1"):
            self.assertFalse(eligible_proposal({**row, "reject_reason": reason}, accepted, 1.2))
        self.assertFalse(eligible_proposal({**row, "overlap_score": 0.1}, accepted, 1.2))
        self.assertFalse(eligible_proposal({**row, "start": 5.5, "end": 7.5}, accepted, 1.2))

    def test_every_model_can_change_review_selection(self):
        def row(index: int, pvad: float, vf: float, spex: float) -> dict:
            return {"start": index * 4.0, "end": index * 4.0 + 2.0,
                    "pvad": {"target_overlap_seconds": pvad},
                    "voicefilter": {"output_over_input_db_model_estimate": vf},
                    "spexplus": {"output_over_input_db": spex}}

        rows = [row(0, 0.5, 4, 4), row(1, 0.5, 3, 3),
                row(2, 0.5, 2, 1), row(3, 0.1, 4, 4)]
        selected, policy = choose_model_proposals(rows)
        self.assertEqual(selected, rows[:2])
        self.assertEqual(policy["three_model_review_selected"], 2)
        self.assertNotIn(rows[3], selected)  # PVAD excluded it.
        rows[0]["voicefilter"]["output_over_input_db_model_estimate"] = 0
        selected, _ = choose_model_proposals(rows)
        self.assertNotIn(rows[0], selected)  # VoiceFilter changed it.
        rows[1]["spexplus"]["output_over_input_db"] = 0
        selected, _ = choose_model_proposals(rows)
        self.assertNotIn(rows[1], selected)  # SpEx+ changed it.


if __name__ == "__main__":
    unittest.main()
