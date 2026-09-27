"""Exercise the production recovery caller, without loading model weights."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extractor.pipeline import ExtractionPipeline
from extractor.speaker import SpeakerBoundary
from extractor.types import TimeSpan


class LocalRecoveryEvidenceTests(unittest.TestCase):
    def test_no_target_parts_still_preserves_all_local_rejection_evidence(self):
        pipeline = ExtractionPipeline.__new__(ExtractionPipeline)
        pipeline.options = SimpleNamespace(min_sentence_seconds=0.55)
        pipeline._verify_speaker_span = Mock(return_value=SimpleNamespace(
            accepted=False, tier="rejected",
        ))
        pipeline._apply_speaker_match = Mock()
        verifier = Mock()
        verifier.exclusion_audit.return_value = None
        scored = []
        result = pipeline._recover_target_segments(
            TimeSpan(1, 4), [SpeakerBoundary(2, 0.1, 0.1, 1)],
            verifier, None, None, [], 0.7, lambda *_args: None, 1, 1,
            scored_parts=scored,
        )
        accepted, rejected = result
        self.assertEqual(accepted, [])
        parts = [part for part in rejected if part.diagnostics.get("recovery_part_index")]
        self.assertEqual([(part.start, part.end) for part in parts], [(1, 2), (2, 4)])
        self.assertTrue(all(part.reject_reason for part in parts))
        self.assertEqual([id(row[1]) for row in scored], [id(part) for part in parts])
        self.assertTrue(all(row[2] is pipeline._verify_speaker_span.return_value for row in scored))
        self.assertEqual([
            part.diagnostics["local_identity_evidence"][0]["state"] for part in parts
        ], ["unresolved", "unresolved"])

    def test_joined_target_keeps_each_parts_evidence_not_just_whole_score(self):
        pipeline = ExtractionPipeline.__new__(ExtractionPipeline)
        pipeline.options = SimpleNamespace(min_sentence_seconds=0.55)
        pipeline._verify_speaker_span = Mock(return_value=SimpleNamespace(
            accepted=True, tier="strong",
        ))
        pipeline._apply_speaker_match = Mock()
        verifier = Mock()
        verifier.exclusion_audit.return_value = None
        scored = []
        accepted, rejected = pipeline._recover_target_segments(
            TimeSpan(1, 4), [SpeakerBoundary(2, 0.1, 0.1, 1)],
            verifier, torch.zeros(4 * 16000), None, [], 0.7,
            lambda *_args: None, 1, 1,
            scored_parts=scored,
        )
        self.assertEqual(rejected, [])
        self.assertEqual(len(accepted), 1)
        evidence = accepted[0][0].diagnostics["local_identity_evidence"]
        self.assertEqual([row["span"] for row in evidence], [[1, 2], [2, 4]])
        self.assertEqual([row["state"] for row in evidence], ["target", "target"])
        self.assertEqual(len(scored), 3)
        self.assertEqual(
            (scored[-1][0].start, scored[-1][0].end),
            (1, 4),
        )
        self.assertIs(scored[-1][1].diagnostics, accepted[0][0].diagnostics)

    def test_failed_join_does_not_add_a_phantom_complete_candidate(self):
        pipeline = ExtractionPipeline.__new__(ExtractionPipeline)
        pipeline.options = SimpleNamespace(min_sentence_seconds=0.55)
        part_match = SimpleNamespace(accepted=True, tier="strong")
        joined_match = SimpleNamespace(accepted=False, tier="rejected")
        pipeline._verify_speaker_span = Mock(
            side_effect=[part_match, part_match, joined_match]
        )
        pipeline._apply_speaker_match = Mock()
        verifier = Mock()
        verifier.exclusion_audit.return_value = None
        verifier.promote_local_with_tertiary.return_value = None
        scored = []
        accepted, rejected = pipeline._recover_target_segments(
            TimeSpan(1, 4), [SpeakerBoundary(2, 0.1, 0.1, 1)],
            verifier, torch.zeros(4 * 16000), None, [], 0.7,
            lambda *_args: None, 1, 1,
            scored_parts=scored,
        )
        self.assertEqual(rejected, [])
        self.assertEqual(len(accepted), 2)
        self.assertEqual(len(scored), 2)
        self.assertEqual(
            [(row[0].start, row[0].end) for row in scored],
            [(1, 2), (2, 4)],
        )


if __name__ == "__main__":
    unittest.main()
