from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extractor.pipeline import ExtractionPipeline, PipelineOptions
from extractor.types import CandidateSentence


def make_pipeline(marked=()):
    return ExtractionPipeline(
        PipelineOptions(user_excluded_spans=marked)
    )


def fragment(start: float, end: float) -> CandidateSentence:
    candidate = CandidateSentence(start, end, "")
    candidate.reject_reason = "声纹匹配不足"
    candidate.speaker_score = 0.55
    candidate.speaker_threshold = 0.7
    return candidate


class ClassifierTargetRescueTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = make_pipeline()
        self.wave = torch.zeros(16000 * 20)

    def run_rescue(self, accepted, rejected, score):
        return self.pipeline._classifier_target_rescue(
            accepted,
            rejected,
            self.wave,
            lambda _v, _m: None,
            scorer=lambda _span, _wave: score,
            threshold=2.0,
        )

    def test_confident_fragment_is_accepted(self):
        accepted = [CandidateSentence(10.0, 14.0, "", accepted=True)]
        hit = fragment(14.0, 15.5)
        rejected = [hit]
        rescued = self.run_rescue(accepted, rejected, 3.0)
        self.assertEqual(rescued, 1)
        self.assertIn(hit, accepted)
        self.assertNotIn(hit, rejected)
        self.assertEqual(hit.reject_reason, "")
        self.assertEqual(hit.diagnostics["classifier_target_rescue"]["mean_score"], 3.0)

    def test_below_threshold_fragment_stays_rejected(self):
        hit = fragment(14.0, 15.5)
        accepted: list[CandidateSentence] = []
        rejected = [hit]
        rescued = self.run_rescue(accepted, rejected, 1.0)
        self.assertEqual(rescued, 0)
        self.assertEqual(accepted, [])
        self.assertEqual(hit.reject_reason, "声纹匹配不足")

    def test_user_marked_span_is_never_rescued(self):
        pipeline = make_pipeline(marked=[(14.0, 15.5)])
        hit = fragment(14.0, 15.5)
        accepted: list[CandidateSentence] = []
        rejected = [hit]
        rescued = pipeline._classifier_target_rescue(
            accepted,
            rejected,
            self.wave,
            lambda _v, _m: None,
            scorer=lambda _span, _wave: 3.0,
            threshold=2.0,
        )
        self.assertEqual(rescued, 0)
        self.assertEqual(hit.reject_reason, "声纹匹配不足")

    def test_structural_reject_is_never_rescued(self):
        hit = fragment(14.0, 15.5)
        hit.diagnostics["structural_hard_reject"] = True
        accepted: list[CandidateSentence] = []
        rejected = [hit]
        rescued = self.run_rescue(accepted, rejected, 3.0)
        self.assertEqual(rescued, 0)

    def test_fragment_inside_accepted_clip_is_never_rescued(self):
        accepted = [CandidateSentence(116.02, 122.51, "", accepted=True)]
        duplicate = fragment(119.62, 121.52)
        rejected = [duplicate]
        rescued = self.run_rescue(accepted, rejected, 3.0)
        self.assertEqual(rescued, 0)
        self.assertEqual(accepted, [CandidateSentence(116.02, 122.51, "", accepted=True)])
        self.assertEqual(duplicate.reject_reason, "声纹匹配不足")

    def test_fragment_partially_overlapping_accepted_is_not_rescued(self):
        accepted = [CandidateSentence(10.0, 14.0, "", accepted=True)]
        overlap = fragment(13.5, 15.5)
        rejected = [overlap]
        rescued = self.run_rescue(accepted, rejected, 3.0)
        self.assertEqual(rescued, 0)

    def test_short_fragment_below_export_floor_is_skipped(self):
        hit = fragment(14.0, 14.9)
        accepted: list[CandidateSentence] = []
        rejected = [hit]
        rescued = self.run_rescue(accepted, rejected, 3.0)
        self.assertEqual(rescued, 0)


if __name__ == "__main__":
    unittest.main()
