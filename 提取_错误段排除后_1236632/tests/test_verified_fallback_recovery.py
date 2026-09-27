from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extractor.pipeline import ExtractionPipeline, PipelineOptions
from extractor.types import CandidateSentence


def verified_part(start: float, end: float) -> CandidateSentence:
    """A locally verified recovery part, the way _recover_target_segments emits it."""
    candidate = CandidateSentence(start, end, "")
    candidate.speaker_score = 0.75
    candidate.speaker_threshold = 0.7
    candidate.diagnostics.update(
        {
            "local_boundary_recovery": True,
            "local_identity_evidence": [
                {
                    "span": [start, end],
                    "state": "target",
                    "method": "boundary_part_verification",
                }
            ],
        }
    )
    return candidate


class VerifiedFallbackTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = ExtractionPipeline(PipelineOptions())

    def test_evicted_core_is_queued_and_certified(self):
        core = verified_part(11.5, 13.3)
        wider = CandidateSentence(
            10.0, 13.3, "", diagnostics={"speaker_tier": "strong"}
        )
        self.pipeline._attach_verified_fallbacks(wider, [core])
        accepted = [wider]
        added = self.pipeline._collect_verified_fallbacks(accepted)
        self.assertEqual(added, 1)
        fallback = accepted[1]
        self.assertEqual((fallback.start, fallback.end), (11.5, 13.3))
        self.assertEqual(fallback.speaker_score, 0.75)
        self.assertTrue(fallback.diagnostics["verified_subspan_fallback"])
        self.assertEqual(fallback.diagnostics["fallback_parent_span"], [10.0, 13.3])
        self.assertTrue(
            ExtractionPipeline._install_acoustic_identity_certificate(fallback)
        )
        self.assertEqual(
            fallback.diagnostics["acoustic_identity_audit"]["method"],
            "verified_boundary_part",
        )
        self.assertIsNone(
            ExtractionPipeline._stt_fragment_identity_veto(fallback, 3)
        )

    def test_fallback_export_removed_when_parent_survives(self):
        fallback = CandidateSentence(11.5, 13.3, "text", accepted=True)
        fallback.diagnostics.update(
            {
                "verified_subspan_fallback": True,
                "fallback_parent_span": [10.0, 13.3],
            }
        )
        parent = CandidateSentence(10.0, 13.3, "parent text", accepted=True)
        accepted = [parent, fallback]
        rejected: list[CandidateSentence] = []
        removed = ExtractionPipeline._dedup_verified_fallback_exports(
            accepted, rejected
        )
        self.assertEqual(removed, 1)
        self.assertEqual(accepted, [parent])
        self.assertEqual(rejected, [fallback])
        self.assertFalse(fallback.accepted)
        self.assertIn("完整父回合", fallback.reject_reason)

    def test_fallback_export_kept_when_parent_withheld(self):
        fallback = CandidateSentence(11.5, 13.3, "text", accepted=True)
        fallback.diagnostics.update(
            {
                "verified_subspan_fallback": True,
                "fallback_parent_span": [10.0, 13.3],
            }
        )
        accepted = [fallback]
        rejected: list[CandidateSentence] = []
        removed = ExtractionPipeline._dedup_verified_fallback_exports(
            accepted, rejected
        )
        self.assertEqual(removed, 0)
        self.assertEqual(accepted, [fallback])
        self.assertEqual(rejected, [])

    def test_certificate_requires_exact_target_evidence(self):
        unresolved = CandidateSentence(11.5, 13.3, "")
        unresolved.diagnostics.update(
            {
                "local_boundary_recovery": True,
                "local_identity_evidence": [
                    {
                        "span": [11.5, 13.3],
                        "state": "unresolved",
                        "method": "boundary_part_verification",
                    }
                ],
            }
        )
        self.assertFalse(
            ExtractionPipeline._install_acoustic_identity_certificate(unresolved)
        )

        mismatched = CandidateSentence(11.5, 13.3, "")
        mismatched.diagnostics.update(
            {
                "local_boundary_recovery": True,
                "local_identity_evidence": [
                    {
                        "span": [11.7, 13.3],
                        "state": "target",
                        "method": "boundary_part_verification",
                    }
                ],
            }
        )
        self.assertFalse(
            ExtractionPipeline._install_acoustic_identity_certificate(mismatched)
        )
        self.assertNotIn("acoustic_identity_audit", mismatched.diagnostics)

        edge_only = verified_part(11.5, 13.3)
        edge_only.diagnostics["local_edge_only"] = True
        self.assertFalse(
            ExtractionPipeline._install_acoustic_identity_certificate(edge_only)
        )

    def test_certificate_accepts_merged_target_parts(self):
        merged = CandidateSentence(11.5, 15.0, "")
        merged.diagnostics.update(
            {
                "local_boundary_recovery": True,
                "local_target_parts_merged": 2,
                "local_identity_evidence": [
                    {
                        "span": [11.5, 13.3],
                        "state": "target",
                        "method": "boundary_part_verification",
                    },
                    {
                        "span": [13.3, 15.0],
                        "state": "target",
                        "method": "boundary_part_verification",
                    },
                ],
            }
        )
        self.assertTrue(
            ExtractionPipeline._install_acoustic_identity_certificate(merged)
        )
        self.assertEqual(
            merged.diagnostics["acoustic_identity_audit"]["method"],
            "verified_boundary_part",
        )

    def test_collect_skips_fallback_overlapping_unrelated_accepted(self):
        core = verified_part(11.5, 13.3)
        wider = CandidateSentence(10.0, 13.3, "", diagnostics={})
        self.pipeline._attach_verified_fallbacks(wider, [core])
        other = CandidateSentence(12.0, 14.0, "", diagnostics={})
        accepted = [wider, other]
        added = self.pipeline._collect_verified_fallbacks(accepted)
        self.assertEqual(added, 0)
        self.assertEqual(len(accepted), 2)

    def test_nested_eviction_chain_is_followed(self):
        core = verified_part(11.5, 13.3)
        middle = CandidateSentence(10.0, 13.3, "", diagnostics={})
        self.pipeline._attach_verified_fallbacks(middle, [core])
        outer = CandidateSentence(9.0, 13.3, "", diagnostics={})
        self.pipeline._attach_verified_fallbacks(outer, [middle])
        accepted = [outer]
        added = self.pipeline._collect_verified_fallbacks(accepted)
        self.assertEqual(added, 2)
        spans = [(item.start, item.end) for item in accepted[1:]]
        self.assertEqual(spans, [(10.0, 13.3), (11.5, 13.3)])

    def test_withheld_parent_fallback_is_still_collected(self):
        # The final island consensus withholds a mixed parent after the
        # fallback was registered; the verified core must not be lost with it.
        core = verified_part(526.81, 528.61)
        parent = CandidateSentence(525.31, 528.61, "", diagnostics={})
        self.pipeline._attach_verified_fallbacks(parent, [core])
        accepted: list[CandidateSentence] = []
        added = self.pipeline._collect_verified_fallbacks(accepted)
        self.assertEqual(added, 1)
        self.assertEqual((accepted[0].start, accepted[0].end), (526.81, 528.61))
        self.assertTrue(accepted[0].diagnostics["verified_subspan_fallback"])
        self.assertEqual(accepted[0].diagnostics["fallback_parent_span"], [525.31, 528.61])

    def test_withheld_parent_same_span_fallback_stays_withheld(self):
        # A fallback covering the withheld parent's exact audio carries the
        # same final-review evidence; resurrecting it would undo the veto.
        same_span = CandidateSentence(1242.99, 1245.46, "", diagnostics={})
        parent = CandidateSentence(1242.99, 1245.46, "", diagnostics={})
        self.pipeline._attach_verified_fallbacks(parent, [same_span])
        accepted: list[CandidateSentence] = []
        added = self.pipeline._collect_verified_fallbacks(accepted)
        self.assertEqual(added, 0)

    def test_surviving_parent_same_span_fallback_still_collected(self):
        # For a surviving parent the fallback is only a duplicate export for
        # one STT pass; the post-STT dedup removes it, so collection is fine.
        same_span = CandidateSentence(10.0, 13.3, "", diagnostics={})
        parent = CandidateSentence(10.0, 13.3, "", diagnostics={})
        self.pipeline._attach_verified_fallbacks(parent, [same_span])
        accepted = [parent]
        added = self.pipeline._collect_verified_fallbacks(accepted)
        self.assertEqual(added, 1)

    def test_withheld_parent_fallback_skipped_when_covered_elsewhere(self):
        core = verified_part(11.5, 13.3)
        parent = CandidateSentence(10.0, 13.3, "", diagnostics={})
        self.pipeline._attach_verified_fallbacks(parent, [core])
        survivor = CandidateSentence(10.0, 13.3, "", diagnostics={})
        accepted = [survivor]
        added = self.pipeline._collect_verified_fallbacks(accepted)
        self.assertEqual(added, 0)
        self.assertEqual(accepted, [survivor])


if __name__ == "__main__":
    unittest.main()
