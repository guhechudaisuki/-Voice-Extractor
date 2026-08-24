from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extractor.pipeline import ExtractionPipeline
from extractor.speaker import LocalSpeakerTurnSplitter, SpeakerBoundary
from extractor.types import CandidateSentence


def candidate(*, coverage: float, tier: str) -> CandidateSentence:
    value = CandidateSentence(0.0, 2.0, "")
    value.diagnostics.update({"target_coverage": coverage, "speaker_tier": tier})
    return value


class FinalMultimodelExclusionTests(unittest.TestCase):
    def test_full_coverage_turn_does_not_load_third_model(self) -> None:
        value = candidate(coverage=1.0, tier="strong")
        exclusion = {"excluded_primary_vote": True, "excluded_secondary_vote": False}
        self.assertFalse(ExtractionPipeline._needs_final_multimodel_exclusion(value, exclusion))

    def test_empty_target_region_uses_primary_vote(self) -> None:
        value = candidate(coverage=0.0, tier="rejected")
        exclusion = {"excluded_primary_vote": True, "excluded_secondary_vote": False}
        self.assertTrue(ExtractionPipeline._needs_final_multimodel_exclusion(value, exclusion))
        self.assertTrue(
            ExtractionPipeline._needs_final_multimodel_exclusion(
                value,
                {**exclusion, "excluded_wespeaker_vote": True},
                require_third_model=True,
            )
        )

    def test_recall_edge_requires_low_coverage_and_third_vote(self) -> None:
        value = candidate(coverage=0.54, tier="recall")
        exclusion = {"excluded_primary_vote": True, "excluded_secondary_vote": False}
        self.assertTrue(ExtractionPipeline._needs_final_multimodel_exclusion(value, exclusion))
        self.assertFalse(
            ExtractionPipeline._needs_final_multimodel_exclusion(
                value,
                exclusion,
                require_third_model=True,
            )
        )
        self.assertTrue(
            ExtractionPipeline._needs_final_multimodel_exclusion(
                value,
                {"excluded_primary_vote": False, "excluded_secondary_vote": True, "excluded_wespeaker_vote": True},
                require_third_model=True,
            )
        )

    def test_recall_with_enough_target_coverage_is_unchanged(self) -> None:
        value = candidate(coverage=0.70, tier="recall")
        exclusion = {"excluded_primary_vote": True, "excluded_secondary_vote": False}
        self.assertFalse(ExtractionPipeline._needs_final_multimodel_exclusion(value, exclusion))

    def test_boundary_risk_opens_third_model_audit_for_strong_turn(self) -> None:
        value = candidate(coverage=1.0, tier="strong")
        value.diagnostics["locator_boundary_count"] = 3
        exclusion = {"excluded_primary_vote": False, "excluded_secondary_vote": False}
        self.assertFalse(
            ExtractionPipeline._needs_final_multimodel_exclusion(value, exclusion)
        )
        self.assertTrue(
            ExtractionPipeline._needs_final_multimodel_exclusion(
                value,
                {**exclusion, "excluded_wespeaker_vote": True},
            )
        )
        self.assertTrue(
            ExtractionPipeline._needs_final_multimodel_exclusion(
                value,
                {**exclusion, "excluded_wespeaker_vote": True},
                require_third_model=True,
            )
        )

    def test_third_vote_alone_is_not_a_veto_on_clean_turn(self) -> None:
        value = candidate(coverage=1.0, tier="strong")
        exclusion = {
            "excluded_primary_vote": False,
            "excluded_secondary_vote": False,
            "excluded_wespeaker_vote": True,
        }
        self.assertFalse(
            ExtractionPipeline._needs_final_multimodel_exclusion(
                value, exclusion, require_third_model=True
            )
        )


class AnimeRecoveryGateTests(unittest.TestCase):
    def test_anime_recovery_requires_episode_consensus(self) -> None:
        diagnostics = {
            "multi_model_base_consensus": False,
            "multi_model_common_anchor_count": 0,
            "multi_model_wespeaker_support": False,
            "multi_model_continuity_ratio": 1.0,
        }
        self.assertFalse(ExtractionPipeline._anime_recovery_base_gate(diagnostics))

    def test_anime_recovery_rejects_direct_exclusion_conflict(self) -> None:
        diagnostics = {
            "multi_model_base_consensus": True,
            "multi_model_common_anchor_count": 2,
            "multi_model_wespeaker_support": True,
            "multi_model_continuity_ratio": 0.85,
            "excluded_primary_direct_margin": -0.01,
            "excluded_secondary_direct_margin": 0.12,
        }
        self.assertFalse(ExtractionPipeline._anime_recovery_base_gate(diagnostics))

    def test_anime_recovery_accepts_independent_clean_evidence(self) -> None:
        diagnostics = {
            "multi_model_base_consensus": True,
            "multi_model_common_anchor_count": 2,
            "multi_model_wespeaker_support": True,
            "multi_model_continuity_ratio": 0.75,
            "excluded_primary_direct_margin": 0.04,
            "excluded_secondary_direct_margin": 0.08,
        }
        self.assertTrue(ExtractionPipeline._anime_recovery_base_gate(diagnostics))


class MultiscaleBoundaryTests(unittest.TestCase):
    def test_serialized_boundary_requires_cross_model_or_legacy_corroboration(self) -> None:
        self.assertTrue(
            ExtractionPipeline._is_reliable_boundary_record(
                {
                    "time": 10.0,
                    "scale_votes": 2,
                    "primary_similarity": 0.44,
                    "secondary_similarity": 0.10,
                }
            )
        )
        self.assertFalse(
            ExtractionPipeline._is_reliable_boundary_record(
                {
                    "time": 10.0,
                    "scale_votes": 1,
                    "confidence": 0.62,
                    "primary_similarity": 0.42,
                    "secondary_similarity": 0.20,
                    "primary_drop": 0.04,
                    "secondary_drop": 0.01,
                }
            )
        )
        self.assertTrue(
            ExtractionPipeline._is_reliable_boundary_record(
                {
                    "time": 10.0,
                    "confidence": 0.84,
                    "primary_similarity": 0.43,
                    "secondary_similarity": 0.12,
                    "primary_drop": 0.08,
                    "secondary_drop": 0.04,
                }
            )
        )

    def test_candidate_boundary_times_ignore_edge_and_weak_records(self) -> None:
        value = CandidateSentence(1.0, 5.0, "")
        value.diagnostics["multi_model_anchor_boundaries"] = [
            {
                "time": 1.20,
                "scale_votes": 2,
                "primary_similarity": 0.40,
                "secondary_similarity": 0.10,
            },
            {
                "time": 3.00,
                "scale_votes": 2,
                "primary_similarity": 0.40,
                "secondary_similarity": 0.10,
            },
            {
                "time": 4.80,
                "scale_votes": 2,
                "primary_similarity": 0.40,
                "secondary_similarity": 0.10,
            },
        ]
        self.assertEqual(
            ExtractionPipeline._candidate_boundary_times(value),
            [3.0],
        )

    def test_only_cross_scale_boundary_survives(self) -> None:
        observed = [
            (
                0.70,
                SpeakerBoundary(
                    10.00, 0.42, 0.18, 0.80, 0.10, 0.08
                ),
            ),
            (
                0.90,
                SpeakerBoundary(
                    10.22, 0.31, 0.04, 0.95, 0.20, 0.15
                ),
            ),
            (
                0.70,
                SpeakerBoundary(
                    20.00, 0.20, 0.03, 1.00, 0.30, 0.20
                ),
            ),
        ]
        boundaries = LocalSpeakerTurnSplitter._cluster_multiscale_boundaries(
            observed,
            cluster_seconds=0.40,
            minimum_context_votes=2,
        )
        self.assertEqual(len(boundaries), 1)
        self.assertEqual(boundaries[0].scale_votes, 2)
        self.assertEqual(boundaries[0].scale_contexts, (0.7, 0.9))
        self.assertTrue(ExtractionPipeline._is_structural_boundary(boundaries[0]))

    def test_single_scale_boundary_is_not_structural(self) -> None:
        boundary = SpeakerBoundary(10.0, 0.22, 0.18, 0.8, 0.2, 0.1)
        self.assertFalse(ExtractionPipeline._is_structural_boundary(boundary))


if __name__ == "__main__":
    unittest.main()
