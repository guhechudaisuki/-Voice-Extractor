from __future__ import annotations

import unittest
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extractor.pipeline import ExtractionPipeline, PipelineOptions
from extractor.speaker import LocalSpeakerTurnSplitter, SpeakerBoundary
from extractor.types import CandidateSentence, TimeSpan


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
    def test_silence_lower_bound_merges_detector_jitter(self) -> None:
        spans = [
            TimeSpan(0.00, 1.00),
            TimeSpan(1.12, 2.00),
            TimeSpan(2.50, 3.00),
        ]
        merged = ExtractionPipeline._merge_vad_spans(spans, gap=0.20)
        self.assertEqual(merged, [TimeSpan(0.00, 2.00), TimeSpan(2.50, 3.00)])

    def test_pipeline_options_keep_public_silence_upper_bound(self) -> None:
        options = PipelineOptions(silence_min_seconds=0.25, silence_max_seconds=0.90)
        self.assertEqual(options.silence_split_seconds, 0.90)
        self.assertEqual(options.silence_max_seconds, 0.90)

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

    def test_short_local_side_is_kept_as_edge_evidence(self) -> None:
        spans = LocalSpeakerTurnSplitter._split_span_with_edge_evidence(
            TimeSpan(0.0, 3.0),
            [0.25, 2.50],
            minimum_turn_seconds=0.30,
            minimum_edge_seconds=0.20,
        )
        self.assertEqual(
            spans,
            [
                TimeSpan(0.0, 0.25),
                TimeSpan(0.25, 2.50),
                TimeSpan(2.50, 3.0),
            ],
        )

    def test_short_internal_side_is_not_dropped(self) -> None:
        spans = LocalSpeakerTurnSplitter._split_span_with_edge_evidence(
            TimeSpan(0.0, 4.0),
            [1.50, 1.72],
            minimum_turn_seconds=0.30,
            minimum_edge_seconds=0.20,
        )
        self.assertEqual(
            spans,
            [
                TimeSpan(0.0, 1.50),
                TimeSpan(1.50, 1.72),
                TimeSpan(1.72, 4.0),
            ],
        )


class LocalRecoveryGateTests(unittest.TestCase):
    @staticmethod
    def _match(
        *,
        tier: str,
        primary: float,
        secondary: float,
        primary_reference: float = 0.55,
        secondary_reference: float = 0.52,
        paired: float = 0.50,
    ):
        return SimpleNamespace(
            tier=tier,
            paired_reference_median=paired,
            primary=SimpleNamespace(
                score=primary,
                reference_max_score=primary_reference,
                window_p20_score=0.50,
                window_vote_ratio=0.50,
            ),
            secondary=SimpleNamespace(
                score=secondary,
                reference_max_score=secondary_reference,
                window_p20_score=0.50,
                window_vote_ratio=0.50,
            ),
        )

    def test_short_recovery_side_needs_both_models(self) -> None:
        match = self._match(tier="recall", primary=0.585, secondary=0.595)
        self.assertTrue(
            ExtractionPipeline._target_side_recovery_support(match, 1.60, 0.684)
        )
        weak_secondary = self._match(tier="recall", primary=0.62, secondary=0.44)
        self.assertFalse(
            ExtractionPipeline._target_side_recovery_support(
                weak_secondary, 1.60, 0.684
            )
        )

    def test_complete_or_rejected_tiers_are_not_short_side_support(self) -> None:
        strong = self._match(tier="strong", primary=0.70, secondary=0.65)
        self.assertFalse(
            ExtractionPipeline._target_side_recovery_support(strong, 1.60, 0.684)
        )
        rejected = self._match(tier="rejected", primary=0.60, secondary=0.60)
        self.assertFalse(
            ExtractionPipeline._target_side_recovery_support(rejected, 1.60, 0.684)
        )

    def test_subsentence_edge_requires_two_model_direct_evidence(self) -> None:
        edge = self._match(
            tier="rejected",
            primary=0.56,
            secondary=0.53,
            primary_reference=0.50,
            secondary_reference=0.45,
            paired=0.48,
        )
        self.assertTrue(
            ExtractionPipeline._short_edge_recovery_support(edge, 0.35, 0.684)
        )
        weak_secondary = self._match(
            tier="rejected",
            primary=0.60,
            secondary=0.42,
            primary_reference=0.55,
            secondary_reference=0.35,
            paired=0.43,
        )
        self.assertFalse(
            ExtractionPipeline._short_edge_recovery_support(
                weak_secondary, 0.35, 0.684
            )
        )

    def test_subsentence_edge_cannot_be_exported_by_duration_alone(self) -> None:
        edge = self._match(
            tier="short_strong",
            primary=0.90,
            secondary=0.90,
            primary_reference=0.90,
            secondary_reference=0.90,
            paired=0.90,
        )
        self.assertFalse(
            ExtractionPipeline._short_edge_recovery_support(edge, 0.15, 0.684)
        )


if __name__ == "__main__":
    unittest.main()
