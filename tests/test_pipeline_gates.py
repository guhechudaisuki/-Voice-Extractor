from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extractor.pipeline import ExtractionPipeline
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


if __name__ == "__main__":
    unittest.main()
