from __future__ import annotations

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extractor.nextgen.purity_consensus import (
    eligible_identity_island,
    local_other_witnesses,
    target_runs,
    whole_identity_conflict,
)


class PurityConsensusTests(unittest.TestCase):
    def test_short_target_does_not_inherit_local_exclusion_false_positive(self) -> None:
        # Both production models can vote for an exclusion on a true short
        # target syllable; independent VA raw/stem evidence must concur.
        margins = {"char_stem": -0.12, "char_raw": -0.01,
                   "va_stem": -0.11, "va_raw": 0.01}
        self.assertEqual(local_other_witnesses(
            margins, {"excluded_role_rejected": True,
                      "excluded_primary_margin": -0.16}), ())

    def test_independent_island_witnesses_do_not_use_parent_score(self) -> None:
        margins = {"char_stem": -0.03, "char_raw": 0.03,
                   "va_stem": -0.09, "va_raw": -0.14}
        self.assertEqual(local_other_witnesses(
            margins, {"excluded_role_rejected": True,
                      "excluded_primary_margin": -0.07}),
            ())
        margins["char_raw"] = -0.01
        self.assertEqual(local_other_witnesses(
            margins, {"excluded_role_rejected": True,
                      "excluded_primary_margin": -0.07}),
            ("domain_all_four_and_production_primary_negative",))

    def test_explicit_production_exclusion_counts_even_if_one_margin_ties(self) -> None:
        margins = {"char_stem": -0.24, "char_raw": -0.20,
                   "va_stem": -0.21, "va_raw": -0.23}
        self.assertEqual(local_other_witnesses(
            margins, {"excluded_role_rejected": True,
                      "excluded_primary_margin": 0.019}),
            ("domain_all_four_and_production_exclusion_rejected",))
        self.assertEqual(local_other_witnesses(
            margins, {"excluded_role_rejected": False,
                      "excluded_primary_margin": 0.019}), ())

    def test_exactly_200_ms_is_not_lost_to_float_roundoff(self) -> None:
        self.assertTrue(eligible_identity_island(934.835, 935.035))
        self.assertFalse(eligible_identity_island(1.0, 1.19))

    def test_whole_conflict_is_abstention_not_other_person_claim(self) -> None:
        self.assertTrue(whole_identity_conflict(
            {"char_stem": 0.0077, "char_raw": 0.108},
            {"char_stem": -0.002, "char_raw": 0.075}, 0.008,
        ))
        self.assertFalse(whole_identity_conflict(
            {"char_stem": 0.03, "char_raw": 0.036},
            {"char_stem": 0.03, "char_raw": -0.01}, 0.11,
        ))
        self.assertFalse(whole_identity_conflict(
            {"char_stem": 0.15, "char_raw": -0.002},
            {"char_stem": 0.15, "char_raw": 0.017}, 0.01,
        ))
        self.assertFalse(whole_identity_conflict(
            {"char_stem": 0.007, "char_raw": 0.04},
            {"char_stem": -0.002, "char_raw": 0.05}, None,
        ))

    def test_target_runs_do_not_bridge_suspect_island(self) -> None:
        islands = [
            {"span": [0.0, 1.5], "state": "target_supported"},
            {"span": [1.8, 2.1], "state": "unresolved"},
            {"span": [2.4, 4.0], "state": "target_supported"},
        ]
        self.assertEqual(target_runs(islands, {(1.8, 2.1)},
                                     maximum_gap_seconds=0.85),
                         [(0.0, 1.5), (2.4, 4.0)])


if __name__ == "__main__":
    unittest.main()
