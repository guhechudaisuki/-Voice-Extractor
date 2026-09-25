from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))

from evaluate_utterance_lattice import assess_reachability


class LatticeEvaluationTests(unittest.TestCase):
    def test_known_change_candidate_is_counted_as_risk_not_a_success(self):
        cases = {"cases": [
            {"id": "target", "left": [1.0, 1.5], "right": [1.8, 2.4],
             "same_target": True},
            {"id": "mixed", "left": [3.0, 3.5], "right": [3.7, 4.0],
             "same_target": False},
        ]}
        lattice = {"analysis_sample_rate": 16000, "proposals": [
            {"source_id": "a", "start_sample": 16000,
             "end_sample": 38400, "island_indexes": [0, 1],
             "identity_state": "unresolved"},
            {"source_id": "b", "start_sample": 48000,
             "end_sample": 64000, "island_indexes": [2, 3],
             "identity_state": "unresolved"},
        ]}
        report = assess_reachability(cases, lattice)
        self.assertEqual(report["positive_candidates_found"], 1)
        self.assertEqual(report["positive_reviewed_spans_enclosed"], 1)
        self.assertEqual(report["known_change_candidates_found"], 1)
        self.assertIn("contamination risks", report["warning"])

    def test_tolerance_match_does_not_claim_complete_reviewed_span(self):
        cases = {"cases": [{
            "id": "short_tail", "left": [1.0, 1.4],
            "right": [1.7, 2.4], "same_target": True,
        }]}
        lattice = {"analysis_sample_rate": 16000, "proposals": [{
            "source_id": "short", "start_sample": 16000,
            "end_sample": 37600, "island_indexes": [0],
            "identity_state": "unresolved",
        }]}
        report = assess_reachability(cases, lattice)
        self.assertEqual(report["positive_candidates_found"], 1)
        self.assertEqual(report["positive_reviewed_spans_enclosed"], 0)
        self.assertEqual(
            report["results"][0]["candidate"]["missing_end_seconds"], 0.05
        )

    def test_invalid_sample_index_is_rejected(self):
        with self.assertRaises(ValueError):
            assess_reachability(
                {"cases": []},
                {"analysis_sample_rate": 16000, "proposals": [{
                    "source_id": "bad", "start_sample": 1.5,
                    "end_sample": 16000,
                }]},
            )

    def test_duplicate_candidates_are_rejected(self):
        with self.assertRaises(ValueError):
            assess_reachability(
                {"cases": []},
                {"analysis_sample_rate": 16000,
                 "proposals": [{"source_id": "a"}, {"source_id": "a"}]},
            )


if __name__ == "__main__":
    unittest.main()
