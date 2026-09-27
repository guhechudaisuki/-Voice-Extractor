from __future__ import annotations

import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
from check_episode_review import check, covered_seconds  # noqa: E402


def manifest(*spans: tuple[float, float]) -> dict:
    return {"sentences": [
        {"start": start, "end": end, "accepted": True}
        for start, end in spans
    ]}


class EpisodeReviewRegressionTests(unittest.TestCase):
    def test_entire_other_is_rejected_even_if_renumbered(self) -> None:
        review = {"flagged_outputs": [
            {"index": 28, "span": [10.0, 12.0], "kind": "entire_other"}
        ]}
        report = check(manifest((11.0, 11.5)), review, manifest((0.0, 1.0)))
        self.assertEqual(report["known_bad_blocked"], 0)
        self.assertFalse(report["cases"][0]["passed"])

    def test_mixed_original_interval_must_not_survive_whole(self) -> None:
        review = {"flagged_outputs": [
            {"index": 25, "span": [10.0, 15.0], "kind": "mixed"}
        ]}
        report = check(manifest((10.0, 15.0)), review, manifest((20.0, 21.0)))
        self.assertEqual(report["known_bad_blocked"], 0)
        repaired = check(manifest((10.0, 11.0), (13.0, 15.0)), review,
                         manifest((20.0, 21.0)))
        self.assertEqual(repaired["known_bad_blocked"], 1)

    def test_baseline_coverage_uses_union_without_double_counting(self) -> None:
        self.assertAlmostEqual(
            covered_seconds((0.0, 2.0), [(0.0, 1.5), (1.0, 2.0)]), 2.0,
        )
        report = check(manifest((0.0, 1.5), (1.0, 2.0)),
                       {"flagged_outputs": []}, manifest((0.0, 2.0)))
        self.assertEqual(report["baseline_preserved"], 1)


if __name__ == "__main__":
    unittest.main()
