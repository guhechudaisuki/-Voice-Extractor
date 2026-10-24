"""P2 must count reviewed clips, preserve uncertainty, and report mixed spans."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.cluster_report import load_review, summarize
from extractor.nextgen.identity_cluster import AssignResult, TurnEvidence


def report(spans, membership, review, *, outliers=(), target_id=0):
    turns = [
        TurnEvidence(i, start, end, end - start, np.ones(2), np.ones(2))
        for i, (start, end) in enumerate(spans)
    ]
    result = AssignResult(
        clusters=(), target_cluster_id=target_id, membership=tuple(membership),
        cluster_of=tuple(0 if accepted or i in outliers else 1
                         for i, accepted in enumerate(membership)),
        reason="target_cluster_found" if target_id is not None else "no_cluster",
        outliers=outliers,
    )
    return summarize(result, turns, review)["review_check"]


def wrong(ordinal, start, end):
    return {"ordinal": ordinal, "span": (start, end), "kind": "entire_other"}


class ClusterReportTests(unittest.TestCase):
    def test_two_review_clips_in_one_turn_remain_two_errors(self):
        check = report([(0, 10)], [True], [wrong(11, 1, 3), wrong(12, 6, 9)])
        self.assertEqual(check["whole_wrong_count"], 2)
        self.assertEqual(check["whole_wrong_inside_target"], 2)
        self.assertEqual([row["ordinal"] for row in check["details"]["entire_other"]], [11, 12])

    def test_one_clip_spanning_two_turns_counts_once(self):
        check = report([(0, 2), (2, 4)], [False, True], [wrong(1, 0, 4)])
        self.assertEqual(check["whole_wrong_count"], 1)
        self.assertEqual(check["whole_wrong_inside_target"], 1)

    def test_uncovered_wrong_audio_is_unresolved(self):
        check = report([(0, 1)], [False], [wrong(1, 0, 4)])
        self.assertEqual(check["whole_wrong_unresolved"], 1)
        self.assertEqual(check["whole_wrong_elsewhere"], 0)
        self.assertEqual(check["verdict"], "incomplete_review_coverage")

    def test_outliers_and_other_clusters_are_separate(self):
        check = report([(0, 2), (3, 5)], [False, False],
                       [wrong(1, 0, 2), wrong(2, 3, 5)], outliers=(0,))
        self.assertEqual(check["whole_wrong_caught_by_outlier_rule"], 1)
        self.assertEqual(check["whole_wrong_elsewhere"], 1)

    def test_mixed_clip_checks_only_confirmed_negative_span(self):
        review = [{"ordinal": 8, "span": (0, 4), "negative_span": (2, 4), "kind": "mixed"}]
        check = report([(0, 2), (2, 4)], [True, False], review)
        self.assertEqual(check["mixed_clips_total"], 1)
        self.assertEqual(check["mixed_clips_in_target"], 0)

    def test_abstaining_is_not_success(self):
        mixed = {"ordinal": 8, "span": (0, 2), "negative_span": (1, 2), "kind": "mixed"}
        check = report([(0, 2)], [False], [wrong(1, 0, 2), mixed], target_id=None)
        self.assertEqual(check["verdict"], "no_target_cluster")
        self.assertEqual(check["whole_wrong_withheld_no_target"], 1)
        self.assertEqual(check["whole_wrong_elsewhere"], 0)
        self.assertEqual(check["mixed_clips_withheld_no_target"], 1)
        self.assertEqual(check["details"]["entire_other"][0]["status"], "withheld_no_target")

    def test_missing_requested_review_fails(self):
        with self.assertRaises(FileNotFoundError):
            load_review(ROOT / "tests" / "missing_cluster_review.json")


if __name__ == "__main__":
    unittest.main()
