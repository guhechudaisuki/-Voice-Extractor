from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extractor.pipeline import ExtractionPipeline, PipelineOptions
from extractor.types import CandidateSentence, TimeSpan


def flagged_spans_file_shape(spans):
    """The JSON input the CLI hands over: a list of [start, end] pairs."""
    return [[float(start), float(end)] for start, end in spans]


class UserMarkedExclusionTests(unittest.TestCase):
    def test_overlapping_turns_are_deleted_with_ledger_reason(self):
        keep = CandidateSentence(10.0, 12.0, "")
        hit = CandidateSentence(20.0, 23.0, "")
        accepted = [keep, hit]
        rejected: list[CandidateSentence] = []
        removed = ExtractionPipeline._apply_user_marked_exclusions(
            accepted,
            rejected,
            [TimeSpan(22.0, 24.0)],
        )
        self.assertEqual(removed, 1)
        self.assertEqual(accepted, [keep])
        self.assertEqual(rejected, [hit])
        self.assertEqual(hit.reject_reason, "用户标记排除区间，已按标记删除")
        self.assertTrue(hit.diagnostics["user_marked_exclusion"])

    def test_fallback_child_inside_marked_span_is_also_deleted(self):
        child = CandidateSentence(526.81, 528.61, "")
        child.diagnostics["verified_subspan_fallback"] = True
        accepted = [child]
        rejected: list[CandidateSentence] = []
        removed = ExtractionPipeline._apply_user_marked_exclusions(
            accepted,
            rejected,
            [TimeSpan(525.31, 528.61)],
        )
        self.assertEqual(removed, 1)
        self.assertEqual(accepted, [])

    def test_empty_marks_and_clean_turns_are_untouched(self):
        one = CandidateSentence(1.0, 2.0, "")
        two = CandidateSentence(3.0, 4.0, "")
        accepted = [one, two]
        rejected: list[CandidateSentence] = []
        self.assertEqual(
            ExtractionPipeline._apply_user_marked_exclusions(
                accepted, rejected, [],
            ),
            0,
        )
        self.assertEqual(
            ExtractionPipeline._apply_user_marked_exclusions(
                accepted, rejected, [TimeSpan(100.0, 101.0)],
            ),
            0,
        )
        self.assertEqual(accepted, [one, two])
        self.assertEqual(rejected, [])

    def test_options_normalize_marked_spans(self):
        options = PipelineOptions(
            user_excluded_spans=flagged_spans_file_shape(
                [(413.0, 418.05), (10, 10)]
            )
        )
        self.assertEqual(
            options.user_excluded_spans, ((413.0, 418.05),)
        )
        self.assertTrue(options.experimental_final_island_consensus)


if __name__ == "__main__":
    unittest.main()
