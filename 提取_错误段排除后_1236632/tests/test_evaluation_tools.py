from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))

from audit_stage_reachability import summarize  # noqa: E402
from build_review_queue import build_queue  # noqa: E402
from check_join_cases import assess as assess_join_cases  # noqa: E402
from probe_adjacent_pairs import parse_pair  # noqa: E402
from replay_subtitle_completion import parse_window  # noqa: E402
from check_known_episode1 import assess as assess_known_cases  # noqa: E402


class EvaluationToolTests(unittest.TestCase):
    def test_reviewed_mixed_clip_is_not_a_recall_gain(self):
        cases = {"cases": [{"id": "mixed", "kind": "must_not_cover_reviewed_mixed_clip",
                             "span": [10, 13.3]}]}
        contaminated = {"sentences": [{"start": 10, "end": 13.3, "accepted": True}]}
        clean_candidate = {"sentences": [{"start": 11.5, "end": 13.3, "accepted": True}]}
        self.assertFalse(assess_known_cases(contaminated, cases)[0]["passed"])
        # This only excludes the known whole-clip error; it does NOT certify
        # the partial candidate's exact start or identity.
        self.assertTrue(assess_known_cases(clean_candidate, cases)[0]["passed"])

    def test_local_replay_window_rejects_invalid_ranges(self):
        window = parse_window("303:308")
        self.assertEqual((window.start, window.end), (303, 308))
        for invalid in ("3:2", "-1:2", "1:1", "NaN:2", "1:inf", "1:2:3"):
            with self.subTest(invalid=invalid), self.assertRaises(argparse.ArgumentTypeError):
                parse_window(invalid)

    def test_pair_parser_requires_ordered_nonoverlapping_spans(self):
        left, right = parse_pair("1.0:2.0,2.5:3.0")
        self.assertEqual((left.start, left.end, right.start, right.end), (1.0, 2.0, 2.5, 3.0))
        for invalid in ("1:2,1.9:3", "1:1,2:3", "2:1,3:4", "not-a-pair"):
            with self.subTest(invalid=invalid), self.assertRaises(argparse.ArgumentTypeError):
                parse_pair(invalid)

    def test_stage_coverage_does_not_call_geometric_presence_a_correct_output(self):
        reviewed = {"sentences": [{"start": 1.0, "end": 3.0, "accepted": True}]}
        audit = {"stages": {
            "initial_vad": {"count": 1, "spans": [[1.0, 3.0]]},
            "final_accepted": {"count": 1, "spans": [[2.0, 3.0]]},
            "overlap_blocked_islands": {"count": 1, "spans": [[1.0, 1.5]]},
        }}
        report = summarize(reviewed, audit)
        row = report["reviewed_coverage"][0]
        self.assertEqual(row["stage_coverage"]["initial_vad"]["seconds"], 2.0)
        self.assertEqual(row["stage_coverage"]["final_accepted"]["seconds"], 1.0)
        self.assertEqual(row["blocked_overlap_seconds"]["overlap_blocked_islands"], 0.5)
        self.assertIn("geometric only", report["warning"])

    def test_stage_transitions_show_loss_and_later_recovery_without_double_counting(self):
        reviewed = {"sentences": [{"start": 1.0, "end": 5.0, "accepted": True}]}
        audit = {"stages": {
            "initial_vad": {"count": 2, "spans": [[1.0, 3.0], [2.0, 5.0]]},
            "subtitle_assisted_vad": {"count": 2, "spans": [[1.0, 2.0], [4.0, 5.0]]},
            "atomic_speech_islands": {"count": 1, "spans": [[1.0, 5.0]]},
        }}
        row = summarize(reviewed, audit)["reviewed_coverage"][0]
        self.assertEqual(row["stage_coverage"]["initial_vad"]["seconds"], 4.0)
        self.assertEqual(
            row["stage_coverage"]["subtitle_assisted_vad"]["missing_intervals"],
            [[2.0, 4.0]],
        )
        self.assertEqual(
            row["stage_transitions"]["subtitle_assisted_vad"]["lost_intervals"],
            [[2.0, 4.0]],
        )
        self.assertEqual(
            row["stage_transitions"]["atomic_speech_islands"]["recovered_intervals"],
            [[2.0, 4.0]],
        )
        self.assertEqual(row["stage_transitions"]["atomic_speech_islands"]["lost_seconds"], 0.0)

    def test_parallel_locator_is_not_counted_as_a_required_gate(self):
        reviewed = {"sentences": [{"start": 1.0, "end": 3.0, "accepted": True}]}
        audit = {"stages": {
            "speaker_turns": {"count": 1, "spans": [[1.0, 3.0]]},
            "target_locator_proposals": {"count": 1, "spans": [[2.0, 3.0]]},
            "identity_accepted_before_stt": {"count": 1, "spans": [[1.0, 3.0]]},
        }}
        row = summarize(reviewed, audit)["reviewed_coverage"][0]
        self.assertEqual(row["stage_coverage"]["target_locator_proposals"]["seconds"], 1.0)
        self.assertNotIn("target_locator_proposals", row["stage_transitions"])
        self.assertEqual(
            row["stage_transitions"]["identity_accepted_before_stt"]["from_stage"],
            "speaker_turns",
        )
        self.assertEqual(row["stage_transitions"]["identity_accepted_before_stt"]["lost_seconds"], 0.0)

    def test_review_queue_uses_stable_source_samples_without_inventing_truth(self):
        reviewed = {"sentences": [{"start": 1.0, "end": 2.0, "accepted": True}]}
        current = {"sentences": [
            {"start": 1.0, "end": 1.5, "accepted": False, "reject_reason": "声纹匹配不足"},
            {"start": 1.5, "end": 2.0, "accepted": True},
        ]}
        audit = {"stages": {"atomic_speech_islands": {
            "count": 1, "spans": [[1.0, 2.0]],
        }}}
        provenance = {"inputs_sha256": {"target": "a" * 64}}
        queue = build_queue(reviewed, current, audit, provenance)
        row = queue["islands"][0]
        self.assertEqual(row["source_id"], f"{'a' * 64}:16000:32000")
        self.assertEqual(row["historical_reviewed_overlap_seconds"], 1.0)
        self.assertEqual(row["current_accepted_overlap_seconds"], 0.5)
        self.assertEqual(row["current_rejection_reasons"], ["声纹匹配不足"])
        self.assertIsNone(row["human_label"])
        self.assertIsNone(row["human_speaker_boundary_intervals"])
        self.assertIn("same source", queue["warning"])

    def test_review_queue_rejects_invalid_source_fingerprint(self):
        with self.assertRaises(ValueError):
            build_queue(
                {"sentences": []},
                {"sentences": []},
                {"stages": {"atomic_speech_islands": {"spans": []}}},
                {"inputs_sha256": {"target": "not-a-hash"}},
            )

    def test_join_regression_accepts_new_policy_predictions_without_changing_labels(self):
        cases = {"cases": [
            {"id": "same", "left": [1.0, 2.0], "right": [2.3, 3.0], "same_target": True},
            {"id": "change", "left": [4.0, 5.0], "right": [5.2, 6.0], "same_target": False},
        ]}
        probe = {"pairs": [
            {"left": case["left"], "right": case["right"], "gap_seconds": 0.2,
             "primary_pair_similarity": 0.5, "secondary_pair_similarity": 0.5,
             "whole_result": {"accepted_by_whole_span_verifier": True, "excluded_role_rejected": False}}
            for case in cases["cases"]
        ]}
        reviewed = {"sentences": [{"start": 1.0, "end": 3.0, "accepted": True}]}
        baseline = assess_join_cases(probe, cases, reviewed)
        self.assertEqual((baseline["passed"], baseline["total"]), (1, 2))
        candidate = assess_join_cases(probe, cases, reviewed, {"same": True, "change": False})
        self.assertEqual((candidate["passed"], candidate["total"]), (2, 2))
        with self.assertRaises(ValueError):
            assess_join_cases(probe, cases, reviewed, {"same": True})
        invalid = {"cases": [dict(cases["cases"][0], right=[2.3, 3.2])]}
        with self.assertRaises(ValueError):
            assess_join_cases(probe, invalid, reviewed)


if __name__ == "__main__":
    unittest.main()
