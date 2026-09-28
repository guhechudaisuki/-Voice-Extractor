from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.train_anime_t3_reviewed import (
    assign_splits, build_episode_cases, calibration_thresholds,
    validate_veto_metrics,
)


def manifest(spans):
    return {"sentences": [
        {"start": start, "end": end, "accepted": True} for start, end in spans
    ]}


class AnimeTrainingLabelsTests(unittest.TestCase):
    def build(self, verified, rescued, latest, review, old=None, bad=()):
        reviews = [(manifest(latest), {"excluded_clips": review})]
        return build_episode_cases(
            manifest(verified), manifest(rescued), reviews,
            old or {"flagged_outputs": []},
            rescued_bad_ordinals=bad,
        )

    def test_latest_negative_overrides_overlapping_reviewed_positive(self):
        cases, conflicts = self.build(
            [(10, 14)], [(11, 15)], [(12, 14)],
            [{"ordinal": 1, "span": [12, 14], "reason": "other"}],
        )
        self.assertEqual(
            [(c["start"], c["end"]) for c in cases if c["label"] == 1],
            [(10, 12), (14, 15)],
        )
        negative = next(c for c in cases if c["label"] == 0)
        self.assertTrue(negative["force_train"])
        self.assertTrue(conflicts)

    def test_duplicate_feedback_does_not_create_identity_negative(self):
        cases, _ = self.build(
            [(10, 12)], [(10, 12)], [(10, 12), (20, 22)],
            [{"ordinal": 2, "span": [20, 22], "duplicate_of": 1}],
        )
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["label"], 1)

    def test_mixed_old_feedback_is_unknown_not_wholly_negative(self):
        cases, _ = self.build([], [], [], [], old={"flagged_outputs": [
            {"index": 1, "span": [10, 14], "kind": "mixed"},
            {"index": 2, "span": [20, 24], "kind": "entire_other"},
        ]})
        self.assertEqual([(c["start"], c["end"]) for c in cases], [(20, 24)])

    def test_partial_negative_splits_clip_into_kept_positive(self):
        cases, _ = self.build(
            [], [], [(10, 16)],
            [{"ordinal": 1, "span": [10, 16], "negative_span": [13, 16],
              "reason": "second half other"}],
        )
        negative = next(c for c in cases if c["label"] == 0)
        kept = next(c for c in cases if c["label"] == 1)
        self.assertEqual((negative["start"], negative["end"]), (13, 16))
        self.assertTrue(negative["force_train"])
        self.assertEqual((kept["start"], kept["end"]), (10, 13))
        self.assertTrue(kept["force_train"])

    def test_reviewed_remaining_clips_become_reviewed_positives(self):
        cases, _ = self.build(
            [], [], [(10, 12), (20, 22)],
            [{"ordinal": 1, "span": [10, 12], "reason": "other"}],
        )
        self.assertEqual(
            [(c["start"], c["end"]) for c in cases if c["label"] == 1],
            [(20, 22)],
        )

    def test_earlier_wrong_ordinals_are_negative_and_not_positive(self):
        cases, _ = self.build([], [(10, 12), (20, 22)], [], [], bad=(2,))
        self.assertEqual([(c["start"], c["label"]) for c in cases], [(10, 1), (20, 0)])

    def test_review_must_match_its_manifest_ordinal(self):
        with self.assertRaisesRegex(ValueError, "Review span"):
            self.build([], [], [(10, 12)], [{"ordinal": 1, "span": [20, 22]}])

    def test_nearby_cases_share_a_group_and_correction_forces_training(self):
        rows = [
            {"id": "a", "source_id": "episode", "start": 1, "end": 3,
             "label": 1, "force_train": False},
            {"id": "b", "source_id": "episode", "start": 8, "end": 10,
             "label": 0, "force_train": True},
            {"id": "c", "source_id": "other", "start": 8, "end": 10,
             "label": 0, "force_train": False},
        ]
        assign_splits(rows)
        self.assertEqual(rows[0]["group"], rows[1]["group"])
        self.assertEqual(rows[0]["split"], "train")
        self.assertNotEqual(rows[0]["group"], rows[2]["group"])

    def test_split_is_deterministic_and_never_splits_a_case_group(self):
        rows = [{"id": str(i), "source_id": "episode", "start": i * 20,
                 "end": i * 20 + 2, "label": i % 2, "force_train": False}
                for i in range(20)]
        assign_splits(rows)
        repeated = [dict(r) for r in reversed(rows)]
        assign_splits(repeated)
        self.assertEqual({r["id"]: r["split"] for r in rows},
                         {r["id"]: r["split"] for r in repeated})
        for split in ("train", "calibration", "test"):
            self.assertEqual({r["label"] for r in rows if r["split"] == split}, {0, 1})

    def test_thresholds_use_only_calibration_not_training_or_test(self):
        rows = [
            {"split": "calibration", "label": 0, "mean": 1.0, "negative_evidence": -3.0},
            {"split": "calibration", "label": 1, "mean": 4.0, "negative_evidence": 2.0},
            {"split": "test", "label": 0, "mean": 100.0, "negative_evidence": -100.0},
            {"split": "train", "label": 1, "mean": 20.0, "negative_evidence": -20.0},
        ]
        self.assertEqual(calibration_thresholds(rows), (2.0, -1.0))

    def test_single_window_positive_cannot_supply_veto_calibration(self):
        with self.assertRaisesRegex(ValueError, "calibration"):
            calibration_thresholds([
                {"split": "calibration", "label": 0, "mean": 1.0, "negative_evidence": -3.0},
                {"split": "calibration", "label": 1, "mean": 4.0, "negative_evidence": None},
            ])

    def test_veto_can_pass_with_one_heldout_negative_and_reports_split_miss(self):
        metrics = {
            "train": {"positive_cases": 2, "negative_cases": 2,
                      "positive_false_veto": 0, "negative_detected": 2},
            "calibration": {"positive_cases": 2, "negative_cases": 2,
                            "positive_false_veto": 0, "negative_detected": 0},
            "test": {"positive_cases": 2, "negative_cases": 2,
                     "positive_false_veto": 0, "negative_detected": 1},
        }
        reasons, warnings = validate_veto_metrics(metrics, [(0, True), (0, True), (0, True)])
        self.assertEqual(reasons, [])
        self.assertEqual(warnings, ["calibration detects no negative cases"])

    def test_veto_fails_if_any_reviewed_positive_is_rejected(self):
        metrics = {
            split: {"positive_cases": 1, "negative_cases": 1,
                    "positive_false_veto": int(split == "test"), "negative_detected": 1}
            for split in ("train", "calibration", "test")
        }
        reasons, _ = validate_veto_metrics(metrics, [(0, True)])
        self.assertIn("test contains positive false vetoes", reasons)

    def test_veto_fails_when_positive_correction_is_vetoed(self):
        metrics = {
            split: {"positive_cases": 1, "negative_cases": 1,
                    "positive_false_veto": 0, "negative_detected": 1}
            for split in ("train", "calibration", "test")
        }
        reasons, _ = validate_veto_metrics(metrics, [(0, True), (1, True)])
        self.assertIn("Explicit positive corrections are vetoed", reasons)

    def test_veto_fails_when_no_heldout_negative_is_detected(self):
        metrics = {
            split: {"positive_cases": 1, "negative_cases": 1,
                    "positive_false_veto": 0, "negative_detected": int(split == "train")}
            for split in ("train", "calibration", "test")
        }
        reasons, warnings = validate_veto_metrics(metrics, [(0, True)])
        self.assertIn("held-out splits detect no negative cases", reasons)
        self.assertEqual(len(warnings), 2)


if __name__ == "__main__":
    unittest.main()
