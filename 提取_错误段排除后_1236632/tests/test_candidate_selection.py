from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extractor.candidate_selection import RecoverySelector
from extractor.types import CandidateSentence


def clip(start, end):
    return CandidateSentence(start, end, "")


class RecoverySelectionTests(unittest.TestCase):
    def test_whole_score_cannot_erase_local_unknown_prefix(self):
        selector = RecoverySelector()
        prefix, core, whole = clip(10, 11.5), clip(11.5, 13.3), clip(10, 13.3)
        selector.observe(prefix, "unresolved")
        selector.observe(core, "target")
        accepted, rejected = [core], []
        self.assertFalse(selector.install(whole, accepted, rejected))
        self.assertEqual(accepted, [core])

    def test_local_evidence_is_not_lost_when_candidate_object_is_mutated(self):
        selector = RecoverySelector()
        unknown = clip(10, 11)
        selector.observe(unknown, "unresolved")
        unknown.start, unknown.end = 30, 31
        self.assertFalse(selector.install(clip(10, 13), [], []))

    def test_new_exact_local_support_can_resolve_unknown(self):
        selector = RecoverySelector()
        selector.observe(clip(10, 11), "unresolved")
        selector.observe(clip(10, 11), "target")
        accepted = [clip(11, 13)]
        whole = clip(10, 13)
        self.assertTrue(selector.install(whole, accepted, []))
        self.assertEqual(accepted, [whole])

    def test_longer_target_proposal_cannot_lend_identity_to_prefix(self):
        selector = RecoverySelector()
        selector.observe(clip(10, 11), "unresolved")
        selector.observe(clip(10, 13), "target")
        self.assertFalse(selector.install(clip(10, 13), [], []))

    def test_explicit_other_cannot_be_overridden_by_target_claim(self):
        selector = RecoverySelector()
        selector.observe(clip(10, 11), "other")
        selector.observe(clip(10, 11), "target")
        self.assertFalse(selector.install(clip(10, 13), [], []))

    def test_nonoverlapping_local_rejection_does_not_block_core(self):
        selector = RecoverySelector()
        selector.observe(clip(10, 11), "other")
        self.assertTrue(selector.install(clip(11, 13), [], []))

    def test_subinterval_with_own_local_support_is_not_blocked_by_parent(self):
        selector = RecoverySelector()
        selector.observe(clip(10, 15), "unresolved")
        selector.observe(clip(12, 14), "target")
        self.assertTrue(selector.install(clip(12, 14), [], []))

    def test_replacement_must_not_trim_even_small_piece_of_existing_core(self):
        selector = RecoverySelector()
        core = clip(10, 13)
        accepted = [core]
        self.assertFalse(selector.install(clip(10.1, 14), accepted, []))
        self.assertEqual(accepted, [core])

    def test_other_short_sound_inside_long_target_has_no_duration_exemption(self):
        selector = RecoverySelector()
        selector.observe(clip(11, 11.01), "other")
        self.assertFalse(selector.install(clip(10, 20), [], []))

    def test_deferred_proposal_is_kept_and_not_called_other(self):
        selector = RecoverySelector()
        selector.observe(clip(10, 11), "unresolved")
        proposed = clip(10, 13)
        rejected = []
        self.assertFalse(selector.install(proposed, [], rejected))
        self.assertEqual(rejected, [proposed])
        self.assertEqual(proposed.diagnostics["recovery_selection"]["state"], "unresolved")

    def test_reconsider_after_new_local_evidence_does_not_leave_duplicate(self):
        selector = RecoverySelector()
        selector.observe(clip(10, 11), "unresolved")
        proposed, accepted, rejected = clip(10, 13), [], []
        self.assertFalse(selector.install(proposed, accepted, rejected))
        selector.observe(clip(10, 11), "target")
        self.assertTrue(selector.install(proposed, accepted, rejected))
        self.assertEqual(accepted, [proposed])
        self.assertEqual(rejected, [])
        self.assertEqual(proposed.reject_reason, "")

    def test_one_sample_uncovered_is_not_silently_filled(self):
        selector = RecoverySelector()
        selector.observe(clip(10, 11), "unresolved")
        selector.observe(clip(10 + 1 / 16000, 11), "target")
        self.assertFalse(selector.install(clip(10, 13), [], []))

    def test_union_of_local_support_can_resolve_unknown(self):
        selector = RecoverySelector()
        selector.observe(clip(10, 12), "unresolved")
        selector.observe(clip(10, 11), "target")
        selector.observe(clip(11, 12), "target")
        self.assertTrue(selector.install(clip(10, 13), [], []))

    def test_invalid_intervals_are_errors_not_empty_safe_regions(self):
        for start, end in [(float("nan"), 2), (1, float("inf")), (2, 1), (-1, 1)]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                RecoverySelector().observe(clip(start, end), "other")

    def test_diagnostic_consumers_cannot_mutate_ledger_history(self):
        selector = RecoverySelector()
        proposed = clip(10, 13)
        selector.install(proposed, [], [])
        proposed.diagnostics["recovery_selection"]["installed"] = False
        report = selector.to_dict()
        report["decisions"].clear()
        self.assertTrue(selector.to_dict()["decisions"][0]["installed"])


if __name__ == "__main__":
    unittest.main()
