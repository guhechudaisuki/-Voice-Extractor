from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.recovery_installation import install_recovered_candidate
from extractor.types import CandidateSentence


class RecoveryInstallationTests(unittest.TestCase):
    def test_strong_whole_cannot_replace_core_with_rejected_prefix(self):
        core = CandidateSentence(11.5, 13.3, "")
        prefix = CandidateSentence(10, 11.5, "", reject_reason="identity unresolved",
            diagnostics={"local_boundary_recovery": True, "recovery_part_index": 1})
        whole = CandidateSentence(10, 13.3, "", diagnostics={"speaker_tier": "strong"})
        accepted = [core]
        rejected = [prefix]
        self.assertFalse(install_recovered_candidate(whole, accepted, rejected))
        self.assertEqual(accepted, [core])
        self.assertIn(whole, rejected)
        self.assertEqual(whole.diagnostics["recovery_local_identity_conflicts"], [[10, 11.5]])

    def test_rejected_suffix_is_not_absorbed_either(self):
        core = CandidateSentence(1, 3, "")
        suffix = CandidateSentence(3, 4, "", reject_reason="other role",
            diagnostics={"local_boundary_recovery": True, "recovery_part_index": 2})
        accepted, rejected = [core], [suffix]
        self.assertFalse(install_recovered_candidate(CandidateSentence(1, 4, ""), accepted, rejected))
        self.assertEqual(accepted, [core])

    def test_neighboring_reject_outside_candidate_does_not_block_target(self):
        other = CandidateSentence(1, 3, "", reject_reason="other",
            diagnostics={"local_boundary_recovery": True, "recovery_part_index": 1})
        target = CandidateSentence(3, 5, "")
        accepted = []
        self.assertTrue(install_recovered_candidate(target, accepted, [other]))
        self.assertEqual(accepted, [target])

    def test_whole_parent_reject_is_not_confused_with_local_identity(self):
        parent = CandidateSentence(1, 10, "", reject_reason="whole uncertain",
            diagnostics={"local_boundary_recovery": True})
        accepted = []
        target = CandidateSentence(3, 5, "")
        self.assertTrue(install_recovered_candidate(target, accepted, [parent]))

    def test_clean_expansion_keeps_existing_replacement_semantics(self):
        original = CandidateSentence(2, 4, "")
        whole = CandidateSentence(1, 5, "")
        accepted = [original]
        self.assertTrue(install_recovered_candidate(whole, accepted, []))
        self.assertEqual(accepted, [whole])


if __name__ == "__main__":
    unittest.main()
