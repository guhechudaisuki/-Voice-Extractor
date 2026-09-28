from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extractor.pipeline import ExtractionPipeline, PipelineOptions
from extractor.types import CandidateSentence, TimeSpan


class _StubProfilePrimary:
    reference_floor = 0.65
    calibration_base = 0.7


class _StubProfile:
    primary = _StubProfilePrimary()
    reference_paths = []
    base_threshold = 0.7


class _StubPrimary:
    score = 0.75
    window_min_score = 0.72
    window_p20_score = 0.73
    vote_ratio = 1.0
    window_vote_ratio = 1.0
    reference_median_score = 0.62
    reference_max_score = 0.66
    reference_spread = 0.04


class _StubMatch:
    primary = _StubPrimary()
    secondary = None
    paired_reference_median = 0.60
    match_mode = "strong"
    tier = "strong"
    diagnostics = {}

    def __init__(self, accepted: bool = True, diagnostics=None) -> None:
        self.accepted = accepted
        self.diagnostics = dict(diagnostics or {})


class _StubVerifier:
    def __init__(self, reject_exclusion: bool = False) -> None:
        self.reject_exclusion = reject_exclusion

    def exclusion_audit(self, match, profile, exclusion_profiles, **_kwargs):
        if self.reject_exclusion:
            return {"excluded_role_rejected": True, "excluded_role": "排除角色 1"}
        return {"excluded_role_rejected": False, "excluded_primary_margin": 0.23}


def accepted_turn(start: float, end: float) -> CandidateSentence:
    turn = CandidateSentence(start, end, "", accepted=True)
    turn.speaker_score = 0.8
    turn.speaker_threshold = 0.7
    return turn


def flagged_spans_file_shape(spans):
    return [[float(start), float(end)] for start, end in spans]


class UserMarkedExclusionTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = ExtractionPipeline(PipelineOptions())
        self.wave = torch.zeros(16000 * 30)
        self.verifier = _StubVerifier()

    def run_gate(self, accepted, rejected, marks, verifier=None,
                 verify_accepted: bool = True, classifier_score=None,
                 match_diagnostics=None):
        if classifier_score is None:
            self.pipeline._ensure_tail_scorer = lambda: None
        else:
            import types

            class _FakeEnc:
                config = types.SimpleNamespace(use_weighted_layer_sum=False)

                def wavlm(self, x, output_hidden_states=False, return_dict=True):
                    return types.SimpleNamespace(
                        last_hidden_state=torch.ones(1, 4, 8)
                    )

                projector = torch.nn.Identity()
                tdnn = torch.nn.ModuleList([torch.nn.Identity()])

            class _FakeProc:
                def __call__(self, wave, sampling_rate, return_tensors):
                    return types.SimpleNamespace(input_values=torch.zeros(1, wave.shape[0]))

            clf = torch.nn.Linear(24, 1)
            with torch.no_grad():
                clf.weight.fill_(1.0)
                clf.bias.fill_(float(classifier_score) - 16.0)
            pack = (_FakeEnc(), _FakeProc(), clf, 2.0, torch.device('cpu'))
            self.pipeline._ensure_tail_scorer = lambda: pack
        self.pipeline._verify_speaker_span = (
            lambda _verifier, _wave, _span, _profile, _threshold, **_kw: _StubMatch(
                accepted=verify_accepted, diagnostics=match_diagnostics,
            )
        )
        return self.pipeline._apply_user_marked_exclusions(
            accepted,
            rejected,
            [TimeSpan(*span) for span in marks],
            verifier or self.verifier,
            _StubProfile(),
            self.wave,
            [],
            0.7,
        )

    def test_turn_is_trimmed_to_verified_remainder(self):
        turn = accepted_turn(11.17, 14.89)
        accepted = [turn]
        rejected: list[CandidateSentence] = []
        removed = self.run_gate(accepted, rejected, [(12.99, 15.46)])
        self.assertEqual(removed, 1)
        self.assertEqual(len(accepted), 1)
        trimmed = accepted[0]
        self.assertAlmostEqual(trimmed.start, 11.17, places=2)
        self.assertAlmostEqual(trimmed.end, 12.99, places=2)
        self.assertIn("user_marked_trim", trimmed.diagnostics)
        self.assertEqual(turn.reject_reason, "用户标记排除区间，已按标记删除")
        self.assertTrue(turn.diagnostics["user_marked_exclusion"])

    def test_turn_is_deleted_when_remainder_fails_verification(self):
        turn = accepted_turn(11.17, 14.89)
        accepted = [turn]
        rejected: list[CandidateSentence] = []
        removed = self.run_gate(
            accepted, rejected, [(12.99, 15.46)], verify_accepted=False
        )
        self.assertEqual(removed, 1)
        self.assertEqual(accepted, [])
        self.assertIn(turn, rejected)

    def test_turn_inside_mark_is_deleted_whole(self):
        turn = accepted_turn(13.0, 14.5)
        accepted = [turn]
        rejected: list[CandidateSentence] = []
        removed = self.run_gate(accepted, rejected, [(12.99, 15.46)])
        self.assertEqual(removed, 1)
        self.assertEqual(accepted, [])

    def test_non_overlapping_turns_are_untouched(self):
        keep = accepted_turn(10.0, 14.0)
        accepted = [keep]
        rejected: list[CandidateSentence] = []
        removed = self.run_gate(accepted, rejected, [(100.0, 110.0)])
        self.assertEqual(removed, 0)
        self.assertEqual(accepted, [keep])

    def test_excluded_role_remainder_is_deleted(self):
        verifier = _StubVerifier(reject_exclusion=True)
        turn = accepted_turn(11.17, 14.89)
        accepted = [turn]
        rejected: list[CandidateSentence] = []
        removed = self.run_gate(accepted, rejected, [(12.99, 15.46)], verifier)
        self.assertEqual(removed, 1)
        self.assertEqual(accepted, [])

    def test_exclusion_rejection_takes_priority_over_classifier_score(self):
        turn = accepted_turn(11.17, 14.89)
        accepted = [turn]
        rejected = []
        self.run_gate(
            accepted, rejected, [(12.99, 15.46)],
            verifier=_StubVerifier(reject_exclusion=True),
            verify_accepted=True, classifier_score=4.2,
        )
        self.assertEqual(accepted, [])
        self.assertIn(turn, rejected)

    def test_rejected_parent_cannot_remain_an_exported_accepted_clip(self):
        for bounds in ((11.17, 14.89), (13.0, 14.5)):
            with self.subTest(bounds=bounds):
                turn = accepted_turn(*bounds)
                turn.audio_file = "audio/old.wav"
                turn.text_file = "text/old.txt"
                turn.video_file = "video/old.mp4"
                accepted = [turn]
                rejected = []
                self.run_gate(accepted, rejected, [(12.99, 15.46)])
                self.assertIn(turn, rejected)
                self.assertFalse(turn.accepted)
                self.assertEqual(turn.audio_file, "")
                self.assertEqual(turn.text_file, "")
                self.assertEqual(turn.video_file, "")

    def test_classifier_cannot_keep_remainder_without_formal_verification(self):
        turn = accepted_turn(11.17, 14.89)
        accepted = [turn]
        rejected: list[CandidateSentence] = []
        removed = self.run_gate(
            accepted, rejected, [(12.99, 15.46)],
            verify_accepted=False, classifier_score=4.2,
        )
        self.assertEqual(removed, 1)
        self.assertEqual(accepted, [])
        self.assertIn(turn, rejected)

    def test_trimmed_remainder_records_its_own_identity_evidence(self):
        turn = accepted_turn(11.17, 14.89)
        turn.diagnostics = {
            "duration": 3.72,
            "speaker_tier": "raw_rescue",
            "excluded_primary_margin": 0.8,
            "target_locator_recovery": True,
            "local_identity_audit": {
                "span": [11.17, 14.89], "passed": True,
            },
        }
        accepted = [turn]
        self.run_gate(
            accepted, [], [(12.99, 15.46)],
            match_diagnostics={"duration": 1.82, "raw_channel_available": True},
        )
        trimmed = accepted[0]
        self.assertEqual(trimmed.speaker_score, 0.75)
        self.assertEqual(trimmed.diagnostics["duration"], 1.82)
        self.assertEqual(trimmed.diagnostics["speaker_tier"], "strong")
        self.assertTrue(trimmed.diagnostics["raw_channel_available"])
        self.assertEqual(trimmed.diagnostics["excluded_primary_margin"], 0.23)
        self.assertNotIn("local_identity_audit", trimmed.diagnostics)
        self.assertNotIn("target_locator_recovery", trimmed.diagnostics)
        provenance = trimmed.diagnostics["user_marked_trim"]["provenance"]
        self.assertEqual(provenance["diagnostics"]["duration"], 3.72)
        self.assertEqual(
            provenance["diagnostics"]["local_identity_audit"]["span"],
            [11.17, 14.89],
        )

    def test_options_normalize_marked_spans(self):
        options = PipelineOptions(
            user_excluded_spans=flagged_spans_file_shape(
                [(413.0, 418.05), (10, 10)]
            )
        )
        self.assertEqual(options.user_excluded_spans, ((413.0, 418.05),))
        self.assertTrue(options.experimental_final_island_consensus)


if __name__ == "__main__":
    unittest.main()
