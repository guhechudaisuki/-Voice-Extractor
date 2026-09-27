from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extractor.nextgen.timeline import SampleSpan as Span, SourceTimeline, ViewAlignment, union_length
from extractor.nextgen.ledger import Candidate, CandidateLedger, Decision, EvidenceKind, LocalEvidence, State
from extractor.nextgen.reference_bank import Reference, ReferenceBank
from extractor.nextgen.decision_policy import Calibration, FramePrediction, SpanPrediction, assess
from extractor.nextgen.boundary_decoder import SilenceRange, SpeechIsland, propose, select_verified

SOURCE, REFS, MODEL = "a" * 64, "b" * 64, "c" * 64


def candidate(start=100, end=300, **kwargs):
    return Candidate(SOURCE, 16000, Span(start, end), Span(0, 1000),
                     (Span(start, end),), "test", start_complete=True,
                     end_complete=True, **kwargs)


def local(start, end, kind):
    return LocalEvidence(SOURCE, 16000, Span(start, end), kind, REFS, MODEL, "test")


def prediction(item, frames=None):
    return SpanPrediction(item.key, REFS, MODEL, tuple(frames or [
        FramePrediction(item.output, .95, .01, .01, True),
    ]), .99)


def calibration():
    # Explicit test fixture, not deployment defaults.
    return Calibration(MODEL, "d" * 64, "unit-fixture", .8, .4, .3, .9)


class TimelineTests(unittest.TestCase):
    def test_resampling_and_delay_map_back_to_source(self):
        source = SourceTimeline(SOURCE, 48000, 480000)
        view = ViewAlignment(source, 16000, 160017, 17, True)
        self.assertEqual(view.from_source(Span(48000, 96000)), Span(16017, 32017))
        self.assertEqual(view.to_source(Span(16017, 32017)), Span(48000, 96000))

    def test_unknown_delay_is_not_assumed_zero(self):
        view = ViewAlignment(SourceTimeline(SOURCE, 16000, 1000), 16000, 1000, 0, False)
        with self.assertRaises(ValueError):
            view.to_source(Span(100, 200))

    def test_round_trip_does_not_cut_fractional_source_edges(self):
        source = SourceTimeline(SOURCE, 44100, 441000)
        view = ViewAlignment(source, 16000, 160000, 0, True)
        original = Span(40001, 80003)
        self.assertTrue(view.to_source(view.from_source(original)).contains(original))

    def test_union_does_not_count_duplicate_audio(self):
        self.assertEqual(union_length([Span(1, 5), Span(2, 4), Span(5, 8)]), 7)


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.ledger = CandidateLedger(SourceTimeline(SOURCE, 16000, 1000), REFS, MODEL)

    def test_distinct_proposals_with_same_interval_do_not_overwrite(self):
        one = candidate()
        two = replace(one, origin="subtitle_timing")
        self.ledger.add(one)
        self.ledger.add(two)
        self.assertEqual(len(self.ledger.candidates), 2)
        self.assertEqual(one.span_key, two.span_key)

    def test_parent_and_local_negative_survive_new_large_candidate(self):
        core = candidate(200, 300)
        self.ledger.add(core)
        index = self.ledger.observe(local(100, 200, EvidenceKind.OTHER))
        bigger = candidate(parents=(core.key,))
        self.ledger.add(bigger)
        self.ledger.decide(Decision(bigger.key, State.REJECTED, "other", (index,), "test"))
        self.assertEqual(len(self.ledger.candidates), 2)
        self.assertEqual(len(self.ledger.evidence), 1)

    def test_model_reference_and_source_fingerprints_are_mandatory(self):
        evidence = local(100, 200, EvidenceKind.TARGET)
        for key in ("model_digest", "reference_digest", "source_sha256"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.ledger.observe(replace(evidence, **{key: "e" * 64}))


class ReferenceTests(unittest.TestCase):
    def ref(self, index, role="target", group=None):
        digest = f"{index:064x}"
        return Reference(digest, SOURCE, Span(index * 100, index * 100 + 100),
                         role, digest, digest, group)

    def test_order_and_exact_duplicates_do_not_change_bank(self):
        a, b = self.ref(1), self.ref(2)
        self.assertEqual(ReferenceBank([a, b, a]).digest, ReferenceBank([b, a]).digest)

    def test_target_and_exclusion_groups_remain_separate(self):
        bank = ReferenceBank([self.ref(1), self.ref(2, "role_1"), self.ref(3, "role_1")])
        self.assertEqual(len(bank.target), 1)
        self.assertEqual(len(bank.excluded["role_1"]), 2)

    def test_role_conflict_cannot_hide_behind_near_duplicate_removal(self):
        a, b = self.ref(1, group="same-audio"), self.ref(2, group="same-audio")
        conflicting = replace(b, role="role_1", duplicate_group=None)
        with self.assertRaises(ValueError):
            ReferenceBank([a, b, conflicting])

    def test_query_cannot_confirm_itself(self):
        bank = ReferenceBank([self.ref(1), self.ref(2)])
        independent = bank.independent_of(SOURCE, Span(100, 200))
        self.assertEqual(independent, (self.ref(2),))


class PolicyTests(unittest.TestCase):
    def check(self, item, pred=None, evidence=()):
        return assess(item, pred or prediction(item), calibration(), reference_digest=REFS,
                      local_evidence=evidence)

    def test_other_short_sound_wins_over_high_whole_purity(self):
        item = candidate()
        verdict = self.check(item, evidence=(local(299, 300, EvidenceKind.OTHER),))
        self.assertEqual(verdict.state, State.REJECTED)

    def test_nonobservable_consonant_does_not_require_own_embedding(self):
        item = candidate()
        pred = prediction(item, [FramePrediction(Span(100, 110), .1, .01, .01, False),
                                 FramePrediction(Span(110, 300), .95, .01, .01, True)])
        self.assertEqual(self.check(item, pred).state, State.ACCEPTED)

    def test_observable_weak_voice_is_unresolved_not_automatically_other(self):
        item = candidate()
        pred = prediction(item, [FramePrediction(item.output, .3, .1, .2, True)])
        self.assertEqual(self.check(item, pred).state, State.UNRESOLVED)

    def test_missing_frame_and_incomplete_boundary_are_not_silently_trimmed(self):
        item = replace(candidate(), end_complete=False)
        pred = prediction(item, [FramePrediction(Span(100, 290), .95, .01, .01, True)])
        result = self.check(item, pred)
        self.assertEqual(result.state, State.UNRESOLVED)
        self.assertIn("missing_temporal_coverage", result.reasons)
        self.assertIn("acoustic_boundary_incomplete", result.reasons)

    def test_edited_boundary_invalidates_prediction(self):
        original = candidate()
        changed = replace(original, output=Span(100, 400), speech=(Span(100, 400),))
        with self.assertRaises(ValueError):
            self.check(changed, prediction(original))

    def test_context_only_other_voice_is_not_export_contamination(self):
        item = candidate()
        self.assertEqual(self.check(item, evidence=(local(400, 500, EvidenceKind.OTHER),)).state,
                         State.ACCEPTED)


class DecoderTests(unittest.TestCase):
    def test_silence_endpoints_match_user_rule(self):
        silence = SilenceRange(3200, 13600)
        self.assertEqual([silence.kind(gap) for gap in (3199, 3200, 13600, 13601)],
                         ["inside_utterance", "identity_join", "identity_join", "hard_split"])

    def test_tiny_pause_does_not_authorize_partial_sentence(self):
        source = SourceTimeline(SOURCE, 16000, 100000)
        proposals = propose(source, (SpeechIsland(Span(1000, 20000), True, True),
                                     SpeechIsland(Span(21000, 40000), True, True)),
                            SilenceRange(3200, 13600), maximum_samples=90000)
        one, combined, two = proposals
        self.assertFalse(one.end_complete)
        self.assertFalse(two.start_complete)
        self.assertTrue(combined.start_complete and combined.end_complete)

    def test_blocked_gap_cannot_be_reinterpreted_as_silence(self):
        source = SourceTimeline(SOURCE, 16000, 100000)
        proposals = propose(source, (SpeechIsland(Span(1000, 20000), True, True),
                                     SpeechIsland(Span(25000, 40000), True, True)),
                            SilenceRange(3200, 13600), maximum_samples=90000,
                            blocked=(Span(21000, 22000),))
        self.assertEqual(len(proposals), 2)

    def test_decoder_maximizes_speech_not_file_count_or_silence(self):
        one, two = candidate(100, 200), candidate(200, 300)
        combined = candidate(100, 300)
        items = (one, two, combined)
        results = tuple(assess(row, prediction(row), calibration(), reference_digest=REFS)
                        for row in items)
        self.assertEqual(select_verified(items, results), (combined,))


if __name__ == "__main__":
    unittest.main()
