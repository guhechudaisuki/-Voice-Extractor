from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extractor.nextgen.artifacts import (InferenceCalibration, load_bundle,
                                         write_research_calibration, write_research_checkpoint)
from extractor.nextgen.acoustic_gaps import find_confirmed_gaps
from extractor.nextgen.boundary_decoder import (AcousticGap, SilenceRange, propose_change_parts,
                                                propose_unresolved_change_sides)
from extractor.nextgen.decision_policy import Calibration, FramePrediction, SpanPrediction, assess
from extractor.nextgen.features import FeatureSequence, FrameGeometry, require_paired
from extractor.nextgen.engine import run_prepared
from extractor.nextgen.inference import EncodedReference, IdentitySession, InferenceResult
from extractor.nextgen.prepared_audio import PairedAudio, estimate_stem_delay
from extractor.nextgen.reference_bank import Reference, ReferenceBank
from extractor.nextgen.finalization import ExportItem, ExportPolicy, deliver, final_audit, normalize_clip
from extractor.nextgen.identity_model import HEADS, ReferenceFeatures, TemporalIdentityModel, identity_loss
from extractor.nextgen.ledger import Candidate, EvidenceKind, LocalEvidence, State
from extractor.nextgen.timeline import SampleSpan as Span, SourceTimeline, ViewAlignment
from evaluation.nextgen_metrics import AcousticTruth, evaluate
from evaluation.replay_nextgen import replay
from training.nextgen_calibration import (BinaryHeadExample, CalibrationExample,
                                          choose_policy, fit_complete_calibration)
from training.nextgen_data import DataRecord, audit_records

SOURCE, REFS, MODEL, BACKBONE = (char * 64 for char in "abcd")


def candidate(start=10, end=70):
    return Candidate(SOURCE, 1000, Span(start, end), Span(0, 100), (Span(start, end),),
                     "unit_fixture", start_complete=True, end_complete=True)


def policy():
    return Calibration(MODEL, "e" * 64, "test-only", .8, .4, .3, .9)


def prediction(row, other=.01):
    return SpanPrediction(row.key, REFS, MODEL, (FramePrediction(row.output, .95, other, .01, True),), .99)


def assessment(row):
    return assess(row, prediction(row), policy(), reference_digest=REFS)


class FeatureTests(unittest.TestCase):
    def test_two_view_vad_and_waveform_gap_yields_reviewable_change_parts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_path, stem_path = root / "raw.wav", root / "stem.wav"
            rng = np.random.default_rng(18)
            speech = (Span(5000, 25000), Span(40000, 65000))
            signal = np.zeros(80000, dtype=np.float32)
            for span in speech:
                signal[span.start:span.end] = rng.standard_normal(span.length) * .04
            sf.write(str(raw_path), signal, 16000)
            sf.write(str(stem_path), signal, 16000)
            report = estimate_stem_delay(raw_path, stem_path,
                                          (Span(8000, 16000), Span(45000, 53000)),
                                          maximum_delay_samples=160)
            paired = PairedAudio(raw_path, stem_path, report)
            parent = Candidate(paired.source.source_sha256, 16000, Span(8000, 60000),
                               Span(0, 70000), (Span(8000, 60000),), "vad",
                               start_complete=True, end_complete=True)
            gaps = find_confirmed_gaps(parent, speech, speech, paired, minimum_gap_samples=1600)
            self.assertEqual(tuple(gap.span for gap in gaps), (Span(25000, 40000),))
            frames = (FramePrediction(Span(31000, 31320), .2, .2, .1, True, .95),)
            pieces = propose_change_parts(parent, frames, gaps, change_min=.9,
                                          tolerance_samples=0)
            self.assertEqual([part.output for part in pieces],
                             [Span(8000, 25000), Span(40000, 60000)])
            self.assertEqual(find_confirmed_gaps(parent, (Span(5000, 65000),), speech,
                                                 paired, minimum_gap_samples=1600), ())
            stem_with_noise = signal.copy()
            stem_with_noise[25000:40000] = rng.standard_normal(15000) * .03
            sf.write(str(stem_path), stem_with_noise, 16000)
            noisy_report = estimate_stem_delay(raw_path, stem_path,
                                                (Span(8000, 16000), Span(45000, 53000)),
                                                maximum_delay_samples=160)
            noisy_pair = PairedAudio(raw_path, stem_path, noisy_report)
            self.assertEqual(find_confirmed_gaps(parent, speech, speech,
                                                 noisy_pair, minimum_gap_samples=1600), ())

    def test_measured_delay_reads_exact_paired_source_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_path, stem_path = root / "raw.wav", root / "stem.wav"
            signal = (np.random.default_rng(5).standard_normal(32000) * .03).astype(np.float32)
            shifted = np.concatenate((np.zeros(73, dtype=np.float32), signal[:-73]))
            sf.write(str(raw_path), signal, 16000, subtype="PCM_24")
            sf.write(str(stem_path), shifted, 16000, subtype="PCM_24")
            anchors = (Span(4000, 12000), Span(18000, 26000))
            report = estimate_stem_delay(raw_path, stem_path, anchors, maximum_delay_samples=160)
            self.assertEqual(report.delay_samples, 73)
            views = PairedAudio(raw_path, stem_path, report)

            class FixtureEncoder:
                def encode(self, waveform, alignment, interval):
                    source_span = alignment.to_source(interval)
                    return FeatureSequence(waveform.float().mean().reshape(1, 1), (source_span,),
                                           alignment.source.source_sha256, BACKBONE)

            raw, stem = views.encode(Span(5000, 7000), FixtureEncoder())
            self.assertEqual(raw.cells, stem.cells)
            torch.testing.assert_close(raw.values, stem.values, atol=1e-7, rtol=0)
            sf.write(str(stem_path), np.zeros_like(signal), 16000, subtype="PCM_24")
            with self.assertRaisesRegex(ValueError, "changed after"):
                PairedAudio(raw_path, stem_path, report)

    def test_unrelated_stem_cannot_be_marked_aligned(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_path, stem_path = root / "raw.wav", root / "stem.wav"
            rng = np.random.default_rng(9)
            sf.write(str(raw_path), rng.standard_normal(32000) * .02, 16000)
            sf.write(str(stem_path), rng.standard_normal(32000) * .02, 16000)
            with self.assertRaisesRegex(ValueError, "not reliably time aligned"):
                estimate_stem_delay(raw_path, stem_path, (Span(4000, 12000), Span(18000, 26000)),
                                    maximum_delay_samples=160)

    def test_tdnn_dilation_changes_geometry_and_count(self):
        config = SimpleNamespace(conv_kernel=[10, 3, 3, 3, 3, 2, 2],
                                 conv_stride=[5, 2, 2, 2, 2, 2, 2],
                                 tdnn_kernel=[5, 3, 3, 1, 1], tdnn_dilation=[1, 2, 3, 1, 1])
        geometry = FrameGeometry.from_config(config)
        self.assertEqual((geometry.hop, geometry.convolution_support), (320, 4880))
        self.assertEqual(geometry.count(16000), 35)
        self.assertEqual(geometry.cells(1), (Span(2280, 2600),))
        self.assertEqual(geometry.count(4879), 0)

    def test_unaligned_views_and_nan_features_fail(self):
        raw = FeatureSequence(torch.ones(2, 4), (Span(0, 10), Span(10, 20)), SOURCE, BACKBONE)
        with self.assertRaises(ValueError):
            require_paired(raw, replace(raw, cells=(Span(1, 11), Span(11, 21))))
        with self.assertRaises(ValueError):
            replace(raw, values=torch.full((2, 4), float("nan")))


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(12)
        self.model = TemporalIdentityModel(8, 4, 4).eval()
        self.raw, self.stem = torch.randn(12, 8), torch.randn(12, 8)
        self.ref = ReferenceFeatures(torch.randn(5, 8), torch.randn(5, 8),
                                     torch.ones(5, dtype=torch.bool), torch.ones(5))
        self.allowed = torch.tensor([False] * 2 + [True] * 8 + [False] * 2)

    def call(self, ref=None, exclusions=()):
        return self.model(self.raw, self.stem, ref or self.ref, exclusions, self.allowed)

    def test_reference_order_and_exact_duplicate_do_not_change_prediction(self):
        before = self.call()
        indices = torch.tensor([4, 2, 0, 1, 3, 3])
        ref = ReferenceFeatures(*(value[indices] for value in (
            self.ref.raw, self.ref.stem, self.ref.valid, self.ref.quality)))
        after = self.call(ref)
        torch.testing.assert_close(before.frames, after.frames)
        torch.testing.assert_close(before.purity, after.purity)

    def test_exclusion_group_order_invariant_and_groups_retained(self):
        other = replace(self.ref, raw=-self.ref.raw, stem=-self.ref.stem)
        before, after = self.call(exclusions=(self.ref, other)), self.call(exclusions=(other, self.ref))
        self.assertEqual(len(before.exclusion_scores), 2)
        torch.testing.assert_close(before.frames, after.frames)

    def test_hole_in_output_mask_cannot_stitch_around_other_voice(self):
        self.allowed[5] = False
        with self.assertRaises(ValueError):
            self.call()

    def test_independent_multilabel_heads_allow_target_and_other_together(self):
        output = self.call()
        labels = torch.zeros_like(output.frames)
        labels[3, HEADS.index("target")] = labels[3, HEADS.index("other")] = 1
        loss = identity_loss(output, labels, torch.tensor(0.), torch.tensor([1., 0.]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(self.model.frame_head.weight.grad)

    def test_all_unknown_labels_do_not_train_as_negative(self):
        output = self.call()
        with self.assertRaises(ValueError):
            identity_loss(output, torch.full_like(output.frames, -1), torch.tensor(-1.), torch.tensor([-1., -1.]))

    def test_research_checkpoint_requires_explicit_mode_and_matching_calibration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "bundle"
            card = write_research_checkpoint(root, self.model, backbone_digest=BACKBONE,
                                              dataset_digest="e" * 64, split_digest="f" * 64,
                                              limitations="Synthetic unit fixture only; unusable for real audio.")
            calibration = InferenceCalibration(replace(policy(), model_digest=card.weights_digest),
                                               .5, .5, .5, .9, .5, .9)
            write_research_calibration(root, calibration, dataset_digest=card.dataset_digest)
            with self.assertRaises(FileExistsError):
                write_research_calibration(root, calibration, dataset_digest=card.dataset_digest)
            with self.assertRaises(ValueError):
                load_bundle(root)
            loaded, _, _ = load_bundle(root, research=True)
            torch.testing.assert_close(loaded(self.raw, self.stem, self.ref, (), self.allowed).frames,
                                       self.call().frames)
            wrong_dataset = replace(calibration,
                                    identity=replace(calibration.identity, data_digest="9" * 64))
            (root / "calibration.json").write_text(json.dumps(asdict(wrong_dataset)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_bundle(root, research=True)
            (root / "calibration.json").write_text(json.dumps(asdict(calibration)), encoding="utf-8")
            with (root / "weights.safetensors").open("ab") as stream:
                stream.write(b"corrupt")
            with self.assertRaises(ValueError):
                load_bundle(root, research=True)

    def test_research_calibration_cannot_claim_another_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "bundle"
            card = write_research_checkpoint(root, self.model, backbone_digest=BACKBONE,
                                              dataset_digest="e" * 64, split_digest="f" * 64,
                                              limitations="Synthetic unit fixture only")
            calibration = InferenceCalibration(replace(policy(), model_digest=card.weights_digest),
                                               .5, .5, .5, .9, .5, .9)
            with self.assertRaisesRegex(ValueError, "dataset"):
                write_research_calibration(root, calibration, dataset_digest="f" * 64)
            self.assertFalse((root / "calibration.json").exists())

    def test_session_engine_and_ledger_join_without_production_imports(self):
        from extractor.nextgen.artifacts import ModelCard

        # Controlled logits test plumbing only; not a trained speaker model.
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.zero_()
            for name in ("target", "speech", "observable"):
                self.model.frame_head.bias[HEADS.index(name)] = 8
            for name in ("other", "singing", "overlap", "change", "uncertainty"):
                self.model.frame_head.bias[HEADS.index(name)] = -8
            self.model.purity_head[-1].bias.fill_(8)
            self.model.boundary_head.bias.fill_(8)
        cells = tuple(Span(i * 10, (i + 1) * 10) for i in range(10))
        reference = Reference("1" * 64, "2" * 64, Span(0, 100), "target", "3" * 64, "4" * 64)
        bank = ReferenceBank((reference,))
        ref_features = FeatureSequence(torch.randn(10, 8), cells, reference.source_digest, BACKBONE)
        encoded = EncodedReference(reference.clip_digest, ref_features, ref_features,
                                    torch.ones(10, dtype=torch.bool), torch.ones(10))
        card = ModelCard(self.model.architecture, MODEL, BACKBONE, "e" * 64, "f" * 64,
                         "research", (), "Synthetic plumbing fixture, never usable audio weights.")
        calibration = InferenceCalibration(policy(), .5, .5, .5, .9, .5, .9)
        legacy_card = replace(card, architecture=TemporalIdentityModel.compatible_architectures[0])
        with self.assertRaisesRegex(ValueError, "architecture"):
            IdentitySession(self.model, legacy_card, calibration, bank, (encoded,), research=True)
        session = IdentitySession(self.model, card, calibration, bank, (encoded,), research=True)
        # This plumbing fixture declares target speech, so its query tokens
        # must actually match the reference when the v2 identity anchor runs.
        query = FeatureSequence(ref_features.values.clone(), cells, SOURCE, BACKBONE)
        original = candidate()
        result = run_prepared(SourceTimeline(SOURCE, 1000, 100), (original,), session,
                              lambda row: (query, query), silence=SilenceRange(2, 8))
        self.assertEqual(len(result.selected), 1)
        self.assertEqual(len(result.reviews[0].head_scores), len(cells))
        self.assertEqual(len(result.reviews[0].head_scores[0]), len(HEADS))
        self.assertEqual(len(result.reviews[0].boundary_scores), 2)
        self.assertEqual(result.selected[0].output, original.output)
        self.assertEqual(result.selected[0].parents, (original.key,))
        self.assertEqual(len(result.ledger.decisions), 1)
        incomplete = replace(original, start_complete=False)
        unresolved = run_prepared(SourceTimeline(SOURCE, 1000, 100), (incomplete,), session,
                                  lambda row: (query, query), silence=SilenceRange(2, 8))
        self.assertFalse(unresolved.selected)
        self.assertIn("acoustic_boundary_incomplete", unresolved.reviews[0].assessment.reasons)
        cancelled = run_prepared(SourceTimeline(SOURCE, 1000, 100), (original,), session,
                                 lambda row: self.fail("Cancellation must precede encoding"),
                                 silence=SilenceRange(2, 8), cancelled=lambda: True)
        self.assertTrue(cancelled.cancelled)
        self.assertFalse(cancelled.selected)

    def test_one_epoch_training_roundtrip_is_research_only(self):
        from safetensors.torch import save_file
        from extractor.nextgen.features import file_digest
        from training.train_nextgen import train

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            for number, split in ((1, "train"), (2, "development")):
                path = root / f"{split}.safetensors"
                # Distinct synthetic fixtures on disjoint declared identities.
                example = {"raw": self.raw + number, "stem": self.stem + number,
                           "allowed": self.allowed, "target.raw": self.ref.raw,
                           "target.stem": self.ref.stem, "target.valid": self.ref.valid,
                           "target.quality": self.ref.quality,
                           "labels.frames": torch.zeros(12, len(HEADS)),
                           "labels.purity": torch.tensor(0.), "labels.boundaries": torch.tensor([1., 1.])}
                save_file(example, str(path))
                row = DataRecord(str(number), split, (f"source-{number}",), (f"speaker-{number}",),
                                 (str(number) * 64,), path.name, file_digest(path), BACKBONE,
                                 "Generated unit fixture only", True, False, True)
                records.append(asdict(row))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"schema": 1, "records": records}), encoding="utf-8")
            report = train(manifest, root, root / "bundle", epochs=1,
                           feature_dim=8, projection_dim=4, hidden_dim=4)
            self.assertEqual(report["status"], "requires_calibration_and_independent_evaluation")
            self.assertFalse((root / "bundle" / "calibration.json").exists())
            self.assertEqual(report["training_supervision"]["observable"]["positive"], 0)

    def test_validated_card_requires_supervision_for_every_consumed_head(self):
        from extractor.nextgen.artifacts import ModelCard

        with self.assertRaisesRegex(ValueError, "supervision"):
            ModelCard(self.model.architecture, MODEL, BACKBONE, "e" * 64, "f" * 64,
                      "validated", ("d" * 64,), "Unit fixture only", {}, "1" * 64)
        coverage = {name: {"positive": 2, "negative": 2}
                    for name in (*HEADS, "purity", "boundary_start", "boundary_end")}
        with self.assertRaisesRegex(ValueError, "calibration"):
            ModelCard(self.model.architecture, MODEL, BACKBONE, "e" * 64, "f" * 64,
                      "validated", ("d" * 64,), "Unit fixture only", coverage)
        card = ModelCard(self.model.architecture, MODEL, BACKBONE, "e" * 64, "f" * 64,
                         "validated", ("d" * 64,), "Unit fixture only", coverage,
                         "1" * 64)
        self.assertEqual(card.supervision["other"]["negative"], 2)


class PolicyAndDataTests(unittest.TestCase):
    def test_change_head_needs_independent_complete_gap_before_splitting(self):
        parent = candidate(10, 90)
        frames = (FramePrediction(Span(30, 35), .9, .01, .01, True, .96),
                  FramePrediction(Span(60, 65), .9, .01, .01, True, .94))
        first = AcousticGap(Span(34, 37), True, True, True, "two_view_acoustic_review")
        second = AcousticGap(Span(63, 67), True, True, True, "two_view_acoustic_review")
        parts = propose_change_parts(parent, frames, (first, second),
                                     change_min=.9, tolerance_samples=0)
        self.assertEqual([row.output for row in parts],
                         [Span(10, 34), Span(37, 63), Span(67, 90)])
        self.assertEqual([row.parents for row in parts], [(parent.key,)] * 3)
        unverified = replace(first, verified=False)
        self.assertEqual(propose_change_parts(parent, frames, (unverified,),
                                              change_min=.9, tolerance_samples=0), ())
        incomplete = replace(first, right_complete=False)
        self.assertEqual(propose_change_parts(parent, frames, (incomplete,),
                                              change_min=.9, tolerance_samples=0), ())
        distant = replace(first, span=Span(45, 47))
        self.assertEqual(propose_change_parts(parent, frames, (distant,),
                                              change_min=.9, tolerance_samples=0), ())

    def test_continuous_change_keeps_both_sides_unresolved_until_cut_verified(self):
        parent = candidate(10, 90)
        frames = (FramePrediction(Span(46, 50), .9, .01, .01, True, .96),)
        parts = propose_unresolved_change_sides(parent, frames, change_min=.9)
        self.assertEqual([row.output for row in parts], [Span(10, 46), Span(50, 90)])
        self.assertFalse(parts[0].end_complete)
        self.assertFalse(parts[1].start_complete)
        self.assertEqual([row.parents for row in parts], [(parent.key,), (parent.key,)])

    def test_engine_reviews_new_exact_child_spans_after_change_proposal(self):
        root = candidate()
        gap = AcousticGap(Span(38, 42), True, True, True, "independent_acoustic_pause")

        class FixtureSession:
            bank = SimpleNamespace(digest=REFS)
            card = SimpleNamespace(weights_digest=MODEL)
            calibration = InferenceCalibration(policy(), .5, .5, .5, .9, .5, .9)

            def review(self, row, raw, stem, evidence):
                mixed = row.output == root.output
                frames = (
                    FramePrediction(Span(10, 38), .95, .01, .01, True, .01),
                    FramePrediction(gap.span, .95, .01, .01, True, .95),
                    FramePrediction(Span(42, 70), .95, .01, .01, True, .01),
                ) if mixed else (FramePrediction(row.output, .95, .01, .01, True, .01),)
                pred = SpanPrediction(row.key, REFS, MODEL, frames, .99)
                local = (
                    LocalEvidence(SOURCE, 1000, gap.span, EvidenceKind.UNCERTAIN,
                                  REFS, MODEL, "temporal_change"),
                    LocalEvidence(SOURCE, 1000, Span(15, 20), EvidenceKind.OTHER,
                                  REFS, MODEL, "parent_only_model_prediction"),
                ) if mixed else ()
                result = assess(row, pred, policy(), reference_digest=REFS, local_evidence=local)
                return InferenceResult(row, pred, result, local)

        result = run_prepared(SourceTimeline(SOURCE, 1000, 100), (root,), FixtureSession(),
                              lambda row: (None, None), silence=SilenceRange(2, 8),
                              boundary_options=lambda review: (gap,))
        self.assertEqual([row.output for row in result.selected], [Span(10, 38), Span(42, 70)])
        self.assertEqual(len(result.reviews), 5)
        self.assertEqual(result.reviews[0].assessment.state, State.REJECTED)
        self.assertTrue(all(row.assessment.state != State.ACCEPTED
                            for row in result.reviews if row.candidate.origin == "unresolved_continuous_change"))
        upstream_other = (LocalEvidence(SOURCE, 1000, Span(15, 20), EvidenceKind.OTHER,
                                        REFS, MODEL, "independently_verified_event"),)
        blocked = run_prepared(SourceTimeline(SOURCE, 1000, 100), (root,), FixtureSession(),
                               lambda row: (None, None), silence=SilenceRange(2, 8),
                               evidence=upstream_other, boundary_options=lambda review: (gap,))
        self.assertEqual([row.output for row in blocked.selected], [Span(42, 70)])

    def test_uncovered_vad_gap_cannot_hide_other_sound(self):
        row = replace(candidate(), speech=(Span(10, 30), Span(50, 70)))
        predicted = replace(prediction(row), frames=(FramePrediction(Span(10, 30), .95, .01, .01, True),
                                                     FramePrediction(Span(50, 70), .95, .01, .01, True)))
        result = assess(row, predicted, policy(), reference_digest=REFS)
        self.assertIn("missing_temporal_coverage", result.reasons)

    def test_normal_silence_uncertainty_does_not_require_speaker_identity(self):
        row = replace(candidate(), speech=(Span(10, 30), Span(50, 70)))
        predicted = replace(prediction(row), frames=(FramePrediction(Span(10, 30), .95, .01, .01, True),
                                                     FramePrediction(Span(30, 50), .1, .01, 1., False),
                                                     FramePrediction(Span(50, 70), .95, .01, .01, True)))
        self.assertEqual(assess(row, predicted, policy(), reference_digest=REFS).state, State.ACCEPTED)

    def record(self, split="train", ident="1"):
        return DataRecord(ident, split, ("source-" + ident,), ("person-" + ident,),
                          (ident * 64,), ident + ".safetensors", "e" * 64, BACKBONE,
                          "unit-test data declaration", True, True, False)

    def test_source_and_reference_person_leakage_fail_before_training(self):
        train = self.record()
        for field in ("sources", "speakers", "content_digests"):
            heldout = replace(self.record("test", "2"), **{field: getattr(train, field)})
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit_records((train, heldout), for_training=True)

    def test_feature_shard_path_rejects_windows_root_only_path(self):
        with self.assertRaisesRegex(ValueError, "relative"):
            replace(self.record(), features_path="\\shard.safetensors")

    def test_unknown_training_rights_stop_before_reading_audio(self):
        with self.assertRaises(ValueError):
            audit_records((replace(self.record(), training_allowed=False),), for_training=True)

    def test_calibration_never_uses_test_or_accepts_zero_as_success(self):
        good, bad = candidate(), candidate(70, 90)
        examples = (CalibrationExample(good, prediction(good), (), True, "calibration"),
                    CalibrationExample(bad, prediction(bad, .95), (), False, "calibration"))
        chosen, report = choose_policy(examples, (policy(),), minimum_correct=1)
        self.assertEqual(chosen, policy())
        self.assertEqual(report["scores"][0]["wrong"], 0)
        with self.assertRaises(ValueError):
            choose_policy((replace(examples[0], partition="test"), examples[1]), (policy(),), minimum_correct=1)
        with self.assertRaises(ValueError):
            choose_policy(examples, (replace(policy(), target_min=1.),), minimum_correct=1)

    def test_full_calibration_requires_separable_provenanced_heads(self):
        good, bad = candidate(), candidate(70, 90)
        train = self.record("train", "1")
        calibration_record = replace(self.record("calibration", "2"),
                                     content_digests=(SOURCE,))
        audit = audit_records((train, calibration_record), for_training=True)
        calibrated_policy = replace(policy(), data_digest=audit["dataset_digest"])
        examples = (
            CalibrationExample(good, prediction(good), (), True, "calibration", "2"),
            CalibrationExample(bad, prediction(bad, .95), (), False, "calibration", "2"),
        )
        heads = ("observable", "singing", "overlap", "boundary", "speech", "change")
        scores = tuple(BinaryHeadExample("2", SOURCE, name, score, label, "calibration")
                       for name in heads for score, label in ((.1, False), (.9, True)))
        calibrated, report = fit_complete_calibration(
            examples, (calibrated_policy,), scores, (train, calibration_record),
            minimum_correct=1,
        )
        self.assertEqual(calibrated.boundary_min, .5)
        self.assertEqual(report["audit"]["counts"]["calibration"], 1)
        with self.assertRaisesRegex(ValueError, "separate"):
            fit_complete_calibration(
                examples, (calibrated_policy,),
                tuple(replace(row, score=.8 if not row.label else .2)
                      if row.head == "singing" else row for row in scores),
                (train, calibration_record), minimum_correct=1,
            )
        with self.assertRaisesRegex(ValueError, "provenance"):
            fit_complete_calibration(
                examples, (calibrated_policy,),
                tuple(replace(row, source_sha256="f" * 64) for row in scores),
                (train, calibration_record), minimum_correct=1,
            )

    def test_replay_is_explicitly_not_acoustic_validation(self):
        row = candidate()
        report = replay({"schema": 1, "purpose": "research_policy_replay", "calibration": asdict(policy()),
                         "reference_digest": REFS, "rows": [{"candidate": asdict(row),
                                                               "prediction": asdict(prediction(row))}]})
        self.assertEqual(report["purpose"], "policy_replay_not_acoustic_validation")
        self.assertEqual(len(report["selected"]), 1)


class DeliveryAndMetricTests(unittest.TestCase):
    def test_known_short_other_sound_invalidates_whole_output_in_metrics(self):
        truth = AcousticTruth(SourceTimeline(SOURCE, 1000, 100), (Span(0, 100),),
                              (Span(10, 70),), other=(Span(70, 71),))
        report = evaluate(truth, (candidate(10, 71),))
        self.assertEqual(report["complete_correct_utterances"], 0)
        self.assertEqual(report["other_samples"], 1)
        self.assertEqual(report["complete_utterance_recall"], 0)

    def test_fragments_and_duplicates_do_not_inflate_complete_recall(self):
        truth = AcousticTruth(SourceTimeline(SOURCE, 1000, 100), (Span(0, 100),), (Span(10, 70),))
        report = evaluate(truth, (candidate(10, 40), candidate(40, 70), candidate(10, 40)))
        self.assertEqual(report["complete_correct_utterances"], 0)
        self.assertEqual(report["duplicate_samples"], 30)

    def test_incomplete_review_not_reported_as_full_episode_recall(self):
        truth = AcousticTruth(SourceTimeline(SOURCE, 1000, 100), (Span(0, 80),), (Span(10, 70),))
        self.assertEqual(evaluate(truth, (candidate(),))["scope"], "manually_reviewed_subset_only")

    def test_normalization_raises_quiet_audio_without_changing_length(self):
        samples = np.full(60, .001, dtype=np.float32)
        output, gain = normalize_clip(samples, np.ones(60, dtype=bool), ExportPolicy(0))
        self.assertGreater(gain, 0)
        self.assertEqual(len(output), len(samples))
        self.assertLessEqual(float(np.max(np.abs(output))), 10 ** (-1 / 20))

    def test_hard_silence_or_incomplete_boundary_fails_final_audit(self):
        row = replace(candidate(), speech=(Span(10, 20), Span(40, 70)))
        with self.assertRaises(ValueError):
            final_audit(row, assessment(row), SilenceRange(2, 8))
        row = replace(candidate(), end_complete=False)
        with self.assertRaises(ValueError):
            final_audit(row, assessment(row), SilenceRange(2, 8))

    def item(self, root):
        stem = root / "source.wav"
        sf.write(str(stem), np.sin(np.arange(100) * .3).astype(np.float32) * .02, 1000)
        source = SourceTimeline(SOURCE, 1000, 100)
        row = candidate()
        return ExportItem(row, assessment(row), stem, ViewAlignment(source, 1000, 100, 0, True), SilenceRange(2, 8))

    def test_stt_failure_retains_exact_audio_outside_training_zip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def failure(path):
                raise RuntimeError("simulated STT error")
            result = deliver((self.item(root),), root / "out", ExportPolicy(0), failure)
            row = result["records"][0]
            self.assertEqual(row["stt_status"], "failed")
            self.assertEqual(sf.info(str(root / "out" / row["audio"])).frames, 60)
            self.assertIsNone(result["zip"])

    def test_success_uses_japanese_stt_and_one_zip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = deliver((self.item(root),), root / "out", ExportPolicy(0), lambda path: "よろしくお願いします")
            row = result["records"][0]
            self.assertEqual((root / "out" / row["text"]).read_text(encoding="utf-8").strip(), "よろしくお願いします")
            with zipfile.ZipFile(result["zip"]) as archive:
                self.assertEqual(len([name for name in archive.namelist() if name.endswith(".wav")]), 1)

    def test_zip_failure_does_not_delete_audio_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("extractor.nextgen.finalization.zipfile.ZipFile", side_effect=OSError("disk error")):
                result = deliver((self.item(root),), root / "out", ExportPolicy(0), lambda path: "日文")
            self.assertIn("disk error", result["zip_error"])
            self.assertTrue((root / "out" / result["records"][0]["audio"]).exists())

    def test_cancel_and_length_filter_are_explicit_not_identity_rejections(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            item = self.item(root)
            stopped = deliver((item,), root / "cancelled", ExportPolicy(0), lambda path: "x", cancelled=lambda: True)
            self.assertEqual(stopped["state"], "cancelled")
            short = deliver((item,), root / "filtered", ExportPolicy(61), lambda path: "x")
            self.assertEqual(short["records"][0]["stt_status"], "export_length_policy")


if __name__ == "__main__":
    unittest.main()
