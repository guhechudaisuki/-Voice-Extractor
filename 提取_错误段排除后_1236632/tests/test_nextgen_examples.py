from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extractor.nextgen.features import FeatureSequence
from extractor.nextgen.identity_model import HEADS, ReferenceFeatures, TemporalIdentityModel
from extractor.nextgen.reference_preparation import ReferenceMaterial, prepare_reference_bank
from extractor.nextgen.scene_adapter import make_scene
from extractor.nextgen.timeline import SampleSpan as Span, SourceTimeline
from training.nextgen_data import DataRecord, read_manifest
from training.nextgen_examples import (AcousticAnnotation, build_feature_example,
                                      build_prepared_example,
                                      data_record_for_prepared,
                                      frame_labels, save_feature_example, write_manifest)
from training.train_nextgen import example_loss, load_example


SOURCE, BACKBONE = "a" * 64, "b" * 64


def features():
    cells = tuple(Span(index * 10, (index + 1) * 10) for index in range(8))
    raw = FeatureSequence(torch.randn(8, 4), cells, SOURCE, BACKBONE)
    stem = FeatureSequence(torch.randn(8, 4), cells, SOURCE, BACKBONE)
    return raw, stem


def reference():
    return ReferenceFeatures(torch.randn(4, 4), torch.randn(4, 4),
                             torch.ones(4, dtype=torch.bool), torch.ones(4))


class AudioFixture:
    def __init__(self, digest):
        self.source = SourceTimeline(digest, 16000, 160000)
        self.alignment_report = SimpleNamespace(raw_digest="d" * 64, stem_digest="e" * 64)

    def read_pair(self, span):
        values = np.ones(span.length, dtype=np.float32) * .1
        return values, values

    def encode(self, context, encoder):
        cells = tuple(Span(start, start + 320)
                      for start in range(context.start + 2240, context.end - 2240, 320))
        sequence = FeatureSequence(torch.ones(len(cells), 4), cells,
                                   self.source.source_sha256, encoder.digest)
        return sequence, sequence


class NextgenExampleTests(unittest.TestCase):
    def test_prepared_candidate_uses_only_external_references(self):
        encoder = SimpleNamespace(digest=BACKBONE)
        query = make_scene(AudioFixture(SOURCE), raw_voice=(Span(16000, 32000),),
                           stem_voice=(Span(16000, 32000),))
        separate = make_scene(AudioFixture("f" * 64), raw_voice=(Span(16000, 32000),),
                              stem_voice=(Span(16000, 32000),))
        candidate = query.candidates[0]
        annotation = AcousticAnnotation(candidate.output, target=(candidate.output,),
                                        complete_timeline=True, purity=1,
                                        start_complete=1, end_complete=1)
        bank, encoded = prepare_reference_bank((ReferenceMaterial("target", separate),),
                                               encoder)
        example = build_prepared_example(query, candidate, bank, encoded,
                                         encoder, annotation,
                                         query_source_group="episode-1",
                                         reference_source_groups={bank.entries[0].clip_digest: "episode-2"})
        self.assertEqual(example["raw"].shape[1], 4)
        self.assertTrue(torch.isfinite(example_loss(TemporalIdentityModel(4, 4, 4), example)))
        record = data_record_for_prepared(
            "example-1", "train", query, bank,
            query_source_group="episode-1",
            reference_source_groups={bank.entries[0].clip_digest: "episode-2"},
            role_speaker_ids={"target": "actor-1"},
            query_speaker_ids=("actor-1", "actor-2"),
            features_path="example-1.safetensors", features_digest="c" * 64,
            backbone_digest=BACKBONE, license_reference="test permission fixture",
            training_allowed=True, human_labels=True, synthetic=False,
        )
        self.assertEqual(record.sources, ("episode-1", "episode-2"))
        self.assertEqual(record.speakers, ("actor-1", "actor-2"))
        self.assertEqual(set(record.content_digests), {"d" * 64, "e" * 64})
        same_bank, same_encoded = prepare_reference_bank((ReferenceMaterial("target", query),),
                                                          encoder)
        with self.assertRaisesRegex(ValueError, "own source"):
            build_prepared_example(query, candidate, same_bank, same_encoded,
                                   encoder, annotation,
                                   query_source_group="episode-1",
                                   reference_source_groups={same_bank.entries[0].clip_digest:
                                                            "episode-2"})
        with self.assertRaisesRegex(ValueError, "another crop"):
            build_prepared_example(query, candidate, bank, encoded,
                                   encoder, annotation,
                                   query_source_group="episode-1",
                                   reference_source_groups={bank.entries[0].clip_digest:
                                                            "episode-1"})

    def test_unknown_band_stays_unknown_and_overlap_must_be_declared(self):
        annotation = AcousticAnnotation(
            Span(0, 80), target=(Span(0, 30),), other=(Span(50, 80),),
            change=(Span(50, 60),), unknown=(Span(30, 50),),
            observable=(Span(0, 30),), unobservable=(Span(50, 80),),
            uncertain=(Span(30, 50),), certain=(Span(0, 30), Span(50, 80)),
            complete_timeline=True, purity=0, start_complete=1, end_complete=0,
        )
        raw, stem = features()
        labels = frame_labels(raw.cells, annotation)
        self.assertEqual(float(labels[0, HEADS.index("target")]), 1)
        self.assertEqual(float(labels[0, HEADS.index("other")]), 0)
        self.assertTrue(torch.all(labels[3:5, :6] == -1))
        self.assertEqual(float(labels[5, HEADS.index("other")]), 1)
        self.assertEqual(float(labels[5, HEADS.index("change")]), 1)
        self.assertEqual(float(labels[0, HEADS.index("observable")]), 1)
        self.assertEqual(float(labels[5, HEADS.index("observable")]), 0)
        self.assertEqual(float(labels[3, HEADS.index("uncertainty")]), 1)
        self.assertEqual(float(labels[0, HEADS.index("uncertainty")]), 0)
        with self.assertRaisesRegex(ValueError, "opposite"):
            AcousticAnnotation(Span(0, 80), observable=(Span(0, 20),),
                               unobservable=(Span(10, 30),))
        with self.assertRaisesRegex(ValueError, "overlap"):
            AcousticAnnotation(Span(0, 80), target=(Span(0, 30),),
                               other=(Span(20, 40),), complete_timeline=True)
        tensors = build_feature_example(raw, stem, annotation, reference())
        model = TemporalIdentityModel(4, 4, 4)
        self.assertTrue(torch.isfinite(example_loss(model, tensors)))

    def test_partial_annotation_cannot_invent_negative_or_pure_label(self):
        raw, _ = features()
        labels = frame_labels(raw.cells, AcousticAnnotation(Span(0, 80),
                                                            target=(Span(0, 20),)))
        self.assertEqual(float(labels[0, HEADS.index("target")]), 1)
        self.assertEqual(float(labels[4, HEADS.index("target")]), -1)
        with self.assertRaisesRegex(ValueError, "Partial labels"):
            AcousticAnnotation(Span(0, 80), target=(Span(0, 20),), purity=1)
        with self.assertRaisesRegex(ValueError, "target-only"):
            AcousticAnnotation(Span(0, 80), target=(Span(0, 20),),
                               complete_timeline=True, purity=0)

    def test_saved_example_matches_trainer_and_manifest_rights_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, stem = features()
            annotation = AcousticAnnotation(Span(0, 80), target=(Span(0, 80),),
                                            complete_timeline=True, purity=1,
                                            start_complete=1, end_complete=1)
            tensors = build_feature_example(raw, stem, annotation, reference(),
                                            (reference(),))
            path = root / "features" / "train.safetensors"
            digest = save_feature_example(tensors, path)
            with self.assertRaises(FileExistsError):
                save_feature_example(tensors, path)
            record = DataRecord("sample", "train", ("source-1",), ("speaker-1",),
                                ("c" * 64,), "train.safetensors", digest,
                                BACKBONE, "CC BY 4.0", True, True, False)
            self.assertEqual(load_example(path.parent, record, "cpu")["raw"].shape, (8, 4))
            with self.assertRaisesRegex(ValueError, "authorized"):
                write_manifest((DataRecord(**{**record.__dict__, "training_allowed": False}),),
                               root / "denied.json")
            with self.assertRaisesRegex(ValueError, "authorized"):
                write_manifest((DataRecord(**{**record.__dict__, "split": "development",
                                              "training_allowed": False}),),
                               root / "denied-development.json")
            audit = write_manifest((record,), root / "approved.json")
            self.assertEqual(audit["counts"]["train"], 1)
            self.assertEqual(read_manifest(root / "approved.json"), (record,))


if __name__ == "__main__":
    unittest.main()
