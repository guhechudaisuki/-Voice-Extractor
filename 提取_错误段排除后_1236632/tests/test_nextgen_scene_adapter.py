from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extractor.nextgen.boundary_decoder import SilenceRange
from extractor.nextgen.ledger import EvidenceKind
from extractor.nextgen.scene_adapter import make_scene
from extractor.nextgen.prepare_media import (_alignment_anchors, _samples, _verify_alignment,
                                             load_prepared_scene)
from extractor.nextgen.prepared_audio import PairedAudio, estimate_stem_delay
from extractor.nextgen.features import FeatureSequence
from extractor.nextgen.reference_preparation import ReferenceMaterial, prepare_reference_bank
from extractor.nextgen.runtime import run_scene
from extractor.nextgen.timeline import SampleSpan as Span, SourceTimeline
from extractor.types import TimeSpan


SOURCE, REFS, MODEL = "a" * 64, "b" * 64, "c" * 64


class AudioFixture:
    def __init__(self):
        self.source = SourceTimeline(SOURCE, 16000, 160000)
        self.alignment_report = SimpleNamespace(raw_digest="d" * 64, stem_digest="e" * 64)
        self.encode_contexts = []

    def read_pair(self, span):
        # Speech on either side, a true 0.30-second pause in the middle.
        raw = np.ones(span.length, dtype=np.float32) * .1
        stem = raw.copy()
        left, right = max(span.start, 32000), min(span.end, 36800)
        if left < right:
            stem[left - span.start:right - span.start] = 0
        return raw, stem

    def encode(self, context, encoder):
        self.encode_contexts.append(context)
        cells = tuple(Span(start, start + 320)
                      for start in range(context.start + 2240, context.end - 2240, 320))
        values = torch.ones(len(cells), 4)
        sequence = FeatureSequence(values, cells, SOURCE, encoder.digest)
        return sequence, sequence


class PreparedSceneTests(unittest.TestCase):
    def test_cached_scene_rejects_changed_audio_and_reuses_identical_candidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            signal = np.random.default_rng(23).standard_normal(64000).astype(np.float32) * .02
            raw_path, stem_path = root / "original_16000.wav", root / "stem_16000.wav"
            sf.write(str(raw_path), signal, 16000, subtype="PCM_24")
            sf.write(str(stem_path), signal, 16000, subtype="PCM_24")
            alignment = estimate_stem_delay(raw_path, stem_path,
                                            (Span(5000, 14000), Span(35000, 45000)),
                                            maximum_delay_samples=160)
            audio = PairedAudio(raw_path, stem_path, alignment)
            scene = make_scene(audio, raw_voice=(Span(8000, 25000),),
                               stem_voice=(Span(8000, 25000),))
            report = {
                "schema": 1, "singing_before_uvr": True,
                "source_sha256": audio.source.source_sha256,
                "alignment": asdict(alignment), "silence": asdict(scene.silence),
                "raw_voice": [asdict(row) for row in scene.raw_voice],
                "stem_voice": [asdict(row) for row in scene.stem_voice],
                "singing_masks": [], "overlap_masks": [], "subtitle_hints": [],
                "candidate_count": len(scene.candidates),
                "candidates": [{"output": asdict(row.output), "origin": row.origin,
                                "start_complete": row.start_complete,
                                "end_complete": row.end_complete} for row in scene.candidates],
            }
            (root / "preparation.json").write_text(json.dumps(report), encoding="utf-8")
            loaded = load_prepared_scene(root)
            self.assertEqual(loaded.candidates, scene.candidates)
            sf.write(str(stem_path), np.zeros_like(signal), 16000, subtype="PCM_24")
            with self.assertRaisesRegex(ValueError, "changed after"):
                load_prepared_scene(root)

    def test_scene_runner_binds_features_events_and_matching_backbone(self):
        scene = make_scene(AudioFixture(), raw_voice=(Span(16000, 32000),),
                           stem_voice=(Span(16000, 32000),),
                           singing=(Span(70000, 75000),))
        session = SimpleNamespace(card=SimpleNamespace(backbone_digest="f" * 64,
                                                        weights_digest=MODEL),
                                  bank=SimpleNamespace(digest=REFS))
        encoder = SimpleNamespace(digest="f" * 64)
        with patch("extractor.nextgen.runtime.run_prepared", return_value="sentinel") as downstream:
            self.assertEqual(run_scene(scene, session, encoder), "sentinel")
        self.assertEqual(downstream.call_args.kwargs["evidence"][0].kind,
                         EvidenceKind.SINGING)
        raw, stem = downstream.call_args.args[3](scene.candidates[0])
        self.assertEqual(raw.cells, stem.cells)
        with self.assertRaisesRegex(ValueError, "encoder"):
            run_scene(scene, session, SimpleNamespace(digest="0" * 64))

    def test_reference_bank_keeps_roles_separate_and_removes_same_waveform(self):
        scene = make_scene(AudioFixture(), raw_voice=(Span(16000, 32000),),
                           stem_voice=(Span(16000, 32000),))
        encoder = SimpleNamespace(digest="f" * 64)
        bank, encoded = prepare_reference_bank((ReferenceMaterial("target", scene),
                                                ReferenceMaterial("target", scene)), encoder)
        self.assertEqual(len(bank.entries), 1)
        self.assertEqual(len(encoded), 1)
        self.assertGreaterEqual(int(encoded[0].valid.sum()), 12)
        with self.assertRaisesRegex(ValueError, "target and exclusion"):
            prepare_reference_bank((ReferenceMaterial("target", scene),
                                    ReferenceMaterial("other_1", scene)), encoder)

    def test_reference_context_cannot_attend_to_neighboring_voice(self):
        audio = AudioFixture()
        scene = make_scene(audio, raw_voice=(Span(16000, 32000), Span(33000, 50000)),
                           stem_voice=(Span(16000, 32000), Span(33000, 50000)))
        bank, _ = prepare_reference_bank((ReferenceMaterial("target", scene),),
                                         SimpleNamespace(digest="f" * 64))
        self.assertEqual(len(bank.entries), 2)
        self.assertLessEqual(audio.encode_contexts[0].end, 33000)
        self.assertGreaterEqual(audio.encode_contexts[1].start, 32000)

    def test_real_pair_anchors_are_measured_not_assumed_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_path, stem_path = root / "raw.wav", root / "stem.wav"
            signal = np.random.default_rng(17).standard_normal(128000).astype(np.float32) * .03
            shifted = np.concatenate((np.zeros(73, np.float32), signal[:-73]))
            sf.write(str(raw_path), signal, 16000, subtype="PCM_24")
            sf.write(str(stem_path), shifted, 16000, subtype="PCM_24")
            speech = _samples([TimeSpan(.5, 2.5), TimeSpan(4, 6)], len(signal))
            anchors = _alignment_anchors(speech, speech, (), len(signal))
            self.assertGreaterEqual(len(anchors), 2)
            self.assertEqual(_verify_alignment(raw_path, stem_path, anchors).delay_samples, 73)
            shifted[64000:] = np.concatenate((np.zeros(120, np.float32),
                                               signal[64000:-120]))
            sf.write(str(stem_path), shifted, 16000, subtype="PCM_24")
            with self.assertRaisesRegex(ValueError, "No stable"):
                _verify_alignment(raw_path, stem_path, anchors)

    def test_raw_only_voice_remains_a_proposal_but_subtitle_is_not_identity(self):
        scene = make_scene(
            AudioFixture(), raw_voice=(Span(16000, 24000), Span(48000, 64000)),
            stem_voice=(Span(48000, 64000),),
            subtitle_hints=(Span(14000, 25000),),
        )
        self.assertEqual(len(scene.candidates), 2)
        self.assertEqual({item.origin for item in scene.candidates},
                         {"stem_vad", "raw_vad_rescue+subtitle_time_hint"})
        self.assertEqual(scene.evidence(REFS, MODEL), ())
        self.assertFalse(any(item.output == Span(14000, 25000) for item in scene.candidates))

    def test_confirmed_singing_and_overlap_never_become_joinable_pause(self):
        scene = make_scene(
            AudioFixture(), raw_voice=(Span(16000, 30000), Span(32000, 50000)),
            stem_voice=(Span(16000, 30000), Span(32000, 50000)),
            singing=(Span(30000, 32000),), overlap=(Span(70000, 74000),),
            silence=SilenceRange(3200, 13600),
        )
        self.assertEqual({item.output for item in scene.candidates},
                         {Span(16000, 30000), Span(32000, 50000)})
        self.assertFalse(any(item.output.start < 32000 and item.output.end > 30000
                             for item in scene.candidates))
        self.assertEqual([row.kind for row in scene.evidence(REFS, MODEL)],
                         [EvidenceKind.SINGING, EvidenceKind.OVERLAP])

    def test_mask_inside_vad_keeps_both_sides_unresolved_instead_of_losing_all(self):
        scene = make_scene(
            AudioFixture(), raw_voice=(Span(16000, 70000),),
            stem_voice=(Span(16000, 70000),),
            singing=(Span(35000, 40000),),
        )
        self.assertEqual([row.output for row in scene.candidates],
                         [Span(16000, 35000), Span(40000, 70000)])
        self.assertEqual([(row.start_complete, row.end_complete)
                          for row in scene.candidates],
                         [(True, False), (False, True)])

    def test_one_vad_view_cannot_certify_a_boundary_the_other_marks_inside_speech(self):
        scene = make_scene(
            AudioFixture(), raw_voice=(Span(16000, 30000), Span(31000, 44000)),
            stem_voice=(Span(16000, 30000),),
        )
        first = next(row for row in scene.candidates if row.output == Span(16000, 30000))
        self.assertFalse(first.end_complete)

    def test_gap_requires_both_vad_views_and_waveform_pause(self):
        base = dict(raw_voice=(Span(16000, 32000), Span(36800, 56000)),
                    stem_voice=(Span(16000, 32000), Span(36800, 56000)))
        scene = make_scene(AudioFixture(), **base)
        parent = next(item for item in scene.candidates if item.output == Span(16000, 56000))
        gaps = scene.gap_options(SimpleNamespace(candidate=parent))
        self.assertEqual(tuple(item.span for item in gaps), (Span(32000, 36800),))
        missing = make_scene(AudioFixture(), raw_voice=(Span(16000, 56000),),
                             stem_voice=base["stem_voice"])
        parent = next(item for item in missing.candidates if item.output == Span(16000, 56000))
        self.assertEqual(missing.gap_options(SimpleNamespace(candidate=parent)), ())

    def test_short_context_is_rejected_before_a_model_can_falsely_cover_edges(self):
        with self.assertRaisesRegex(ValueError, "context"):
            make_scene(AudioFixture(), raw_voice=(Span(16000, 32000),),
                       stem_voice=(), context_samples=100)

    def test_single_long_vad_utterance_is_not_silently_dropped(self):
        audio = AudioFixture()
        audio.source = SourceTimeline(SOURCE, 16000, 400000)
        complete_utterance = Span(16000, 336000)  # 20 seconds, above the lattice join cap.
        scene = make_scene(audio, raw_voice=(complete_utterance,),
                           stem_voice=(complete_utterance,))
        self.assertEqual([row.output for row in scene.candidates], [complete_utterance])


if __name__ == "__main__":
    unittest.main()
