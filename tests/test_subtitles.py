from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extractor.pipeline import ExtractionPipeline, PipelineOptions
from extractor.subtitles import SubtitleCue, SubtitleGuide, read_subtitles, validate_subtitle_bindings
from extractor.subtitle_assistance import restore_subtitle_sentences, retry_subtitle_vad, split_at_subtitle_pauses
from extractor.types import CandidateSentence, PipelineResult, TimeSpan


def setUpModule():
    (ROOT / "work").mkdir(parents=True, exist_ok=True)


def guide_for(*cues):
    return SubtitleGuide(list(cues), "dialogue.srt", aligned=True,
                         anchors=[(0, 0), (20, 0)], report={})


class SubtitleParsingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "work")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, text, encoding="utf-8"):
        path = self.root / name
        path.write_text(text, encoding=encoding)
        return path

    def test_srt_vtt_and_utf16(self):
        for name, text, encoding in (
            ("cn.srt", "1\n00:00:01,250 --> 00:00:02,900\n你好\n", "utf-16"),
            ("cn-gb.srt", "1\n00:00:01,250 --> 00:00:02,900\n你好\n", "gb18030"),
            ("jp.vtt", "WEBVTT\n\n00:00:01.250 --> 00:00:02.900\nこんにちは\n", "utf-8"),
        ):
            cues = read_subtitles(self.write(name, text, encoding))
            self.assertEqual(len(cues), 1)
            self.assertEqual((cues[0].start, cues[0].end), (1.25, 2.9))
            self.assertEqual(cues[0].kind, "speech")

    def test_ass_lyrics_signs_comments_and_duplicates(self):
        text = """[Script Info]
ScriptType: v4.00+
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{\\i1}你好
Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,Hello
Dialogue: 0,0:00:03.00,0:00:05.00,oped_ja,,0,0,0,,song
Dialogue: 0,0:00:06.00,0:00:07.00,Default,,0,0,0,,{\\pos(50,80)}sign
Comment: 0,0:00:08.00,0:00:09.00,Default,,0,0,0,,comment
"""
        cues = read_subtitles(self.write("test.ass", text))
        self.assertEqual([c.kind for c in cues], ["speech", "lyrics", "screen_text"])
        self.assertEqual(cues[0].text, "你好")

    def test_wrong_target_binding_fails_before_loading(self):
        path = self.write("test.srt", "1\n00:00:01,000 --> 00:00:02,000\nHi\n")
        with self.assertRaises(ValueError):
            validate_subtitle_bindings([self.root / "episode1.wav"], {self.root / "episode2.wav": path})
        self.assertEqual(validate_subtitle_bindings([self.root / "episode1.wav"], {}), {})

    def test_malformed_subtitle_does_not_silently_enable_assistance(self):
        with self.assertRaises(ValueError):
            read_subtitles(self.write("broken.ass", "no events"))


class SubtitleTimingTests(unittest.TestCase):
    def timeline(self, shift=0):
        starts = [2, 7, 13, 21, 31, 44, 58, 74, 93, 115]
        durations = [1.2, 2.4, 1.5, 3.2, .9, 1.7, 2.2, 1.8, 3.1, 1.4]
        cues = [SubtitleCue(i, t, t + d, "translated") for i, (t, d) in enumerate(zip(starts, durations))]
        speech = [TimeSpan(c.start + shift, c.end + shift) for c in cues]
        return SubtitleGuide(cues, "test.srt"), speech

    def test_fractional_offsets(self):
        for shift in (-.35, 0, .45):
            guide, speech = self.timeline(shift)
            guide.calibrate(speech)
            self.assertTrue(guide.aligned, guide.report)
            self.assertAlmostEqual(guide.offset, shift, places=2)

    def test_mismatched_or_empty_timeline_falls_back(self):
        guide, speech = self.timeline(10)
        guide.calibrate(speech)
        self.assertFalse(guide.aligned)
        guide.calibrate([])
        self.assertFalse(guide.aligned)
        self.assertEqual(guide.groups(speech, .85), [])

    def test_no_local_anchors_no_assistance(self):
        guide = guide_for(SubtitleCue(1, 200, 203, "far away"))
        self.assertIsNone(guide.cue_span(guide.cues[0]))

    def test_long_gap_is_not_merged(self):
        guide = guide_for(SubtitleCue(1, 1, 7, "one subtitle"))
        groups = guide.groups([TimeSpan(1, 3), TimeSpan(4, 7)], .85)
        self.assertEqual([len(parts) for _, parts in groups], [1, 1])

    def test_ambiguous_overlapping_cues_are_not_identity_evidence(self):
        guide = guide_for(SubtitleCue(1, 1, 4, "one"), SubtitleCue(2, 1, 4, "two"))
        self.assertEqual(guide.groups([TimeSpan(1, 4)], .85), [])

    def test_unassigned_intervening_voice_cannot_be_skipped(self):
        guide = guide_for(SubtitleCue(1, 1, 4, "one"), SubtitleCue(2, 2.1, 2.3, "overlap"))
        groups = guide.groups([TimeSpan(1, 2), TimeSpan(2.1, 2.3), TimeSpan(2.4, 4)], .85)
        self.assertFalse(any(len(parts) > 1 for _cue, parts in groups))

    def test_quiet_pause_refines_subtitle_edge_not_voiced_audio(self):
        pipeline = ExtractionPipeline.__new__(ExtractionPipeline)
        pipeline.options = PipelineOptions()
        waveform = torch.full((8 * 16000,), .1)
        waveform[round(3.4 * 16000):round(3.8 * 16000)] = 0
        guide = guide_for(SubtitleCue(1, 1, 3.7, "first"), SubtitleCue(2, 3.7, 7, "second"))
        refined = split_at_subtitle_pauses(pipeline, guide, [TimeSpan(1, 7)], waveform, Mock())
        self.assertEqual(len(refined), 2)
        self.assertLess(refined[0].end, 3.7)
        self.assertGreater(refined[1].start, 3.7)
        self.assertEqual((refined[0].start, refined[1].end), (1, 7))
        self.assertEqual(split_at_subtitle_pauses(pipeline, guide, [TimeSpan(1, 7)], torch.full_like(waveform, .1), Mock()), [TimeSpan(1, 7)])

    def test_sub_lower_bound_pause_not_cut(self):
        pipeline = ExtractionPipeline.__new__(ExtractionPipeline)
        pipeline.options = PipelineOptions(silence_min_seconds=.3)
        waveform = torch.full((8 * 16000,), .1)
        waveform[round(3.5 * 16000):round(3.65 * 16000)] = 0
        guide = guide_for(SubtitleCue(1, 1, 3.7, "first"))
        self.assertEqual(split_at_subtitle_pauses(pipeline, guide, [TimeSpan(1, 7)], waveform, Mock()), [TimeSpan(1, 7)])

    def test_translation_is_provenance_only(self):
        guide = guide_for(SubtitleCue(1, 1, 4, "中文字幕"))
        candidate = CandidateSentence(1, 4, "こんにちは", text="こんにちは", language="ja")
        guide.annotate(candidate)
        self.assertEqual(candidate.text, "こんにちは")
        self.assertEqual(candidate.language, "ja")
        self.assertFalse(candidate.diagnostics["subtitle_text_used_for_stt"])


class SubtitleCompletionTests(unittest.TestCase):
    def setUp(self):
        self.guide = guide_for(SubtitleCue(1, 1, 4, "one line, unknown identity"))
        self.parts = [TimeSpan(1, 2.5), TimeSpan(2.8, 4)]
        self.core = CandidateSentence(1, 2.5, "")
        self.accepted = [self.core]
        self.rejected = []
        self.verifier = Mock()
        self.verifier.SHORT_MIN_DURATION = .55
        self.verifier.exclusion_audit.return_value = None
        self.verifier._tertiary_pair.return_value = (SimpleNamespace(_embeddings_from_waveforms=lambda _p: torch.tensor([[1., 0.], [1., 0.]])), object())
        self.pipeline = Mock()
        self.pipeline.options = PipelineOptions()
        self.pipeline._verify_speaker_span.return_value = SimpleNamespace(accepted=True)
        self.pipeline._merge_short_silence_same_speaker.return_value = [TimeSpan(1, 4)]
        self.pipeline._wavlm_same_speaker_floor.return_value = .82
        self.splitter_patch = patch("extractor.subtitle_assistance.LocalSpeakerTurnSplitter")
        self.splitter = self.splitter_patch.start().return_value
        self.addCleanup(self.splitter_patch.stop)
        self.splitter.detect_multiscale_speaker_boundaries.return_value = []

    def restore(self, blocked=()):
        return restore_subtitle_sentences(self.pipeline, self.guide, self.accepted,
            self.rejected, self.parts, blocked, self.verifier, object(), Path("stem.wav"),
            torch.zeros(1), [], .68, Mock())

    def test_verified_tail_can_complete_without_1200ms_export_requirement(self):
        self.parts[1] = TimeSpan(2.8, 3.5)
        self.assertEqual(self.restore(), 1)
        self.assertEqual((self.accepted[0].start, self.accepted[0].end), (1, 3.5))
        self.assertTrue(self.accepted[0].diagnostics["subtitle_completion"])

    def test_neighbor_cannot_borrow_core_identity(self):
        self.pipeline._verify_speaker_span.side_effect = lambda _v, _w, span, _p, _t: SimpleNamespace(accepted=span.start < 2.7)
        self.assertEqual(self.restore(), 0)
        self.assertIs(self.accepted[0], self.core)

    def test_overlap_singing_and_known_rejection_are_not_crossed(self):
        self.assertEqual(self.restore([TimeSpan(2.55, 2.65)]), 0)
        self.rejected = [CandidateSentence(2.8, 4, "", diagnostics={"structural_hard_reject": True})]
        self.assertEqual(self.restore(), 0)
        self.pipeline._verify_speaker_span.assert_not_called()

    def test_excluded_role_veto_preserves_original(self):
        self.verifier.exclusion_audit.return_value = {"excluded_role_rejected": True}
        self.assertEqual(self.restore(), 0)
        self.assertIs(self.accepted[0], self.core)

    def test_internal_speaker_switch_veto(self):
        self.splitter.detect_multiscale_speaker_boundaries.return_value = [SimpleNamespace(time=3)]
        self.assertEqual(self.restore(), 0)
        self.assertIs(self.accepted[0], self.core)

    def test_sub_550ms_voice_never_inherits_identity(self):
        self.parts[1] = TimeSpan(2.8, 3.2)
        self.assertEqual(self.restore(), 0)
        self.assertEqual(self.guide.report["completion_proposals"][0]["result"], "unverifiable_short_edge")

    def test_no_shortening_existing_output_to_fit_subtitle(self):
        self.core.end = 4.3
        self.assertEqual(self.restore(), 0)
        self.assertEqual(self.core.end, 4.3)


class SubtitlePipelineTests(unittest.TestCase):
    def test_batch_binding_does_not_leak_to_unbound_second_source(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "work") as temp:
            root = Path(temp)
            targets = [root / "ep01.wav", root / "ep02.wav"]
            for target in targets:
                target.touch()
            subtitle = root / "ep01.srt"
            subtitle.write_text("1\n00:00:01,000 --> 00:00:04,000\nHi\n", encoding="utf-8")
            pipeline = ExtractionPipeline.__new__(ExtractionPipeline)
            pipeline.options = PipelineOptions()
            calls = []

            def child(_references, target, **kwargs):
                calls.append((target, kwargs.get("subtitle")))
                output = root / kwargs["job_id"]
                output.mkdir()
                manifest = output / "manifest.json"
                transcript = output / "transcript.srt"
                manifest.write_text("{}", encoding="utf-8")
                transcript.write_text("", encoding="utf-8")
                return PipelineResult(kwargs["job_id"], output, root / "unused.zip", [], [], manifest, transcript)

            pipeline.run = child
            with patch("extractor.pipeline.OUTPUT_ROOT", root):
                result = pipeline.run_many([targets[0]], targets, subtitles={targets[0]: subtitle})
            self.assertEqual(calls, [(targets[0], subtitle), (targets[1], None)])
            self.assertTrue(result.archive_path.is_file())

    def test_vad_retry_requires_real_voice_and_keeps_existing_spans(self):
        guide = guide_for(SubtitleCue(1, 5, 7, "translation"))
        vad = Mock()
        with tempfile.TemporaryDirectory(dir=ROOT / "work") as work:
            with patch("extractor.subtitle_assistance.write_clip"):
                vad.vad_many.side_effect = lambda paths, **_kw: {p: [] for p in paths}
                self.assertEqual(retry_subtitle_vad(guide, [], 10, Path("stem.wav"), work, vad, Mock()), [])
                vad.vad_many.side_effect = lambda paths, **_kw: {p: [TimeSpan(.6, 2.6)] for p in paths}
                original = [TimeSpan(1, 3)]
                result = retry_subtitle_vad(guide, original, 10, Path("stem.wav"), work, vad, Mock())
                self.assertEqual(result, [TimeSpan(1, 3), TimeSpan(5, 7)])
                vad.vad_many.side_effect = lambda paths, **_kw: {p: [TimeSpan(0, 2.6)] for p in paths}
                self.assertEqual(retry_subtitle_vad(guide, [], 10, Path("stem.wav"), work, vad, Mock()), [])

    def test_silence_guard_preserves_lower_and_upper_classification(self):
        pipeline = ExtractionPipeline.__new__(ExtractionPipeline)
        pipeline.options = PipelineOptions()
        for start, end in ((3.50, 3.71), (3.3, 4.16)):
            guide = guide_for(SubtitleCue(1, 1, (start + end) / 2, "first"))
            pipeline._refine_boundary_to_quiet_gap = Mock(return_value=(start, [{"start": start, "end": end, "duration": end - start}]))
            spans = split_at_subtitle_pauses(pipeline, guide, [TimeSpan(1, 7)], torch.zeros(8 * 16000), Mock())
            self.assertEqual(len(spans), 2)
            gap = spans[1].start - spans[0].end
            self.assertGreaterEqual(gap + 1e-6, .2)
            self.assertEqual(gap > .85, end - start > .85)

    def test_pipeline_keeps_singing_first_and_stt_last_with_optional_subtitles(self):
        for with_subtitle in (False, True):
            with self.subTest(with_subtitle=with_subtitle), tempfile.TemporaryDirectory(dir=ROOT / "work") as temp:
                root = Path(temp)
                source = root / "source.wav"
                source.touch()
                subtitle = root / "source.srt"
                subtitle.write_text("1\n00:00:01,000 --> 00:00:04,000\n中文翻译\n", encoding="utf-8")
                events = []
                pipeline = ExtractionPipeline.__new__(ExtractionPipeline)
                pipeline.options = PipelineOptions(export_all_sentences=True)
                pipeline.device = "cpu"
                pipeline._raw_target_waveform = None
                pipeline._raw_blocked_spans = ()
                paths = {"root": root, "output": root / "output", "normalized_target": source,
                         "reference_stems": root, "stems": root, "raw_reference_clips": root}
                pipeline._job_paths = Mock(return_value=paths)
                pipeline._prepare_references = Mock(return_value=([source], [10]))
                pipeline._prepare_negative_references = Mock(return_value=([], []))
                pipeline._make_reference_clips = Mock(return_value=[source])
                pipeline._merge_short_silence_same_speaker = Mock(side_effect=lambda spans, *_a, **_kw: spans)
                with ExitStack() as stack:
                    def patched(name, **kwargs):
                        return stack.enter_context(patch("extractor.pipeline." + name, **kwargs))
                    patched("normalize_audio", return_value=source)
                    patched("probe_duration", return_value=10)
                    patched("pad_for_separator", return_value=10)
                    patched("trim_audio_in_place")
                    patched("write_clip")
                    patched("load_mono", side_effect=lambda *_a: torch.full((10 * 16000,), .1))
                    singer = patched("SingingDetector").return_value
                    singer.clean_spans.side_effect = lambda _path, spans, *_a, **_kw: (events.append("singing") or spans, [])
                    separator = patched("UVR5Separator").return_value
                    separator.separate_many.side_effect = lambda *_a, **_kw: events.append("uvr") or [source, source]
                    vad = patched("FunASRTools").return_value
                    vad.vad_many.side_effect = lambda *_a, **_kw: events.append("vad") or {source: [TimeSpan(1, 4)]}
                    overlap = patched("OverlapDetector").return_value
                    overlap.clean_spans.side_effect = lambda _path, spans, *_a, **_kw: (events.append("overlap") or spans, [])
                    patched("DualSpeakerVerifier")
                    stt = patched("WhisperSegmenter").return_value
                    stt.transcribe_spans.side_effect = lambda *_a, **_kw: events.append("stt") or [CandidateSentence(1, 4, "こんにちは", language="ja", diagnostics={"transcription_span_index": 0})]
                    result = pipeline.run([source], source, subtitle=subtitle if with_subtitle else None, create_archive=False)
                self.assertEqual(events, ["singing", "uvr", "vad", "singing", "overlap", "stt"])
                self.assertEqual(len(result.accepted), 1)
                self.assertEqual(result.accepted[0].text, "こんにちは")
                self.assertEqual(result.accepted[0].language, "ja")
                stt_kwargs = stt.transcribe_spans.call_args.kwargs
                self.assertEqual(set(stt_kwargs), {"progress"})
                manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
                self.assertEqual("subtitle_assistance" in manifest, with_subtitle)
                self.assertIsNone(pipeline._subtitle_guide)
                self.assertEqual(list((root / "output/text").glob("*.txt"))[0].read_text(encoding="utf-8"), "こんにちは")


if __name__ == "__main__":
    unittest.main()
