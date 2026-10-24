"""计划书 Stage 4（字幕/STT 精裁）与第四通道/双轨复核的单元测试。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extractor.subtitle_assistance import energy_valley, split_at_subtitle_pauses
from extractor.subtitles import SubtitleCue, SubtitleGuide, read_subtitles
from extractor.types import CandidateSentence, TimeSpan

ASS_SAMPLE = """[Script Info]
Title: sample

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, Bold, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,MS Gothic,48,&H00FFFFFF,0,2,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:02.00,Default,久美子,0,0,0,,うん…合ってる
Dialogue: 0,0:00:02.00,0:00:02.64,Default,葉月,0,0,0,,でしょ
Dialogue: 0,0:00:03.00,0:00:05.00,Default,,0,0,0,,セリフ
"""


class SubtitleActorTests(unittest.TestCase):
    def test_actor_field_parsed_and_reported(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.ass"
            path.write_text(ASS_SAMPLE, encoding="utf-8")
            cues = read_subtitles(path)
        self.assertEqual([cue.actor for cue in cues], ["久美子", "葉月", ""])
        guide = SubtitleGuide(cues, path.name)
        # 未对齐时重叠区直接按字幕时间计算。
        self.assertEqual(
            [(round(s.start, 2), round(s.end, 2)) for s in guide.overlapping_speech_spans()],
            [])

    def test_overlapping_speech_spans_detected(self):
        # 两条 speech 字幕时间重叠 >=0.25s → 多人同时说话禁区。
        cues = [
            SubtitleCue(1, 10.0, 16.0, "楽譜 返して"),
            SubtitleCue(2, 12.0, 18.0, "前景对白"),
            SubtitleCue(3, 40.0, 42.0, "无关句"),
        ]
        guide = SubtitleGuide(cues, "test")
        spans = guide.overlapping_speech_spans()
        self.assertEqual(len(spans), 1)
        self.assertAlmostEqual(spans[0].start, 12.0)
        self.assertAlmostEqual(spans[0].end, 16.0)


class EnergyValleyTests(unittest.TestCase):
    def setUp(self):
        self.rate = 16000

    def _wave(self, dip_center, dip=0.16):
        wave = torch.full((8 * self.rate,), 0.1)
        start = round((dip_center - dip / 2) * self.rate)
        wave[start:start + round(dip * self.rate)] = 0.0
        return wave

    def test_valley_found_at_dip(self):
        wave = self._wave(4.30)
        valley = energy_valley(wave, 4.40)
        self.assertIsNotNone(valley)
        self.assertLess(abs(valley - 4.30), 0.10)

    def test_flat_speech_has_no_valley(self):
        wave = torch.full((8 * self.rate,), 0.1)
        self.assertIsNone(energy_valley(wave, 4.0))

    def test_split_skips_overlap_zones(self):
        from extractor.pipeline import ExtractionPipeline, PipelineOptions
        pipeline = ExtractionPipeline.__new__(ExtractionPipeline)
        pipeline.options = PipelineOptions()
        waveform = self._wave(4.0)
        # Cue 边界 4.0 处有能量谷，但该处处于双字幕重叠禁区 → 不切。
        cues = [SubtitleCue(1, 1.0, 4.0, "a"), SubtitleCue(2, 3.6, 7.0, "b")]
        guide = SubtitleGuide(cues, "t", aligned=True, offset=0.0,
                              anchors=[[t, 0.0] for t in (1.0, 2.0, 5.0, 6.0)])
        guide.aligned = True
        refined = split_at_subtitle_pauses(
            pipeline, guide, [TimeSpan(1, 7)], waveform, Mock())
        # 重叠禁区（3.6-4.0 与 cue 边界重合）内不切。
        self.assertTrue(all(abs(part.end - 4.0) > 0.05 for part in refined))


class SttTextTests(unittest.TestCase):
    def setUp(self):
        from extractor.pipeline import ExtractionPipeline
        self.pipeline = ExtractionPipeline.__new__(ExtractionPipeline)
        self.pipeline._cluster_accepted_spans = [[10.0, 12.0]]

    def test_text_similarity_ignores_punctuation(self):
        sim = self.pipeline._text_similarity("うん…合ってる", "うん合ってる")
        self.assertGreater(sim, 0.95)
        self.assertEqual(self.pipeline._text_similarity("合ってる", " DETAIL "), 0.0)

    def test_hallucination_pattern_filters_watermark(self):
        pattern = self.pipeline._HALLUCINATION_PATTERN
        self.assertTrue(pattern.match("谢谢观看"))
        self.assertTrue(pattern.match("字幕由XX字幕组提供"))
        self.assertTrue(pattern.match("ご視聴ありがとうございました"))
        self.assertFalse(pattern.match("うん…合ってる"))
        self.assertFalse(pattern.match("ダメ金だけど金賞は金賞"))

    def test_trim_keeps_cluster_clean_fragments(self):
        guide = None
        turn = CandidateSentence(10.0, 15.0, "")
        fragments = [
            CandidateSentence(10.0, 12.0, "うん…合ってる"),
            CandidateSentence(12.2, 14.8, "でしょ"),   # 干净区间之外
        ]
        trimmed = self.pipeline._trim_turn_to_clean_fragments(turn, fragments, guide)
        self.assertIsNotNone(trimmed)
        start, end, kept = trimmed
        self.assertEqual((start, end), (10.0, 12.0))
        self.assertEqual(len(kept), 1)

    def test_fragment_clean_via_subtitle_alignment(self):
        cues = [SubtitleCue(1, 20.0, 22.0, "もっとダメだね")]
        guide = SubtitleGuide(cues, "t", aligned=True, offset=0.0,
                              anchors=[[t, 0.0] for t in (19.0, 21.0, 23.0)])
        fragment = CandidateSentence(20.1, 21.9, "もつとダメだね")
        clean, method = self.pipeline._fragment_clean_evidence(fragment, guide, {})
        self.assertTrue(clean)
        self.assertEqual(method, "subtitle_aligned")

    def test_subtitle_text_correction(self):
        cues = [SubtitleCue(1, 30.0, 32.0, "もっとダメだね")]
        guide = SubtitleGuide(cues, "t", aligned=True, offset=0.0,
                              anchors=[[t, 0.0] for t in (29.0, 31.0, 33.0)])
        corrected, info = self.pipeline._correct_text_with_subtitle(
            "もつとダメだね", "ja", 30.0, 32.0, guide)
        self.assertEqual(corrected, "もっとダメだね")
        self.assertGreaterEqual(info["subtitle_text_correction"]["similarity"], 0.70)
        # 非日文候选不修正。
        same, none_info = self.pipeline._correct_text_with_subtitle(
            "原文", "zh", 30.0, 32.0, guide)
        self.assertEqual(same, "原文")
        self.assertIsNone(none_info)


class WespeakerArbitrationTests(unittest.TestCase):
    def test_rescue_marked_tier_a_demoted_below_floor(self):
        from extractor.nextgen.cluster_runtime import ClusterAnalyzer
        analyzer = ClusterAnalyzer.__new__(ClusterAnalyzer)
        analyzer.quaternary = object()          # 只要非 None 即启用
        analyzer.quaternary_floor = 0.63
        analyzer.wespeaker_score = lambda start, end: 0.55
        accepted = [
            CandidateSentence(1.0, 3.0, "", accepted=True,
                              diagnostics={"target_locator_recovery": True}),
            CandidateSentence(5.0, 7.0, "", accepted=True,
                              diagnostics={}),
        ]
        rejected = []
        demoted = analyzer.wespeaker_arbitrate(accepted, rejected)
        self.assertEqual(demoted, 1)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0].start, 5.0)  # 剩下的是无救援标记候选
        self.assertEqual(rejected[0].diagnostics.get("cluster_tier"), "B")
        self.assertEqual(rejected[0].diagnostics.get("cluster_reason"),
                         "wespeaker_arbitration")

    def test_no_quaternary_is_noop(self):
        from extractor.nextgen.cluster_runtime import ClusterAnalyzer
        analyzer = ClusterAnalyzer.__new__(ClusterAnalyzer)
        analyzer.quaternary = None
        accepted = [CandidateSentence(1.0, 3.0, "", accepted=True)]
        self.assertEqual(analyzer.wespeaker_arbitrate(accepted, []), 0)
        self.assertEqual(len(accepted), 1)


if __name__ == "__main__":
    unittest.main()
