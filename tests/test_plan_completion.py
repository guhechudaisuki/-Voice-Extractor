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


class LocalClassifierMultiVetoTests(unittest.TestCase):
    def _candidate(self, scores, threshold=-4.24, start=0.0, end=5.0):
        windows = []
        for index, score in enumerate(scores):
            windows.append({"start": start + index * 0.5, "end": start + index * 0.5 + 0.5,
                            "score": score, "voiced": True})
        from extractor.types import CandidateSentence
        return CandidateSentence(start, end, "", accepted=True, window_min_score=0.8,
                                 diagnostics={"local_classifier_audit": {
                                     "windows": windows, "reject_threshold": threshold}})

    def test_two_strong_negative_windows_veto(self):
        from extractor.nextgen.cluster_gate import _local_classifier_multi_veto
        candidate = self._candidate([0.5, -5.0, 0.6, -6.0, 0.4])
        self.assertTrue(_local_classifier_multi_veto(candidate))

    def test_single_negative_window_does_not_veto(self):
        from extractor.nextgen.cluster_gate import _local_classifier_multi_veto
        candidate = self._candidate([0.5, -5.0, 0.6, 0.4, 0.7])
        self.assertFalse(_local_classifier_multi_veto(candidate))

    def test_negative_windows_outside_interval_do_not_count(self):
        from extractor.nextgen.cluster_gate import _local_classifier_multi_veto
        candidate = self._candidate([0.5, 0.6, 0.4], start=0.0, end=1.5)
        # 强负窗位于候选区间之外（间隔裁切后的子段场景）。
        candidate.diagnostics["local_classifier_audit"]["windows"] = [
            {"start": 3.0, "end": 3.5, "score": -9.0, "voiced": True},
            {"start": 4.0, "end": 4.5, "score": -9.0, "voiced": True}]
        self.assertFalse(_local_classifier_multi_veto(candidate))

    def test_vetoed_crop_child_goes_tier_b(self):
        from extractor.nextgen.cluster_gate import apply_cluster_gate
        import numpy as np
        from test_high_recall_cluster import make_result
        base = np.ones(8) / np.sqrt(8)
        from extractor.nextgen.identity_cluster import TurnEvidence
        turns = [TurnEvidence(0, 0.0, 4.0, 4.0, base, base)]
        result = make_result(turns, accepted_spans=[[0.0, 4.0]])
        parent = CandidateSentence(0.5, 3.5, "", reject_reason="最终局部身份冲突，原混合片段已撤回",
                                   window_min_score=0.8,
                                   diagnostics={"local_classifier_audit": {
                                       "windows": [
                                           {"start": 1.0, "end": 1.5, "score": -6.0, "voiced": True},
                                           {"start": 2.0, "end": 2.5, "score": -8.0, "voiced": True}],
                                       "reject_threshold": -4.24}})
        accepted, rejected = [], [parent]
        changes = apply_cluster_gate(accepted, rejected, turns, result,
                                     mode="enforce", min_seconds=1.2)
        self.assertEqual(changes["cropped"], 0)
        self.assertEqual(changes["borderline"], 1)
        self.assertEqual(accepted, [])

    def test_tier_a_containment_dedup_and_tier_b_no_cap(self):
        from extractor.nextgen.cluster_gate import _deduplicate_tier_a, _refine_tier_b
        long = CandidateSentence(0.0, 4.0, "", accepted=True)
        inner = CandidateSentence(0.2, 2.0, "", accepted=True)
        separate = CandidateSentence(10.0, 12.0, "", accepted=True)
        kept = _deduplicate_tier_a([long, inner, separate])
        self.assertEqual([(c.start, c.end) for c in kept], [(0.0, 4.0), (10.0, 12.0)])
        # Tier B：被 Tier A 覆盖 >=85% 的隐藏，其余保留且不设数量上限。
        covered = CandidateSentence(0.5, 3.5, "")
        covered.diagnostics["cluster_tier"] = "B"
        survivors = [CandidateSentence(float(i) * 20.0, float(i) * 20.0 + 2.0, "")
                     for i in range(40)]
        for c in survivors:
            c.diagnostics["cluster_tier"] = "B"
            c.speaker_score = 0.5
        borderline = [covered, *survivors]
        _refine_tier_b(borderline, [long, separate])
        kept_b = [c for c in borderline if c.diagnostics.get("cluster_tier") == "B"]
        # 40 个幸存者中 [0,2] 被 Tier A [0,4] 覆盖隐藏，其余 39 个全保留。
        self.assertEqual(len(kept_b), 39)
        self.assertEqual(covered.diagnostics.get("cluster_tier"), "covered_by_tier_a")


class PartialClipTextTests(unittest.TestCase):
    def test_explicit_partial_text_wins(self):
        from launcher.review_export import _partial_clip_text
        row = {"start": 10.0, "end": 14.0, "text": "全文"}
        self.assertEqual(_partial_clip_text(row, {"partial_text": "修正"}, (0.0, 2.0)), "修正")

    def test_subtitle_fragments_sliced_by_time(self):
        from launcher.review_export import _partial_clip_text
        row = {"start": 100.0, "end": 106.0, "text": "整句",
               "diagnostics": {"subtitle_reference": [
                   {"text": "他人插话", "aligned_start": 100.0, "aligned_end": 102.0},
                   {"text": "目标正文", "aligned_start": 102.5, "aligned_end": 105.5}]}}
        # 保留段 [102.0, 106.0]：与"目标正文"重叠 3.0s（>=50%），与"他人插话"
        # 重叠 0.0s（<50%）→ 只保留正文。
        self.assertEqual(_partial_clip_text(row, {}, (2.0, 6.0)), "目标正文")

    def test_proportional_fallback(self):
        from launcher.review_export import _partial_clip_text
        row = {"start": 0.0, "end": 10.0, "text": "abcdefghij"}
        self.assertEqual(_partial_clip_text(row, {}, (2.0, 6.0)), "cdef")
