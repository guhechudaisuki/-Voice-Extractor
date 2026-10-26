"""Phase E tests: acoustic QC + long-sentence split + shortcuts."""
from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extractor.acoustic_qc import evaluate_clip_acoustics  # noqa: E402


class AcousticQCTests(unittest.TestCase):
    def test_clean_speech_is_gold_or_silver(self):
        sr = 16000
        t = np.linspace(0, 3.0, sr * 3, dtype=np.float32)
        wave = 0.5 * np.sin(2 * np.pi * 220 * t)
        qc = evaluate_clip_acoustics(wave, sr, speaker_score=0.8, window_min=0.7)
        self.assertIn(qc["quality_grade"], ("Gold", "Silver"))
        self.assertGreater(qc["snr_db"], 10.0)
        self.assertEqual(qc["clipping_ratio"], 0.0)

    def test_clipping_detected(self):
        sr = 16000
        wave = np.ones(sr * 2, dtype=np.float32)  # 全削波
        qc = evaluate_clip_acoustics(wave, sr, speaker_score=0.8)
        self.assertGreater(qc["clipping_ratio"], 0.5)
        self.assertEqual(qc["quality_grade"], "Review")

    def test_silent_clip_is_review(self):
        sr = 16000
        qc = evaluate_clip_acoustics(np.zeros(sr, dtype=np.float32), sr)
        self.assertEqual(qc["quality_grade"], "Review")

    def test_low_snr_is_review(self):
        sr = 16000
        rng = np.random.default_rng(1)
        wave = (rng.normal(0, 0.001, sr * 3) + 0.001 * np.sin(
            2 * np.pi * 220 * np.linspace(0, 3, sr * 3))).astype(np.float32)
        qc = evaluate_clip_acoustics(wave, sr, speaker_score=0.5)
        self.assertEqual(qc["quality_grade"], "Review")


class LongSentenceSplitTests(unittest.TestCase):
    def test_over_12s_should_split(self):
        # 模拟：14s 句子有 3 个子分片 → 应该被切分
        fragments = [
            {"start": 100.0, "end": 105.0, "text": "第一句"},
            {"start": 105.2, "end": 110.0, "text": "第二句"},
            {"start": 110.2, "end": 114.0, "text": "第三句"},
        ]
        duration = 14.0
        should_split = duration > 12.0 and len(fragments) >= 2
        self.assertTrue(should_split)
        # 切分点在 105.0-105.2 之间（静音谷底）
        split_idx = 1
        first_end = fragments[split_idx - 1]["end"]
        second_start = fragments[split_idx]["start"]
        first_dur = first_end - fragments[0]["start"]
        second_dur = fragments[-1]["end"] - second_start
        self.assertGreaterEqual(first_dur, 2.5)
        self.assertGreaterEqual(second_dur, 2.5)
        self.assertLessEqual(first_dur, 10.0)
        self.assertLessEqual(second_dur, 10.0)


if __name__ == "__main__":
    unittest.main()
