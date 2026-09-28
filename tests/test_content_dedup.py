from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extractor.pipeline import ExtractionPipeline, PipelineOptions
from extractor.types import CandidateSentence


def clip(start: float, end: float) -> CandidateSentence:
    turn = CandidateSentence(start, end, "", accepted=True)
    turn.speaker_score = 0.8
    return turn


def burst(freq: float, seconds: float = 2.0) -> torch.Tensor:
    t = torch.arange(int(16000 * seconds), dtype=torch.float32) / 16000
    return (0.5 * torch.sin(2 * torch.pi * freq * t)).numpy()


class ContentDedupTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = ExtractionPipeline(PipelineOptions())
        self.wave = torch.zeros(16000 * 30)

    def place(self, wave, start, samples):
        a = int(start * 16000)
        wave[a:a + len(samples)] = torch.from_numpy(samples)

    def run_dedup(self, accepted):
        return self.pipeline._deduplicate_content_clips(
            accepted, self.wave, lambda _v, _m: None
        )

    def test_identical_later_clip_is_removed(self):
        tone = burst(440.0)
        self.place(self.wave, 5.0, tone)
        self.place(self.wave, 20.0, tone)
        first, second = clip(5.0, 7.0), clip(20.0, 22.0)
        removed = self.run_dedup([first, second])
        self.assertEqual(removed, 1)
        self.assertEqual(first.reject_reason, "")
        self.assertIn("content_duplicate_of", second.diagnostics)

    def test_different_content_clips_are_kept(self):
        self.place(self.wave, 5.0, burst(440.0))
        self.place(self.wave, 20.0, burst(880.0))
        first, second = clip(5.0, 7.0), clip(20.0, 22.0)
        removed = self.run_dedup([first, second])
        self.assertEqual(removed, 0)
        self.assertEqual(second.reject_reason, "")

    def test_removing_later_does_not_touch_first(self):
        tone = burst(300.0)
        self.place(self.wave, 5.0, tone)
        self.place(self.wave, 20.0, tone)
        first, second = clip(5.0, 7.0), clip(20.0, 22.0)
        self.run_dedup([first, second])
        self.assertEqual(first.reject_reason, "")
        self.assertTrue(first.diagnostics.get("content_duplicate_of") is None)

    def test_duplicate_tail_is_removed_without_trimming_complete_sentence(self):
        speech = np.random.default_rng(42).normal(0, 0.1, 64000).astype("float32")
        self.place(self.wave, 5.0, speech)
        self.place(self.wave, 20.0, speech[40000:])
        whole, tail = clip(5.0, 9.0), clip(20.0, 21.5)
        accepted, rejected = [whole, tail], []
        removed = self.pipeline._deduplicate_content_clips(
            accepted, self.wave, lambda *_: None, rejected=rejected,
        )
        self.assertEqual(removed, 1)
        self.assertEqual(accepted, [whole])
        self.assertEqual((whole.start, whole.end), (5.0, 9.0))
        self.assertEqual(rejected, [tail])
        self.assertFalse(tail.accepted)

    def test_later_complete_sentence_wins_over_earlier_tail(self):
        speech = np.random.default_rng(7).normal(0, 0.1, 64000).astype("float32")
        self.place(self.wave, 5.0, speech[-24000:])
        self.place(self.wave, 20.0, speech)
        tail, whole = clip(5.0, 6.5), clip(20.0, 24.0)
        accepted = [tail, whole]
        self.assertEqual(self.run_dedup(accepted), 1)
        self.assertEqual(accepted, [whole])
        self.assertEqual((whole.start, whole.end), (20.0, 24.0))

    def test_shared_prefix_does_not_authorize_cutting_unique_words(self):
        rng = np.random.default_rng(10)
        shared, first_end, second_end = [
            rng.normal(0, 0.1, 32000).astype("float32") for _ in range(3)
        ]
        self.place(self.wave, 5.0, np.concatenate([shared, first_end]))
        self.place(self.wave, 20.0, np.concatenate([shared, second_end]))
        accepted = [clip(5.0, 9.0), clip(20.0, 24.0)]
        self.assertEqual(self.run_dedup(accepted), 0)
        self.assertEqual([(c.start, c.end) for c in accepted], [(5.0, 9.0), (20.0, 24.0)])

    def test_gain_and_dc_offset_do_not_hide_contained_audio(self):
        speech = np.random.default_rng(3).normal(0, 0.1, 64000).astype("float32")
        self.place(self.wave, 5.0, speech)
        self.place(self.wave, 20.0, speech[12345:36345] * 0.4 + 0.15)
        accepted = [clip(5.0, 9.0), clip(20.0, 21.5)]
        self.assertEqual(self.run_dedup(accepted), 1)
        self.assertEqual(len(accepted), 1)

    def test_silence_is_not_content_identity_evidence(self):
        accepted = [clip(5.0, 7.0), clip(20.0, 22.0)]
        self.assertEqual(self.run_dedup(accepted), 0)
        self.assertEqual(len(accepted), 2)

    def test_quiet_unique_tail_is_not_hidden_by_loud_shared_prefix(self):
        rng = np.random.default_rng(19)
        shared = rng.normal(0, 0.3, 32000).astype("float32")
        endings = [rng.normal(0, 0.03, 32000).astype("float32") for _ in range(2)]
        self.place(self.wave, 5.0, np.concatenate([shared, endings[0]]))
        self.place(self.wave, 20.0, np.concatenate([shared, endings[1]]))
        accepted = [clip(5.0, 9.0), clip(20.0, 24.0)]
        self.assertEqual(self.run_dedup(accepted), 0)
        self.assertEqual(len(accepted), 2)

    def test_removed_clip_cannot_point_to_an_export(self):
        tone = burst(440.0)
        self.place(self.wave, 5.0, tone)
        self.place(self.wave, 20.0, tone)
        first, second = clip(5.0, 7.0), clip(20.0, 22.0)
        second.audio_file, second.text_file, second.video_file = "old.wav", "old.txt", "old.mp4"
        self.run_dedup([first, second])
        self.assertFalse(second.accepted)
        self.assertEqual((second.audio_file, second.text_file, second.video_file), ("", "", ""))


if __name__ == "__main__":
    unittest.main()
