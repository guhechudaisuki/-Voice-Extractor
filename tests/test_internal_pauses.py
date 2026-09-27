from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extractor.nextgen.internal_pauses import (  # noqa: E402
    locate_quiet_gaps, locate_terminal_short_tail_gap,
)


class InternalPauseTests(unittest.TestCase):
    def test_finds_medium_internal_pause_before_short_new_voice(self) -> None:
        rate = 16000
        voice_a = np.full(rate, 0.05, dtype=np.float32)
        pause = np.full(round(0.34 * rate), 0.00002, dtype=np.float32)
        voice_b = np.full(round(0.22 * rate), 0.06, dtype=np.float32)
        gaps = locate_quiet_gaps(
            np.concatenate((voice_a, pause, voice_b)), rate,
            lower_seconds=0.20, upper_seconds=0.85,
        )
        self.assertEqual(len(gaps), 1)
        self.assertAlmostEqual(gaps[0].start / rate, 1.0, delta=0.04)
        self.assertAlmostEqual(gaps[0].end / rate, 1.34, delta=0.04)

    def test_short_intra_word_weakness_does_not_create_a_split(self) -> None:
        rate = 16000
        samples = np.concatenate((
            np.full(rate, 0.05, dtype=np.float32),
            np.full(round(0.08 * rate), 0.00002, dtype=np.float32),
            np.full(rate, 0.05, dtype=np.float32),
        ))
        gaps = locate_quiet_gaps(
            samples, rate, lower_seconds=0.20, upper_seconds=0.85,
        )
        self.assertEqual(gaps, ())

    def test_decaying_word_end_still_exposes_real_pause(self) -> None:
        rate = 16000
        frame = round(0.02 * rate)
        decaying_end = np.concatenate([
            np.full(frame, level, dtype=np.float32)
            for level in (0.017, 0.015, 0.012, 0.011, 0.009, 0.007)
        ])
        samples = np.concatenate((
            np.full(rate, 0.05, dtype=np.float32),
            decaying_end,
            np.full(round(0.34 * rate), 0.00002, dtype=np.float32),
            np.full(round(0.22 * rate), 0.06, dtype=np.float32),
        ))
        gaps = locate_quiet_gaps(
            samples, rate, lower_seconds=0.20, upper_seconds=0.85,
        )
        self.assertEqual(len(gaps), 1)
        self.assertAlmostEqual(gaps[0].start / rate, 1.12, delta=0.04)

    def test_terminal_short_speech_after_medium_gap_is_not_inherited(self) -> None:
        rate = 16000
        samples = np.concatenate((
            np.full(rate, 0.05, dtype=np.float32),
            np.full(round(0.34 * rate), 0.00002, dtype=np.float32),
            np.full(round(0.06 * rate), 0.06, dtype=np.float32),
        ))
        self.assertEqual(locate_quiet_gaps(
            samples, rate, lower_seconds=0.20, upper_seconds=0.85,
        ), ())
        gap = locate_terminal_short_tail_gap(
            samples, rate, lower_seconds=0.20, upper_seconds=0.85,
        )
        self.assertIsNotNone(gap)
        self.assertAlmostEqual(gap.start / rate, 1.0, delta=0.04)
        self.assertAlmostEqual(gap.end / rate, 1.34, delta=0.04)

    def test_terminal_gap_without_post_gap_voice_is_not_a_short_tail(self) -> None:
        rate = 16000
        samples = np.concatenate((
            np.full(rate, 0.05, dtype=np.float32),
            np.full(round(0.34 * rate), 0.00002, dtype=np.float32),
        ))
        self.assertIsNone(locate_terminal_short_tail_gap(
            samples, rate, lower_seconds=0.20, upper_seconds=0.85,
        ))

    def test_terminal_gap_uses_local_floor_after_a_loud_prefix(self) -> None:
        rate = 16000
        samples = np.concatenate((
            np.full(4 * rate, 0.10, dtype=np.float32),
            np.full(round(3.5 * rate), 0.005, dtype=np.float32),
            np.full(round(0.44 * rate), 0.000001, dtype=np.float32),
            np.full(round(0.06 * rate), 0.05, dtype=np.float32),
        ))
        gap = locate_terminal_short_tail_gap(
            samples, rate, lower_seconds=0.20, upper_seconds=0.85,
        )
        self.assertIsNotNone(gap)
        self.assertAlmostEqual(gap.start / rate, 7.5, delta=0.04)

    def test_short_final_syllable_without_quiet_gap_stays_connected(self) -> None:
        rate = 16000
        samples = np.concatenate((
            np.full(rate, 0.05, dtype=np.float32),
            np.full(round(0.06 * rate), 0.06, dtype=np.float32),
        ))
        self.assertIsNone(locate_terminal_short_tail_gap(
            samples, rate, lower_seconds=0.20, upper_seconds=0.85,
        ))


if __name__ == "__main__":
    unittest.main()
