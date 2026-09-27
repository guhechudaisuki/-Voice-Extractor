"""Quiet-gap partition keeps speech and excludes only the proposed gap."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from probe_internal_island_identity import speech_islands  # noqa: E402
from probe_pitch_islands import pitch_stats  # noqa: E402


class InternalIslandProbeTests(unittest.TestCase):
    def test_partitions_multiple_internal_pauses(self):
        self.assertEqual(
            speech_islands(1.0, 4.0, [
                {"span": [1.5, 1.8]}, {"span": [2.5, 2.7]},
            ]),
            [(1.0, 1.5), (1.8, 2.5), (2.7, 4.0)],
        )

    def test_rejects_gap_outside_or_out_of_order(self):
        for gaps in ([{"span": [0.8, 1.2]}],
                     [{"span": [2.0, 2.5]}, {"span": [2.4, 2.8]}]):
            with self.subTest(gaps=gaps), self.assertRaises(ValueError):
                speech_islands(1.0, 4.0, gaps)

    def test_pitch_probe_reports_measurement_not_identity(self):
        time = np.arange(16000) / 16000
        stats = pitch_stats(np.sin(2 * np.pi * 220 * time))
        self.assertGreater(stats["voiced_frames"], 10)
        self.assertAlmostEqual(stats["median_hz"], 220, delta=5)
        self.assertEqual(
            pitch_stats(np.zeros(1000)),
            {"voiced_frames": 0, "median_hz": None},
        )


if __name__ == "__main__":
    unittest.main()
