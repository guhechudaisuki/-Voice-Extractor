"""Original-channel VAD is a proposal source, never an identity verdict."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extractor.candidate_lattice import propose_raw_vad_subturns  # noqa: E402
from extractor.types import TimeSpan  # noqa: E402


class RawVADSubturnTests(unittest.TestCase):
    def test_continuous_stem_keeps_complete_target_alternative(self):
        # Frozen 337-344 s scene: the stem VAD is continuous, but original
        # audio VAD has a 0.67 s internal gap after the target utterance.
        result = propose_raw_vad_subturns(
            [TimeSpan(0.86, 6.98)],
            [TimeSpan(0.92, 3.66), TimeSpan(4.33, 5.22),
             TimeSpan(5.55, 6.98)],
            [TimeSpan(0.86, 3.11)],
            minimum_gap_seconds=0.20,
            minimum_candidate_seconds=0.55,
        )
        self.assertEqual(result, [TimeSpan(0.86, 3.66)])

    def test_sub_lower_bound_jitter_does_not_create_fragments(self):
        result = propose_raw_vad_subturns(
            [TimeSpan(1.0, 4.0)],
            [TimeSpan(1.05, 2.0), TimeSpan(2.10, 3.95)],
            [TimeSpan(1.1, 1.9)],
            minimum_gap_seconds=0.20,
            minimum_candidate_seconds=0.55,
        )
        self.assertEqual(result, [])

    def test_a_raw_candidate_needs_local_target_evidence(self):
        result = propose_raw_vad_subturns(
            [TimeSpan(1.0, 6.0)],
            [TimeSpan(1.05, 2.5), TimeSpan(3.0, 5.95)],
            [TimeSpan(3.0, 5.0)],
            minimum_gap_seconds=0.20,
            minimum_candidate_seconds=0.55,
        )
        self.assertEqual(result, [TimeSpan(3.0, 6.0)])


if __name__ == "__main__":
    unittest.main()
