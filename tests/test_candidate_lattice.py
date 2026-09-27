from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extractor.candidate_lattice import build_utterance_lattice  # noqa: E402
from extractor.types import TimeSpan  # noqa: E402


SOURCE_HASH = "a" * 64


class UtteranceLatticeTests(unittest.TestCase):
    def test_atomic_islands_are_preserved_and_medium_gap_proposes_join(self):
        result = build_utterance_lattice(
            SOURCE_HASH,
            [TimeSpan(1.0, 2.0), TimeSpan(2.5, 3.5)],
            max_gap_seconds=0.85,
            max_utterance_seconds=45,
        )
        self.assertEqual([p.island_indexes for p in result], [(0,), (0, 1), (1,)])
        self.assertEqual(result[1].gap_intervals, ((32000, 40000),))
        self.assertEqual(result[1].identity_state, "unresolved")
        self.assertEqual(result[1].source_id, f"{SOURCE_HASH}:16000:16000:56000")

    def test_hard_gap_and_blocked_gap_cannot_be_crossed(self):
        islands = [TimeSpan(1.0, 2.0), TimeSpan(2.5, 3.0)]
        hard = build_utterance_lattice(
            SOURCE_HASH, islands, max_gap_seconds=0.4,
            max_utterance_seconds=45,
        )
        blocked = build_utterance_lattice(
            SOURCE_HASH, islands, max_gap_seconds=0.85,
            max_utterance_seconds=45, blocked=[TimeSpan(2.1, 2.4)],
        )
        self.assertEqual(len(hard), 2)
        self.assertEqual(len(blocked), 2)

    def test_subtitles_annotate_but_do_not_control_join(self):
        result = build_utterance_lattice(
            SOURCE_HASH,
            [TimeSpan(1.0, 2.0), TimeSpan(2.4, 3.0)],
            max_gap_seconds=0.85,
            max_utterance_seconds=45,
            subtitle_cues=[
                (10, TimeSpan(0.9, 2.1)),
                (11, TimeSpan(2.3, 3.1)),
            ],
        )
        joined = result[1]
        self.assertEqual(joined.subtitle_cue_indexes, (10, 11))

    def test_lattice_has_linear_bound_with_max_atoms(self):
        islands = [TimeSpan(index, index + 0.2) for index in range(100)]
        result = build_utterance_lattice(
            SOURCE_HASH, islands, max_gap_seconds=0.85,
            max_utterance_seconds=45, max_islands_per_proposal=3,
        )
        self.assertEqual(len(result), 100 + 99 + 98)

    def test_single_long_island_is_kept_but_never_joined_past_cap(self):
        result = build_utterance_lattice(
            SOURCE_HASH,
            [TimeSpan(1.0, 21.0), TimeSpan(21.3, 22.0)],
            max_gap_seconds=0.85,
            max_utterance_seconds=16.0,
        )
        self.assertEqual(
            [(item.start_sample, item.end_sample) for item in result],
            [(16000, 336000), (340800, 352000)],
        )
        self.assertTrue(all(len(item.island_indexes) == 1 for item in result))


if __name__ == "__main__":
    unittest.main()
