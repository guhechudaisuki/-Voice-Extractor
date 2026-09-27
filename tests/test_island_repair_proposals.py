from __future__ import annotations

import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extractor.nextgen.purity_consensus import target_runs  # noqa: E402


class IslandRepairProposalTests(unittest.TestCase):
    def test_other_voice_cannot_be_bridged_by_whole_clip_score(self) -> None:
        islands = [
            {"span": [0.0, 1.5], "state": "target_supported"},
            {"span": [1.8, 2.1], "state": "other_supported"},
            {"span": [2.4, 4.0], "state": "target_supported"},
        ]
        self.assertEqual(
            target_runs(islands, {(1.8, 2.1)}, maximum_gap_seconds=0.85),
            [(0.0, 1.5), (2.4, 4.0)],
        )

    def test_unresolved_island_is_not_inherited_from_target_neighbors(self) -> None:
        islands = [
            {"span": [0.0, 1.5], "state": "target_supported"},
            {"span": [1.8, 2.1], "state": "unresolved_short"},
            {"span": [2.4, 4.0], "state": "target_supported"},
        ]
        self.assertEqual(
            target_runs(islands, set(), maximum_gap_seconds=0.85),
            [(0.0, 1.5), (2.4, 4.0)],
        )

    def test_long_gap_still_splits_same_speaker(self) -> None:
        islands = [
            {"span": [0.0, 1.5], "state": "target_supported"},
            {"span": [2.5, 4.0], "state": "target_supported"},
        ]
        self.assertEqual(
            target_runs(islands, set(), maximum_gap_seconds=0.85),
            [(0.0, 1.5), (2.5, 4.0)],
        )


if __name__ == "__main__":
    unittest.main()
