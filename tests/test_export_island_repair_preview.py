from __future__ import annotations

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.export_island_repair_preview import select_spans


class ExportIslandRepairPreviewTests(unittest.TestCase):
    def test_selects_only_requested_accepted_replay_child(self) -> None:
        replay = {"sentences": [
            {"start": 1.0, "end": 2.0, "accepted": True},
            {"start": 3.0, "end": 4.0, "accepted": True},
            {"start": 5.0, "end": 6.0, "accepted": False},
        ]}
        self.assertEqual(select_spans(replay, [3.0]), [
            (2, {"span": [3.0, 4.0]}),
        ])

    def test_existing_proposal_format_is_unchanged(self) -> None:
        proposals = {"rows": [
            {"index": 25, "children": [
                {"span": [1.0, 2.0], "state": "identity_supported_unverified_boundary"},
                {"span": [3.0, 4.0], "state": "withhold_identity_unresolved"},
            ]},
        ]}
        self.assertEqual(select_spans(proposals), [
            (25, {"span": [1.0, 2.0],
                  "state": "identity_supported_unverified_boundary"}),
        ])


if __name__ == "__main__":
    unittest.main()
