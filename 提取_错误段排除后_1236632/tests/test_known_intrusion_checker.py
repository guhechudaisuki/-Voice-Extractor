from __future__ import annotations

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.check_known_intrusions import check
from evaluation.probe_edge_windows import edge_windows


class KnownIntrusionCheckerTests(unittest.TestCase):
    def test_preview_clip_tail_contamination_is_not_hidden_by_parent_repair(self) -> None:
        preview = {"clips": [
            {"span": [930.995, 934.495]},
            {"span": [935.555, 939.095]},
        ]}
        cases = [{"id": "foreign_tail", "other_audio_span": [938.75, 939.095],
                  "minimum_contaminated_seconds": 0.15}]
        report = check(preview, cases)
        self.assertEqual(report["passed"], 0)
        self.assertEqual(report["cases"][0]["contaminating_outputs"][0][
            "accepted_span"], [935.555, 939.095])

    def test_manifest_acceptance_and_source_offset(self) -> None:
        manifest = {"sentences": [
            {"start": 0.0, "end": 1.0, "accepted": False},
            {"start": 1.0, "end": 2.0, "accepted": True},
        ]}
        cases = [{"id": "other", "other_audio_span": [11.6, 11.9],
                  "minimum_contaminated_seconds": 0.15}]
        report = check(manifest, cases, source_offset=10.0)
        self.assertEqual(report["passed"], 0)
        self.assertEqual(report["cases"][0]["contaminating_outputs"][0][
            "overlap_seconds"], 0.3)

    def test_edge_probe_uses_geometry_not_reviewed_change_point(self) -> None:
        rows = edge_windows(2.0, 4.0, (0.25, 0.5))
        self.assertEqual(rows[-2:], [
            {"kind": "before_tail_0.5", "span": [3.0, 3.5]},
            {"kind": "tail_0.5", "span": [3.5, 4.0]},
        ])


if __name__ == "__main__":
    unittest.main()
