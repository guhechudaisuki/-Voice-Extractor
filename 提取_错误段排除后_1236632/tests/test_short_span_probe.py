from __future__ import annotations

import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.probe_short_span_identity import covered_fraction, windows_for_span
sys.path.insert(0, str(ROOT / "evaluation"))
from probe_anime_edge_windows import select_edge_windows


class ShortSpanGeometryTests(unittest.TestCase):
    def test_windows_include_exact_tail_once(self) -> None:
        self.assertEqual(
            windows_for_span(1.0, 2.0, 0.6, 0.3),
            [(1.0, 1.6), (1.3, 1.9), (1.4, 2.0)],
        )

    def test_short_case_has_no_window(self) -> None:
        self.assertEqual(windows_for_span(0.0, 0.5, 0.6, 0.3), [])

    def test_coverage_does_not_double_count_overlapping_spans(self) -> None:
        self.assertAlmostEqual(
            covered_fraction(1.0, 2.0, [(0.0, 1.5), (1.3, 1.8)]),
            0.8,
        )

    def test_invalid_window_geometry_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            windows_for_span(2.0, 1.0, 0.6, 0.3)

    def test_edge_selection_keeps_first_last_and_all_known_other(self) -> None:
        report = {
            "minimum_clean_speech_fraction": 0.8,
            "windows": [
                {"case_id": "a", "label": "target", "start": start,
                 "end": start + 0.6, "window_seconds": 0.6,
                 "clean_speech_fraction": 1.0}
                for start in (1.0, 1.3, 1.6)
            ] + [
                {"case_id": "other", "label": "other", "start": 4.0,
                 "end": 4.6, "window_seconds": 0.6,
                 "clean_speech_fraction": 1.0}
            ],
        }
        selected = select_edge_windows(report)
        self.assertEqual([row["start"] for row in selected], [1.0, 1.6, 4.0])


if __name__ == "__main__":
    unittest.main()
