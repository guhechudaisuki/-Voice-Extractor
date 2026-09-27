from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extractor.boundary_proposals import (
    propose_cross_channel_boundaries, subdivide_islands,
)
from extractor.types import TimeSpan


class BoundaryProposalTests(unittest.TestCase):
    def test_agreement_creates_only_an_alternative_partition(self):
        island = TimeSpan(10.0, 13.0)
        cuts = propose_cross_channel_boundaries(
            [island], [12.3], [12.26],
        )
        self.assertEqual(len(cuts), 1)
        self.assertAlmostEqual(cuts[0].time, 12.28)
        self.assertEqual(
            subdivide_islands([island], cuts),
            [TimeSpan(10.0, 12.28), TimeSpan(12.28, 13.0)],
        )
        self.assertEqual(island, TimeSpan(10.0, 13.0))

    def test_disagreement_and_edge_hypotheses_are_not_used(self):
        cuts = propose_cross_channel_boundaries(
            [TimeSpan(10.0, 13.0)], [10.1, 12.3], [12.0],
        )
        self.assertEqual(cuts, [])

    def test_duplicate_reports_do_not_duplicate_cuts(self):
        cuts = propose_cross_channel_boundaries(
            [TimeSpan(10.0, 13.0)], [12.3, 12.3], [12.26, 12.26],
        )
        self.assertEqual(len(cuts), 1)

    def test_higher_confidence_stem_wins_one_raw_observation(self):
        cuts = propose_cross_channel_boundaries(
            [TimeSpan(10.0, 13.0)],
            [
                {"time": 12.47, "confidence": 0.75},
                {"time": 12.67, "confidence": 0.98},
            ],
            [{"time": 12.57, "confidence": 0.82}],
            include_unmatched_stem=True,
        )
        matched = [cut for cut in cuts if cut.support == "stem_and_raw"]
        self.assertEqual(len(matched), 1)
        self.assertAlmostEqual(matched[0].stem_time, 12.67)
        self.assertTrue(any(cut.support == "stem_only" for cut in cuts))


if __name__ == "__main__":
    unittest.main()
