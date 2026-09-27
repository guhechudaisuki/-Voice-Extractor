from __future__ import annotations

import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extractor.anime_identity import AnimeDomainScore


class AnimeIdentityTests(unittest.TestCase):
    def test_score_states_are_explicit(self):
        score = AnimeDomainScore("unresolved", (), ())
        self.assertEqual(score.state, "unresolved")


if __name__ == "__main__":
    unittest.main()
