from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extractor.anime import AnimeIdentityDecision, AnimeIdentityVerifier


class AnimeRecoveryGateTests(unittest.TestCase):
    def test_both_independent_margins_are_required(self) -> None:
        decision = AnimeIdentityDecision(0.46, 0.18, 0.28, 0.56, 0.17, 0.39)
        self.assertGreater(decision.char_margin, AnimeIdentityVerifier.MARGIN_FLOOR)
        self.assertGreater(decision.va_margin, AnimeIdentityVerifier.MARGIN_FLOOR)

    def test_one_weak_model_does_not_pass(self) -> None:
        decision = AnimeIdentityDecision(0.50, 0.34, 0.16, 0.42, 0.30, 0.12)
        self.assertFalse(
            decision.char_margin > AnimeIdentityVerifier.MARGIN_FLOOR
            and decision.va_margin > AnimeIdentityVerifier.MARGIN_FLOOR
        )


if __name__ == "__main__":
    unittest.main()
