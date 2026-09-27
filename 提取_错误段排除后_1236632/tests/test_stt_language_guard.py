"""Visible Japanese script must not be overwritten by Chinese STT."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extractor.pipeline import _language_from_text  # noqa: E402


class SttLanguageGuardTests(unittest.TestCase):
    def test_kana_outweighs_misreported_chinese_language(self):
        self.assertEqual(
            _language_from_text("私、大前久美子。よろしく。", "zh"), "ja"
        )

    def test_chinese_without_kana_stays_chinese(self):
        self.assertEqual(_language_from_text("你好。", "zh"), "zh")


if __name__ == "__main__":
    unittest.main()
