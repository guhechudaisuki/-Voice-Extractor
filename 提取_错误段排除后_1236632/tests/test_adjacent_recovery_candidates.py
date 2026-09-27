"""The recovery audit must start from physical islands, not past decisions."""

from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))

from probe_adjacent_recovery import candidate_pairs, local_windows, run  # noqa: E402
from extractor.types import TimeSpan  # noqa: E402
from extractor.nextgen.domain_identity import DomainVerdict  # noqa: E402


class AdjacentRecoveryCandidateTests(unittest.TestCase):
    @staticmethod
    def stage(islands, **extra):
        return {"stages": {
            "clean_speech_islands": {"spans": islands},
            **{name: {"spans": spans} for name, spans in extra.items()},
        }}

    def test_two_rejected_sides_are_still_adjacent_candidates(self):
        stage = self.stage([[1.0, 2.0], [2.5, 3.5]])
        pairs = candidate_pairs(stage, gap_min=0.2, gap_max=0.85)
        self.assertEqual(
            [([left.start, left.end], [right.start, right.end])
             for left, right in pairs],
            [([1.0, 2.0], [2.5, 3.5])],
        )

    def test_intervening_other_island_cannot_be_skipped(self):
        stage = self.stage([[1.0, 2.0], [2.2, 2.4], [2.6, 3.5]])
        pairs = candidate_pairs(stage, gap_min=0.2, gap_max=0.85)
        self.assertEqual(len(pairs), 2)
        self.assertNotIn((1.0, 3.5), [(a.start, b.end) for a, b in pairs])

    def test_singing_or_overlap_between_sides_blocks_join_proposal(self):
        stage = self.stage(
            [[1.0, 2.0], [2.5, 3.5]],
            overlap_evidence=[[2.1, 2.3]],
        )
        self.assertEqual(candidate_pairs(stage, gap_min=0.2, gap_max=0.85), [])

    def test_gap_above_upper_bound_is_not_proposed(self):
        stage = self.stage([[1.0, 2.0], [2.851, 3.5]])
        self.assertEqual(candidate_pairs(stage, gap_min=0.2, gap_max=0.85), [])

    def test_local_scan_includes_short_final_voice_and_covers_full_span(self):
        windows = local_windows(TimeSpan(1241.17, 1245.46))
        self.assertIn(TimeSpan(1245.21, 1245.46), windows)
        short = [row for row in windows if abs(row.duration - 0.25) < 1e-6]
        self.assertLessEqual(short[0].start, 1241.17)
        self.assertGreaterEqual(short[-1].end, 1245.46)
        self.assertTrue(all(
            1241.17 - 1e-6 <= row.start < row.end <= 1245.46 + 1e-6
            for row in windows
        ))

    def test_silent_island_is_unresolved_without_aborting_other_pairs(self):
        class Bank:
            def score(self, *, stem, raw, sample_rate):
                if float(stem.square().mean()) < 1e-10:
                    raise ValueError("Silent audio cannot provide speaker identity evidence")
                return DomainVerdict("target_supported", ())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / "work"
            (work / "stems").mkdir(parents=True)
            audio = np.zeros(16000 * 4, dtype=np.float32)
            audio[2 * 16000:3 * 16000] = 0.05
            for file in (work / "target_normalized.wav",
                         work / "stems" / "target_vocals.wav"):
                sf.write(file, audio, 16000)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "sentences": [], "options": {"speaker_threshold": 0.68},
            }), encoding="utf-8")
            stage = root / "stage.json"
            stage.write_text(json.dumps(self.stage([[0.1, 1.1], [1.5, 2.5]])),
                             encoding="utf-8")
            with patch("probe_adjacent_recovery.build_bank", return_value=Bank()):
                report = run(work, manifest, stage)
        self.assertEqual(report["pair_count"], 1)
        self.assertEqual(report["pairs"][0]["left_state"], "unresolved")
        self.assertEqual(report["pairs"][0]["right_state"], "target_supported")
        self.assertEqual(report["pairs"][0]["left_acoustic_issue"],
                         "Silent audio cannot provide speaker identity evidence")
        self.assertFalse(report["pairs"][0]["both_target_supported"])


if __name__ == "__main__":
    unittest.main()
