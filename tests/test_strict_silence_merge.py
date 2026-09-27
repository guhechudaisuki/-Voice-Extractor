from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extractor.pipeline import ExtractionPipeline, PipelineOptions
from extractor.types import TimeSpan


class _FakeEmbedder:
    """Constant embeddings: pairwise similarity is fully controlled."""

    def __init__(self, value: float) -> None:
        self.value = value

    def _embeddings_from_waveforms(self, waves, progress=None):
        return torch.stack([torch.full((4,), self.value) for _ in waves])


class _FakeVerifier:
    def __init__(self, primary_value: float, secondary_value: float) -> None:
        self.primary = _FakeEmbedder(primary_value)
        self.secondary = _FakeEmbedder(secondary_value)

    def _ensure_secondary(self, profile) -> None:
        return None


def noop_progress(value: float, message: str) -> None:
    return None


class StrictSilenceMergeTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = ExtractionPipeline(PipelineOptions())
        self.waveform = torch.zeros(16000 * 4)
        self.spans = [TimeSpan(1.0, 2.0), TimeSpan(2.3, 3.5)]

    def merge(self, *, strict_flags=None, primary=0.1, secondary=0.1, forbidden=()):
        verifier = _FakeVerifier(primary, secondary)
        return self.pipeline._merge_short_silence_same_speaker(
            list(self.spans),
            verifier,
            None,
            self.waveform,
            noop_progress,
            forbidden_joins=list(forbidden),
            strict_flags=strict_flags,
        )

    def test_two_strict_turns_merge_without_fragment_floor(self):
        # Similarity 0.04 is far below the 0.76/0.64 floors, but both sides
        # already passed the formal dual-model verification themselves.
        merged = self.merge(strict_flags=[True, True])
        self.assertEqual(len(merged), 1)
        self.assertEqual((merged[0].start, merged[0].end), (1.0, 3.5))

    def test_strict_plus_edge_still_requires_fragment_floor(self):
        merged = self.merge(strict_flags=[True, False])
        self.assertEqual(len(merged), 2)

    def test_default_behaviour_unchanged_without_flags(self):
        merged = self.merge(strict_flags=None)
        self.assertEqual(len(merged), 2)

    def test_blocked_join_still_guards_strict_pairs(self):
        blocker = TimeSpan(2.05, 2.25)
        merged = self.merge(strict_flags=[True, True], forbidden=[blocker])
        self.assertEqual(len(merged), 2)

    def test_high_similarity_still_merges_with_flags(self):
        merged = self.merge(strict_flags=[True, True], primary=0.5, secondary=0.5)
        self.assertEqual(len(merged), 1)

    def test_misaligned_flags_are_rejected(self):
        with self.assertRaises(ValueError):
            self.merge(strict_flags=[True])

    def test_gap_over_limit_never_merges(self):
        self.spans = [TimeSpan(1.0, 2.0), TimeSpan(3.2, 3.5)]
        merged = self.merge(strict_flags=[True, True])
        self.assertEqual(len(merged), 2)


if __name__ == "__main__":
    unittest.main()
