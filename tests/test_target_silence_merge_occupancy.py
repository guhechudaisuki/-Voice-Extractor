"""A target join must not hide an unselected speech island in its gap."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extractor.pipeline import ExtractionPipeline, PipelineOptions  # noqa: E402
from extractor.types import CandidateSentence, TimeSpan  # noqa: E402


class SameVoiceEncoder:
    def _embeddings_from_waveforms(self, waveforms, **_kwargs):
        return torch.tensor([[1.0, 0.0]] * len(waveforms))


class TargetSilenceMergeOccupancyTests(unittest.TestCase):
    def exercise(
        self, *, intervening_speech: bool, shared_vad_island: bool = False,
        middle_span: tuple[float, float] = (1.2, 1.3),
    ):
        pipeline = object.__new__(ExtractionPipeline)
        pipeline.options = PipelineOptions()
        pipeline._verify_speaker_span = lambda *_args: SimpleNamespace(accepted=True)
        pipeline._apply_speaker_match = lambda *_args: None
        encoder = SameVoiceEncoder()
        verifier = SimpleNamespace(
            primary=encoder, secondary=encoder,
            _ensure_secondary=lambda _profile: encoder,
        )
        core = CandidateSentence(1.5, 3.0, "")
        edge = CandidateSentence(0.0, 1.0, "", reject_reason="声纹匹配不足")
        edge.diagnostics["speaker_tier"] = "recall"
        rejected = [edge]
        islands = [TimeSpan(0.0, 1.0), TimeSpan(1.5, 3.0)]
        if intervening_speech:
            rejected.append(CandidateSentence(
                *middle_span, "", reject_reason="声纹匹配不足",
            ))
            islands.insert(1, TimeSpan(*middle_span))
        # The production call passes every clean atomic VAD island as a join
        # blocker.  A continuous island is possible when VAD misses a very
        # short change of speaker but a later turn check rejects the middle.
        clean_atomic_spans = (
            [TimeSpan(0.0, 3.0)] if shared_vad_island else islands
        )
        accepted = [core]
        merged = pipeline._merge_verified_target_turns(
            accepted, rejected, verifier, object(),
            torch.ones(3 * 16000), 0.65, [TimeSpan(0.0, 3.0)],
            clean_atomic_spans, clean_atomic_spans, lambda *_args: None,
        )
        return merged, accepted, rejected

    def test_separate_third_vad_island_blocks_join(self):
        merged, accepted, rejected = self.exercise(intervening_speech=True)
        self.assertEqual(merged, 0)
        self.assertEqual([(item.start, item.end) for item in accepted], [(1.5, 3.0)])
        self.assertEqual(len(rejected), 2)

    def test_rejected_third_turn_inside_one_vad_island_blocks_join(self):
        merged, accepted, rejected = self.exercise(
            intervening_speech=True, shared_vad_island=True,
        )
        self.assertEqual(merged, 0)
        self.assertEqual([(item.start, item.end) for item in accepted], [(1.5, 3.0)])
        self.assertEqual(len(rejected), 2)

    def test_rejected_turn_straddling_gap_edge_blocks_join(self):
        merged, accepted, rejected = self.exercise(
            intervening_speech=True, shared_vad_island=True,
            middle_span=(0.95, 1.2),
        )
        self.assertEqual(merged, 0)
        self.assertEqual([(item.start, item.end) for item in accepted], [(1.5, 3.0)])
        self.assertEqual(len(rejected), 2)

    def test_confirmed_empty_gap_still_allows_join(self):
        merged, accepted, rejected = self.exercise(intervening_speech=False)
        self.assertEqual(merged, 1)
        self.assertEqual([(item.start, item.end) for item in accepted], [(0.0, 3.0)])
        self.assertEqual(rejected, [])

    def test_shared_vad_without_rejected_middle_is_unchanged(self):
        merged, accepted, rejected = self.exercise(
            intervening_speech=False, shared_vad_island=True,
        )
        self.assertEqual(merged, 1)
        self.assertEqual([(item.start, item.end) for item in accepted], [(0.0, 3.0)])
        self.assertEqual(rejected, [])


if __name__ == "__main__":
    unittest.main()
