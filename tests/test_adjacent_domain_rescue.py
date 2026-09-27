"""Research rescue never lets a high whole score erase a bad local side."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extractor.pipeline import ExtractionPipeline, PipelineOptions  # noqa: E402
from extractor.types import CandidateSentence, TimeSpan  # noqa: E402


class AdjacentDomainRescueTests(unittest.TestCase):
    def exercise(self, *, side_states=("target_supported", "target_supported"),
                 change=False, blocked=(), exclusions=True,
                 intervening=False, both_sides=False, gap_island=False,
                 redundant_core_rejection=False):
        with tempfile.TemporaryDirectory() as directory:
            assets = Path(directory) / "assets"
            for name in ("anime-speaker-char/anime_speaker_char_ecapa.onnx",
                         "anime-speaker-va/anime_speaker_va_ecapa.onnx"):
                path = assets / "models" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

            class Witness:
                def __init__(self):
                    self.states = iter(side_states)

                def score(self, stem, raw):
                    return SimpleNamespace(state=next(self.states))

            class Splitter:
                def __init__(self, *_args, **_kwargs):
                    pass

                def detect_multiscale_speaker_boundaries(self, *_args):
                    return [object()] if change else []

            pipeline = object.__new__(ExtractionPipeline)
            pipeline.options = PipelineOptions()
            pipeline._raw_target_waveform = torch.ones(6 * 16000) * 0.05
            pipeline._install_acoustic_identity_certificate = lambda _core: True
            pipeline._verify_speaker_span = lambda *_args, **_kwargs: SimpleNamespace(
                accepted=True, tier="strong",
            )
            pipeline._apply_speaker_match = lambda *_args: None
            verifier = SimpleNamespace(
                primary=object(), secondary=object(),
                exclusion_audit=lambda *_args, **_kwargs: {
                    "excluded_role_rejected": False,
                },
            )
            edge = CandidateSentence(
                0.97, 1.97 if gap_island or intervening else 2.17, "",
                reject_reason="声纹匹配不足",
            )
            core = CandidateSentence(2.17, 4.97, "")
            accepted, rejected = [core], [edge]
            if intervening:
                rejected.append(CandidateSentence(
                    2.02, 2.12, "", reject_reason="声纹匹配不足",
                ))
            if redundant_core_rejection:
                rejected.append(CandidateSentence(
                    2.50, 2.75, "", reject_reason="声纹匹配不足",
                ))
            if both_sides:
                rejected.append(CandidateSentence(
                    4.97, 5.97, "", reject_reason="声纹匹配不足",
                ))
            with (patch("extractor.pipeline.ASSET_ROOT", assets),
                  patch("extractor.pipeline.WORK_ROOT", Path(directory)),
                  patch("extractor.anime_identity.AnimeDomainWitness",
                        return_value=Witness()),
                  patch("extractor.pipeline.LocalSpeakerTurnSplitter", Splitter),
                  patch("extractor.pipeline.write_clip")):
                recovered = pipeline._experimental_rescue_adjacent_domain_sides(
                    accepted, rejected, verifier, object(),
                    torch.ones(6 * 16000) * 0.05,
                    Path("stem.wav"), Path("raw.wav"), 0.7,
                    [Path("target.wav")], [Path("target.wav")],
                    [[Path("other.wav")]] if exclusions else [],
                    [[Path("other.wav")]] if exclusions else [],
                    [object()] if exclusions else [], blocked,
                    ([TimeSpan(0.97, 1.97), TimeSpan(2.02, 2.12),
                      TimeSpan(2.17, 4.97)] if gap_island
                     else [TimeSpan(0.97, 1.97), TimeSpan(2.17, 4.97)]
                     if intervening
                     else [TimeSpan(0.97, 4.97)]),
                )
            return recovered, accepted, rejected

    def test_independent_sides_can_restore_complete_span(self):
        recovered, accepted, rejected = self.exercise()
        self.assertEqual(recovered, 1)
        self.assertEqual((accepted[0].start, accepted[0].end), (0.97, 4.97))
        self.assertEqual(rejected, [])
        self.assertEqual(accepted[0].diagnostics["rescued_side_span"], [0.97, 2.17])

    def test_other_side_is_not_laundered_by_good_whole_score(self):
        recovered, accepted, rejected = self.exercise(
            side_states=("other_supported", "target_supported"),
        )
        self.assertEqual(recovered, 0)
        self.assertEqual((accepted[0].start, accepted[0].end), (2.17, 4.97))
        self.assertEqual(len(rejected), 1)

    def test_change_and_contamination_are_independent_vetoes(self):
        for kwargs in ({"change": True}, {"blocked": (TimeSpan(1.5, 1.6),)}):
            with self.subTest(kwargs=kwargs):
                recovered, accepted, rejected = self.exercise(**kwargs)
                self.assertEqual(recovered, 0)
                self.assertEqual((accepted[0].start, accepted[0].end), (2.17, 4.97))
                self.assertEqual(len(rejected), 1)

    def test_no_exclusion_references_preserves_old_behavior(self):
        recovered, accepted, rejected = self.exercise(exclusions=False)
        self.assertEqual(recovered, 0)
        self.assertEqual((accepted[0].start, accepted[0].end), (2.17, 4.97))
        self.assertEqual(len(rejected), 1)

    def test_independent_intervening_voice_blocks_whole_join(self):
        recovered, accepted, rejected = self.exercise(intervening=True)
        self.assertEqual(recovered, 0)
        self.assertEqual((accepted[0].start, accepted[0].end), (2.17, 4.97))
        self.assertEqual(len(rejected), 2)

    def test_unscored_vad_island_in_gap_blocks_whole_join(self):
        recovered, accepted, rejected = self.exercise(gap_island=True)
        self.assertEqual(recovered, 0)
        self.assertEqual((accepted[0].start, accepted[0].end), (2.17, 4.97))
        self.assertEqual(len(rejected), 1)

    def test_alternate_core_rejection_is_not_a_third_gap_island(self):
        recovered, accepted, rejected = self.exercise(
            redundant_core_rejection=True,
        )
        self.assertEqual(recovered, 1)
        self.assertEqual((accepted[0].start, accepted[0].end), (0.97, 4.97))
        self.assertEqual(len(rejected), 1)

    def test_both_sides_can_be_recovered_in_sequence(self):
        recovered, accepted, rejected = self.exercise(
            both_sides=True, side_states=("target_supported",) * 4,
        )
        self.assertEqual(recovered, 2)
        self.assertEqual((accepted[0].start, accepted[0].end), (0.97, 5.97))
        self.assertEqual(rejected, [])


if __name__ == "__main__":
    unittest.main()
