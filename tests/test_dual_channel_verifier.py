from __future__ import annotations

import sys
import unittest
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extractor.speaker import (  # noqa: E402
    CAMPlusProfile,
    DualSpeakerVerifier,
    SpeakerDecision,
    SpeakerMatchDecision,
    SpeakerMatchProfile,
    SpeakerProfile,
)
from extractor.pipeline import ExtractionPipeline  # noqa: E402


def _decision(
    *,
    score: float,
    secondary: bool = False,
    reference_max: float | None = None,
    window_p20: float | None = None,
    window_vote: float = 0.90,
    vote: float = 0.90,
) -> SpeakerDecision:
    value = score if reference_max is None else reference_max
    return SpeakerDecision(
        accepted=score >= 0.70,
        score=score,
        window_min_score=score,
        window_p20_score=score if window_p20 is None else window_p20,
        vote_ratio=vote,
        window_vote_ratio=window_vote,
        window_scores=[score, score],
        reference_median_score=value,
        reference_max_score=value,
        reference_spread=0.05,
        embedding=torch.tensor([1.0]),
    )


def _match(
    *,
    accepted: bool,
    tier: str,
    primary_score: float,
    secondary_score: float,
    primary_max: float,
    secondary_max: float,
    paired: float,
) -> SpeakerMatchDecision:
    primary = _decision(score=primary_score, reference_max=primary_max)
    secondary = _decision(score=secondary_score, reference_max=secondary_max)
    primary = primary.__class__(
        **{**primary.__dict__, "accepted": accepted}
    )
    secondary = secondary.__class__(
        **{**secondary.__dict__, "accepted": accepted}
    )
    return SpeakerMatchDecision(
        accepted=accepted,
        primary=primary,
        match_mode=tier,
        secondary=secondary,
        tier=tier,
        paired_reference_median=paired,
        diagnostics={"speaker_tier": tier},
    )


class DualChannelCompositionTests(unittest.TestCase):
    def test_raw_rescue_short_span_still_requires_boundary_audit(self) -> None:
        match = SimpleNamespace(
            tier="raw_rescue",
            accepted=True,
            secondary=SimpleNamespace(score=0.7),
        )
        self.assertTrue(
            ExtractionPipeline._needs_boundary_recovery(match, 1.20)
        )
        self.assertFalse(
            ExtractionPipeline._needs_boundary_recovery(match, 1.19)
        )

    def test_public_channel_api_keeps_uvr_first(self) -> None:
        signature = inspect.signature(DualSpeakerVerifier.verify_dual_channel_waveform)
        names = list(signature.parameters)
        self.assertEqual(names[:5], [
            "self",
            "uvr_waveform",
            "raw_waveform",
            "profile",
            "threshold",
        ])
        self.assertIn("clean_gate", signature.parameters)
        self.assertIn("allow_raw_rescue", signature.parameters)
        self.assertIn("audit_raw_on_uvr_accept", signature.parameters)
        self.assertIn("raw_primary", SpeakerMatchProfile.__dataclass_fields__)
        self.assertIn("raw_secondary", SpeakerMatchProfile.__dataclass_fields__)

    def _verifier(self, uvr: SpeakerMatchDecision, raw: SpeakerMatchDecision):
        verifier = DualSpeakerVerifier.__new__(DualSpeakerVerifier)
        verifier._ensure_secondary = Mock(
            return_value=CAMPlusProfile(
                embeddings=torch.tensor([[1.0]]),
                centroid=torch.tensor([1.0]),
                reference_scores=[1.0],
                reference_indexes=(0,),
            )
        )
        verifier._verify_channel_waveform = Mock(side_effect=[uvr, raw])
        return verifier

    @staticmethod
    def _profile() -> SimpleNamespace:
        # The public dual-channel method only needs profile attributes; model
        # execution is replaced by the verifier seam above.
        primary = SpeakerProfile(
            embeddings=torch.tensor([[1.0]]),
            centroid=torch.tensor([1.0]),
            reference_scores=[1.0],
            suggested_threshold=0.70,
            reference_floor=0.52,
            calibration_base=0.66,
            reference_indexes=(0,),
        )
        secondary = CAMPlusProfile(
            embeddings=torch.tensor([[1.0]]),
            centroid=torch.tensor([1.0]),
            reference_scores=[1.0],
            reference_indexes=(0,),
        )
        return SimpleNamespace(
            primary=primary,
            reference_paths=[Path("uvr.wav")],
            raw_primary=primary,
            raw_secondary=secondary,
            raw_reference_paths=[Path("raw.wav")],
        )

    def test_channel_profile_builds_raw_models_from_raw_references(self) -> None:
        def speaker_profile() -> SpeakerProfile:
            return SpeakerProfile(
                embeddings=torch.ones(1, 2),
                centroid=torch.tensor([1.0, 0.0]),
                reference_scores=[1.0],
                suggested_threshold=0.70,
                reference_floor=0.52,
                calibration_base=0.66,
                reference_indexes=(0,),
            )

        def cam_profile() -> CAMPlusProfile:
            return CAMPlusProfile(
                embeddings=torch.ones(1, 2),
                centroid=torch.tensor([1.0, 0.0]),
                reference_scores=[1.0],
                reference_indexes=(0,),
            )

        verifier = DualSpeakerVerifier.__new__(DualSpeakerVerifier)
        verifier.primary = Mock()
        verifier.primary.build_profile.side_effect = [speaker_profile(), speaker_profile()]
        verifier.secondary = Mock()
        verifier.secondary.build_profile.return_value = cam_profile()
        verifier._ensure_secondary = Mock(return_value=cam_profile())
        result = verifier.build_channel_profile(
            [Path("uvr.wav")],
            [Path("raw.wav")],
            0.70,
        )
        self.assertIsNotNone(result.raw_primary)
        self.assertIsNotNone(result.raw_secondary)
        self.assertEqual(result.raw_reference_paths, [Path("raw.wav")])
        # UVR and raw references are scored in separate profile calls.
        self.assertEqual(verifier.primary.build_profile.call_count, 2)

    def test_uvr_acceptance_remains_authoritative(self) -> None:
        uvr = _match(
            accepted=True,
            tier="strong",
            primary_score=0.78,
            secondary_score=0.76,
            primary_max=0.80,
            secondary_max=0.78,
            paired=0.76,
        )
        raw = _match(
            accepted=False,
            tier="rejected",
            primary_score=0.35,
            secondary_score=0.30,
            primary_max=0.40,
            secondary_max=0.35,
            paired=0.32,
        )
        verifier = self._verifier(uvr, raw)
        result = verifier.verify_dual_channel_waveform(
            torch.zeros(48000),
            torch.zeros(48000),
            self._profile(),
            0.70,
            3.0,
            clean_gate=True,
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.tier, "strong")
        self.assertFalse(result.diagnostics["raw_rescue"])
        self.assertEqual(result.diagnostics["dual_channel_primary"], "uvr")
        self.assertTrue(result.diagnostics["raw_channel_skipped"])
        self.assertEqual(
            result.diagnostics["raw_rescue_reason"],
            "skipped_uvr_accepted",
        )
        self.assertEqual(verifier._verify_channel_waveform.call_count, 1)

    def test_uvr_acceptance_can_opt_into_raw_audit(self) -> None:
        uvr = _match(
            accepted=True,
            tier="strong",
            primary_score=0.78,
            secondary_score=0.76,
            primary_max=0.80,
            secondary_max=0.78,
            paired=0.76,
        )
        raw = _match(
            accepted=True,
            tier="strong",
            primary_score=0.80,
            secondary_score=0.79,
            primary_max=0.82,
            secondary_max=0.81,
            paired=0.78,
        )
        verifier = self._verifier(uvr, raw)
        result = verifier.verify_dual_channel_waveform(
            torch.zeros(48000),
            torch.zeros(48000),
            self._profile(),
            0.70,
            3.0,
            clean_gate=True,
            audit_raw_on_uvr_accept=True,
        )
        self.assertTrue(result.accepted)
        self.assertFalse(result.diagnostics["raw_channel_skipped"])
        self.assertEqual(result.diagnostics["raw_rescue_reason"], "uvr_accepted")
        self.assertIsNotNone(result.raw_primary)
        self.assertEqual(verifier._verify_channel_waveform.call_count, 2)

    def test_raw_can_rescue_only_with_uvr_partial_evidence(self) -> None:
        uvr = _match(
            accepted=False,
            tier="recall",
            primary_score=0.53,
            secondary_score=0.49,
            primary_max=0.54,
            secondary_max=0.50,
            paired=0.47,
        )
        raw = _match(
            accepted=True,
            tier="strong",
            primary_score=0.76,
            secondary_score=0.73,
            primary_max=0.78,
            secondary_max=0.76,
            paired=0.72,
        )
        verifier = self._verifier(uvr, raw)
        result = verifier.verify_dual_channel_waveform(
            torch.zeros(48000),
            torch.zeros(48000),
            self._profile(),
            0.70,
            3.0,
            clean_gate=True,
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.tier, "raw_rescue")
        self.assertTrue(result.diagnostics["raw_rescue"])
        self.assertEqual(result.diagnostics["raw_rescue_reason"], "accepted")
        self.assertAlmostEqual(result.diagnostics["raw_eres_score"], 0.76)

    def test_raw_alone_never_bypasses_uvr_gate(self) -> None:
        uvr = _match(
            accepted=False,
            tier="rejected",
            primary_score=0.20,
            secondary_score=0.18,
            primary_max=0.25,
            secondary_max=0.22,
            paired=0.20,
        )
        raw = _match(
            accepted=True,
            tier="strong",
            primary_score=0.82,
            secondary_score=0.80,
            primary_max=0.84,
            secondary_max=0.82,
            paired=0.80,
        )
        verifier = self._verifier(uvr, raw)
        result = verifier.verify_dual_channel_waveform(
            torch.zeros(48000),
            torch.zeros(48000),
            self._profile(),
            0.70,
            3.0,
            clean_gate=True,
        )
        self.assertFalse(result.accepted)
        self.assertIn(
            result.diagnostics["raw_rescue_reason"],
            {
                "uvr_insufficient_evidence",
                "uvr_discontinuous",
                "uvr_prefilter_below_floor",
            },
        )

    def test_clean_gate_is_required_for_raw_rescue(self) -> None:
        uvr = _match(
            accepted=False,
            tier="recall",
            primary_score=0.53,
            secondary_score=0.49,
            primary_max=0.54,
            secondary_max=0.50,
            paired=0.47,
        )
        raw = _match(
            accepted=True,
            tier="strong",
            primary_score=0.76,
            secondary_score=0.73,
            primary_max=0.78,
            secondary_max=0.76,
            paired=0.72,
        )
        verifier = self._verifier(uvr, raw)
        result = verifier.verify_dual_channel_waveform(
            torch.zeros(48000),
            torch.zeros(48000),
            self._profile(),
            0.70,
            3.0,
            clean_gate=False,
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.diagnostics["raw_rescue_reason"], "clean_gate_disabled")

    def test_misaligned_raw_waveform_cannot_rescue(self) -> None:
        uvr = _match(
            accepted=False,
            tier="recall",
            primary_score=0.53,
            secondary_score=0.49,
            primary_max=0.54,
            secondary_max=0.50,
            paired=0.47,
        )
        raw = _match(
            accepted=True,
            tier="strong",
            primary_score=0.76,
            secondary_score=0.73,
            primary_max=0.78,
            secondary_max=0.76,
            paired=0.72,
        )
        verifier = self._verifier(uvr, raw)
        result = verifier.verify_dual_channel_waveform(
            torch.zeros(48000),
            torch.zeros(64000),
            self._profile(),
            0.70,
            3.0,
            clean_gate=True,
        )
        self.assertFalse(result.accepted)
        self.assertEqual(
            result.diagnostics["raw_rescue_reason"],
            "channel_duration_mismatch",
        )


if __name__ == "__main__":
    unittest.main()
