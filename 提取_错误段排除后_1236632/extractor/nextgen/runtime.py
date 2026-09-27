"""One explicit bridge from prepared real audio to the independent engine."""
from __future__ import annotations

from functools import lru_cache
from typing import Callable

from .engine import EngineResult, run_prepared
from .features import WavLMSpeakerFeatures
from .inference import IdentitySession
from .scene_adapter import PreparedScene
from .timeline import SampleSpan


def run_scene(scene: PreparedScene, session: IdentitySession,
              encoder: WavLMSpeakerFeatures, *,
              cancelled: Callable[[], bool] = lambda: False,
              progress: Callable[[int, int, str], None] = lambda *args: None) -> EngineResult:
    """Run only with a matching fingerprinted model/calibration session.

    There is intentionally no random/default model or old-score stand-in.
    Calling code must have loaded a validated bundle and encoded references.
    """
    if encoder.digest != session.card.backbone_digest:
        raise ValueError("Scene encoder does not match the identity model")
    if not scene.candidates:
        raise ValueError("Prepared scene contains no acoustic speech candidate")

    @lru_cache(maxsize=16)
    def features(context: SampleSpan):
        return scene.audio.encode(context, encoder)

    return run_prepared(
        scene.audio.source, scene.candidates, session,
        lambda candidate: features(candidate.context), silence=scene.silence,
        evidence=scene.evidence(session.bank.digest, session.card.weights_digest),
        boundary_options=scene.gap_options,
        change_tolerance_samples=max(1600, scene.silence.lower_samples),
        cancelled=cancelled, progress=progress,
    )
