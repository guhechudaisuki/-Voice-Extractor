"""Reference-conditioned, frozen anime-speaker evidence for local speech parts.

This is an independent acoustic witness, not a calibrated open-set classifier.
The optional exclusion roles are anonymous: their names convey no identity to
the models. Without exclusions this module abstains rather than treating the
nearest target reference as proof that no other speaker exists.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
from typing import Literal, Mapping, Sequence

import numpy as np
import torch

from .anime_embedding import AnimeSpeakerOnnx


CHANNELS = ("stem", "raw")
VARIANTS = ("char", "va")
IdentityVerdict = Literal["target_supported", "other_supported", "unresolved"]


@dataclass(frozen=True)
class ReferenceAudio:
    role: str
    stem: torch.Tensor
    raw: torch.Tensor
    sample_rate: int

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("Reference role is required")
        if self.sample_rate != 16000:
            raise ValueError("Anime reference audio must be resampled to 16 kHz")
        if self.stem.ndim != 1 or self.raw.ndim != 1:
            raise ValueError("Reference audio must be mono")
        # Existing UVR reference crops can end up to ~35 ms shorter than the
        # matching original crop. That is acceptable for utterance embeddings,
        # but a larger difference suggests the files were paired incorrectly.
        if abs(self.stem.numel() - self.raw.numel()) > self.sample_rate // 20:
            raise ValueError("Reference raw/stem audio must align within 50 ms")


@dataclass(frozen=True)
class ModelViewScore:
    variant: str
    channel: str
    target_median: float
    nearest_other: float | None
    nearest_role: str | None

    @property
    def margin(self) -> float | None:
        return (self.target_median - self.nearest_other
                if self.nearest_other is not None else None)


@dataclass(frozen=True)
class DomainVerdict:
    state: IdentityVerdict
    scores: tuple[ModelViewScore, ...]

    def score(self, variant: str, channel: str) -> ModelViewScore:
        return next(row for row in self.scores
                    if row.variant == variant and row.channel == channel)


@dataclass(frozen=True)
class JoinVerdict:
    allowed: bool
    reason: str


def classify(scores: tuple[ModelViewScore, ...]) -> DomainVerdict:
    if len(scores) != 4 or {(row.variant, row.channel) for row in scores} != {
        (variant, channel) for variant in VARIANTS for channel in CHANNELS
    }:
        raise ValueError("All four frozen model/channel observations are required")
    if any(row.margin is None for row in scores):
        return DomainVerdict("unresolved", scores)

    supporting = [row for row in scores if row.margin is not None and row.margin > 0]
    if len(supporting) >= 2 and any(row.channel == "stem" for row in supporting):
        return DomainVerdict("target_supported", scores)

    # A negative identity is only a local witness when the same user-supplied
    # role wins in both differently trained stem models and in a raw view.
    # Different roles winning separate models are disagreement, not certainty.
    stem = [row for row in scores if row.channel == "stem"]
    if (stem[0].nearest_role is not None
            and stem[0].nearest_role == stem[1].nearest_role
            and all(row.margin is not None and row.margin < 0 for row in stem)
            and any(row.channel == "raw" and row.nearest_role == stem[0].nearest_role
                    and row.margin is not None and row.margin < 0 for row in scores)):
        return DomainVerdict("other_supported", scores)
    return DomainVerdict("unresolved", scores)


def review_join(left: DomainVerdict, right: DomainVerdict, *,
                whole_verified: bool, confirmed_change: bool,
                gap_allowed: bool, contaminated: bool = False) -> JoinVerdict:
    """Only propose a join; downstream full-boundary and export gates remain."""
    if contaminated:
        return JoinVerdict(False, "singing_or_overlap")
    if not gap_allowed:
        return JoinVerdict(False, "hard_silence_boundary")
    if confirmed_change:
        return JoinVerdict(False, "confirmed_speaker_change")
    if not whole_verified:
        return JoinVerdict(False, "whole_span_not_verified")
    if left.state != "target_supported" or right.state != "target_supported":
        return JoinVerdict(False, "local_identity_unresolved")
    return JoinVerdict(True, "independent_local_target_support")


class DomainReferenceBank:
    """Encodes supplied clips once; query audio cannot become its own reference."""

    def __init__(self, encoders: Mapping[str, AnimeSpeakerOnnx],
                 references: Sequence[ReferenceAudio]):
        if set(encoders) != set(VARIANTS) or not references:
            raise ValueError("Both frozen anime models and target references are required")
        if not any(item.role == "target" for item in references):
            raise ValueError("At least one target reference is required")
        owners: dict[str, str] = {}
        unique: list[ReferenceAudio] = []
        for item in references:
            waveform = item.raw.detach().cpu().float().contiguous()
            if not torch.isfinite(waveform).all():
                raise ValueError("Reference audio contains nonfinite samples")
            digest = hashlib.sha256(waveform.numpy().tobytes()).hexdigest()
            previous = owners.get(digest)
            if previous is not None:
                if previous != item.role:
                    raise ValueError("Identical reference audio belongs to conflicting roles")
                continue
            owners[digest] = item.role
            unique.append(item)

        grouped: dict[tuple[str, str, str], list[np.ndarray]] = defaultdict(list)
        for item in unique:
            for variant, encoder in encoders.items():
                for channel in CHANNELS:
                    vector = encoder.encode(getattr(item, channel), sample_rate=item.sample_rate)
                    grouped[(variant, channel, item.role)].append(vector)
        self.encoders = dict(encoders)
        self.roles = tuple(sorted({item.role for item in unique if item.role != "target"}))
        self.vectors = {key: np.stack(rows) for key, rows in grouped.items()}

    def score(self, *, stem: torch.Tensor, raw: torch.Tensor,
              sample_rate: int) -> DomainVerdict:
        if sample_rate != 16000:
            raise ValueError("Anime query audio must be resampled to 16 kHz")
        if abs(stem.numel() - raw.numel()) > 160:
            raise ValueError("Query raw/stem audio must be aligned within 10 ms")
        scores: list[ModelViewScore] = []
        for variant, encoder in self.encoders.items():
            for channel in CHANNELS:
                vector = encoder.encode(stem if channel == "stem" else raw,
                                        sample_rate=sample_rate)
                target = float(np.median(self.vectors[(variant, channel, "target")] @ vector))
                role_scores = {
                    role: float((self.vectors[(variant, channel, role)] @ vector).max())
                    for role in self.roles
                }
                nearest_role = (max(role_scores, key=role_scores.get) if role_scores else None)
                scores.append(ModelViewScore(variant, channel, target,
                                             role_scores.get(nearest_role), nearest_role))
        return classify(tuple(scores))
