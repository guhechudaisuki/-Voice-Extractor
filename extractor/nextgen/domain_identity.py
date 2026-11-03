"""Compare local speech against supplied target and exclusion voices with frozen anime encoders."""
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
        # Allow 35 ms of UVR truncation when pairing utterance references.
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

    # Require the same exclusion role to win both stem models and one raw view.
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
        self.target_floors: dict[tuple[str, str], float] = {}
        for variant in VARIANTS:
            for channel in CHANNELS:
                targets = self.vectors[(variant, channel, "target")]
                if len(targets) >= 2:
                    similarities = targets @ targets.T
                    off_diagonal = similarities[~np.eye(len(targets), dtype=bool)]
                    lower, upper = (0.28, 0.45) if variant == "char" else (0.18, 0.35)
                    floor = float(np.clip(np.quantile(off_diagonal, 0.20) - 0.04,
                                          lower, upper))
                else:
                    floor = 0.32 if variant == "char" else 0.20
                self.target_floors[(variant, channel)] = floor

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
                dots = self.vectors[(variant, channel, "target")] @ vector
                target = (max(float(np.median(dots)),
                              float(np.sort(dots)[-2:].mean()) - 0.02)
                          if len(dots) >= 2 else float(dots[0]))
                role_scores = {}
                for role in self.roles:
                    r_dots = self.vectors[(variant, channel, role)] @ vector
                    role_scores[role] = (float(np.sort(r_dots)[-2:].mean())
                                         if len(r_dots) >= 3 else float(r_dots.max()))
                if not self.roles:
                    nearest_other = self.target_floors[(variant, channel)]
                    nearest_role = "baseline_floor"
                else:
                    nearest_role = max(role_scores, key=role_scores.get)
                    nearest_other = role_scores[nearest_role]
                scores.append(ModelViewScore(variant, channel, target,
                                             nearest_other, nearest_role))
        return classify(tuple(scores))
