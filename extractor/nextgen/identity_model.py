"""Trainable reference-conditioned temporal identity/contamination head.

References remain separate tokens, never a mean speaker centroid. Matching is
order invariant; known exclusion groups remain inspectable while the independent
other-voice head can detect an unregistered person. This architecture has NO
usable default weights. Construction is for training/tests; inference must load
a fingerprinted checkpoint and its matching calibration.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


HEADS = ("target", "other", "speech", "singing", "overlap", "change", "observable", "uncertainty")


@dataclass(frozen=True)
class ReferenceFeatures:
    raw: torch.Tensor
    stem: torch.Tensor
    valid: torch.Tensor
    quality: torch.Tensor

    def validate(self, dimension: int, device: torch.device) -> None:
        if (self.raw.ndim != 2 or self.raw.shape != self.stem.shape
                or self.raw.shape[1] != dimension or self.valid.shape != self.raw.shape[:1]
                or self.quality.shape != self.valid.shape or self.valid.dtype != torch.bool
                or not self.valid.any()):
            raise ValueError("Reference requires paired [R,D] tokens and valid [R] masks")
        if any(t.device != device for t in (self.raw, self.stem, self.valid, self.quality)):
            raise ValueError("References and query must use the same device")
        if (not all(torch.isfinite(t).all() for t in (self.raw, self.stem, self.quality))
                or torch.any((self.quality < 0) | (self.quality > 1))):
            raise ValueError("Nonfinite reference features or quality outside [0,1]")


@dataclass(frozen=True)
class IdentityOutput:
    frames: torch.Tensor  # [T,len(HEADS)] independent logits, not a softmax
    purity: torch.Tensor  # scalar: whole allowed output is target-only
    boundaries: torch.Tensor  # [2]: complete start / complete end
    exclusion_scores: tuple[torch.Tensor, ...]  # each [T,2], raw/stem


class TemporalIdentityModel(nn.Module):
    architecture = "paired-tdnn-reference-local-v2"
    compatible_architectures = ("paired-tdnn-reference-local-v1", architecture)

    def __init__(self, feature_dim: int = 1500, projection_dim: int = 128,
                 hidden_dim: int = 96, architecture_version: int = 2):
        super().__init__()
        if any(type(v) is not int or v <= 0 for v in (feature_dim, projection_dim, hidden_dim)):
            raise ValueError("Invalid model dimensions")
        if type(architecture_version) is not int or architecture_version not in (1, 2):
            raise ValueError("Unsupported identity architecture version")
        self.architecture_version = architecture_version
        self.architecture = self.compatible_architectures[architecture_version - 1]
        self.config = dict(feature_dim=feature_dim, projection_dim=projection_dim,
                           hidden_dim=hidden_dim, architecture_version=architecture_version)
        self.project = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, projection_dim))
        self.temporal = nn.GRU(projection_dim * 7 + 8, hidden_dim, num_layers=2,
                               batch_first=True, bidirectional=True)
        self.frame_head = nn.Linear(hidden_dim * 4, len(HEADS))
        # Extremes over the allowed output, never a duration-weighted mean.
        self.purity_head = nn.Sequential(nn.Linear(hidden_dim * 4 + len(HEADS) * 2, hidden_dim),
                                         nn.GELU(), nn.Linear(hidden_dim, 1))
        self.boundary_head = nn.Linear(hidden_dim * 4, 2)

    def _match(self, raw: torch.Tensor, stem: torch.Tensor,
               reference: ReferenceFeatures) -> tuple[torch.Tensor, ...]:
        r = F.normalize(self.project(reference.raw), dim=-1)
        s = F.normalize(self.project(reference.stem), dim=-1)
        raw_scores = (raw @ r.T).masked_fill(~reference.valid[None, :], -torch.inf)
        stem_scores = (stem @ s.T).masked_fill(~reference.valid[None, :], -torch.inf)
        rs, ri = raw_scores.max(dim=-1)
        ss, si = stem_scores.max(dim=-1)
        return r[ri], s[si], torch.stack((rs, ss), dim=-1), torch.stack((
            reference.quality[ri], reference.quality[si],
        ), dim=-1)

    def forward(self, raw: torch.Tensor, stem: torch.Tensor, target: ReferenceFeatures,
                exclusions: tuple[ReferenceFeatures, ...], allowed: torch.Tensor) -> IdentityOutput:
        if (raw.ndim != 2 or raw.shape != stem.shape or raw.shape[0] == 0
                or raw.shape[1] != self.config["feature_dim"]
                or allowed.shape != raw.shape[:1] or allowed.dtype != torch.bool
                or allowed.device != raw.device or stem.device != raw.device or not allowed.any()):
            raise ValueError("Expected paired query [T,D] and a nonempty boolean output mask")
        if not torch.isfinite(raw).all() or not torch.isfinite(stem).all():
            raise ValueError("Nonfinite query features")
        positions = torch.where(allowed)[0]
        if int(positions[-1] - positions[0] + 1) != len(positions):
            raise ValueError("Output mask cannot stitch around contaminated audio")
        for reference in (target, *exclusions):
            reference.validate(raw.shape[1], raw.device)
        r, s = F.normalize(self.project(raw), dim=-1), F.normalize(self.project(stem), dim=-1)
        tr, ts, target_scores, quality = self._match(r, s, target)
        groups = tuple(self._match(r, s, reference)[2] for reference in exclusions)
        negative = torch.stack(groups).amax(dim=0) if groups else torch.zeros_like(target_scores)
        present = torch.full_like(target_scores[:, :1], float(bool(groups)))
        features = torch.cat((r, s, tr, ts, (r - tr).abs(), (s - ts).abs(), r * s,
                              target_scores, negative, quality, (r * s).sum(-1, keepdim=True),
                              present), dim=-1)
        hidden, _ = self.temporal(features[None])
        hidden = hidden[0]
        previous = torch.cat((hidden[:1], hidden[:-1]), dim=0)
        logits = self.frame_head(torch.cat((hidden, (hidden - previous).abs()), dim=-1))
        if self.architecture_version == 2:
            # The v1 learned projection/head erased stronger frozen-speaker
            # evidence on held-out scenes. Keep that evidence as a local skip;
            # the bounded temporal correction may refine, not overwrite it.
            direct_target = (F.normalize(raw, dim=-1)
                             @ F.normalize(target.raw, dim=-1).T)
            direct_target = direct_target.masked_fill(~target.valid[None, :], -torch.inf)
            direct_target = direct_target.amax(dim=-1, keepdim=True)
            anchored_target = 8.0 * (direct_target - 0.65) + torch.tanh(logits[:, :1])
            logits = torch.cat((anchored_target, logits[:, 1:]), dim=-1)
        selected, probs = hidden[allowed], logits[allowed].sigmoid()
        pooled = torch.cat((selected.amax(0), selected.amin(0), probs.amax(0), probs.amin(0)))
        purity = self.purity_head(pooled).squeeze(-1)
        boundary = self.boundary_head(torch.cat((hidden[positions[0]], hidden[positions[-1]])))
        return IdentityOutput(logits, purity, boundary, groups)


def masked_bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor | None:
    if logits.shape != labels.shape or not torch.isfinite(labels).all():
        raise ValueError("Mismatched/nonfinite labels")
    if torch.any((labels != -1) & ((labels < 0) | (labels > 1))):
        raise ValueError("Use -1 for unknown labels, never an invented negative")
    valid = labels >= 0
    return F.binary_cross_entropy_with_logits(logits[valid], labels[valid]) if valid.any() else None


def identity_loss(output: IdentityOutput, frames: torch.Tensor,
                  purity: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
    losses = [masked_bce(output.frames, frames), masked_bce(output.purity, purity),
              masked_bce(output.boundaries, boundaries)]
    known = [loss for loss in losses if loss is not None]
    if not known:
        raise ValueError("No known labels in this example")
    return torch.stack(known).sum()
