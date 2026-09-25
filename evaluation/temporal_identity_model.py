"""Experimental reference-conditioned frame classifier, not production inference.

The model receives acoustic frame features only. Target/other/speech/change are
independent logits; an overlap may activate target and other simultaneously.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


OUTPUTS = ("target", "other", "speech", "change")


class TemporalIdentityHead(nn.Module):
    def __init__(self, feature_dim: int = 768, projection_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        if min(feature_dim, projection_dim, hidden_dim) <= 0:
            raise ValueError("Model dimensions must be positive")
        self.project = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, projection_dim))
        self.encoder = nn.GRU(
            input_size=projection_dim * 5 + 6,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_dim * 4, len(OUTPUTS))

    def forward(
        self,
        stem_frames: torch.Tensor,
        target_references: torch.Tensor,
        *,
        raw_frames: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
        negative_references: torch.Tensor | None = None,
        negative_mask: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if stem_frames.ndim != 3 or target_references.ndim != 3:
            raise ValueError("Expected stem [B,T,D] and target references [B,R,D]")
        batch, frames, feature_dim = stem_frames.shape
        if frames == 0 or target_references.shape[0] != batch or target_references.shape[2] != feature_dim:
            raise ValueError("Query and reference shapes do not match")
        if raw_frames is not None and raw_frames.shape != stem_frames.shape:
            raise ValueError("Raw and UVR feature timelines must align")
        if target_mask is None:
            target_mask = torch.ones(target_references.shape[:2], dtype=torch.bool, device=stem_frames.device)
        if target_mask.shape != target_references.shape[:2] or target_mask.dtype != torch.bool:
            raise ValueError("Target reference mask must be boolean [B,R]")
        target_mask = target_mask.to(stem_frames.device)
        if not torch.all(target_mask.any(dim=1)):
            raise ValueError("Each item requires at least one valid target reference")
        if lengths is None:
            lengths = torch.full((batch,), frames, dtype=torch.long, device=stem_frames.device)
        lengths = lengths.to(device=stem_frames.device, dtype=torch.long)
        if lengths.shape != (batch,) or torch.any(lengths < 1) or torch.any(lengths > frames):
            raise ValueError("Invalid query lengths")

        query = F.normalize(self.project(stem_frames), dim=-1)
        raw = F.normalize(self.project(raw_frames), dim=-1) if raw_frames is not None else query
        targets = F.normalize(self.project(target_references), dim=-1)
        target_scores = torch.einsum("bth,brh->btr", query, targets)
        target_scores = target_scores.masked_fill(~target_mask[:, None, :], -1e4)
        target_context = torch.einsum(
            "btr,brh->bth", torch.softmax(target_scores, dim=-1), targets,
        )
        target_max = target_scores.max(dim=-1).values.unsqueeze(-1)
        target_count = torch.log1p(target_mask.sum(dim=1).float())[:, None, None].expand_as(target_max)

        if negative_references is None:
            negative_max = torch.zeros_like(target_max)
            negative_count = torch.zeros_like(target_max)
        else:
            if (
                negative_references.ndim != 4
                or negative_references.shape[0] != batch
                or negative_references.shape[3] != feature_dim
                or min(negative_references.shape[1:3]) == 0
            ):
                raise ValueError("Expected negative references [B,G,R,D]")
            if negative_mask is None:
                negative_mask = torch.ones(negative_references.shape[:3], dtype=torch.bool, device=stem_frames.device)
            if negative_mask.shape != negative_references.shape[:3] or negative_mask.dtype != torch.bool:
                raise ValueError("Negative reference mask must be boolean [B,G,R]")
            negative_mask = negative_mask.to(stem_frames.device)
            negatives = F.normalize(self.project(negative_references), dim=-1)
            negative_scores = torch.einsum("bth,bgrh->btgr", query, negatives)
            negative_scores = negative_scores.masked_fill(~negative_mask[:, None, :, :], -1e4)
            negative_max = negative_scores.flatten(2).max(dim=-1).values.unsqueeze(-1)
            negative_max = torch.where(
                negative_mask.flatten(1).any(dim=1)[:, None, None],
                negative_max,
                torch.zeros_like(negative_max),
            )
            negative_count = torch.log1p(negative_mask.flatten(1).sum(dim=1).float())[:, None, None].expand_as(target_max)

        raw_present = torch.full_like(target_max, float(raw_frames is not None))
        raw_agreement = (query * raw).sum(dim=-1, keepdim=True)
        features = torch.cat(
            (
                query, raw, target_context, (query - target_context).abs(), query * raw,
                target_max, negative_max, target_count, negative_count,
                raw_agreement, raw_present,
            ),
            dim=-1,
        )
        packed = nn.utils.rnn.pack_padded_sequence(
            features, lengths.detach().cpu(), batch_first=True, enforce_sorted=False,
        )
        encoded, _ = self.encoder(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(
            encoded, batch_first=True, total_length=frames,
        )
        previous = torch.cat((encoded[:, :1], encoded[:, :-1]), dim=1)
        return self.classifier(torch.cat((encoded, (encoded - previous).abs()), dim=-1))


def temporal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    lengths: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked BCE; -1 labels are unknown and never counted as negatives."""
    if logits.shape != labels.shape or logits.ndim != 3 or logits.shape[-1] != len(OUTPUTS):
        raise ValueError("Expected matching [B,T,4] logits and labels")
    if lengths.shape != (logits.shape[0],):
        raise ValueError("Invalid lengths shape")
    lengths = lengths.to(device=logits.device, dtype=torch.long)
    if torch.any(lengths < 1) or torch.any(lengths > logits.shape[1]):
        raise ValueError("Invalid valid-frame lengths")
    if not torch.isfinite(labels).all() or torch.any((labels != -1) & ((labels < 0) | (labels > 1))):
        raise ValueError("Labels must be -1 (unknown) or a value from 0 to 1")
    time_mask = torch.arange(logits.shape[1], device=logits.device)[None, :] < lengths[:, None]
    valid = (labels >= 0) & time_mask[:, :, None]
    if not valid.any():
        raise ValueError("No labeled valid frames")
    if class_weights is not None:
        if class_weights.shape != (len(OUTPUTS),) or not torch.isfinite(class_weights).all() or torch.any(class_weights <= 0):
            raise ValueError("Expected one positive finite weight per output head")
        class_weights = class_weights.to(logits.device)
    losses = F.binary_cross_entropy_with_logits(logits, labels.clamp(0, 1), reduction="none")
    if class_weights is not None:
        losses = losses * class_weights[None, None, :]
    return losses[valid].mean()
