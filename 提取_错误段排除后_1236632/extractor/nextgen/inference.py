"""Independent inference adapter. No imports from the production pipeline."""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from .artifacts import InferenceCalibration, ModelCard
from .decision_policy import Assessment, FramePrediction, SpanPrediction, assess
from .features import FeatureSequence, require_paired
from .identity_model import HEADS, ReferenceFeatures, TemporalIdentityModel
from .ledger import Candidate, EvidenceKind, LocalEvidence
from .reference_bank import ReferenceBank


@dataclass(frozen=True)
class EncodedReference:
    clip_digest: str
    raw: FeatureSequence
    stem: FeatureSequence
    valid: torch.Tensor
    quality: torch.Tensor


@dataclass(frozen=True)
class InferenceResult:
    candidate: Candidate
    prediction: SpanPrediction
    assessment: Assessment
    evidence: tuple[LocalEvidence, ...]
    # Raw sigmoid scores are diagnostics, not calibrated probabilities.
    # Frame order matches prediction.frames and column order matches HEADS.
    head_scores: tuple[tuple[float, ...], ...] = ()
    boundary_scores: tuple[float, float] | None = None


class IdentitySession:
    def __init__(self, model: TemporalIdentityModel, card: ModelCard,
                 calibration: InferenceCalibration, bank: ReferenceBank,
                 encoded: tuple[EncodedReference, ...], *, research: bool = False):
        if model.architecture != card.architecture:
            raise ValueError("Model and card architecture mismatch")
        if card.stage != "validated" and not research:
            raise ValueError("A research session must be explicitly requested")
        if card.weights_digest != calibration.identity.model_digest:
            raise ValueError("Model/calibration mismatch")
        if model.training:
            raise ValueError("Inference model must be in eval mode")
        if len({row.clip_digest for row in encoded}) != len(encoded):
            raise ValueError("Duplicate encoded references")
        mapping = {row.clip_digest: row for row in encoded}
        if set(mapping) != {row.clip_digest for row in bank.entries}:
            raise ValueError("Encoded reference bank is incomplete or contains undeclared samples")
        for reference in bank.entries:
            features = mapping[reference.clip_digest]
            require_paired(features.raw, features.stem)
            if (features.raw.source_digest != reference.source_digest
                    or features.raw.backbone_digest != card.backbone_digest
                    or any(not reference.source_span.contains(cell) for cell in features.raw.cells)):
                raise ValueError("Reference features do not match their declared source/encoder")
        self.model, self.card, self.calibration, self.bank = model, card, calibration, bank
        self.encoded = mapping

    def _references(self, candidate: Candidate):
        device = next(self.model.parameters()).device
        # Even analysis context cannot be used as its own independent reference.
        independent = self.bank.independent_of(candidate.source_sha256, candidate.context)
        roles = sorted({row.role for row in independent})
        if "target" not in roles:
            raise ValueError("No independent target reference remains for this candidate")
        grouped = {}
        for role in roles:
            rows = [self.encoded[row.clip_digest] for row in independent if row.role == role]
            grouped[role] = ReferenceFeatures(
                torch.cat([row.raw.values for row in rows]).to(device),
                torch.cat([row.stem.values for row in rows]).to(device),
                torch.cat([row.valid for row in rows]).to(device),
                torch.cat([row.quality for row in rows]).to(device),
            )
        return grouped.pop("target"), tuple(grouped[role] for role in sorted(grouped))

    @torch.inference_mode()
    def review(self, candidate: Candidate, raw: FeatureSequence, stem: FeatureSequence,
               evidence: tuple[LocalEvidence, ...] = ()) -> InferenceResult:
        require_paired(raw, stem)
        if (raw.source_digest != candidate.source_sha256
                or raw.backbone_digest != self.card.backbone_digest
                or any(not candidate.context.contains(cell) for cell in raw.cells)):
            raise ValueError("Query features do not match candidate context/encoder")
        device = next(self.model.parameters()).device
        allowed = torch.tensor([cell.intersection(candidate.output) is not None for cell in raw.cells],
                               dtype=torch.bool, device=device)
        target, excluded = self._references(candidate)
        output = self.model(raw.values.to(device), stem.values.to(device), target, excluded, allowed)
        probabilities = output.frames.sigmoid().cpu()
        column = {name: probabilities[:, i].tolist() for i, name in enumerate(HEADS)}
        boundary = output.boundaries.sigmoid().cpu().tolist()
        # These are newly computed acoustic endpoints, not inherited VAD truth.
        reviewed = replace(candidate,
                           start_complete=candidate.start_complete and boundary[0] >= self.calibration.boundary_min,
                           end_complete=candidate.end_complete and boundary[1] >= self.calibration.boundary_min)
        frames = tuple(FramePrediction(
            cell, column["target"][i], column["other"][i], column["uncertainty"][i],
            column["observable"][i] >= self.calibration.observable_min,
            column["change"][i],
        ) for i, cell in enumerate(raw.cells))
        predicted = SpanPrediction(reviewed.key, self.bank.digest, self.card.weights_digest,
                                   frames, float(output.purity.sigmoid()))
        local = list(evidence)
        for i, cell in enumerate(raw.cells):
            if cell.intersection(reviewed.output) is None:
                continue
            # Includes VAD-negative internal gaps: a short unknown voice cannot
            # disappear merely because the initial VAD missed it.
            risks = ((EvidenceKind.SINGING, column["singing"][i] >= self.calibration.singing_max),
                     (EvidenceKind.OVERLAP, column["overlap"][i] >= self.calibration.overlap_max),
                     (EvidenceKind.UNCERTAIN, column["change"][i] >= self.calibration.change_min),
                     (EvidenceKind.UNCERTAIN, column["speech"][i] >= self.calibration.speech_min
                      and column["uncertainty"][i] > self.calibration.identity.uncertainty_max))
            for risk_index, (kind, active) in enumerate(risks):
                if active:
                    local.append(LocalEvidence(reviewed.source_sha256, reviewed.sample_rate, cell, kind,
                                               self.bank.digest, self.card.weights_digest,
                                               "temporal_change" if risk_index == 2 else "temporal_head"))
        result = assess(reviewed, predicted, self.calibration.identity,
                        reference_digest=self.bank.digest, local_evidence=tuple(local))
        head_scores = tuple(tuple(float(value) for value in frame) for frame in probabilities)
        return InferenceResult(reviewed, predicted, result, tuple(local),
                               head_scores, (float(boundary[0]), float(boundary[1])))
