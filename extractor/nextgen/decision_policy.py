"""Pure candidate assessment. Thresholds must come from a calibration artifact.

No arithmetic mean of local identity scores authorizes output. Definite local
contamination wins over the learned span-purity score. Unobservable consonants
are not required to carry a standalone speaker embedding: the contextual purity
head must support the entire span, and acoustic endpoints must be complete.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .ledger import Candidate, EvidenceKind, LocalEvidence, State
from .timeline import SampleSpan, require_digest, union_length


def probability(value: float) -> None:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Expected a finite probability in [0, 1]")


@dataclass(frozen=True)
class Calibration:
    model_digest: str
    data_digest: str
    version: str
    target_min: float
    other_max: float
    uncertainty_max: float
    purity_min: float

    def __post_init__(self) -> None:
        require_digest(self.model_digest)
        require_digest(self.data_digest)
        if not self.version:
            raise ValueError("Calibration requires a version")
        for value in (self.target_min, self.other_max, self.uncertainty_max, self.purity_min):
            probability(value)


@dataclass(frozen=True)
class FramePrediction:
    span: SampleSpan
    target: float
    other: float
    uncertainty: float
    identity_observable: bool
    change: float = 0.0

    def __post_init__(self) -> None:
        for value in (self.target, self.other, self.uncertainty, self.change):
            probability(value)
        if type(self.identity_observable) is not bool:
            raise ValueError("Identity observability must be explicit")


@dataclass(frozen=True)
class SpanPrediction:
    candidate_key: str
    reference_digest: str
    model_digest: str
    frames: tuple[FramePrediction, ...]
    purity: float

    def __post_init__(self) -> None:
        for value in (self.reference_digest, self.model_digest):
            require_digest(value)
        probability(self.purity)
        if not self.candidate_key or not self.frames:
            raise ValueError("Prediction requires an exact candidate scope and local frames")
        if any(left.span.end > right.span.start for left, right in zip(self.frames, self.frames[1:])):
            raise ValueError("Frame cells must be ordered and nonoverlapping")


@dataclass(frozen=True)
class Assessment:
    candidate_key: str
    state: State
    reasons: tuple[str, ...]
    risk_spans: tuple[SampleSpan, ...]
    policy_version: str


def assess(
    candidate: Candidate, prediction: SpanPrediction, calibration: Calibration,
    *, reference_digest: str, local_evidence: tuple[LocalEvidence, ...] = (),
) -> Assessment:
    if prediction.candidate_key != candidate.key:
        raise ValueError("Prediction cannot be reused after a boundary/context change")
    if prediction.model_digest != calibration.model_digest:
        raise ValueError("Model does not match calibration")
    if prediction.reference_digest != reference_digest:
        raise ValueError("Reference bank changed after prediction")
    for evidence in local_evidence:
        if (evidence.source_sha256, evidence.sample_rate, evidence.reference_digest,
            evidence.model_digest) != (candidate.source_sha256, candidate.sample_rate,
                                      reference_digest, calibration.model_digest):
            raise ValueError("Stale or cross-source local evidence")
    relevant = [evidence for evidence in local_evidence
                if evidence.span.intersection(candidate.output) is not None]
    hard = [evidence for evidence in relevant if evidence.kind in (
        EvidenceKind.OTHER, EvidenceKind.SINGING, EvidenceKind.OVERLAP,
    )]
    frames = [frame for frame in prediction.frames
              if frame.span.intersection(candidate.output) is not None]
    # Other-voice evidence is never diluted by duration or by a target majority.
    contamination = [frame.span for frame in frames if frame.other >= calibration.other_max]
    if hard or contamination:
        return Assessment(candidate.key, State.REJECTED, ("local_contamination",),
                          tuple([item.span for item in hard] + contamination), calibration.version)
    reasons: list[str] = []
    risk: list[SampleSpan] = []
    # Check the entire exported span, including VAD-negative gaps: those gaps
    # can contain a missed short other-speaker sound. Context is not output.
    for speech in (candidate.output,):
        coverage = [part for frame in frames
                    if (part := frame.span.intersection(speech)) is not None]
        if union_length(coverage) != speech.length:
            reasons.append("missing_temporal_coverage")
            risk.append(speech)
    observed = [frame for frame in frames if frame.identity_observable
                and any(frame.span.intersection(part) for part in candidate.speech)]
    if not observed:
        reasons.append("no_observable_identity")
    weak = [frame.span for frame in observed if frame.target < calibration.target_min]
    uncertain = [frame.span for frame in frames
                 if frame.uncertainty > calibration.uncertainty_max
                 and any(frame.span.intersection(part) for part in candidate.speech)]
    uncertainty_evidence = [item.span for item in relevant if item.kind == EvidenceKind.UNCERTAIN]
    if weak or uncertain or uncertainty_evidence:
        reasons.append("local_identity_unresolved")
        risk.extend([*weak, *uncertain, *uncertainty_evidence])
    if prediction.purity < calibration.purity_min:
        reasons.append("span_purity_unresolved")
    if not candidate.start_complete or not candidate.end_complete:
        reasons.append("acoustic_boundary_incomplete")
    return Assessment(candidate.key, State.UNRESOLVED if reasons else State.ACCEPTED,
                      tuple(dict.fromkeys(reasons)) or ("acoustically_verified",),
                      tuple(sorted(set(risk))), calibration.version)
