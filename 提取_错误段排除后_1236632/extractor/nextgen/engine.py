"""Prepared-candidate engine for independent testing, not desktop integration.

Preprocessing has to finish singing -> UVR -> VAD before this entry. It cannot
silently regenerate the source, bypass event masks, or use transcript labels.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from .boundary_decoder import (AcousticGap, SilenceRange, propose_change_parts,
                               propose_unresolved_change_sides, select_verified)
from .features import FeatureSequence
from .finalization import final_audit
from .decision_policy import assess
from .inference import IdentitySession, InferenceResult
from .ledger import Candidate, CandidateLedger, Decision, EvidenceKind, LocalEvidence
from .timeline import SourceTimeline


@dataclass(frozen=True)
class EngineResult:
    ledger: CandidateLedger
    reviews: tuple[InferenceResult, ...]
    selected: tuple[Candidate, ...]
    cancelled: bool


def run_prepared(source: SourceTimeline, candidates: tuple[Candidate, ...], session: IdentitySession,
                 features: Callable[[Candidate], tuple[FeatureSequence, FeatureSequence]], *,
                 silence: SilenceRange, evidence: tuple[LocalEvidence, ...] = (),
                 boundary_options: Callable[[InferenceResult], tuple[AcousticGap, ...]] | None = None,
                 change_tolerance_samples: int = 0,
                 cancelled: Callable[[], bool] = lambda: False,
                 progress: Callable[[int, int, str], None] = lambda *args: None) -> EngineResult:
    if type(change_tolerance_samples) is not int or change_tolerance_samples < 0:
        raise ValueError("Invalid change-boundary tolerance")
    queued_keys = {row.key for row in candidates}
    if len(queued_keys) != len(candidates):
        raise ValueError("Duplicate prepared candidates")
    ledger = CandidateLedger(source, session.bank.digest, session.card.weights_digest)
    for row in candidates:
        ledger.add(row)
    for row in evidence:
        ledger.observe(row)
    reviews = []
    queue = [(row, False) for row in candidates]
    for index, (candidate, child) in enumerate(queue):
        if cancelled():
            return EngineResult(ledger, tuple(reviews), (), True)
        progress(index, len(queue), "local_identity_and_boundary")
        raw, stem = features(candidate)
        proposal = replace(candidate, parents=(*candidate.parents, candidate.key))
        review = session.review(proposal, raw, stem, evidence)
        ledger.add(review.candidate)
        indexes = []
        for item in review.evidence:
            key = ledger.observe(item)
            if item.span.intersection(review.candidate.output) is not None:
                indexes.append(key)
        # Persist local weak/other/target observations, not just one whole score.
        for frame in review.prediction.frames:
            span = frame.span.intersection(review.candidate.output)
            if span is None:
                continue
            calibration = session.calibration.identity
            if frame.other >= calibration.other_max:
                kind = EvidenceKind.OTHER
            elif frame.identity_observable and frame.target >= calibration.target_min:
                kind = EvidenceKind.TARGET
            else:
                kind = EvidenceKind.UNCERTAIN
            indexes.append(ledger.observe(LocalEvidence(source.source_sha256, source.sample_rate, span,
                                                        kind, session.bank.digest, session.card.weights_digest,
                                                        "candidate_local_prediction")))
        if not indexes:
            indexes.append(ledger.observe(LocalEvidence(source.source_sha256, source.sample_rate,
                                                        candidate.output, EvidenceKind.UNCERTAIN,
                                                        session.bank.digest, session.card.weights_digest,
                                                        "missing_temporal_evidence")))
        ledger.decide(Decision(review.candidate.key, review.assessment.state,
                               ";".join(review.assessment.reasons), tuple(dict.fromkeys(indexes)),
                               review.assessment.policy_version))
        reviews.append(review)
        if boundary_options is not None and not child:
            parts = propose_change_parts(
                review.candidate, review.prediction.frames, boundary_options(review),
                change_min=session.calibration.change_min,
                tolerance_samples=change_tolerance_samples,
            )
            for part in parts:
                if part.key not in queued_keys:
                    ledger.add(part)
                    queued_keys.add(part.key)
                    queue.append((part, True))
        if not child:
            for part in propose_unresolved_change_sides(
                review.candidate, review.prediction.frames,
                change_min=session.calibration.change_min,
            ):
                if part.key not in queued_keys:
                    ledger.add(part)
                    queued_keys.add(part.key)
                    queue.append((part, True))
    # Independently verified upstream events apply across all proposals. Model
    # observations are scoped to the candidate/context that produced them: a
    # mistaken parent prediction must not veto a newly reviewed child.
    hard = tuple(row for row in evidence if row.kind in (
        EvidenceKind.OTHER, EvidenceKind.SINGING, EvidenceKind.OVERLAP,
    ))
    resolved = []
    for review in reviews:
        combined = tuple(dict.fromkeys((*review.evidence, *hard)))
        assessment = assess(review.candidate, review.prediction, session.calibration.identity,
                            reference_digest=session.bank.digest, local_evidence=combined)
        if assessment != review.assessment:
            indexes = tuple(i for i, row in enumerate(ledger.evidence)
                            if row in hard and row.span.intersection(review.candidate.output))
            ledger.decide(Decision(review.candidate.key, assessment.state,
                                   ";".join(assessment.reasons), indexes, assessment.policy_version))
        resolved.append(replace(review, assessment=assessment, evidence=combined))
    reviews = resolved
    selected = select_verified(tuple(row.candidate for row in reviews),
                               tuple(row.assessment for row in reviews))
    blocked = tuple(row.span for row in evidence if row.kind in (
        EvidenceKind.OTHER, EvidenceKind.SINGING, EvidenceKind.OVERLAP,
    ))
    assessment = {row.candidate.key: row.assessment for row in reviews}
    for row in selected:
        final_audit(row, assessment[row.key], silence, blocked)
    progress(len(queue), len(queue), "acoustic_review_complete")
    return EngineResult(ledger, tuple(reviews), selected, False)
