"""Identity operating-point selection on a declared calibration partition only.

Zero observed errors is a calibration constraint, NOT proof of zero future
errors. No hidden default, first-episode special case, or test-set fitting.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from extractor.nextgen.artifacts import InferenceCalibration
from extractor.nextgen.decision_policy import Calibration, SpanPrediction, assess
from extractor.nextgen.ledger import Candidate, LocalEvidence, State
from extractor.nextgen.timeline import require_digest
from training.nextgen_data import DataRecord, audit_records


@dataclass(frozen=True)
class CalibrationExample:
    candidate: Candidate
    prediction: SpanPrediction
    evidence: tuple[LocalEvidence, ...]
    complete_correct: bool
    partition: str
    source_group: str | None = None


_HEADS = ("observable", "singing", "overlap", "boundary", "speech", "change")


@dataclass(frozen=True)
class BinaryHeadExample:
    source_group: str
    source_sha256: str
    head: str
    score: float
    label: bool
    partition: str

    def __post_init__(self) -> None:
        require_digest(self.source_sha256)
        if (not self.source_group or self.head not in _HEADS
                or not math.isfinite(self.score) or not 0 <= self.score <= 1
                or type(self.label) is not bool or self.partition != "calibration"):
            raise ValueError("Invalid supervised calibration head example")


def choose_policy(examples: tuple[CalibrationExample, ...], policies: tuple[Calibration, ...],
                  *, minimum_correct: int) -> tuple[Calibration, dict]:
    if type(minimum_correct) is not int or minimum_correct < 1 or not examples or not policies:
        raise ValueError("Calibration needs both data and an explicit nonzero acceptance requirement")
    if any(row.partition != "calibration" or type(row.complete_correct) is not bool for row in examples):
        raise ValueError("Do not fit policy on training, development feedback or sealed tests")
    if not any(row.complete_correct for row in examples) or all(row.complete_correct for row in examples):
        raise ValueError("Calibration requires verified positive and negative candidates")
    if len({row.candidate.key for row in examples}) != len(examples):
        raise ValueError("Duplicate calibration candidates")
    scores = []
    for policy in policies:
        accepted = [row for row in examples if assess(
            row.candidate, row.prediction, policy, reference_digest=row.prediction.reference_digest,
            local_evidence=row.evidence,
        ).state == State.ACCEPTED]
        correct = sum(row.complete_correct for row in accepted)
        wrong = len(accepted) - correct
        scores.append({"version": policy.version, "correct": correct, "wrong": wrong})
    eligible = [(score["correct"], -i, policy) for i, (policy, score) in enumerate(zip(policies, scores))
                if score["wrong"] == 0 and score["correct"] >= minimum_correct]
    if not eligible:
        raise ValueError("No calibration policy meets both contamination and nonzero recall constraints")
    selected = max(eligible, key=lambda row: row[:2])[2]
    return selected, {"status": "calibration_only_not_release_validation", "scores": scores,
                      "selected": selected.version, "examples": len(examples)}


def fit_complete_calibration(
    examples: tuple[CalibrationExample, ...], policies: tuple[Calibration, ...],
    heads: tuple[BinaryHeadExample, ...], records: tuple[DataRecord, ...], *,
    minimum_correct: int,
) -> tuple[InferenceCalibration, dict]:
    """Fit every runtime decision threshold on an audited calibration split.

    This is an operating-point selector, not a deployable validated model.
    A head whose known positive and negative scores overlap cannot be made
    zero-error by choosing a threshold; the caller must improve data/model.
    """
    audit = audit_records(records, for_training=True)
    calibration_rows = {row.example_id: row for row in records if row.split == "calibration"}
    if not calibration_rows or any(not row.training_allowed or not row.human_labels
                                   for row in calibration_rows.values()):
        raise ValueError("Calibration selection needs authorized human-verified labels")
    if any(policy.data_digest != audit["dataset_digest"] for policy in policies):
        raise ValueError("Calibration policy does not match the audited dataset")
    if any(row.source_group not in calibration_rows
           or row.candidate.source_sha256 not in calibration_rows[row.source_group].content_digests
           for row in examples):
        raise ValueError("Candidate calibration provenance is incomplete or mismatched")
    if any(row.source_group not in calibration_rows
           or row.source_sha256 not in calibration_rows[row.source_group].content_digests
           for row in heads):
        raise ValueError("Head calibration provenance is incomplete or mismatched")
    selected, identity_report = choose_policy(examples, policies, minimum_correct=minimum_correct)
    thresholds = {}
    head_counts = {}
    for name in _HEADS:
        group = [row for row in heads if row.head == name]
        positive = [row.score for row in group if row.label]
        negative = [row.score for row in group if not row.label]
        if not positive or not negative:
            raise ValueError(f"Head {name} lacks positive or negative calibration labels")
        if max(negative) >= min(positive):
            raise ValueError(f"Head {name} cannot separate known calibration positives and negatives")
        thresholds[name] = (max(negative) + min(positive)) / 2
        head_counts[name] = {"positive": len(positive), "negative": len(negative)}
    calibration = InferenceCalibration(
        selected, thresholds["observable"], thresholds["singing"],
        thresholds["overlap"], thresholds["boundary"], thresholds["speech"],
        thresholds["change"],
    )
    return calibration, {"status": "calibration_only_not_release_validation",
                         "audit": audit, "identity": identity_report,
                         "heads": head_counts}
