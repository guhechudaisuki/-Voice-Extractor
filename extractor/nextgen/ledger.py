"""Append-only, source-scoped candidates, local evidence and decision history."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json

from .timeline import SampleSpan, SourceTimeline, require_digest


class State(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNRESOLVED = "unresolved"


class EvidenceKind(str, Enum):
    TARGET = "target"
    OTHER = "other"
    SINGING = "singing"
    OVERLAP = "overlap"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class Candidate:
    source_sha256: str
    sample_rate: int
    output: SampleSpan
    context: SampleSpan
    speech: tuple[SampleSpan, ...]
    origin: str
    parents: tuple[str, ...] = ()
    # Complete is acoustic boundary evidence, not punctuation or ASR fluency.
    start_complete: bool = False
    end_complete: bool = False

    def __post_init__(self) -> None:
        require_digest(self.source_sha256)
        if type(self.sample_rate) is not int or self.sample_rate <= 0:
            raise ValueError("Invalid candidate sample rate")
        if any(type(value) is not bool for value in (self.start_complete, self.end_complete)):
            raise ValueError("Boundary completeness must be explicit")
        if not self.origin or not self.context.contains(self.output) or not self.speech:
            raise ValueError("Candidate needs provenance, context and speech intervals")
        if any(not self.output.contains(part) for part in self.speech):
            raise ValueError("Speech must be contained in the output, not merely context")
        if tuple(sorted(self.speech)) != self.speech or any(
            left.end > right.start for left, right in zip(self.speech, self.speech[1:])
        ):
            raise ValueError("Speech intervals must be ordered and nonoverlapping")

    @property
    def span_key(self) -> str:
        return f"{self.source_sha256}:{self.sample_rate}:{self.output.start}:{self.output.end}"

    @property
    def key(self) -> str:
        metadata = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return f"{self.span_key}:{hashlib.sha256(metadata.encode()).hexdigest()[:16]}"


@dataclass(frozen=True)
class LocalEvidence:
    source_sha256: str
    sample_rate: int
    span: SampleSpan
    kind: EvidenceKind
    reference_digest: str
    model_digest: str
    method: str

    def __post_init__(self) -> None:
        for value in (self.source_sha256, self.reference_digest, self.model_digest):
            require_digest(value)
        if not isinstance(self.kind, EvidenceKind) or not self.method:
            raise ValueError("Evidence requires a typed kind and producer")
        if type(self.sample_rate) is not int or self.sample_rate <= 0:
            raise ValueError("Invalid evidence sample rate")


@dataclass(frozen=True)
class Decision:
    candidate_key: str
    state: State
    reason: str
    evidence_indexes: tuple[int, ...]
    policy_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.state, State) or not self.reason or not self.policy_version:
            raise ValueError("Decision requires state, reason and policy version")


class CandidateLedger:
    """Records observations and decisions; recording is not export authority."""
    def __init__(self, source: SourceTimeline, reference_digest: str, model_digest: str):
        require_digest(reference_digest)
        require_digest(model_digest)
        self.source = source
        self.reference_digest = reference_digest
        self.model_digest = model_digest
        self._candidates: dict[str, Candidate] = {}
        self._evidence: list[LocalEvidence] = []
        self._decisions: list[Decision] = []

    @property
    def candidates(self) -> tuple[Candidate, ...]:
        return tuple(self._candidates.values())

    @property
    def evidence(self) -> tuple[LocalEvidence, ...]:
        return tuple(self._evidence)

    @property
    def decisions(self) -> tuple[Decision, ...]:
        return tuple(self._decisions)

    def add(self, candidate: Candidate) -> str:
        if (candidate.source_sha256, candidate.sample_rate) != (
            self.source.source_sha256, self.source.sample_rate
        ):
            raise ValueError("Candidate belongs to a different source timeline")
        self.source.validate(candidate.context)
        if any(parent not in self._candidates for parent in candidate.parents):
            raise ValueError("Unknown candidate parent")
        previous = self._candidates.get(candidate.key)
        if previous is not None and previous != candidate:
            raise ValueError("Same interval has conflicting metadata; keep the original proposal")
        self._candidates[candidate.key] = candidate
        return candidate.key

    def observe(self, evidence: LocalEvidence) -> int:
        if (evidence.source_sha256, evidence.sample_rate, evidence.reference_digest,
            evidence.model_digest) != (self.source.source_sha256, self.source.sample_rate,
                                      self.reference_digest, self.model_digest):
            raise ValueError("Evidence belongs to a different source, reference set or model")
        self.source.validate(evidence.span)
        if evidence in self._evidence:
            return self._evidence.index(evidence)
        self._evidence.append(evidence)
        return len(self._evidence) - 1

    def decide(self, decision: Decision) -> None:
        if decision.candidate_key not in self._candidates:
            raise ValueError("Decision has no candidate")
        candidate = self._candidates[decision.candidate_key]
        if not decision.evidence_indexes:
            raise ValueError("A decision must cite acoustic evidence, not just a score")
        for index in decision.evidence_indexes:
            if type(index) is not int or not 0 <= index < len(self._evidence):
                raise ValueError("Invalid evidence index")
            if self._evidence[index].span.intersection(candidate.output) is None:
                raise ValueError("Cited evidence does not intersect the output")
        self._decisions.append(decision)

    def to_dict(self) -> dict:
        return {"schema": 1, "source": asdict(self.source),
                "reference_digest": self.reference_digest, "model_digest": self.model_digest,
                "candidates": [asdict(row) for row in self.candidates],
                "evidence": [asdict(row) for row in self.evidence],
                "decisions": [asdict(row) for row in self.decisions]}
