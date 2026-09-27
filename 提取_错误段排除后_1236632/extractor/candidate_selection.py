"""Non-destructive recovery selection on a single source's 16 kHz timeline.

No score, subtitle or human label is consumed here. Callers provide scoped
acoustic findings; a high whole-span score cannot erase a local finding.
This selector is being evaluated offline before replacing production policy.
"""
from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Literal

from .types import CandidateSentence


IdentityState = Literal["target", "other", "unresolved"]
SAMPLE_RATE = 16000


def _interval(candidate: CandidateSentence) -> tuple[int, int]:
    if not all(math.isfinite(value) for value in (candidate.start, candidate.end)):
        raise ValueError("Candidate times must be finite")
    start, end = round(candidate.start * SAMPLE_RATE), round(candidate.end * SAMPLE_RATE)
    if start < 0 or end <= start:
        raise ValueError("Candidate must occupy a positive source interval")
    return start, end


@dataclass(frozen=True)
class LocalEvidence:
    start_sample: int
    end_sample: int
    state: IdentityState


def _covered(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    cursor = start
    for left, right in sorted(spans):
        if right <= cursor:
            continue
        if left > cursor:
            return False
        cursor = max(cursor, right)
        if cursor >= end:
            return True
    return False


class RecoverySelector:
    """One instance per source, not shared across files or reference profiles.

    ``observe`` accepts *local* findings, not arbitrary recovery proposals.
    An unresolved interval may be resolved by independent target findings
    wholly inside it. Adding target speech outside it is not new evidence
    about that interval. Explicit other-speaker findings remain vetoes.
    """

    def __init__(self) -> None:
        self._evidence: set[LocalEvidence] = set()
        self.decisions: list[dict] = []

    def observe(self, candidate: CandidateSentence, state: IdentityState) -> None:
        if state not in ("target", "other", "unresolved"):
            raise ValueError("Unknown local identity state")
        start, end = _interval(candidate)
        self._evidence.add(LocalEvidence(start, end, state))

    def conflicts(self, candidate: CandidateSentence) -> list[LocalEvidence]:
        start, end = _interval(candidate)
        conflicts = []
        for evidence in sorted(self._evidence, key=lambda item: (
            item.start_sample, item.end_sample, item.state,
        )):
            left, right = max(start, evidence.start_sample), min(end, evidence.end_sample)
            if left >= right or evidence.state == "target":
                continue
            if evidence.state == "unresolved":
                support = [
                    (part.start_sample, part.end_sample) for part in self._evidence
                    if part.state == "target"
                    and evidence.start_sample <= part.start_sample
                    and part.end_sample <= evidence.end_sample
                ]
                if _covered(left, right, support):
                    continue
            conflicts.append(evidence)
        return conflicts

    def install(
        self, candidate: CandidateSentence, accepted: list[CandidateSentence],
        rejected: list[CandidateSentence],
    ) -> bool:
        start, end = _interval(candidate)
        conflicts = self.conflicts(candidate)
        overlapping = [existing for existing in accepted if
                       min(end, _interval(existing)[1]) > max(start, _interval(existing)[0])]
        loses_existing = any(
            start > _interval(existing)[0] or end < _interval(existing)[1]
            for existing in overlapping
        )
        state = ("other" if any(part.state == "other" for part in conflicts)
                 else "unresolved" if conflicts else "uncontested")
        installed = not conflicts and not loses_existing
        decision = {
            "span_samples": [start, end], "state": state, "installed": installed,
            "reason": ("local_identity_conflict" if conflicts else
                       "would_truncate_existing" if loses_existing else "eligible"),
            "conflicts": [asdict(part) for part in conflicts],
            "previous_spans_samples": [list(_interval(part)) for part in overlapping],
        }
        self.decisions.append(decision)
        candidate.diagnostics["recovery_selection"] = deepcopy(decision)
        if not installed:
            # A deferred proposal is retained, not relabelled as a wrong person.
            # Leave already-selected objects intact when installation is retried.
            if not any(part is candidate for part in accepted):
                candidate.reject_reason = (
                    "局部证据存在冲突，候选保留待复核" if conflicts
                    else "替换会截短已验证讲话，保留原候选"
                )
                if not any(part is candidate for part in rejected):
                    rejected.append(candidate)
            return False
        removed = {id(part) for part in overlapping}
        accepted[:] = [part for part in accepted if id(part) not in removed]
        rejected[:] = [part for part in rejected if part is not candidate]
        candidate.reject_reason = ""
        accepted.append(candidate)
        return True

    def to_dict(self) -> dict:
        return {
            "sample_rate": SAMPLE_RATE,
            "evidence": [asdict(part) for part in sorted(self._evidence, key=lambda part: (
                part.start_sample, part.end_sample, part.state,
            ))],
            "decisions": deepcopy(self.decisions),
        }
