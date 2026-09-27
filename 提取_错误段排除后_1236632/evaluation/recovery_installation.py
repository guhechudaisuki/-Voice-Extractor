"""Experimental local-evidence guard; NOT used in production.

It blocks the reviewed mixed prefix but also blocks known correct recoveries.
Keep this as a regression/architecture probe, not a deployment policy.
"""
from __future__ import annotations

from extractor.types import CandidateSentence


def install_recovered_candidate(
    candidate: CandidateSentence,
    accepted: list[CandidateSentence],
    rejected: list[CandidateSentence],
) -> bool:
    # A local failure is not necessarily another speaker, but it is unresolved
    # audio. Whole-span similarity may not overrule it. Only rejected atomic
    # recovery parts count here; a rejected mixed parent can contain clean
    # target children and must not prevent their independent recovery.
    conflicts = [part for part in rejected
                 if part.reject_reason
                 and part.diagnostics.get("local_boundary_recovery")
                 and part.diagnostics.get("recovery_part_index") is not None
                 and min(candidate.end, part.end) - max(candidate.start, part.start) > 0.01]
    if conflicts:
        candidate.diagnostics["recovery_local_identity_conflicts"] = [
            [part.start, part.end] for part in conflicts
        ]
        candidate.reject_reason = "恢复候选跨越未通过局部身份的声音"
        if not any(part is candidate for part in rejected):
            rejected.append(candidate)
        return False
    overlapping = [existing for existing in accepted
                   if min(candidate.end, existing.end) - max(candidate.start, existing.start) > 0.10]
    if overlapping and not all(
        candidate.start <= existing.start + 0.25 and candidate.end >= existing.end - 0.25
        for existing in overlapping
    ):
        return False
    for existing in overlapping:
        accepted.remove(existing)
    accepted.append(candidate)
    return True
