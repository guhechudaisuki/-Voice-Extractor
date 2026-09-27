"""Conservative, reference-conditioned local identity conflict rules.

The rules produce an *abstain* signal, never a positive identity certificate.
They require independent evidence from the production verifier and the frozen
anime character/voice-actor encoders; a high parent score cannot clear a
short embedded speech island. No episode-specific labels or times are used.
"""

from __future__ import annotations

from collections.abc import Mapping


def _negative(value: object) -> bool:
    return value is not None and float(value) < 0


def _all_views_prefer_other(margins: Mapping[str, object]) -> bool:
    return all(_negative(margins.get(key)) for key in (
        "char_stem", "char_raw", "va_stem", "va_raw",
    ))


def local_other_witnesses(
    domain_margins: Mapping[str, object],
    production_exclusion: Mapping[str, object],
) -> tuple[str, ...]:
    """Return independent reasons to withhold a locally mixed parent.

    The production exclusion requires user-supplied negative examples. Missing
    or contradictory evidence returns no witness, not a target verdict.
    """
    reasons = []
    if _all_views_prefer_other(domain_margins):
        if _negative(production_exclusion.get("excluded_primary_margin")):
            reasons.append("domain_all_four_and_production_primary_negative")
        elif production_exclusion.get("excluded_role_rejected") is True:
            # The production exclusion decision combines multiple model and
            # channel votes. A single positive component margin does not
            # invalidate its explicit rejection of this local speech island.
            reasons.append("domain_all_four_and_production_exclusion_rejected")
    return tuple(reasons)


def whole_identity_conflict(
    domain_margins: Mapping[str, object],
    alternate_margins: Mapping[str, object],
    production_primary_direct_margin: float | None,
) -> bool:
    """Abstain on unstable character-stem sign plus a production near tie.

    0.02 is the existing production exclusion tie tolerance, not a tuned
    target-similarity threshold. The same query is decoded by two legitimate
    resampling paths; a changed sign is uncertainty, not another-person proof.
    """
    if production_primary_direct_margin is None:
        return False
    exact = domain_margins.get("char_stem")
    alternate = alternate_margins.get("char_stem")
    if exact is None or alternate is None:
        return False
    return (production_primary_direct_margin <= 0.02
            and _negative(exact) != _negative(alternate))


def eligible_identity_island(start: float, end: float,
                             *, sample_rate: int = 16000,
                             minimum_samples: int = 3200) -> bool:
    """Apply the encoder's minimum length in samples, not floating seconds."""
    return round(end * sample_rate) - round(start * sample_rate) >= minimum_samples


def target_runs(
    islands: list[dict],
    suspect_spans: set[tuple[float, float]],
    *,
    maximum_gap_seconds: float,
) -> list[tuple[float, float]]:
    """Never bridge a suspected/unknown speech island or a hard silence."""
    runs: list[tuple[float, float]] = []
    current: tuple[float, float] | None = None
    for island in islands:
        start, end = map(float, island["span"])
        if (start, end) in suspect_spans or island["state"] != "target_supported":
            if current is not None:
                runs.append(current)
                current = None
            continue
        if current is None:
            current = (start, end)
        elif start - current[1] <= maximum_gap_seconds:
            current = (current[0], end)
        else:
            runs.append(current)
            current = (start, end)
    if current is not None:
        runs.append(current)
    return runs
