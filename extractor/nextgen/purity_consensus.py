"""Combine independent local identity conflicts into abstention decisions."""

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
    """Return local conflict witnesses; missing evidence leaves identity unresolved."""
    reasons = []
    if _all_views_prefer_other(domain_margins):
        if _negative(production_exclusion.get("excluded_primary_margin")):
            reasons.append("domain_all_four_and_production_primary_negative")
        elif production_exclusion.get("excluded_role_rejected") is True:
            # A single positive component cannot override a joint exclusion verdict.
            reasons.append("domain_all_four_and_production_exclusion_rejected")
    return tuple(reasons)


def whole_identity_conflict(
    domain_margins: Mapping[str, object],
    alternate_margins: Mapping[str, object],
    production_primary_direct_margin: float | None,
) -> bool:
    """Abstain on an unstable character score near the production exclusion tie."""
    if production_primary_direct_margin is None:
        return False
    exact = domain_margins.get("char_stem")
    alternate = alternate_margins.get("char_stem")
    va_stem = domain_margins.get("va_stem")
    if exact is None or alternate is None or va_stem is None:
        return False
    return (production_primary_direct_margin <= 0.01
            and float(va_stem) <= 0.02
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
