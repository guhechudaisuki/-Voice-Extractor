"""Cross-channel internal boundary hypotheses; never speaker decisions.

Two noisy local change detectors can agree on a time even for a prosody change.
These cuts only create *alternative* candidates. The original speech island
must remain available, and no cut may be used as an export/veto rule alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

from .types import TimeSpan


@dataclass(frozen=True)
class BoundaryProposal:
    time: float
    island_index: int
    stem_time: float
    raw_time: float | None
    timing_difference: float | None
    stem_confidence: float = 0.0
    raw_confidence: float | None = None
    support: str = "stem_and_raw"

    def to_dict(self) -> dict:
        return {
            "time": round(self.time, 5),
            "island_index": self.island_index,
            "stem_time": round(self.stem_time, 5),
            "raw_time": (
                round(self.raw_time, 5) if self.raw_time is not None else None
            ),
            "timing_difference": (
                round(self.timing_difference, 5)
                if self.timing_difference is not None else None
            ),
            "stem_confidence": round(self.stem_confidence, 5),
            "raw_confidence": (
                round(self.raw_confidence, 5)
                if self.raw_confidence is not None else None
            ),
            "support": self.support,
            "identity_state": "unresolved",
        }


def _deduplicate(
    observations: Iterable[float | Mapping[str, float]], tolerance: float,
) -> list[tuple[float, float]]:
    ordered: list[tuple[float, float]] = []
    for observation in observations:
        if isinstance(observation, Mapping):
            value = float(observation["time"])
            confidence = float(observation.get("confidence", 0.0))
        else:
            value = float(observation)
            confidence = 0.0
        if not math.isfinite(value) or not math.isfinite(confidence):
            raise ValueError("Non-finite boundary observation")
        ordered.append((value, confidence))
    ordered.sort()
    clusters: list[list[tuple[float, float]]] = []
    for observation in ordered:
        if clusters and observation[0] - clusters[-1][-1][0] <= tolerance:
            clusters[-1].append(observation)
        else:
            clusters.append([observation])
    return [max(cluster, key=lambda item: item[1]) for cluster in clusters]


def propose_cross_channel_boundaries(
    islands: Iterable[TimeSpan],
    stem_times: Iterable[float | Mapping[str, float]],
    raw_times: Iterable[float | Mapping[str, float]],
    *,
    tolerance_seconds: float = 0.15,
    minimum_side_seconds: float = 0.25,
    max_cuts_per_island: int = 4,
    include_unmatched_stem: bool = False,
) -> list[BoundaryProposal]:
    if not 0 < tolerance_seconds <= 0.5:
        raise ValueError("Invalid cross-channel timing tolerance")
    if minimum_side_seconds <= 0 or max_cuts_per_island < 1:
        raise ValueError("Invalid boundary geometry")
    ordered = sorted(islands, key=lambda item: (item.start, item.end))
    stem = _deduplicate(stem_times, tolerance_seconds * 0.5)
    raw = _deduplicate(raw_times, tolerance_seconds * 0.5)
    proposals: list[BoundaryProposal] = []
    for index, island in enumerate(ordered):
        if not 0 <= island.start < island.end:
            raise ValueError("Invalid speech island")
        local_stem = [
            observation for observation in stem
            if island.start + minimum_side_seconds <= observation[0]
            <= island.end - minimum_side_seconds
        ]
        local_raw = [
            observation for observation in raw
            if island.start + minimum_side_seconds <= observation[0]
            <= island.end - minimum_side_seconds
        ]
        pairs = sorted(
            (
                -(left[1] + right[1]),
                abs(left[0] - right[0]),
                left_index,
                right_index,
            )
            for left_index, left in enumerate(local_stem)
            for right_index, right in enumerate(local_raw)
            if abs(left[0] - right[0]) <= tolerance_seconds
        )
        candidates: list[BoundaryProposal] = []
        used_stem: set[int] = set()
        used_raw: set[int] = set()
        for _priority, distance, left_index, right_index in pairs:
            if left_index in used_stem or right_index in used_raw:
                continue
            left, left_confidence = local_stem[left_index]
            right, right_confidence = local_raw[right_index]
            center = round((left + right) / 2, 5)
            if not (
                island.start + minimum_side_seconds <= center
                <= island.end - minimum_side_seconds
            ):
                continue
            used_stem.add(left_index)
            used_raw.add(right_index)
            candidates.append(BoundaryProposal(
                time=center,
                island_index=index,
                stem_time=left,
                raw_time=right,
                timing_difference=distance,
                stem_confidence=left_confidence,
                raw_confidence=right_confidence,
            ))
        if include_unmatched_stem:
            candidates.extend(
                BoundaryProposal(
                    time=left,
                    island_index=index,
                    stem_time=left,
                    raw_time=None,
                    timing_difference=None,
                    stem_confidence=confidence,
                    support="stem_only",
                )
                for left_index, (left, confidence) in enumerate(local_stem)
                if left_index not in used_stem
            )
        # Cap alternatives, prioritizing two-channel support. All candidates
        # remain unresolved even if both models align on the same prosody dip.
        selected = sorted(
            candidates,
            key=lambda item: (
                item.support != "stem_and_raw",
                -item.stem_confidence,
                -(item.raw_confidence or 0.0),
                item.timing_difference or 0.0,
            ),
        )[:max_cuts_per_island]
        proposals.extend(
            selected
        )
    return sorted(proposals, key=lambda item: (item.island_index, item.time))


def subdivide_islands(
    islands: Iterable[TimeSpan], cuts: Iterable[BoundaryProposal],
) -> list[TimeSpan]:
    """Create an alternate partition; callers must retain original islands."""

    ordered = sorted(islands, key=lambda item: (item.start, item.end))
    by_island: dict[int, list[float]] = {}
    for cut in cuts:
        if not 0 <= cut.island_index < len(ordered):
            raise ValueError("Cut refers to an unknown island")
        source = ordered[cut.island_index]
        if not source.start < cut.time < source.end:
            raise ValueError("Cut is outside its source island")
        by_island.setdefault(cut.island_index, []).append(cut.time)
    result: list[TimeSpan] = []
    for index, island in enumerate(ordered):
        points = [island.start, *sorted(set(by_island.get(index, []))), island.end]
        result.extend(
            TimeSpan(left, right)
            for left, right in zip(points, points[1:])
            if right > left
        )
    return result
