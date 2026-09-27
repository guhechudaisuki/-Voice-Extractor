"""Silence-constrained candidate generation and nonoverlapping final selection."""
from __future__ import annotations

import bisect
from dataclasses import dataclass
import math

from .decision_policy import Assessment, FramePrediction
from .ledger import Candidate, State
from .timeline import SampleSpan, SourceTimeline, union_length


@dataclass(frozen=True)
class SilenceRange:
    lower_samples: int
    upper_samples: int

    def __post_init__(self) -> None:
        if any(type(value) is not int for value in (self.lower_samples, self.upper_samples)):
            raise ValueError("Silence range must use integer samples")
        if not 0 <= self.lower_samples <= self.upper_samples:
            raise ValueError("Invalid silence range")

    def kind(self, gap: int) -> str:
        if type(gap) is not int or gap < 0:
            raise ValueError("Speech islands overlap")
        if gap < self.lower_samples:
            return "inside_utterance"
        return "identity_join" if gap <= self.upper_samples else "hard_split"


@dataclass(frozen=True)
class SpeechIsland:
    span: SampleSpan
    start_complete: bool
    end_complete: bool

    def __post_init__(self) -> None:
        if any(type(value) is not bool for value in (self.start_complete, self.end_complete)):
            raise ValueError("Boundary completeness must be explicit")


@dataclass(frozen=True)
class AcousticGap:
    """A real pause located by an independent waveform/VAD boundary check.

    The change head only requests an inspection here. A caller must explicitly
    certify both endpoints; a nearby low energy sample alone cannot do that.
    """
    span: SampleSpan
    left_complete: bool
    right_complete: bool
    verified: bool
    method: str

    def __post_init__(self) -> None:
        if any(type(flag) is not bool for flag in (
            self.left_complete, self.right_complete, self.verified,
        )) or not self.method:
            raise ValueError("Acoustic gap requires explicit provenance and boundary states")


def propose_change_parts(
    parent: Candidate, frames: tuple[FramePrediction, ...], gaps: tuple[AcousticGap, ...],
    *, change_min: float, tolerance_samples: int,
) -> tuple[Candidate, ...]:
    """Keep the parent and propose independently reviewable speech parts.

    The frame score does not grant a sample-exact cut. Only a verified local
    pause with independently complete neighboring speech edges can be used.
    Fragments remain alternatives until the new exact spans are reviewed.
    """
    if (not math.isfinite(change_min) or not 0 <= change_min <= 1
            or type(tolerance_samples) is not int or tolerance_samples < 0):
        raise ValueError("Invalid calibrated change rule")
    if any(a.span.end > b.span.start for a, b in zip(frames, frames[1:])):
        raise ValueError("Change frames must be ordered")
    changes = [frame.span for frame in frames if frame.change >= change_min
               and frame.span.intersection(parent.output) is not None]
    usable = []
    for gap in sorted(gaps, key=lambda row: row.span):
        if (not parent.output.contains(gap.span) or gap.span.start <= parent.output.start
                or gap.span.end >= parent.output.end):
            raise ValueError("Gap is outside parent or touches its endpoint")
        if usable and usable[-1].span.end > gap.span.start:
            raise ValueError("Acoustic gaps overlap")
        if not gap.verified or not (gap.left_complete and gap.right_complete):
            continue
        distance = min((max(0, change.start - gap.span.end, gap.span.start - change.end)
                        for change in changes), default=None)
        if distance is not None and distance <= tolerance_samples:
            usable.append(gap)
    if not usable:
        return ()
    bounds = [parent.output.start, *[edge for gap in usable
                                    for edge in (gap.span.start, gap.span.end)], parent.output.end]
    result = []
    for index in range(len(usable) + 1):
        span = SampleSpan(bounds[index * 2], bounds[index * 2 + 1])
        speech = tuple(part for original in parent.speech
                       if (part := original.intersection(span)) is not None)
        if not speech:
            continue
        result.append(Candidate(
            parent.source_sha256, parent.sample_rate, span, parent.context, speech,
            "change_with_verified_acoustic_gap", (parent.key,),
            parent.start_complete if index == 0 else usable[index - 1].right_complete,
            parent.end_complete if index == len(usable) else usable[index].left_complete,
        ))
    return tuple(result)


def propose_unresolved_change_sides(
    parent: Candidate, frames: tuple[FramePrediction, ...], *, change_min: float,
) -> tuple[Candidate, ...]:
    """Preserve both identities around a continuous, unverified voice change.

    The high-change cells remain an unknown band. These proposals deliberately
    have incomplete inner endpoints, so they can gather local evidence but can
    never be exported until a separate boundary refinement verifies the cut.
    """
    if not math.isfinite(change_min) or not 0 <= change_min <= 1:
        raise ValueError("Invalid calibrated change rule")
    if any(a.span.end > b.span.start for a, b in zip(frames, frames[1:])):
        raise ValueError("Change frames must be ordered")
    active = [part for frame in frames if frame.change >= change_min
              if (part := frame.span.intersection(parent.output)) is not None]
    if not active:
        return ()
    bands: list[SampleSpan] = []
    for part in active:
        if bands and part.start <= bands[-1].end:
            bands[-1] = SampleSpan(bands[-1].start, max(bands[-1].end, part.end))
        else:
            bands.append(part)
    boundaries = [parent.output.start, *[edge for band in bands
                                         for edge in (band.start, band.end)], parent.output.end]
    result = []
    for index in range(len(bands) + 1):
        start, end = boundaries[2 * index:2 * index + 2]
        if start >= end:
            continue
        output = SampleSpan(start, end)
        speech = tuple(part for original in parent.speech
                       if (part := original.intersection(output)) is not None)
        if speech:
            result.append(Candidate(
                parent.source_sha256, parent.sample_rate, output, parent.context, speech,
                "unresolved_continuous_change", (parent.key,),
                parent.start_complete if index == 0 else False,
                parent.end_complete if index == len(bands) else False,
            ))
    return tuple(result)


def propose(
    source: SourceTimeline, islands: tuple[SpeechIsland, ...], silence: SilenceRange,
    *, maximum_samples: int, context_samples: int = 0,
    blocked: tuple[SampleSpan, ...] = (),
) -> tuple[Candidate, ...]:
    if (type(maximum_samples) is not int or type(context_samples) is not int
            or maximum_samples <= 0 or context_samples < 0):
        raise ValueError("Invalid candidate limits")
    if any(left.span.end > right.span.start for left, right in zip(islands, islands[1:])):
        raise ValueError("Speech islands must be ordered and nonoverlapping")
    for island in islands:
        source.validate(island.span)
    for mask in blocked:
        source.validate(mask)
    proposals = []
    for first, left in enumerate(islands):
        for last in range(first, len(islands)):
            right = islands[last]
            if last > first and silence.kind(right.span.start - islands[last - 1].span.end) == "hard_split":
                break
            span = SampleSpan(left.span.start, right.span.end)
            # Limit combinatorial multi-island joins, not a single intact VAD
            # island: silently dropping a long utterance loses all recall and
            # cannot be repaired by later boundary or identity review.
            if span.length > maximum_samples and last > first:
                break
            if any(span.intersection(mask) for mask in blocked):
                break
            # Tiny gaps do not authorize partial-utterance exports. Those
            # atomic alternatives remain available for further boundary work.
            valid_start = first == 0 or silence.kind(left.span.start - islands[first - 1].span.end) != "inside_utterance"
            valid_end = last == len(islands) - 1 or silence.kind(islands[last + 1].span.start - right.span.end) != "inside_utterance"
            proposals.append(Candidate(
                source.source_sha256, source.sample_rate, span,
                SampleSpan(max(0, span.start - context_samples),
                           min(source.total_samples, span.end + context_samples)),
                tuple(item.span for item in islands[first:last + 1]), "silence",
                start_complete=left.start_complete and valid_start,
                end_complete=right.end_complete and valid_end,
            ))
    return tuple(proposals)


def select_verified(
    candidates: tuple[Candidate, ...], assessments: tuple[Assessment, ...],
) -> tuple[Candidate, ...]:
    """Maximize unique verified speech samples; silence is never a reward.

    Weighted interval scheduling avoids greedy long candidates deleting two
    better complete alternatives. Ties prefer fewer files, then stable keys.
    """
    if len({row.key for row in candidates}) != len(candidates):
        raise ValueError("Duplicate candidate keys")
    if len({(row.source_sha256, row.sample_rate) for row in candidates}) > 1:
        raise ValueError("Decode one source timeline at a time")
    if len({row.candidate_key for row in assessments}) != len(assessments):
        raise ValueError("Select with one final assessment per candidate")
    verdict = {row.candidate_key: row.state for row in assessments}
    if set(verdict) - {row.key for row in candidates}:
        raise ValueError("Assessment names an unknown candidate")
    if any(verdict.get(row.key) == State.ACCEPTED and
           not (row.start_complete and row.end_complete) for row in candidates):
        raise ValueError("An incomplete candidate cannot be marked verified")
    eligible = sorted((row for row in candidates if verdict.get(row.key) == State.ACCEPTED),
                      key=lambda row: (row.output.end, row.output.start, row.key))
    ends = [row.output.end for row in eligible]
    best: list[tuple[int, tuple[int, ...]]] = [(0, ())]
    for index, row in enumerate(eligible):
        previous = bisect.bisect_right(ends, row.output.start, hi=index)
        score, chosen = best[previous]
        take = (score + union_length(list(row.speech)), (*chosen, index))
        skip = best[-1]
        def key(choice):
            return choice[0], -len(choice[1])
        best.append(take if key(take) > key(skip) else skip)
    return tuple(eligible[index] for index in best[-1][1])
