"""Non-destructive utterance proposals on the 16 kHz analysis timeline.

The lattice deliberately does not assign a speaker or authorize export. Each
atomic speech island remains available even if a longer proposal is created.
Subtitle timing can annotate proposals but never creates a voiced boundary.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable

from .types import TimeSpan


ANALYSIS_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class UtteranceProposal:
    source_id: str
    start_sample: int
    end_sample: int
    island_indexes: tuple[int, ...]
    speech_intervals: tuple[tuple[int, int], ...]
    gap_intervals: tuple[tuple[int, int], ...]
    subtitle_cue_indexes: tuple[int, ...] = ()
    identity_state: str = "unresolved"

    @property
    def span(self) -> TimeSpan:
        return TimeSpan(
            self.start_sample / ANALYSIS_SAMPLE_RATE,
            self.end_sample / ANALYSIS_SAMPLE_RATE,
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _samples(span: TimeSpan) -> tuple[int, int]:
    start = round(span.start * ANALYSIS_SAMPLE_RATE)
    end = round(span.end * ANALYSIS_SAMPLE_RATE)
    if start < 0 or end <= start:
        raise ValueError("Speech and evidence spans must have positive duration")
    return start, end


def build_utterance_lattice(
    source_sha256: str,
    islands: Iterable[TimeSpan],
    *,
    max_gap_seconds: float,
    max_utterance_seconds: float,
    blocked: Iterable[TimeSpan] = (),
    subtitle_cues: Iterable[tuple[int, TimeSpan]] = (),
    max_islands_per_proposal: int = 6,
) -> list[UtteranceProposal]:
    """Enumerate contiguous island groups without deleting any atomic island.

    A blocked event in a gap forbids crossing it. A group is merely a proposal:
    the combined waveform may still contain multiple speakers inside one island
    and must receive a separate local identity/contamination audit.
    """

    if re.fullmatch(r"[0-9a-fA-F]{64}", source_sha256) is None:
        raise ValueError("source_sha256 must be a complete SHA-256 digest")
    if max_gap_seconds < 0 or max_utterance_seconds <= 0:
        raise ValueError("Invalid gap or maximum utterance duration")
    if max_islands_per_proposal < 1:
        raise ValueError("max_islands_per_proposal must be positive")
    atoms = sorted((_samples(span) for span in islands), key=lambda row: row)
    evidence = sorted((_samples(span) for span in blocked), key=lambda row: row)
    cues = sorted((int(index), _samples(span)) for index, span in subtitle_cues)
    if len({index for index, _span in cues}) != len(cues):
        raise ValueError("Subtitle cue indexes must be unique")
    max_gap = round(max_gap_seconds * ANALYSIS_SAMPLE_RATE)
    max_duration = round(max_utterance_seconds * ANALYSIS_SAMPLE_RATE)
    proposals: list[UtteranceProposal] = []
    for first in range(len(atoms)):
        speech: list[tuple[int, int]] = []
        gaps: list[tuple[int, int]] = []
        for last in range(first, min(len(atoms), first + max_islands_per_proposal)):
            start, end = atoms[last]
            if last > first:
                previous_end = atoms[last - 1][1]
                if start < previous_end or start - previous_end > max_gap:
                    break
                if any(
                    min(start, blocked_end) > max(previous_end, blocked_start)
                    for blocked_start, blocked_end in evidence
                ):
                    break
                if start > previous_end:
                    gaps.append((previous_end, start))
            # The duration cap limits *joins*, not an indivisible VAD island.
            # Dropping an overlong atom here makes every utterance inside it
            # unreachable before identity or boundary review can split it.
            if last > first and end - atoms[first][0] > max_duration:
                break
            speech.append((start, end))
            cue_indexes = tuple(
                index
                for index, (cue_start, cue_end) in cues
                if min(end, cue_end) > max(atoms[first][0], cue_start)
            )
            proposals.append(
                UtteranceProposal(
                    source_id=(
                        f"{source_sha256.lower()}:{ANALYSIS_SAMPLE_RATE}:"
                        f"{atoms[first][0]}:{end}"
                    ),
                    start_sample=atoms[first][0],
                    end_sample=end,
                    island_indexes=tuple(range(first, last + 1)),
                    speech_intervals=tuple(speech),
                    gap_intervals=tuple(gaps),
                    subtitle_cue_indexes=cue_indexes,
                )
            )
    return proposals


def propose_raw_vad_subturns(
    stem_turns: Iterable[TimeSpan],
    raw_speech: Iterable[TimeSpan],
    target_regions: Iterable[TimeSpan],
    *,
    minimum_gap_seconds: float,
    minimum_candidate_seconds: float,
    edge_snap_seconds: float = 0.12,
) -> list[TimeSpan]:
    """Propose raw-channel subturns hidden inside a continuous UVR VAD turn.

    Original-channel VAD can find an inter-speaker pause obscured by the UVR
    stem. It is only a candidate source: its gaps are *not* certified silence,
    and callers must still perform exact-span identity, exclusion, overlap,
    boundary and completeness review before exporting anything. Locator overlap
    restricts work, but never lends its identity to the candidate's extra audio.
    """

    if minimum_gap_seconds <= 0 or minimum_candidate_seconds <= 0:
        raise ValueError("Raw VAD proposal durations must be positive")
    if not 0 <= edge_snap_seconds <= 0.25:
        raise ValueError("Invalid raw VAD edge snap distance")
    turns = sorted(stem_turns, key=lambda span: (span.start, span.end))
    raw = sorted(raw_speech, key=lambda span: (span.start, span.end))
    targets = sorted(target_regions, key=lambda span: (span.start, span.end))
    for span in [*turns, *raw, *targets]:
        if not 0 <= span.start < span.end:
            raise ValueError("Invalid raw/stem VAD interval")
    proposals: dict[tuple[float, float], TimeSpan] = {}
    for turn in turns:
        clipped = [
            TimeSpan(max(turn.start, span.start), min(turn.end, span.end))
            for span in raw
            if min(turn.end, span.end) > max(turn.start, span.start)
        ]
        groups: list[TimeSpan] = []
        for span in clipped:
            if groups and span.start - groups[-1].end < minimum_gap_seconds:
                groups[-1] = TimeSpan(groups[-1].start, max(groups[-1].end, span.end))
            else:
                groups.append(span)
        # A single raw interval merely differs in edge placement; without an
        # internal gap it cannot explain a multi-speaker continuous stem turn.
        if len(groups) < 2:
            continue
        for group in groups:
            if group.duration < minimum_candidate_seconds:
                continue
            if not any(
                min(group.end, target.end) - max(group.start, target.start)
                >= min(0.35, group.duration * 0.25)
                for target in targets
            ):
                continue
            start = (turn.start if group.start - turn.start <= edge_snap_seconds
                     else group.start)
            end = (turn.end if turn.end - group.end <= edge_snap_seconds
                   else group.end)
            if end - start < minimum_candidate_seconds:
                continue
            candidate = TimeSpan(start, end)
            if candidate == turn:
                continue
            proposals[(round(start, 5), round(end, 5))] = candidate
    return sorted(proposals.values(), key=lambda span: (span.start, span.end))
