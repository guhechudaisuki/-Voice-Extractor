"""Propose speech-internal pauses for subsequent identity and boundary review."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class QuietGap:
    start: int
    end: int
    hard_split: bool
    left_level_db: float
    gap_level_db: float
    right_level_db: float
    spectral_change: float | None = None
    crossing_change: float | None = None
    transition_evidence: float | None = None
    pause_kind: str = "unresolved"

    @property
    def length(self) -> int:
        return self.end - self.start


def _flank_cues(left: np.ndarray, right: np.ndarray) -> dict:
    """Compare a boundary change with ordinary variation within its two flanks.

    Spectral shape is amplitude-independent. Zero crossings provide a second
    view of voicing; neither measurement identifies a speaker on its own.
    """
    epsilon = np.finfo(np.float64).eps
    frame_size = left.shape[1]
    taper = np.hanning(frame_size) if frame_size > 2 else np.ones(frame_size)

    def features(frames):
        spectrum = np.abs(np.fft.rfft(frames * taper, axis=1)) ** 2
        mass = spectrum.sum(axis=1, keepdims=True)
        spectrum = np.sqrt(np.divide(spectrum, mass, out=np.zeros_like(spectrum),
                                     where=mass > epsilon))
        crossings = (np.mean(np.signbit(frames[:, 1:]) != np.signbit(frames[:, :-1]),
                             axis=1) if frame_size > 1 else np.zeros(len(frames)))
        return spectrum, crossings

    left_spectrum, left_crossings = features(left)
    right_spectrum, right_crossings = features(right)
    left_center, right_center = left_spectrum.mean(axis=0), right_spectrum.mean(axis=0)
    spectral_change = float(np.linalg.norm(left_center - right_center))
    spectral_variance = float(
        np.mean(np.sum((left_spectrum - left_center) ** 2, axis=1))
        + np.mean(np.sum((right_spectrum - right_center) ** 2, axis=1)))
    crossing_change = float(abs(left_crossings.mean() - right_crossings.mean()))
    crossing_variance = float(left_crossings.var() + right_crossings.var())
    # A crossing count has one-sample resolution even on a perfectly steady tone.
    crossing_resolution = 1. / max(1, frame_size - 1)
    spectral_ratio = spectral_change / math.sqrt(max(spectral_variance, epsilon))
    crossing_ratio = crossing_change / math.sqrt(
        crossing_variance + crossing_resolution ** 2)
    transition = min(spectral_ratio, crossing_ratio) - 1.
    continuous = max(spectral_ratio, crossing_ratio) <= 1.
    return {
        "spectral_change": round(spectral_change, 6),
        "crossing_change": round(crossing_change, 6),
        "transition_evidence": round(transition, 6),
        "pause_kind": ("handoff_candidate" if transition > 0.
                       else "continuous_spectrum" if continuous else "unresolved"),
    }


def locate_quiet_gaps(
    samples: np.ndarray,
    sample_rate: int,
    *,
    lower_seconds: float,
    upper_seconds: float,
    frame_seconds: float = 0.02,
    flank_seconds: float = 0.12,
) -> tuple[QuietGap, ...]:
    """Find low-energy runs with voiced flanks; hard_split marks the upper silence bound."""
    if (type(sample_rate) is not int or sample_rate <= 0
            or not all(math.isfinite(value) for value in (
                lower_seconds, upper_seconds, frame_seconds, flank_seconds,
            ))
            or lower_seconds < 0 or upper_seconds < lower_seconds
            or frame_seconds <= 0 or flank_seconds <= 0):
        raise ValueError("Invalid sample rate or silence range")
    waveform = np.asarray(samples, dtype=np.float32)
    if waveform.ndim != 1 or not np.isfinite(waveform).all():
        raise ValueError("Expected finite mono audio")
    frame = max(1, round(frame_seconds * sample_rate))
    flank_frames = max(1, math.ceil(flank_seconds * sample_rate / frame))
    frame_count = len(waveform) // frame
    if frame_count < flank_frames * 2 + 2:
        return ()
    frames = waveform[:frame_count * frame].reshape(frame_count, frame)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    levels = 20.0 * np.log10(rms + 1e-10)
    speech_level = float(np.quantile(levels, 0.75))
    quiet_floor = min(-46.0, speech_level - 15.0)
    minimum_samples = math.ceil(lower_seconds * sample_rate)
    upper_samples = round(upper_seconds * sample_rate)

    def runs(floor, *, median_drop, minimum_drop=None, require_handoff=False):
        quiet = levels <= floor
        transitions = np.diff(np.pad(quiet.astype(np.int8), (1, 1)))
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1)
        for first, last in zip(starts, ends):
            start, end = int(first) * frame, int(last) * frame
            if (end - start < minimum_samples or first < flank_frames
                    or frame_count - last < flank_frames):
                continue
            left_level = float(np.quantile(levels[first - flank_frames:first], 0.75))
            right_level = float(np.quantile(levels[last:last + flank_frames], 0.75))
            flank_level = min(left_level, right_level)
            gap_levels = levels[first:last]
            gap_level = float(np.median(gap_levels))
            if minimum_drop is None:
                # Absolute silence still needs two voiced flanks.
                if flank_level < floor + 3.0:
                    continue
            elif float(gap_levels.min()) > flank_level - minimum_drop:
                continue
            if gap_level > flank_level - median_drop:
                continue
            cues = _flank_cues(frames[first - flank_frames:first],
                               frames[last:last + flank_frames])
            if require_handoff and cues["pause_kind"] != "handoff_candidate":
                continue
            yield QuietGap(
                start=start, end=end, hard_split=end - start > upper_samples,
                left_level_db=round(left_level, 3),
                gap_level_db=round(gap_level, 3),
                right_level_db=round(right_level, 3),
                **cues,
            )

    proposals = list(runs(quiet_floor, median_drop=12.0))
    valley_floor = min(-38.0, speech_level - 11.0)
    for valley in runs(valley_floor, median_drop=9.5, minimum_drop=11.5):
        # Retain the narrower verified silence if both detectors find the same pause.
        if not any(min(valley.end, gap.end) > max(valley.start, gap.start)
                   for gap in proposals):
            proposals.append(valley)
    # With a high background floor, require both acoustic views to corroborate
    # a relative energy valley. Existing silence boundaries remain unchanged.
    if speech_level - 11.0 > valley_floor:
        for valley in runs(speech_level - 11.0, median_drop=9.5,
                           minimum_drop=11.5, require_handoff=True):
            if not any(min(valley.end, gap.end) > max(valley.start, gap.start)
                       for gap in proposals):
                proposals.append(valley)
    return tuple(sorted(proposals, key=lambda gap: (gap.start, gap.end)))


def locate_terminal_short_tail_gap(
    samples: np.ndarray,
    sample_rate: int,
    *,
    lower_seconds: float,
    upper_seconds: float,
    frame_seconds: float = 0.02,
    flank_seconds: float = 0.12,
    minimum_tail_seconds: float = 0.04,
    analysis_seconds: float = 4.0,
) -> QuietGap | None:
    """Find terminal pause candidates whose right speech flank is shorter than 120 ms."""
    if (type(sample_rate) is not int or sample_rate <= 0
            or not all(math.isfinite(value) for value in (
                lower_seconds, upper_seconds, frame_seconds, flank_seconds,
                minimum_tail_seconds, analysis_seconds,
            ))
            or lower_seconds < 0 or upper_seconds < lower_seconds
            or frame_seconds <= 0 or flank_seconds <= minimum_tail_seconds
            or minimum_tail_seconds <= 0 or analysis_seconds <= 0):
        raise ValueError("Invalid terminal pause parameters")
    waveform = np.asarray(samples, dtype=np.float32)
    if waveform.ndim != 1 or not np.isfinite(waveform).all():
        raise ValueError("Expected finite mono audio")
    frame = max(1, round(frame_seconds * sample_rate))
    count = len(waveform) // frame
    left_frames = max(1, math.ceil(flank_seconds * sample_rate / frame))
    minimum_tail_frames = max(1, math.ceil(minimum_tail_seconds * sample_rate / frame))
    maximum_tail_frames = max(1, math.ceil(flank_seconds * sample_rate / frame))
    if count < left_frames + minimum_tail_frames + 2:
        return None
    frames = waveform[:count * frame].reshape(count, frame)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    levels = 20.0 * np.log10(rms + 1e-10)
    # Estimate the terminal noise floor locally to avoid a loud prefix bias.
    local_frames = max(1, round(analysis_seconds * sample_rate / frame))
    quiet_floor = min(-46.0, float(np.quantile(levels[-local_frames:], 0.75)) - 15.0)
    quiet = levels <= quiet_floor

    # Permit at most one quiet frame of export padding after the final sound.
    last_voiced = count - 1
    while last_voiced >= 0 and quiet[last_voiced]:
        last_voiced -= 1
    if last_voiced < count - 2:
        return None
    first_tail = last_voiced
    while first_tail > 0 and not quiet[first_tail - 1]:
        first_tail -= 1
    tail_frames = last_voiced + 1 - first_tail
    if not minimum_tail_frames <= tail_frames < maximum_tail_frames:
        return None

    gap_end = first_tail
    gap_start = gap_end
    while gap_start > 0 and quiet[gap_start - 1]:
        gap_start -= 1
    if ((gap_end - gap_start) * frame < math.ceil(lower_seconds * sample_rate)
            or gap_start < left_frames):
        return None
    left_level = float(np.quantile(levels[gap_start - left_frames:gap_start], 0.75))
    right_level = float(np.quantile(levels[first_tail:last_voiced + 1], 0.75))
    gap_level = float(np.median(levels[gap_start:gap_end]))
    # Sustained non-quiet frames retain softly decaying endings.
    left_active = int((~quiet[gap_start - left_frames:gap_start]).sum())
    if (left_active < math.ceil(left_frames / 2)
            or right_level <= quiet_floor
            or gap_level > min(left_level, right_level) - 12.0):
        return None
    start, end = gap_start * frame, gap_end * frame
    return QuietGap(
        start=start,
        end=end,
        hard_split=end - start > round(upper_seconds * sample_rate),
        left_level_db=round(left_level, 3),
        gap_level_db=round(gap_level, 3),
        right_level_db=round(right_level, 3),
        **_flank_cues(frames[gap_start - left_frames:gap_start],
                      frames[first_tail:last_voiced + 1]),
    )
