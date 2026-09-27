"""Find possible speech-internal pauses missed by a coarse VAD.

This is deliberately a proposal generator, not a voice/identity decision.
The original channel, local speaker evidence and complete boundaries must be
reviewed before a proposed pause changes any exported audio.
"""

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

    @property
    def length(self) -> int:
        return self.end - self.start


def locate_quiet_gaps(
    samples: np.ndarray,
    sample_rate: int,
    *,
    lower_seconds: float,
    upper_seconds: float,
    frame_seconds: float = 0.02,
    flank_seconds: float = 0.12,
) -> tuple[QuietGap, ...]:
    """Return medium/hard pause proposals with voiced flanks on both sides.

    A low-energy run is a candidate only if it lasts at least the user's
    lower silence bound and each side has local speech-like energy. It may
    still be UVR damage or a weak consonant, so no caller may export based on
    this result alone. ``hard_split`` marks a run above the upper bound.
    """
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
    quiet = levels <= quiet_floor
    minimum_samples = math.ceil(lower_seconds * sample_rate)
    upper_samples = round(upper_seconds * sample_rate)
    proposals: list[QuietGap] = []
    index = 0
    while index < frame_count:
        if not quiet[index]:
            index += 1
            continue
        first = index
        while index + 1 < frame_count and quiet[index + 1]:
            index += 1
        last = index + 1
        index += 1
        start, end = first * frame, last * frame
        if (end - start < minimum_samples or first < flank_frames
                or frame_count - last < flank_frames):
            continue
        left = levels[first - flank_frames:first]
        right = levels[last:last + flank_frames]
        left_level = float(np.quantile(left, 0.75))
        right_level = float(np.quantile(right, 0.75))
        gap_level = float(np.median(levels[first:last]))
        # Real word endings decay before silence.  A hard +10 dB flank floor
        # misses them even when the middle is tens of dB quieter; the local
        # flank-to-gap contrast is the actual evidence for a pause.
        if (min(left_level, right_level) < quiet_floor + 3.0
                or gap_level > min(left_level, right_level) - 12.0):
            continue
        proposals.append(QuietGap(
            start=start, end=end, hard_split=end - start > upper_samples,
            left_level_db=round(left_level, 3),
            gap_level_db=round(gap_level, 3),
            right_level_db=round(right_level, 3),
        ))
    return tuple(proposals)


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
    """Propose a final pause missed because its right speech flank is short.

    The ordinary gap finder correctly demands a 120 ms right flank to propose
    an internal split. At an export's right edge, a distinct syllable can be
    shorter than that and still contaminate the output. This function only
    reports a possible gap; it does not label the tail or authorize a cut.
    """
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
    # A long, loud prefix must not set the local noise floor for the final
    # syllable. Analyze level statistics only near the export's right edge.
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
    # A softly decaying word ending may lie only just above the adaptive
    # quiet floor. Require sustained non-quiet frames instead of a fixed
    # extra dB margin; the strong gap contrast is the pause evidence.
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
    )
