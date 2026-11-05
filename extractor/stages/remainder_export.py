"""Plan all unselected clean speech for review without applying identity gates."""
from __future__ import annotations

import math

from ..types import TimeSpan


def _sample_intervals(spans, sample_rate):
    intervals = []
    for span in spans:
        start, end = (span.start, span.end) if hasattr(span, 'start') else span
        if not all(math.isfinite(v) for v in (start, end)) or not 0 <= start < end:
            raise ValueError('Expected a finite, positive, nonnegative audio interval')
        start, end = round(start * sample_rate), round(end * sample_rate)
        # A positive sub-sample proposal contains no exportable samples.
        if end > start:
            intervals.append((start, end))
    return sorted(set(intervals))


def remaining_spans(clean_spans, selected_spans, blocked=(), sample_rate=16000):
    """Subtract selected/blocked unions, preserving every remaining sample.

    Clean proposals retain their sentence boundaries when adjacent. When they
    overlap, the earlier proposal owns the shared samples; a later proposal
    contributes only its uncovered tail. Invalid geometry raises rather than
    silently dropping an exclusion mask. No target scores are consulted.
    """
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError('Sample rate must be a positive integer')
    clean = _sample_intervals(clean_spans, sample_rate)
    excluded = _sample_intervals(selected_spans, sample_rate)
    excluded.extend(_sample_intervals(blocked, sample_rate))
    masks = []
    for start, end in sorted(excluded):
        if masks and start <= masks[-1][1]:
            masks[-1] = (masks[-1][0], max(end, masks[-1][1]))
        else:
            masks.append((start, end))

    remaining = []
    covered_until = 0
    mask_index = 0
    for start, end in clean:
        start = max(start, covered_until)
        covered_until = max(covered_until, end)
        if end <= start:
            continue
        while mask_index < len(masks) and masks[mask_index][1] <= start:
            mask_index += 1
        cursor = start
        index = mask_index
        while index < len(masks) and masks[index][0] < end:
            left, right = masks[index]
            if left > cursor:
                remaining.append(TimeSpan(cursor / sample_rate, left / sample_rate))
            cursor = max(cursor, right)
            if cursor >= end:
                break
            index += 1
        if cursor < end:
            remaining.append(TimeSpan(cursor / sample_rate, end / sample_rate))
    return remaining
