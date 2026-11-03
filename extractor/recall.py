"""Recall policies shared by the acoustic candidate and export stages."""
from __future__ import annotations

from .types import TimeSpan

MIN_TARGET_SECONDS = .35


def subtract_intervals(spans, blocked):
    """Subtract a union of half-open intervals without rounding their edges."""
    barriers = sorted(blocked)
    result = []
    for start, end in spans:
        cursor = start
        for a, b in barriers:
            if b <= cursor:
                continue
            if a >= end:
                break
            if a > cursor:
                result.append((cursor, min(a, end)))
            cursor = max(cursor, b)
            if cursor >= end:
                break
        if cursor < end:
            result.append((cursor, end))
    return result


def certificate_matches(start, end, spans, minimum=MIN_TARGET_SECONDS):
    if end - start < minimum - 1e-9:
        return False
    return any(max(0., min(end, b) - max(start, a))
               / max(end - start, b - a, 1e-9) >= .75 for a, b in spans)


def recover_raw_vad(stem_spans, raw_spans, blocked=()):
    """Only add audio missing from stem VAD, never bridge existing islands."""
    pieces = subtract_intervals(((s.start, s.end) for s in raw_spans),
                               [(s.start, s.end) for s in [*stem_spans, *blocked]])
    recovered = [TimeSpan(a, b) for a, b in pieces if b - a >= MIN_TARGET_SECONDS - 1e-9]
    return sorted([*stem_spans, *recovered], key=lambda s: (s.start, s.end)), recovered


def annotate_raw_recovery(candidates, recovered):
    for candidate in candidates:
        spans = [[max(candidate.start, s.start), min(candidate.end, s.end)]
                 for s in recovered if min(candidate.end, s.end) > max(candidate.start, s.start)]
        if spans:
            candidate.diagnostics.update(raw_vad_recovered=True, raw_recovered_spans=spans,
                                         recovery_identity_source='raw', recovery_audio_source='stem')
