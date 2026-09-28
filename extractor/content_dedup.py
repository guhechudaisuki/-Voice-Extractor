"""Remove contained copies without changing any retained sentence boundary."""

from __future__ import annotations

import numpy as np

from .types import CandidateSentence


def contained_audio_match(query: np.ndarray, longer: np.ndarray) -> tuple[float, int]:
    """Return the strongest full-query Pearson correlation and sample offset.

    Center each sliding window independently: global normalization of the
    longer clip incorrectly penalizes copies with a different gain/DC level.
    Silence and invalid samples provide no evidence of repeated content.
    """
    query = np.asarray(query, dtype=np.float64)
    longer = np.asarray(longer, dtype=np.float64)
    if query.size == 0 or query.size > longer.size:
        return 0.0, 0
    if not np.isfinite(query).all() or not np.isfinite(longer).all():
        return 0.0, 0
    query = query - query.mean()
    energy = float(query @ query)
    if energy / query.size < 1e-12:
        return 0.0, 0
    longer = longer - longer.mean()
    nfft = 1 << (query.size + longer.size - 2).bit_length()
    cross = np.fft.irfft(
        np.fft.rfft(query[::-1], nfft) * np.fft.rfft(longer, nfft), nfft,
    )[query.size - 1:longer.size]
    sums = np.concatenate(([0.0], np.cumsum(longer)))
    squares = np.concatenate(([0.0], np.cumsum(longer * longer)))
    window_sum = sums[query.size:] - sums[:-query.size]
    window_energy = np.maximum(
        0.0, squares[query.size:] - squares[:-query.size]
        - window_sum * window_sum / query.size,
    )
    denominator = np.sqrt(energy * window_energy)
    scores = np.divide(
        cross, denominator, out=np.zeros_like(cross),
        where=window_energy / query.size >= 1e-12,
    )
    offset = int(np.argmax(scores))
    return float(np.clip(scores[offset], -1.0, 1.0)), offset


def deduplicate_content(
    accepted: list[CandidateSentence],
    waveform: np.ndarray,
    sample_rate: int = 16000,
) -> list[CandidateSentence]:
    """Prefer an existing complete clip over any wholly contained copy.

    Only near-identical waveforms count; text or speaker similarity alone is
    insufficient. Partial common prefixes/tails never authorize new cuts.
    Run after final acceptance so a rejected parent cannot suppress a fallback.
    """
    waves = {}
    for candidate in accepted:
        begin = max(0, int(candidate.start * sample_rate))
        end = int(candidate.end * sample_rate)
        if end <= len(waveform) and end > begin:
            waves[id(candidate)] = waveform[begin:end]
    # Longest first is deliberate: chronology must not favor an early fragment
    # over a complete sentence that appears later in the candidate list.
    ranked = sorted(accepted, key=lambda c: (-c.duration, c.start, c.end))
    kept: list[CandidateSentence] = []
    removed: list[CandidateSentence] = []
    for candidate in ranked:
        query = waves.get(id(candidate))
        if query is not None and query.size >= sample_rate // 2:
            for whole in kept:
                longer = waves.get(id(whole))
                if longer is None or query.size > longer.size:
                    continue
                score, offset = contained_audio_match(query, longer)
                if score < 0.98:
                    continue
                # Loud common words must not hide a quiet unique ending. The
                # FFT score locates the copy; every 20 ms window must agree.
                matched = longer[offset:offset + query.size]
                width = max(1, sample_rate // 50)
                starts = list(range(0, max(1, query.size - width + 1), max(1, width // 2)))
                starts.append(max(0, query.size - width))
                local_match = True
                for begin in starts:
                    left = np.asarray(query[begin:begin + width], dtype=np.float64)
                    right = np.asarray(matched[begin:begin + width], dtype=np.float64)
                    left, right = left - left.mean(), right - right.mean()
                    left_rms = float(np.sqrt(np.mean(left * left)))
                    right_rms = float(np.sqrt(np.mean(right * right)))
                    if max(left_rms, right_rms) < 1e-6:
                        continue
                    denominator = np.linalg.norm(left) * np.linalg.norm(right)
                    if denominator <= 1e-12 or float(left @ right) / denominator < 0.98:
                        local_match = False
                        break
                if not local_match:
                    continue
                candidate.accepted = False
                candidate.reject_reason = "音频已完整包含于保留句中，删除重复片段"
                candidate.audio_file = candidate.text_file = candidate.video_file = ""
                candidate.diagnostics["content_duplicate_of"] = [whole.start, whole.end]
                candidate.diagnostics["content_duplicate_match"] = {
                    "correlation": round(score, 6),
                    "offset_seconds": offset / sample_rate,
                    "compared_seconds": query.size / sample_rate,
                    "retained_boundary_unchanged": True,
                }
                removed.append(candidate)
                break
            else:
                kept.append(candidate)
        else:
            kept.append(candidate)
    removed_ids = {id(candidate) for candidate in removed}
    accepted[:] = [candidate for candidate in accepted if id(candidate) not in removed_ids]
    return removed
