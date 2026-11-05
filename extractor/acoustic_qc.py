"""Waveform boundary refinement, text-safe long-turn splitting and export QC."""
from __future__ import annotations

from copy import deepcopy
from collections.abc import Callable
import re

import numpy as np


def _wave(waveform) -> np.ndarray:
    if hasattr(waveform, "detach"):
        waveform = waveform.detach().cpu().numpy()
    # Keep float32 episode arrays as views: this function is also called once
    # per crop, and copying a full episode for every boundary is prohibitive.
    wave = np.asarray(waveform, dtype=np.float32)
    if wave.ndim == 2:
        channel_axis = 0 if wave.shape[0] <= 8 and wave.shape[1] > wave.shape[0] else 1
        wave = wave.mean(axis=channel_axis)
    return wave.reshape(-1)


def _rms_at(wave: np.ndarray, indices: np.ndarray, frame: int) -> np.ndarray:
    """Centered 10 ms energy, evaluated at sample/1 ms resolution."""
    if not len(indices):
        return np.empty(0)
    left = np.maximum(0, indices - frame // 2)
    right = np.minimum(len(wave), indices + max(1, frame - frame // 2))
    # Restrict the cumulative sum to the requested neighborhood: long episodes
    # must not allocate a full-duration envelope for each sentence boundary.
    origin, finish = int(left.min()), int(right.max())
    local = np.nan_to_num(wave[origin:finish], nan=0., posinf=0., neginf=0.)
    cumulative = np.r_[0., np.cumsum(np.square(local), dtype=np.float64)]
    return np.sqrt(np.maximum(0., cumulative[right - origin] - cumulative[left - origin])
                   / np.maximum(1, right - left))


def _zero_crossing(wave, index, low, high, radius):
    low, high = max(low, index - radius), min(high, index + radius)
    if high <= low:
        return int(np.clip(index, low, high))
    points = np.arange(low, high + 1)
    values = np.nan_to_num(wave[points], nan=0., posinf=0., neginf=0.)
    exact = points[np.abs(values) <= 1e-10]
    changes = points[1:][np.signbit(values[1:]) != np.signbit(values[:-1])]
    choices = np.r_[exact, changes]
    if len(choices):
        return int(choices[np.argmin(np.abs(choices - index))])
    # No crossing exists (e.g. a DC offset): do not invent one far away.
    return int(np.clip(index, low, high))


def _valley(wave, center, low, high, rate):
    """Return a deep energy valley and the contiguous quiet range around it."""
    step, frame = max(1, round(rate * .001)), max(1, round(rate * .01))
    if high < low:
        return None
    indices = np.arange(low, high + 1, step, dtype=np.int64)
    if not len(indices):
        return None
    rms = _rms_at(wave, indices, frame)
    minimum, typical = float(rms.min()), float(np.percentile(rms, 80))
    if minimum > max(1e-5, typical * .7):
        return None
    # Center the closest contiguous minimum plateau, avoiding the first-bin
    # bias that otherwise puts every cut at the leading edge of a silence.
    minima = np.flatnonzero(rms <= minimum + max(1e-10, minimum * .01))
    groups = np.split(minima, np.flatnonzero(np.diff(minima) > 1) + 1)
    group = min(groups, key=lambda g: abs(float(indices[g].mean()) - center))
    chosen = int(group[len(group) // 2])
    quiet = rms <= max(1e-5, typical * 10 ** (-18 / 20))
    if not quiet[chosen]:
        quiet = rms <= minimum * 1.05 + 1e-10
    first = last = chosen
    while first > 0 and quiet[first - 1]:
        first -= 1
    while last + 1 < len(indices) and quiet[last + 1]:
        last += 1
    return int(indices[chosen]), int(indices[first]), int(indices[last])


def snap_crop_boundaries_to_valley(
    waveform: np.ndarray, start: float, end: float, sample_rate: int = 16000,
    max_shift: float = .18, pad_seconds: float = .04,
    hard_bounds: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Snap mechanical crop edges while preserving hard parent/blocked bounds.

    Call only for mechanically cropped candidates. Already quiet edges move at
    most 5 ms to a zero crossing. Padding stays inside the selected quiet valley,
    so a 40 ms pause is never turned back into a cut through neighboring speech.
    """
    if sample_rate <= 0 or max_shift < 0 or pad_seconds < 0:
        raise ValueError("Invalid sample rate or boundary refinement limits")
    wave = _wave(waveform)
    if not len(wave):
        return start, end
    duration = len(wave) / sample_rate
    low, high = hard_bounds if hard_bounds is not None else (0., duration)
    low, high = max(0., low), min(duration, high)
    if not (np.isfinite([start, end, low, high]).all() and low < high and start < end):
        raise ValueError("Invalid crop or hard bounds")
    start, end = max(low, start), min(high, end)
    if start >= end:
        raise ValueError("Crop does not intersect hard bounds")
    rate = sample_rate
    lower = int(np.ceil(low * rate))
    upper = min(len(wave) - 1, int(np.floor(high * rate)))
    frame, step = max(1, round(rate * .01)), max(1, round(rate * .001))
    interior = np.arange(round(start * rate), min(len(wave), round(end * rate)), frame)
    reference = float(np.percentile(_rms_at(wave, interior, frame), 80)) if len(interior) else 0.
    quiet_threshold = max(1e-5, reference * 10 ** (-18 / 20))

    def snap(boundary, direction):
        index = int(np.clip(round(boundary * rate), lower, upper))
        radius = round(max_shift * rate)
        lo, hi = max(lower, index - radius), min(upper, index + radius)
        local_rms = float(_rms_at(wave, np.array([index]), frame)[0])
        if local_rms <= quiet_threshold:
            return _zero_crossing(wave, index, lo, hi, round(.005 * rate)) / rate
        valley = _valley(wave, index, lo, hi, rate)
        if valley is None:
            return boundary
        point, quiet_low, quiet_high = valley
        padded = int(np.clip(point + direction * round(pad_seconds * rate), quiet_low, quiet_high))
        return _zero_crossing(wave, padded, quiet_low, quiet_high, step * 5) / rate

    snapped = (snap(start, -1), snap(end, 1))
    if snapped[0] >= snapped[1]:
        return start, end
    return snapped


def _longest_voiced_run(mask: np.ndarray, hop: int, frame: int, rate: int) -> float:
    longest = current = 0
    for voiced in mask:
        current = current + 1 if voiced else 0
        longest = max(longest, current)
    return (longest * hop + frame - hop) / rate if longest else 0.


def estimate_voiced_f0_stats(waveform, sample_rate: int = 16000) -> dict:
    """Estimate voiced pitch with YIN differences and a periodicity check.

    Unvoiced frames do not contribute a zero pitch. Missing estimates remain
    explicit, so callers cannot confuse silence with a low-register speaker.
    The first credible period is used to avoid selecting its octave multiples.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    wave = np.nan_to_num(_wave(waveform), nan=0., posinf=0., neginf=0.)
    result = {
        "f0_median_hz": None,
        "f0_p15_hz": None,
        "f0_p85_hz": None,
        "voiced_seconds": 0.,
        "voiced_ratio": 0.,
        "voiced_run_seconds": 0.,
        "f0_low_register_run_seconds": 0.,
        "f0_high_register_run_seconds": 0.,
        "f0_periodicity": 0.,
        "f0_stable": False,
    }
    frame, hop = max(8, round(.05 * sample_rate)), max(1, round(.01 * sample_rate))
    min_lag = max(2, int(sample_rate / 650))
    max_lag = min(frame - 2, int(np.ceil(sample_rate / 65)))
    if len(wave) < frame or max_lag <= min_lag:
        return result
    starts = np.arange(0, len(wave) - frame + 1, hop)
    powers = _rms_at(wave, starts + frame // 2, frame)
    energy_floor = max(.003, float(np.percentile(powers, 85)) * .08)
    pitches = np.full(len(starts), np.nan)
    periodicities = np.full(len(starts), np.nan)
    lags = np.arange(1, max_lag + 1)
    fft_size = 1 << (2 * frame - 1).bit_length()
    active = np.flatnonzero(powers >= energy_floor)
    for begin in range(0, len(active), 256):
        indices = active[begin:begin + 256]
        frames = wave[starts[indices, None] + np.arange(frame)]
        frames = frames.astype(np.float64)
        frames -= frames.mean(axis=1, keepdims=True)
        spectrum = np.fft.rfft(frames, n=fft_size, axis=1)
        autocorrelation = np.fft.irfft(spectrum * spectrum.conj(), n=fft_size, axis=1)
        energy = np.column_stack((np.zeros(len(frames)), np.cumsum(frames ** 2, axis=1)))
        left_energy = energy[:, frame - lags]
        right_energy = energy[:, -1, None] - energy[:, lags]
        difference = np.maximum(0., left_energy + right_energy - 2 * autocorrelation[:, lags])
        normalized = difference * lags / np.maximum(np.cumsum(difference, axis=1), 1e-20)
        periodicity = autocorrelation[:, lags] / np.sqrt(np.maximum(left_energy * right_energy, 1e-20))
        for row, output_index in enumerate(indices):
            curve = normalized[row]
            # Index k represents lag k + 1. Only local minima inside the
            # requested pitch range are eligible for parabolic interpolation.
            minima = np.flatnonzero((curve[1:-1] <= curve[:-2]) & (curve[1:-1] < curve[2:])) + 1
            minima = minima[(minima + 1 >= min_lag) & (curve[minima] <= .20)
                            & (periodicity[row, minima] >= .75)]
            if not len(minima):
                continue
            selected = int(minima[0])
            before, center, after = curve[selected - 1:selected + 2]
            curvature = before - 2 * center + after
            offset = .5 * (before - after) / curvature if curvature > 1e-12 else 0.
            period = selected + 1 + float(np.clip(offset, -.5, .5))
            frequency = sample_rate / period
            if 65 <= frequency <= 650:
                pitches[output_index] = frequency
                periodicities[output_index] = periodicity[row, selected]
    voiced = np.isfinite(pitches)
    if not voiced.any():
        return result
    median, low, high = np.percentile(pitches[voiced], [50, 15, 85])
    continuity = float(np.median(periodicities[voiced]))
    # Count the union of voiced frame intervals, including overlap across
    # isolated unvoiced frames without counting those samples twice.
    voiced_starts = starts[voiced]
    voiced_seconds = (frame + np.minimum(frame, np.diff(voiced_starts)).sum()) / sample_rate
    result.update({
        "f0_median_hz": round(float(median), 2),
        "f0_p15_hz": round(float(low), 2),
        "f0_p85_hz": round(float(high), 2),
        "voiced_seconds": round(voiced_seconds, 3),
        "voiced_ratio": round(float(voiced.mean()), 4),
        "voiced_run_seconds": round(_longest_voiced_run(voiced, hop, frame, sample_rate), 3),
        "f0_low_register_run_seconds": round(_longest_voiced_run(voiced & (pitches < 145), hop, frame, sample_rate), 3),
        "f0_high_register_run_seconds": round(_longest_voiced_run(voiced & (pitches > 235), hop, frame, sample_rate), 3),
        "f0_periodicity": round(continuity, 4),
        "f0_stable": bool(high / max(low, 1.) <= 1.8 and continuity >= .8),
    })
    return result


def detect_speaker_change_valleys(
    waveform, start: float = 0., end: float | None = None, *,
    embed_windows: Callable, target_centroid=None, target_threshold: float = .55,
    sample_rate: int = 16000,
) -> list[float]:
    """Locate speaker changes with 600 ms embeddings every 200 ms.

    ``embed_windows`` receives a list of mono NumPy slices and returns an N x D
    matrix. Cuts use absolute source times. A target-affinity change requires
    at least one second of target support; a cosine drop above 0.30 is also
    sufficient. Every child still needs independent identity verification.
    """
    if sample_rate <= 0 or not np.isfinite([start, target_threshold]).all():
        raise ValueError("Invalid sample rate or speaker-change limits")
    wave = _wave(waveform)
    duration = len(wave) / sample_rate
    end = duration if end is None else end
    if not np.isfinite(end) or end < start:
        raise ValueError("Invalid speaker-change span")
    start, end = max(0., start), min(duration, end)
    if end - start < 2.2:
        return []
    window, step = .6, .2
    starts = np.arange(start, end - window + 1e-8, step)
    if end - window - starts[-1] > 1e-6:
        starts = np.r_[starts, end - window]
    windows = [wave[round(value * sample_rate):round((value + window) * sample_rate)] for value in starts]
    embeddings = embed_windows(windows)
    if hasattr(embeddings, "detach"):
        embeddings = embeddings.detach().cpu().numpy()
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim != 2 or len(embeddings) != len(windows):
        raise ValueError("embed_windows must return one embedding per window")
    norms = np.linalg.norm(embeddings, axis=1)
    valid = np.isfinite(embeddings).all(axis=1) & (norms > 1e-8)
    embeddings = embeddings / np.maximum(norms[:, None], 1e-8)
    affinity = None
    if target_centroid is not None:
        if hasattr(target_centroid, "detach"):
            target_centroid = target_centroid.detach().cpu().numpy()
        centroid = np.asarray(target_centroid, dtype=np.float64).reshape(-1)
        if len(centroid) != embeddings.shape[1]:
            raise ValueError("Target centroid and window embedding dimensions differ")
        norm = float(np.linalg.norm(centroid))
        if np.isfinite(centroid).all() and norm > 1e-8:
            affinity = embeddings @ (centroid / norm)
    centers = starts + window / 2
    candidates = []
    for index in range(len(starts) - 1):
        center = float((centers[index] + centers[index + 1]) / 2)
        if center - start < .6 or end - center < .6:
            continue
        similarity = float(embeddings[index] @ embeddings[index + 1])
        drop = 1. - similarity if valid[index:index + 2].all() else 0.
        affinity_jump = 0.
        if affinity is not None:
            left = np.flatnonzero((starts + window <= center + 1e-6) & (starts >= center - 1.2))
            right = np.flatnonzero((starts >= center - 1e-6) & (starts + window <= center + 1.2))
            for target, other in ((left, right), (right, left)):
                if not len(target) or not len(other) or not valid[np.r_[target, other]].all():
                    continue
                coverage = starts[target[-1]] + window - starts[target[0]]
                if coverage < 1. - 1e-6 or float(affinity[target].min()) < target_threshold:
                    continue
                target_score, other_score = float(np.median(affinity[target])), float(np.median(affinity[other]))
                if other_score <= target_threshold - .12 and target_score - other_score >= .20:
                    affinity_jump = max(affinity_jump, target_score - other_score)
        if drop > .30 or affinity_jump > 0:
            candidates.append((max(drop, affinity_jump), center))
    cuts = []
    for strength, center in sorted(candidates, reverse=True):
        low, high = max(start + .6, center - .18), min(end - .6, center + .18)
        if high < low:
            continue
        first = int(np.ceil(low * sample_rate))
        last = min(len(wave) - 1, int(np.floor(high * sample_rate)))
        if first > last:
            continue
        cut, _ = snap_crop_boundaries_to_valley(
            wave, center, end, sample_rate, max_shift=.18, pad_seconds=0., hard_bounds=(low, end))
        cut = float(np.clip(cut, low, high))
        if abs(cut - center) < 1 / sample_rate:
            # Seamless handoffs may have no deep silence. Their identity change
            # still supplies the cut; use the least energetic 10 ms nearby.
            points = np.arange(first, last + 1, max(1, round(.001 * sample_rate)))
            energy = _rms_at(wave, points, max(1, round(.01 * sample_rate)))
            minimum = energy.min()
            choices = points[energy <= minimum + max(1e-10, minimum * .01)]
            point = int(choices[np.argmin(abs(choices / sample_rate - center))])
            cut = _zero_crossing(wave, point, first, last, round(.005 * sample_rate)) / sample_rate
        else:
            cut = _zero_crossing(wave, round(cut * sample_rate), first, last,
                                 round(.005 * sample_rate)) / sample_rate
        if all(abs(cut - previous) >= .6 for previous in cuts):
            cuts.append(cut)
    return sorted(cuts)


def _fragment_value(fragment, name, default=None):
    return fragment.get(name, default) if isinstance(fragment, dict) else getattr(fragment, name, default)


def _estimated(fragment):
    return bool(_fragment_value(fragment, "timing_estimated", False)
                or (_fragment_value(fragment, "diagnostics", {}) or {}).get("timing_estimated"))


def _text_cuts(text, fragments):
    """Map fragment boundaries onto original text without changing any character."""
    normalized = re.sub(r"\s+", "", text)
    parts = [re.sub(r"\s+", "", str(_fragment_value(f, "whisper_text", None)
                                   or _fragment_value(f, "text", ""))) for f in fragments]
    if not all(parts) or "".join(parts) != normalized:
        return None
    positions = [index for index, character in enumerate(text) if not character.isspace()]
    offsets, consumed = [0], 0
    for part in parts[:-1]:
        consumed += len(part)
        offsets.append(positions[consumed])
    return offsets + [len(text)]


def split_overlong_sentences(
    accepted: list, stem_waveform: np.ndarray, by_turn_fragments: dict,
    sample_rate: int = 16000, max_seconds: float = 12., min_part_seconds: float = 2.2,
) -> list:
    """Recursively split only at timestamped, text-aligned acoustic boundaries.

    No inferred character-time proportions are accepted. Corrected text which
    cannot be mapped exactly to the original fragments is retained for review.
    Children carry their own timing metadata and require new identity scoring.
    """
    if sample_rate <= 0 or min_part_seconds <= 0 or max_seconds < 2 * min_part_seconds:
        raise ValueError("Invalid long-sentence split limits")
    wave = _wave(stem_waveform)
    output = []

    def split(candidate, fragments, origin):
        if candidate.end - candidate.start <= max_seconds + 1e-9:
            return [candidate]
        # Mixed timed/untimed Whisper windows can include an estimated fragment
        # spanning the whole turn. Checking only the two adjacent fragments
        # would then assign its text to one half while its audio spans both.
        invalid_timing = any(_estimated(fragment) for fragment in fragments)
        invalid_timing = invalid_timing or any(
            float(_fragment_value(left, "end")) > float(_fragment_value(right, "start")) + .001
            for left, right in zip(fragments, fragments[1:]))
        if invalid_timing:
            candidate.diagnostics["overlong_unsplit"] = True
            candidate.diagnostics["overlong_unsplit_reason"] = "estimated_or_overlapping_timestamps"
            return [candidate]
        text = candidate.text or candidate.whisper_text
        text_offsets = _text_cuts(text, fragments)
        whisper_offsets = _text_cuts(candidate.whisper_text or text, fragments)
        reason = "no_reliable_timed_fragments"
        if len(fragments) >= 2 and text_offsets is not None and whisper_offsets is not None:
            choices = []
            midpoint = (candidate.start + candidate.end) / 2
            for index in range(1, len(fragments)):
                left, right = fragments[index - 1:index + 1]
                if any(_estimated(f) for f in (left, right)):
                    continue
                left_end = float(_fragment_value(left, "end"))
                right_start = float(_fragment_value(right, "start"))
                left_text = str(_fragment_value(left, "whisper_text", "") or _fragment_value(left, "text", ""))
                if right_start < left_end - .001:
                    continue
                if right_start - left_end < .04 - 1e-8 and not re.search(r"[。！？!?][\"'”’）】》]*$", left_text.strip()):
                    continue
                center = (left_end + right_start) / 2
                low = max(candidate.start + min_part_seconds, center - .25)
                high = min(candidate.end - min_part_seconds, center + .25, len(wave) / sample_rate)
                if right_start - left_end >= .01:
                    low, high = max(low, left_end), min(high, right_start)
                if low >= high:
                    continue
                valley = _valley(wave, round(center * sample_rate),
                                 max(0, int(np.ceil(low * sample_rate))),
                                 min(len(wave) - 1, int(np.floor(high * sample_rate))), sample_rate)
                if valley is None:
                    continue
                point, qlow, qhigh = valley
                cut = _zero_crossing(wave, point, qlow, qhigh, round(.005 * sample_rate)) / sample_rate
                # A nearby vowel minimum can lie inside a word. Preserve the
                # shared boundary and both text halves; never duplicate a word.
                from .word_boundaries import real_words
                crossing = next((w for w in real_words(candidate.diagnostics.get('word_timestamps',()))
                                 if w['start'] < cut < w['end']),None)
                if crossing is not None:
                    options = [edge for edge in (crossing['start'],crossing['end'])
                               if abs(edge-cut)<=.08 and left_end-.001<=edge<=right_start+.001
                               and low<=edge<=high]
                    if not options:
                        continue
                    cut = min(options,key=lambda edge:abs(edge-cut))
                choices.append((abs(cut - midpoint), index, cut))
            if choices:
                _, index, cut = min(choices)
                children = []
                for begin, finish, a, b in ((candidate.start, cut, 0, index),
                                           (cut, candidate.end, index, len(fragments))):
                    child = deepcopy(candidate)
                    child.start, child.end = begin, finish
                    child.text = text[text_offsets[a]:text_offsets[b]]
                    whisper = candidate.whisper_text or text
                    child.whisper_text = whisper[whisper_offsets[a]:whisper_offsets[b]]
                    child.audio_file = child.text_file = child.video_file = ""
                    for field in ('speaker_score', 'window_min_score', 'window_p20_score',
                                  'speaker_vote_ratio', 'window_vote_ratio'):
                        setattr(child, field, 0.)
                    for key in ('acoustic_identity_audit', 'ledger_candidate_key', 'review_navigation'):
                        child.diagnostics.pop(key, None)
                    child.diagnostics['exact_audio_verified'] = False
                    for key in ('word_timestamps', 'stt_fragments', 'transcription_fragments'):
                        entries = child.diagnostics.get(key)
                        if isinstance(entries, list):
                            child.diagnostics[key] = [entry for entry in entries
                                if isinstance(entry, dict)
                                and isinstance(entry.get('start'), (int, float))
                                and isinstance(entry.get('end'), (int, float))
                                and begin <= entry['start'] < entry['end'] <= finish]
                            if any(isinstance(entry, dict)
                                   and isinstance(entry.get('start'), (int, float))
                                   and isinstance(entry.get('end'), (int, float))
                                   and min(finish, entry['end']) > max(begin, entry['start'])
                                   and entry not in child.diagnostics[key] for entry in entries):
                                child.diagnostics['split_timing_incomplete'] = True
                    child.diagnostics.pop("acoustic_qc", None)
                    child.diagnostics.pop("overlong_unsplit", None)
                    child.diagnostics.pop("overlong_unsplit_reason", None)
                    child.diagnostics["overlong_split_from"] = list(origin)
                    child.diagnostics["overlong_split_cut"] = cut
                    child.diagnostics["stt_fragment_count"] = b - a
                    children.extend(split(child, fragments[a:b], origin))
                return children
            reason = "no_safe_acoustic_fragment_boundary"
        elif fragments and (text_offsets is None or whisper_offsets is None):
            reason = "fragment_text_mismatch"
        candidate.diagnostics["overlong_unsplit"] = True
        candidate.diagnostics["overlong_unsplit_reason"] = reason
        return [candidate]

    for candidate in accepted:
        key = (candidate.start, candidate.end)
        fragments = by_turn_fragments.get(key)
        if fragments is None:
            index = candidate.diagnostics.get("transcription_span_index")
            fragments = by_turn_fragments.get(index, [])
        # The segmenter may join several genuinely timestamped chunks into one
        # sentence. Preserve their real pause boundaries, while its proportional
        # punctuation splits remain explicitly estimated and cannot drive cuts.
        expanded = []
        for fragment in fragments or []:
            nested = (_fragment_value(fragment, "diagnostics", {}) or {}).get("transcription_fragments")
            expanded.extend(nested if nested else [fragment])
        fragments = expanded
        fragments = sorted(fragments or [], key=lambda f: (float(_fragment_value(f, "start")),
                                                           float(_fragment_value(f, "end"))))
        fragments = [f for f in fragments
                     if float(_fragment_value(f, "start")) >= candidate.start - .05
                     and float(_fragment_value(f, "end")) <= candidate.end + .05]
        output.extend(split(candidate, fragments, key))
    return output


def evaluate_clip_acoustics(
    waveform: np.ndarray, sample_rate: int = 16000,
    speaker_score: float = 0., window_min: float = 0.,
) -> dict:
    """Estimate SNR, clipping and speech duration; grade the actual exported WAV.

    SNR is a no-reference estimate. With observed pauses it uses the measured
    lower envelope, floored at -60 dBFS. Without pauses, spectral flatness gives
    a conservative noise estimate (a steady vowel/tone must not score 0 dB,
    but steady white noise must not receive an invented high SNR).
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    raw = _wave(waveform)
    invalid = not np.isfinite(raw).all()
    wave = np.nan_to_num(_wave(waveform), nan=0., posinf=0., neginf=0.)
    duration = len(wave) / sample_rate
    peak = float(np.max(np.abs(wave))) if len(wave) else 0.
    clipping = float(np.mean(np.abs(wave) >= .985)) if len(wave) else 0.
    frame = max(1, round(.02 * sample_rate))
    starts = np.arange(0, len(wave), frame)
    energies = [float(np.sqrt(np.mean(wave[start:start + frame] ** 2))) for start in starts]
    rms = np.asarray(energies)
    snr, effective = 0., 0.
    if len(rms):
        signal = float(np.percentile(rms, 85))
        noise = max(.001, float(np.percentile(rms, 10)))
        if signal > .001 and noise > signal * .5:
            # Analyze a representative voiced frame sequence, bounded to 1 s.
            center = int(starts[int(np.argmax(rms))])
            segment = wave[max(0, center - sample_rate // 2):min(len(wave), center + sample_rate // 2)]
            if len(segment) >= 8:
                spectrum = np.abs(np.fft.rfft((segment - segment.mean()) * np.hanning(len(segment)))) ** 2
                spectrum = spectrum[1:]
                flatness = float(np.exp(np.mean(np.log(np.maximum(spectrum, 1e-20))))
                                 / max(float(spectrum.mean()), 1e-20))
                noise = max(.001, signal * np.sqrt(min(1., flatness)))
        snr = max(0., 20 * np.log10(max(signal, .001) / noise))
        threshold = max(10 ** (-50 / 20), signal * 10 ** (-18 / 20))
        effective = sum(min(frame, len(wave) - int(start))
                        for start, energy in zip(starts, rms) if energy > threshold) / sample_rate
    grade = "Review"
    if not invalid and effective >= .15:
        if (speaker_score >= .70 and window_min >= .60 and snr >= 18
                and clipping < .001 and 1.5 <= duration <= 10.):
            grade = "Gold"
        elif speaker_score >= .55 and snr >= 12 and clipping < .005 and duration >= .25:
            grade = "Silver"
    return {
        **estimate_voiced_f0_stats(wave, sample_rate),
        "snr_db": round(float(snr), 2),
        "clipping_ratio": round(clipping, 6),
        "effective_speech_seconds": round(float(effective), 3),
        "quality_grade": grade,
        "duration_seconds": round(duration, 3),
        "peak_dbfs": round(float(20 * np.log10(max(peak, 1e-10))), 2),
    }
