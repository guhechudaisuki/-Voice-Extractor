"""Paired-track pitch, vocal-tract and coherent spectral-transfer measurements.

These measurements describe a separator's effect on aligned audio. They do not
identify a speaker and must never supply independent target-identity evidence.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, correlate, sosfilt


VERSION = "pitch-lpc-coherent-transfer-1"
_EPS = np.finfo(np.float64).eps


def _empty(reason):
    return {"available": False, "reason": reason, "version": VERSION,
            "f0_agreement": 0., "formant_agreement": 0., "coupling_agreement": 0.,
            "core_coherence": 0., "high_coherence": 0., "high_band_loss": 0.,
            "alignment_confidence": 0., "voiced_coverage": 0.,
            "reliability": 0., "artifact_confidence": 0.}


def _frames(audio, width, hop):
    return np.lib.stride_tricks.sliding_window_view(audio, width)[::hop].copy()


def _pitch_batch(frames, sample_rate):
    """Batched YIN, preserving the first credible local minimum rule."""
    frames = np.asarray(frames, dtype=np.float64)
    count, size = frames.shape
    frames = frames - frames.mean(axis=1, keepdims=True)
    first, last = max(2, math.floor(sample_rate / 800)), min(size // 2, math.ceil(sample_rate / 50))
    if last <= first or not count:
        return np.full(count, np.nan), np.zeros(count)
    transform = np.fft.rfft(frames, n=2 * size, axis=1)
    acf = np.fft.irfft(transform * transform.conj(), n=2 * size, axis=1)[:, :last + 1]
    energy = np.pad(np.cumsum(frames * frames, axis=1), ((0, 0), (1, 0)))
    lags = np.arange(last + 1)
    difference = np.maximum(0., energy[:, size - lags] + energy[:, -1:] - energy[:, lags] - 2 * acf)
    difference /= size - lags
    average = np.cumsum(difference[:, 1:], axis=1) / np.arange(1, last + 1)
    cmnd = np.ones_like(difference)
    cmnd[:, 1:] = difference[:, 1:] / np.maximum(average, _EPS)
    credible = cmnd[:, first:last] < .15
    valid = credible.any(axis=1) & (energy[:, -1] > size * _EPS)
    initial = first + credible.argmax(axis=1)
    positions = np.arange(last)
    stops = ((cmnd[:, 1:last + 1] >= cmnd[:, :last]) | (positions == last - 1))
    lag = ((positions >= initial[:, None]) & stops).argmax(axis=1)
    rows = np.arange(count)
    left, center, right = cmnd[rows, lag - 1], cmnd[rows, lag], cmnd[rows, lag + 1]
    curvature = left - 2 * center + right
    offset = np.divide(left - right, 2 * curvature, out=np.zeros(count), where=curvature > _EPS)
    period = lag + np.clip(offset, -.5, .5)
    pitch = np.divide(sample_rate, period, out=np.full(count, np.nan), where=valid & (period > 0))
    return pitch, np.where(valid, np.clip(1. - center, 0., 1.), 0.)


def _pitch(frame, sample_rate):
    pitch, confidence = _pitch_batch(np.asarray(frame)[None], sample_rate)
    return float(pitch[0]), float(confidence[0])


def _formants_batch(frames, sample_rate, count_formants=2):
    """Batched Toeplitz systems and companion eigenvalues for LPC resonances."""
    frames = np.asarray(frames, dtype=np.float64)
    count, width = frames.shape
    output = np.full((count, count_formants), np.nan)
    bandwidths = output.copy()
    if not count or width < 9:
        return output, bandwidths
    alpha = math.exp(-2 * math.pi * 50 / sample_rate)
    emphasized = frames[:, 1:] - alpha * frames[:, :-1]
    emphasized = (emphasized - emphasized.mean(axis=1, keepdims=True)) * np.hamming(width - 1)
    order = min(2 + sample_rate // 1000, (width - 1) // 4)
    spectrum = np.fft.rfft(emphasized, n=2 * (width - 1), axis=1)
    acf = np.fft.irfft(spectrum * spectrum.conj(), axis=1)[:, :order + 1]
    valid = np.flatnonzero(acf[:, 0] > _EPS)
    acf = acf[valid]
    if not len(valid):
        return output, bandwidths
    acf[:, 0] *= 1. + math.sqrt(_EPS)
    offsets = np.abs(np.arange(order)[:, None] - np.arange(order))
    # Bound the temporary matrices for long recordings.
    for begin in range(0, len(valid), 256):
        chunk = acf[begin:begin + 256]
        try:
            coefficients = np.linalg.solve(chunk[:, offsets], -chunk[:, 1:, None])[:, :, 0]
            companion = np.zeros((len(chunk), order, order))
            companion[:, 0, :] = -coefficients
            companion[:, np.arange(1, order), np.arange(order - 1)] = 1.
            poles = np.linalg.eigvals(companion)
        except np.linalg.LinAlgError:
            continue  # Unresolved vocal structure cannot confer confidence.
        radius = np.abs(poles)
        frequency = np.angle(poles) * sample_rate / (2 * math.pi)
        bandwidth = -np.log(np.maximum(radius, _EPS)) * sample_rate / math.pi
        good = ((poles.imag > 0.) & (radius < 1.) & (radius > 0.)
                & (frequency >= 150.) & (frequency < min(3500. if count_formants == 2 else 5500., sample_rate / 2))
                & (bandwidth > 0.) & (bandwidth < frequency / 2))
        if count_formants > 2:
            good &= bandwidth < np.minimum(500., frequency * .2)
        ranked = np.argsort(np.where(good, frequency, np.inf), axis=1)[:, :count_formants]
        fr = np.take_along_axis(frequency, ranked, axis=1)
        bw = np.take_along_axis(bandwidth, ranked, axis=1)
        resolved = (good.sum(axis=1) >= count_formants) & (fr[:, 0] <= 1500.) & (fr[:, 1] >= 500.)
        separation = np.maximum(sample_rate / width, (bw[:, 1:] + bw[:, :-1]) / 2)
        resolved &= (np.diff(fr, axis=1) >= separation).all(axis=1)
        indexes = valid[begin:begin + len(chunk)][resolved]
        output[indexes], bandwidths[indexes] = fr[resolved], bw[resolved]
    return output, bandwidths


def _formants(frame, sample_rate):
    frequency, bandwidth = _formants_batch(np.asarray(frame)[None], sample_rate)
    return (frequency[0], bandwidth[0]) if np.isfinite(frequency[0]).all() else None


def vocal_tract_frames(waveform, sample_rate=16000):
    """Reliable high-formant ratios and log register, never an identity vote.

    Analyze bounded blocks to avoid full-recording FFT/LPC temporary arrays.
    Unvoiced, under-resolved and non-finite samples supply no geometry evidence.
    """
    if hasattr(waveform, "detach"):
        waveform = waveform.detach().cpu().numpy()
    audio = np.asarray(waveform, dtype=float).reshape(-1)
    width, hop = round(.064 * sample_rate), round(.02 * sample_rate)
    if len(audio) < width or not np.isfinite(audio).all():
        return np.empty((0, 4))
    output = []
    frames = np.lib.stride_tricks.sliding_window_view(audio, width)[::hop]
    for start in range(0, len(frames), 256):
        block = frames[start:start+256]
        pitch, confidence = _pitch_batch(block, sample_rate)
        formants, bandwidth = _formants_batch(block, sample_rate, count_formants=4)
        valid = np.isfinite(pitch) & (confidence > .85) & np.isfinite(formants).all(axis=1)
        valid &= (formants[:, 1] > 3 * pitch) & (bandwidth/formants < .2).all(axis=1)
        f, p = formants[valid], pitch[valid]
        if len(f):
            # Spacing estimates inverse vocal tract length; ratios tolerate scale.
            output.append(np.column_stack((np.log(f[:,2]/f[:,1]), np.log(f[:,3]/f[:,2]),
                                            np.log(np.mean(np.diff(f, axis=1), axis=1)), np.log(p))))
    return np.concatenate(output) if output else np.empty((0,4))


def prosody_signature(waveform, sample_rate=16000):
    """Register-independent contour/rhythm summary, usable only as a prior."""
    if hasattr(waveform, 'detach'):
        waveform = waveform.detach().cpu().numpy()
    audio = np.asarray(waveform,dtype=float).reshape(-1)
    width, hop = round(.064*sample_rate), round(.02*sample_rate)
    if len(audio) < sample_rate*.8 or not np.isfinite(audio).all():
        return None
    frames = np.lib.stride_tricks.sliding_window_view(audio,width)[::hop]
    blocks = [_pitch_batch(frames[i:i+256],sample_rate) for i in range(0,len(frames),256)]
    pitch = np.concatenate([p for p,_ in blocks])
    confidence = np.concatenate([c for _,c in blocks])
    voiced = np.isfinite(pitch) & (confidence > .85)
    if voiced.sum() < 20:
        return None
    contour = np.log(np.maximum(pitch,1.))
    continuous = voiced[1:] & voiced[:-1]
    delta = np.diff(contour)[continuous]
    if len(delta) < 10:
        return None
    tail = continuous & (np.arange(len(continuous)) >= .7*len(continuous))
    slope = float(np.median(np.diff(contour)[tail])) if tail.any() else 0.
    low, high = np.quantile(contour[voiced],[.1,.9])
    return np.array([high-low, np.median(abs(delta)), slope, voiced.mean()])


def _align(stem, raw, sample_rate):
    """Correct a small separator latency using only the vocal core band."""
    sos = butter(3, [80., min(3000., sample_rate * .45)], btype="bandpass",
                 fs=sample_rate, output="sos")
    a, b = sosfilt(sos, stem), sosfilt(sos, raw)
    radius = min(round(.02 * sample_rate), len(a) // 4)
    mid = len(a) - 1
    correlation = correlate(a, b, mode="full", method="fft")[mid - radius:mid + radius + 1]
    lag = int(np.argmax(np.abs(correlation))) - radius
    if lag > 0:
        a, b, stem, raw = a[lag:], b[:-lag], stem[lag:], raw[:-lag]
    elif lag < 0:
        a, b, stem, raw = a[:lag], b[-lag:], stem[:lag], raw[-lag:]
    confidence = abs(float(a @ b)) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), _EPS)
    # A maximum at the search boundary is not an identified latency.
    if abs(lag) == radius:
        confidence = 0.
    return stem, raw, lag, float(np.clip(confidence, 0., 1.))


def analyze_pair(stem: np.ndarray, raw: np.ndarray, sample_rate: int = 16000) -> dict:
    """Measure paired voiced structure and attenuation of coherent high harmonics.

    Frame settings cover at least three periods at 50 Hz. Welch estimates average
    independent time observations; a single-frame coherence is never reported.
    Missing or ambiguous pitch/formants yield zero compensation confidence.
    """
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate < 8000:
        return _empty("unsupported_sample_rate")
    a, b = np.asarray(stem, dtype=np.float64), np.asarray(raw, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or len(a) != len(b):
        return _empty("unaligned_shapes")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return _empty("nonfinite_audio")
    width, hop = round(.064 * sample_rate), round(.02 * sample_rate)
    if len(a) < width + 2 * hop:
        return _empty("insufficient_frames")
    a, b = a - a.mean(), b - b.mean()
    scales = float(np.sqrt(np.mean(a * a))), float(np.sqrt(np.mean(b * b)))
    if min(scales) <= math.sqrt(_EPS):
        return _empty("silent_audio")
    a, b = a / scales[0], b / scales[1]
    a, b, lag, alignment = _align(a, b, sample_rate)
    frames_a, frames_b = _frames(a, width, hop), _frames(b, width, hop)
    fa, ca = _pitch_batch(frames_a, sample_rate)
    fb, cb = _pitch_batch(frames_b, sample_rate)
    valid_frames = np.flatnonzero(np.isfinite(fa) & np.isfinite(fb))
    pitch_a, pitch_b = fa[valid_frames], fb[valid_frames]
    ca, cb = ca[valid_frames], cb[valid_frames]
    confidence = np.sqrt(ca * cb)
    pitch_error = np.sqrt((1. - ca) ** 2 + (1. - cb) ** 2
                          + (pitch_a / sample_rate) ** 2 + (pitch_b / sample_rate) ** 2)
    matches = np.exp(-.5 * (np.log(pitch_a / pitch_b) / np.maximum(pitch_error, _EPS)) ** 2)
    ra, ba = _formants_batch(frames_a[valid_frames], sample_rate)
    rb, bb = _formants_batch(frames_b[valid_frames], sample_rate)
    resolved = np.isfinite(ra).all(axis=1) & np.isfinite(rb).all(axis=1)
    resolved &= np.minimum(ra[:, 0] / pitch_a, rb[:, 0] / pitch_b) > 1.
    formant_a, formant_b = ra[resolved], rb[resolved]
    ra, rb, ba, bb = ra[resolved], rb[resolved], ba[resolved], bb[resolved]
    resolution = sample_rate / width
    error = np.sqrt((ba / ra) ** 2 + (bb / rb) ** 2
                    + (resolution / ra) ** 2 + (resolution / rb) ** 2)
    formant_matches = np.exp(-.5 * np.mean((np.log(ra / rb) / error) ** 2, axis=1))
    coupled = np.log((ra / pitch_a[resolved, None]) / (rb / pitch_b[resolved, None]))
    coupling_matches = np.exp(-.5 * np.mean(
        coupled ** 2 / (error ** 2 + pitch_error[resolved, None] ** 2), axis=1))
    if len(valid_frames) < 3:
        result = _empty("insufficient_voiced_frames")
        result.update(alignment_confidence=alignment, alignment_lag_samples=lag,
                      frame_count=len(frames_a), voiced_frame_count=len(valid_frames))
        return result

    taper = np.hanning(width)
    stft_a = np.fft.rfft(frames_a[valid_frames] * taper, axis=1)
    stft_b = np.fft.rfft(frames_b[valid_frames] * taper, axis=1)
    power_a, power_b = np.mean(abs(stft_a) ** 2, axis=0), np.mean(abs(stft_b) ** 2, axis=0)
    cross = np.mean(stft_a * stft_b.conj(), axis=0)
    coherence = np.clip(abs(cross) ** 2 / np.maximum(power_a * power_b, _EPS), 0., 1.)
    # Remove the independent-noise finite-sample bias of magnitude squared coherence.
    effective_frames = max(2., len(valid_frames) * hop / width)
    coherence = np.maximum(0., (coherence - 1. / effective_frames) / (1. - 1. / effective_frames))
    frequencies = np.fft.rfftfreq(width, 1. / sample_rate)
    # A family-wise 95% null bound prevents removed broadband background from
    # masquerading as an attenuated voice harmonic through finite-sample bias.
    null_bound = 1. - (.05 / len(coherence)) ** (1. / (effective_frames - 1.))
    coherent_support = np.clip((coherence - null_bound) / (1. - null_bound), 0., 1.)
    coherent_power = power_b * coherence * coherent_support
    # Stem-to-raw coherent transfer separates removed background energy from voice
    # attenuation: with raw = voice + noise and stem = gain * voice, this is gain.
    transfer = power_a / np.maximum(abs(cross), _EPS)
    pitch_center = np.sqrt(np.asarray(pitch_a) * np.asarray(pitch_b))
    distance = np.abs(frequencies[None] / pitch_center[:, None]
                      - np.rint(frequencies[None] / pitch_center[:, None])) * pitch_center[:, None]
    harmonic_weights = np.exp(-.5 * (distance / (sample_rate / width)) ** 2).mean(axis=0)
    weights = coherent_power * harmonic_weights

    def band_measure(low, high):
        mask = (frequencies >= low) & (frequencies < high)
        weight = weights[mask]
        mass = float(weight.sum())
        if mass <= _EPS:
            return 0., 0., 0.
        gain = float(np.exp(np.average(np.log(np.maximum(transfer[mask], _EPS)), weights=weight)))
        return float(np.average(coherence[mask], weights=weight)), gain, mass

    core_coherence, core_gain, core_mass = band_measure(150., min(3500., sample_rate / 2))
    high_coherence, high_gain, high_mass = band_measure(4000., sample_rate / 2)
    # Speech's high harmonics can be quiet yet measurable. Use their coherence
    # precision, not their fraction of the much louder first formant's energy.
    resolved = high_mass / (high_mass + _EPS * max(core_mass, _EPS))
    high_support = resolved * high_coherence * (1. - 1. / effective_frames)
    loss = float(np.clip(1. - high_gain / max(core_gain, _EPS), 0., 1.)) if high_mass > _EPS else 0.
    coverage = len(valid_frames) / len(frames_a)
    f0_match = float(np.average(matches, weights=confidence))
    formant_match = float(np.mean(formant_matches)) if len(formant_matches) else 0.
    coupling_match = float(np.mean(coupling_matches)) if len(coupling_matches) else 0.
    formant_coverage = len(formant_matches) / len(valid_frames)
    reliability = (alignment * coverage * formant_coverage * float(np.mean(confidence))
                   * f0_match * math.sqrt(formant_match * coupling_match) * core_coherence)
    artifact = float(np.clip(reliability * high_support * loss, 0., 1.))
    return {"available": True, "reason": "measured", "version": VERSION,
            "sample_rate": int(sample_rate), "frame_count": len(frames_a),
            "voiced_frame_count": len(valid_frames), "formant_frame_count": len(formant_matches),
            "alignment_lag_samples": lag, "alignment_confidence": alignment,
            "voiced_coverage": float(coverage), "formant_coverage": float(formant_coverage),
            "f0_agreement": f0_match, "formant_agreement": formant_match,
            "coupling_agreement": coupling_match,
            "stem_f0_hz": float(np.median(pitch_a)), "raw_f0_hz": float(np.median(pitch_b)),
            "stem_formants_hz": np.median(formant_a, axis=0).tolist() if len(formant_a) else None,
            "raw_formants_hz": np.median(formant_b, axis=0).tolist() if len(formant_b) else None,
            "core_coherence": core_coherence, "high_coherence": high_coherence,
            "high_coherence_deficit": max(0., core_coherence - high_coherence),
            "high_band_loss": loss, "high_band_support": float(high_support),
            "reliability": float(np.clip(reliability, 0., 1.)), "artifact_confidence": artifact}
