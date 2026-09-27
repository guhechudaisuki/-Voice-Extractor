"""Bounded paired reads from already decoded 16 kHz raw and UVR WAV files.

The singing pass and UVR must have run upstream. This module measures a stable
raw-to-stem delay from multiple speech anchors, then maps both views back to the
same original sample timeline. Ambiguous alignment is an error, not zero lag.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .features import FeatureSequence, WavLMSpeakerFeatures, file_digest, require_paired
from .timeline import SampleSpan, SourceTimeline, ViewAlignment, require_digest


def _mono_geometry(path: Path) -> tuple[int, int]:
    info = sf.info(str(path))
    if info.samplerate != 16000 or info.channels != 1 or info.frames <= 0:
        raise ValueError(f"Expected a nonempty mono 16 kHz WAV: {path}")
    return info.samplerate, info.frames


def _read(path: Path, span: SampleSpan) -> np.ndarray:
    with sf.SoundFile(str(path)) as stream:
        if span.end > len(stream):
            raise ValueError("Requested interval exceeds the prepared file")
        stream.seek(span.start)
        data = stream.read(span.length, dtype="float32", always_2d=False)
    if len(data) != span.length or not np.isfinite(data).all():
        raise ValueError("Prepared audio is truncated or nonfinite")
    return data


@dataclass(frozen=True)
class AlignmentReport:
    raw_digest: str
    stem_digest: str
    delay_samples: int
    anchors: tuple[SampleSpan, ...]
    correlations: tuple[float, ...]
    maximum_disagreement_samples: int

    def __post_init__(self) -> None:
        require_digest(self.raw_digest)
        require_digest(self.stem_digest)
        if (type(self.delay_samples) is not int
                or type(self.maximum_disagreement_samples) is not int
                or self.maximum_disagreement_samples < 0
                or len(self.anchors) != len(self.correlations)
                or len(self.anchors) < 2
                or any(not np.isfinite(value) or not 0 < value <= 1 for value in self.correlations)):
            raise ValueError("Invalid measured-alignment report")


def estimate_stem_delay(raw_path: Path, stem_path: Path,
                        anchors: tuple[SampleSpan, ...], *,
                        maximum_delay_samples: int = 3200,
                        minimum_correlation: float = 0.55,
                        maximum_disagreement_samples: int = 8) -> AlignmentReport:
    """Sample-lag cross correlation on independent voiced spans.

    This verifies timing only, not identity, singing or separation quality.
    Anchors must be disjoint and long enough for a distinctive waveform.
    """
    from scipy.signal import correlate

    _, raw_length = _mono_geometry(raw_path)
    _, stem_length = _mono_geometry(stem_path)
    if (type(maximum_delay_samples) is not int or maximum_delay_samples < 0
            or type(maximum_disagreement_samples) is not int or maximum_disagreement_samples < 0
            or not np.isfinite(minimum_correlation) or not 0 < minimum_correlation <= 1
            or len(anchors) < 2 or tuple(sorted(anchors)) != anchors
            or any(a.end > b.start for a, b in zip(anchors, anchors[1:]))):
        raise ValueError("Alignment needs ordered independent anchors and valid limits")
    lags, strengths = [], []
    for anchor in anchors:
        if (anchor.length < 4000 or anchor.start < maximum_delay_samples
                or anchor.end + maximum_delay_samples > min(raw_length, stem_length)):
            raise ValueError("Alignment anchor is short or too close to a file edge")
        raw = _read(raw_path, anchor).astype(np.float64)
        stem = _read(stem_path, SampleSpan(anchor.start - maximum_delay_samples,
                                           anchor.end + maximum_delay_samples)).astype(np.float64)
        raw -= raw.mean()
        raw_power = float(raw @ raw)
        if raw_power < 1e-8:
            raise ValueError("Alignment anchor is silent")
        dot = correlate(stem, raw, mode="valid", method="fft")
        squared = np.concatenate(([0.0], np.cumsum(stem * stem)))
        summed = np.concatenate(([0.0], np.cumsum(stem)))
        width = len(raw)
        count = len(stem) - width + 1
        variance = squared[width:] - squared[:count] - (
            summed[width:] - summed[:count]) ** 2 / width
        score = np.abs(dot) / np.sqrt(np.maximum(variance, 1e-12) * raw_power)
        position = int(np.argmax(score))
        strength = min(1.0, float(score[position]))
        if strength < minimum_correlation:
            raise ValueError("Raw and UVR speech are not reliably time aligned")
        lags.append(position - maximum_delay_samples)
        strengths.append(strength)
    delay = int(round(float(np.median(lags))))
    if max(abs(lag - delay) for lag in lags) > maximum_disagreement_samples:
        raise ValueError("Raw/UVR delay varies across speech anchors")
    return AlignmentReport(file_digest(raw_path), file_digest(stem_path), delay, anchors,
                           tuple(strengths), maximum_disagreement_samples)


class PairedAudio:
    """Read exact candidate context after the alignment report has been checked."""

    def __init__(self, raw_path: Path, stem_path: Path, report: AlignmentReport):
        raw_path, stem_path = raw_path.resolve(strict=True), stem_path.resolve(strict=True)
        rate, raw_length = _mono_geometry(raw_path)
        _, stem_length = _mono_geometry(stem_path)
        raw_digest = file_digest(raw_path)
        if raw_digest != report.raw_digest or file_digest(stem_path) != report.stem_digest:
            raise ValueError("Prepared audio changed after delay measurement")
        source = SourceTimeline(raw_digest, rate, raw_length)
        self.source = source
        self.raw_path, self.stem_path = raw_path, stem_path
        self.raw_alignment = ViewAlignment(source, rate, raw_length, 0, True)
        self.stem_alignment = ViewAlignment(source, rate, stem_length, report.delay_samples, True)
        for anchor in report.anchors:
            source.validate(anchor)
            self.stem_alignment.from_source(anchor)
        self.alignment_report = report

    def encode(self, context: SampleSpan, encoder: WavLMSpeakerFeatures) -> tuple[FeatureSequence, FeatureSequence]:
        raw_span, stem_span = self._view_spans(context)
        raw_audio, stem_audio = self.read_pair(context)
        raw = encoder.encode(torch.from_numpy(raw_audio), self.raw_alignment, raw_span)
        stem = encoder.encode(torch.from_numpy(stem_audio), self.stem_alignment, stem_span)
        require_paired(raw, stem)
        return raw, stem

    def _view_spans(self, context: SampleSpan) -> tuple[SampleSpan, SampleSpan]:
        raw_span = self.raw_alignment.from_source(context)
        stem_span = self.stem_alignment.from_source(context)
        if raw_span.length != stem_span.length:
            raise ValueError("Raw/stem paired windows have different real sample counts")
        return raw_span, stem_span

    def read_pair(self, context: SampleSpan) -> tuple[np.ndarray, np.ndarray]:
        raw_span, stem_span = self._view_spans(context)
        return _read(self.raw_path, raw_span), _read(self.stem_path, stem_span)
