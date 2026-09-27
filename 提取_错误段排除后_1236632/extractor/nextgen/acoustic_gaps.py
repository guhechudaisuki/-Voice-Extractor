"""Conservative independent gap confirmation for change-head proposals.

Both raw and separated VAD must observe the pause, and separated waveform
energy must actually fall relative to speech on both sides. These checks only
certify an acoustic space between utterances; the new child still needs its
own identity, endpoint, purity and contamination assessment.
"""
from __future__ import annotations

import math

import numpy as np

from .boundary_decoder import AcousticGap
from .ledger import Candidate
from .prepared_audio import PairedAudio
from .timeline import SampleSpan


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))


def _gaps(spans: tuple[SampleSpan, ...], parent: Candidate) -> tuple[SampleSpan, ...]:
    if tuple(sorted(spans)) != spans or any(a.end > b.start for a, b in zip(spans, spans[1:])):
        raise ValueError("Voice intervals must be ordered and nonoverlapping")
    if any(not parent.context.contains(span) for span in spans):
        raise ValueError("Voice interval outside analysis context")
    return tuple(SampleSpan(a.end, b.start) for a, b in zip(spans, spans[1:]) if a.end < b.start)


def find_confirmed_gaps(parent: Candidate, raw_voice: tuple[SampleSpan, ...],
                        stem_voice: tuple[SampleSpan, ...], audio: PairedAudio, *,
                        minimum_gap_samples: int, flank_samples: int = 1600,
                        maximum_gap_to_speech_rms: float = 0.2) -> tuple[AcousticGap, ...]:
    if (type(minimum_gap_samples) is not int or minimum_gap_samples < 1
            or type(flank_samples) is not int or flank_samples < 1
            or not math.isfinite(maximum_gap_to_speech_rms)
            or not 0 < maximum_gap_to_speech_rms < 1):
        raise ValueError("Invalid independent gap check")
    if (parent.source_sha256, parent.sample_rate) != (audio.source.source_sha256, audio.source.sample_rate):
        raise ValueError("Gap review used another source recording")
    raw_gaps, stem_gaps = _gaps(raw_voice, parent), _gaps(stem_voice, parent)
    candidates = []
    for raw_gap in raw_gaps:
        for stem_gap in stem_gaps:
            common = raw_gap.intersection(stem_gap)
            if (common is None or common.length < minimum_gap_samples
                    or common.start <= parent.output.start or common.end >= parent.output.end):
                continue
            if (common.start - flank_samples < parent.context.start
                    or common.end + flank_samples > parent.context.end):
                continue
            before = SampleSpan(common.start - flank_samples, common.start)
            after = SampleSpan(common.end, common.end + flank_samples)
            _, stem_left = audio.read_pair(before)
            _, stem_pause = audio.read_pair(common)
            _, stem_right = audio.read_pair(after)
            speech_level = min(_rms(stem_left), _rms(stem_right))
            if speech_level <= 1e-5 or _rms(stem_pause) > maximum_gap_to_speech_rms * speech_level:
                continue
            candidates.append(AcousticGap(common, True, True, True,
                                          "raw_and_stem_vad_with_stem_waveform_pause"))
    # Intersecting two VAD streams may produce the same gap through duplicate
    # side intervals; preserve one deterministic proposal per source interval.
    return tuple({gap.span: gap for gap in sorted(candidates, key=lambda row: row.span)}.values())
