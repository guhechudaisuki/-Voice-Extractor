"""Convert verified preprocessing events into one source-timeline scene.

This module is deliberately independent of the desktop pipeline. Singing must
have been checked before separation. Subtitle text never reaches identity or
STT; aligned cue times may only request another acoustic VAD pass upstream.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .acoustic_gaps import find_confirmed_gaps
from .boundary_decoder import AcousticGap, SilenceRange, SpeechIsland, propose
from .inference import InferenceResult
from .ledger import Candidate, EvidenceKind, LocalEvidence
from .prepared_audio import PairedAudio
from .timeline import SampleSpan


def _ordered(spans: tuple[SampleSpan, ...], audio: PairedAudio) -> tuple[SampleSpan, ...]:
    """Union overlapping detector windows without filling any actual pause."""
    merged: list[SampleSpan] = []
    for span in sorted(spans):
        audio.source.validate(span)
        if merged and span.start <= merged[-1].end:
            merged[-1] = SampleSpan(merged[-1].start, max(merged[-1].end, span.end))
        else:
            merged.append(span)
    return tuple(merged)


def _within(spans: tuple[SampleSpan, ...], scope: SampleSpan) -> tuple[SampleSpan, ...]:
    return tuple(part for item in spans if (part := item.intersection(scope)) is not None)


def _unmasked_islands(voice: tuple[SampleSpan, ...],
                      blocked: tuple[SampleSpan, ...]) -> tuple[SpeechIsland, ...]:
    """Preserve clean sides of a mask without certifying the new cut as complete."""
    result: list[SpeechIsland] = []
    for span in voice:
        cursor = span.start
        start_complete = True
        for mask in blocked:
            if mask.end <= cursor:
                continue
            if mask.start >= span.end:
                break
            if cursor < mask.start:
                result.append(SpeechIsland(SampleSpan(cursor, mask.start),
                                           start_complete, False))
            cursor = max(cursor, mask.end)
            start_complete = False
            if cursor >= span.end:
                break
        if cursor < span.end:
            result.append(SpeechIsland(SampleSpan(cursor, span.end),
                                       start_complete, True))
    return tuple(result)


@dataclass(frozen=True)
class PreparedScene:
    audio: PairedAudio
    candidates: tuple[Candidate, ...]
    raw_voice: tuple[SampleSpan, ...]
    stem_voice: tuple[SampleSpan, ...]
    singing: tuple[SampleSpan, ...]
    overlap: tuple[SampleSpan, ...]
    subtitle_hints: tuple[SampleSpan, ...]
    silence: SilenceRange

    def evidence(self, reference_digest: str, model_digest: str) -> tuple[LocalEvidence, ...]:
        source = self.audio.source
        return tuple(LocalEvidence(source.source_sha256, source.sample_rate, span, kind,
                                   reference_digest, model_digest, method)
                     for kind, spans, method in (
                         (EvidenceKind.SINGING, self.singing, "pre_uvr_singing"),
                         (EvidenceKind.OVERLAP, self.overlap, "independent_overlap"),
                     ) for span in spans)

    def gap_options(self, review: InferenceResult) -> tuple[AcousticGap, ...]:
        candidate = review.candidate
        if (candidate.source_sha256 != self.audio.source.source_sha256
                or candidate.sample_rate != self.audio.source.sample_rate):
            raise ValueError("Change proposal belongs to another source")
        # A subtitle edge is never itself a cut. The independent dual-VAD and
        # waveform check must find a real pause inside this candidate.
        return find_confirmed_gaps(
            candidate, _within(self.raw_voice, candidate.context),
            _within(self.stem_voice, candidate.context), self.audio,
            minimum_gap_samples=max(1, self.silence.lower_samples // 2),
        )


def make_scene(audio: PairedAudio, *, raw_voice: tuple[SampleSpan, ...],
               stem_voice: tuple[SampleSpan, ...],
               singing: tuple[SampleSpan, ...] = (),
               overlap: tuple[SampleSpan, ...] = (),
               subtitle_hints: tuple[SampleSpan, ...] = (),
               silence: SilenceRange | None = None,
               maximum_samples: int = 16 * 16000,
               context_samples: int = 6000) -> PreparedScene:
    """Build a recall-oriented lattice; this does not accept any voice.

    Stem VAD is primary, but raw-only speech gets an independent proposal so
    UVR damage cannot erase it before identity review. Hard singing/overlap
    masks are never converted into a joinable silence gap. All final decisions
    still require a calibrated identity session and endpoint review.
    """
    if audio.source.sample_rate != 16000:
        raise ValueError("Prepared scene needs the verified 16 kHz source timeline")
    silence = silence or SilenceRange(3200, 13600)
    raw_voice, stem_voice = _ordered(raw_voice, audio), _ordered(stem_voice, audio)
    singing, overlap = _ordered(singing, audio), _ordered(overlap, audio)
    subtitle_hints = _ordered(subtitle_hints, audio)
    if type(maximum_samples) is not int or maximum_samples < 1:
        raise ValueError("Invalid maximum candidate duration")
    # The unpooled TDNN has a wide receptive field. Enough real surrounding
    # audio is required for every output sample to receive model evidence.
    if type(context_samples) is not int or context_samples < 4000:
        raise ValueError("Candidate context must contain real TDNN edge support")
    blocked = _ordered((*singing, *overlap), audio)
    candidates: dict[tuple[SampleSpan, tuple[SampleSpan, ...]], Candidate] = {}
    for origin, voice in (("stem_vad", stem_voice), ("raw_vad_rescue", raw_voice)):
        if not voice:
            continue
        proposed = propose(
            audio.source, _unmasked_islands(voice, blocked), silence,
            maximum_samples=maximum_samples, context_samples=context_samples,
            blocked=blocked,
        )
        for item in proposed:
            # Keep stem provenance when both views propose exactly the same
            # audio; otherwise raw-only alternatives remain available.
            key = item.output, item.speech
            if key in candidates:
                previous = candidates[key]
                start_complete = previous.start_complete and item.start_complete
                end_complete = previous.end_complete and item.end_complete
                if ((previous.start_complete, previous.end_complete)
                        != (item.start_complete, item.end_complete)):
                    candidates[key] = replace(
                        previous, start_complete=start_complete, end_complete=end_complete,
                        origin=previous.origin + "+cross_view_boundary_conflict",
                    )
                continue
            # Aligned subtitle time is provenance for a VAD-created
            # proposal, never a speaker label or an acoustic cut.
            hinted = any((part := item.output.intersection(hint)) is not None
                         and part.length >= min(3200, item.output.length // 4)
                         for hint in subtitle_hints)
            candidates[key] = Candidate(
                item.source_sha256, item.sample_rate, item.output,
                item.context, item.speech,
                origin + ("+subtitle_time_hint" if hinted else ""),
                start_complete=item.start_complete,
                end_complete=item.end_complete,
            )
    return PreparedScene(audio, tuple(sorted(candidates.values(),
                                             key=lambda row: (row.output.start, row.output.end, row.origin))),
                         raw_voice, stem_voice, singing, overlap, subtitle_hints, silence)
