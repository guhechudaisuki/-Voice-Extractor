"""Bounded real-media preparation for the independent engine.

This reuses the installed singing, UVR, VAD and overlap components without
changing the desktop result. It stops before identity inference: a scene is a
set of possibilities, not a claim that any clip belongs to the target.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import json
from pathlib import Path
from typing import Callable

import soundfile as sf

from ..audio import (UVR5Separator, mute_spans, normalize_audio,
                     pad_for_separator, probe_duration, trim_audio_in_place)
from ..filters import OverlapDetector, SingingDetector
from ..subtitle_assistance import retry_subtitle_vad
from ..subtitles import SubtitleGuide
from ..transcription import FunASRTools
from ..types import TimeSpan
from .boundary_decoder import SilenceRange
from .prepared_audio import AlignmentReport, PairedAudio, estimate_stem_delay
from .scene_adapter import PreparedScene, make_scene
from .timeline import SampleSpan


Progress = Callable[[str, float, str], None]


def _samples(spans: list[TimeSpan], total: int) -> tuple[SampleSpan, ...]:
    result = []
    for span in spans:
        start, end = round(span.start * 16000), round(span.end * 16000)
        if start < 0 or end > total + 160 or end <= start:
            raise ValueError("Detector interval is invalid or too far outside decoded audio")
        end = min(end, total)
        if start < end:
            result.append(SampleSpan(start, end))
    return tuple(result)


def _alignment_anchors(raw: tuple[SampleSpan, ...], stem: tuple[SampleSpan, ...],
                       singing: tuple[SampleSpan, ...], total: int) -> tuple[SampleSpan, ...]:
    """Propose independent voiced anchors; the correlation test decides validity."""
    options: list[SampleSpan] = []
    for left in raw:
        for right in stem:
            common = left.intersection(right)
            if common is None or common.length < 8000:
                continue
            cursor = max(common.start + 1600, 1601)
            while cursor + 5600 <= min(common.end - 1600, total - 1601):
                anchor = SampleSpan(cursor, cursor + 5600)
                if not any(anchor.intersection(mask) for mask in singing):
                    options.append(anchor)
                cursor += 8800
    options.sort()
    unique = []
    for anchor in options:
        if not unique or anchor.start >= unique[-1].end:
            unique.append(anchor)
    if len(unique) > 12:
        # Testing only the first few seconds would silently miss a late UVR
        # timing drift in a longer prepared scene.
        indexes = {round(i * (len(unique) - 1) / 11) for i in range(12)}
        unique = [unique[index] for index in sorted(indexes)]
    return tuple(unique)


def _verify_alignment(raw_path: Path, stem_path: Path,
                      anchors: tuple[SampleSpan, ...]):
    if len(anchors) < 2:
        raise ValueError("Two independent clean speech anchors are needed to verify UVR timing")
    if tuple(sorted(anchors)) != anchors or any(a.end > b.start for a, b in zip(anchors, anchors[1:])):
        raise ValueError("Alignment anchors must be ordered and disjoint")
    coverage = anchors[-1].end - anchors[0].start
    pairs = sorted((pair for pair in combinations(anchors, 2)
                    if pair[1].end - pair[0].start >= coverage * .7),
                   key=lambda pair: pair[1].end - pair[0].start, reverse=True)
    failure = None
    for first, second in pairs:
        middle = [anchor for anchor in anchors if first.end <= anchor.start
                  and anchor.end <= second.start]
        trials = ([(first, anchor, second) for anchor in middle]
                  if len(anchors) >= 3 else [(first, second)])
        for trial in trials:
            try:
                return estimate_stem_delay(raw_path, stem_path, trial,
                                           maximum_delay_samples=1600)
            except ValueError as error:
                failure = error
    raise ValueError("No stable raw/UVR speech alignment could be verified") from failure


@dataclass(frozen=True)
class MediaPreparation:
    scene: PreparedScene
    source_media: Path
    raw_16k: Path
    stem_16k: Path
    report_path: Path


def prepare_bounded_media(source: str | Path, destination: str | Path, *,
                          subtitle: str | Path | None = None,
                          device: str = "cuda", max_seconds: float = 90.0,
                          silence: SilenceRange | None = None,
                          progress: Progress | None = None) -> MediaPreparation:
    """Prepare a short scene for code/debug checks, without running identity.

    The destination must be new. A full episode is intentionally rejected by
    default so development cannot accidentally repeat a multi-hour run.
    Production may later call this per bounded chunk, with explicit overlap.
    """
    source = Path(source).resolve(strict=True)
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite a prepared scene: {destination}")
    duration = probe_duration(source)
    if not 0 < duration <= max_seconds:
        raise ValueError(f"Expected an audio/video scene no longer than {max_seconds:g} seconds")
    progress = progress or (lambda *_args: None)
    destination.mkdir(parents=True)
    normalized = normalize_audio(source, destination / "original_44100.wav")
    detector = SingingDetector("cpu")
    try:
        progress("singing", 0.0, "Scan human singing before UVR")
        _, pre_singing = detector.clean_spans(normalized, [TimeSpan(0.0, duration)])
        separator_input = (mute_spans(normalized, destination / "singing_muted_44100.wav",
                                      pre_singing) if pre_singing else normalized)
        progress("singing", 1.0, f"Human-singing masks: {len(pre_singing)}")
        original_duration = pad_for_separator(separator_input)
        stem = UVR5Separator(device).separate(separator_input, destination / "stem_44100.wav",
                                               progress=lambda value, message:
                                               progress("uvr", value, message))
        trim_audio_in_place(stem, original_duration)
        if separator_input == normalized:
            trim_audio_in_place(normalized, original_duration)
        else:
            trim_audio_in_place(separator_input, original_duration)
        raw_16k = normalize_audio(normalized, destination / "original_16000.wav",
                                  sample_rate=16000, stereo=False)
        stem_16k = normalize_audio(stem, destination / "stem_16000.wav",
                                   sample_rate=16000, stereo=False)
        source_frames = sf.info(raw_16k).frames
        vad = FunASRTools(device)
        progress("vad", 0.0, "Check original and separated speech activity")
        found = vad.vad_many([raw_16k, stem_16k])
        raw_voice, stem_voice = found[raw_16k], found[stem_16k]
        guide = SubtitleGuide.load(Path(subtitle)) if subtitle else None
        if guide is not None:
            guide.calibrate(stem_voice)
            if guide.aligned:
                stem_voice = retry_subtitle_vad(guide, stem_voice, duration,
                                                stem_16k, destination, vad,
                                                lambda value, message:
                                                progress("subtitle_vad", value, message))
        progress("vad", 1.0, f"Raw/stem voice islands: {len(raw_voice)}/{len(stem_voice)}")
        _, residual_singing = detector.clean_spans(stem_16k, stem_voice)
    finally:
        detector.close()
    overlap = []
    if stem_voice:
        progress("overlap", 0.0, "Check simultaneous voices")
        _, overlap = OverlapDetector().clean_spans(stem_16k, stem_voice)
        progress("overlap", 1.0, f"Overlap masks: {len(overlap)}")
    raw_samples = _samples(raw_voice, source_frames)
    stem_samples = _samples(stem_voice, source_frames)
    singing_samples = _samples([*pre_singing, *residual_singing], source_frames)
    anchors = _alignment_anchors(raw_samples, stem_samples, singing_samples, source_frames)
    alignment = _verify_alignment(raw_16k, stem_16k, anchors)
    paired = PairedAudio(raw_16k, stem_16k, alignment)
    hints = ([span for cue in guide.cues if (span := guide.cue_span(cue)) is not None
              and 0 <= span.start < span.end <= duration + 0.005]
             if guide is not None and guide.aligned else [])
    scene = make_scene(paired, raw_voice=raw_samples, stem_voice=stem_samples,
                       singing=singing_samples, overlap=_samples(overlap, source_frames),
                       subtitle_hints=_samples(hints, source_frames), silence=silence)
    report = {
        "schema": 1, "source_sha256": paired.source.source_sha256,
        "duration_seconds": duration, "singing_before_uvr": True,
        "raw_voice_count": len(scene.raw_voice), "stem_voice_count": len(scene.stem_voice),
        "singing_mask_count": len(scene.singing), "overlap_mask_count": len(scene.overlap),
        "subtitle_hint_count": len(scene.subtitle_hints),
        "candidate_count": len(scene.candidates),
        "silence": asdict(scene.silence),
        "raw_voice": [asdict(span) for span in scene.raw_voice],
        "stem_voice": [asdict(span) for span in scene.stem_voice],
        "singing_masks": [asdict(span) for span in scene.singing],
        "overlap_masks": [asdict(span) for span in scene.overlap],
        "subtitle_hints": [asdict(span) for span in scene.subtitle_hints],
        "candidates": [{"output": asdict(item.output), "origin": item.origin,
                        "start_complete": item.start_complete,
                        "end_complete": item.end_complete}
                       for item in scene.candidates],
        "subtitle": guide.report if guide is not None else None,
        "alignment": asdict(alignment),
        "note": "Candidates are unverified. No new speaker model has approved any output.",
    }
    report_path = destination / "preparation.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return MediaPreparation(scene, source, raw_16k, stem_16k, report_path)


def load_prepared_scene(directory: str | Path) -> PreparedScene:
    """Reuse a scene only if its audio digests, geometry and proposals agree."""
    directory = Path(directory).resolve(strict=True)
    report = json.loads((directory / "preparation.json").read_text(encoding="utf-8"))
    if report.get("schema") != 1 or report.get("singing_before_uvr") is not True:
        raise ValueError("Prepared scene has no verified preprocessing contract")
    alignment = report["alignment"]
    alignment_report = AlignmentReport(
        alignment["raw_digest"], alignment["stem_digest"],
        alignment["delay_samples"],
        tuple(SampleSpan(**row) for row in alignment["anchors"]),
        tuple(alignment["correlations"]), alignment["maximum_disagreement_samples"],
    )
    audio = PairedAudio(directory / "original_16000.wav", directory / "stem_16000.wav",
                        alignment_report)
    if audio.source.source_sha256 != report["source_sha256"]:
        raise ValueError("Prepared source digest changed")
    def spans(field: str) -> tuple[SampleSpan, ...]:
        return tuple(SampleSpan(**row) for row in report[field])

    scene = make_scene(
        audio, raw_voice=spans("raw_voice"), stem_voice=spans("stem_voice"),
        singing=spans("singing_masks"), overlap=spans("overlap_masks"),
        subtitle_hints=spans("subtitle_hints"),
        silence=SilenceRange(**report["silence"]),
    )
    documented = [(row["output"], row["origin"], row["start_complete"], row["end_complete"])
                  for row in report["candidates"]]
    actual = [(asdict(row.output), row.origin, row.start_complete, row.end_complete)
              for row in scene.candidates]
    if (len(scene.candidates) != report["candidate_count"] or actual != documented):
        raise ValueError("Prepared candidates no longer match the recorded preprocessing policy")
    return scene
