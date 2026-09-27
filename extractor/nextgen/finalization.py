"""Final source-span audit, separate export length policy and STT-last delivery.

This module never changes audio boundaries to satisfy STT or minimum duration.
STT errors retain acoustically accepted audio in a separate review area, not in
the ready-to-train ZIP. One ZIP contains all successful clips in the batch.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable
import zipfile

import numpy as np
import soundfile as sf

from .boundary_decoder import SilenceRange
from .decision_policy import Assessment
from .ledger import Candidate, State
from .timeline import SampleSpan, ViewAlignment


class Cancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class ExportPolicy:
    minimum_samples: int  # explicit choice, not an identity threshold
    target_rms_db: float = -23.0
    maximum_gain_db: float = 12.0
    peak_ceiling_db: float = -1.0

    def __post_init__(self) -> None:
        if type(self.minimum_samples) is not int or self.minimum_samples < 0:
            raise ValueError("Invalid output-length policy")
        if (not all(np.isfinite(v) for v in (self.target_rms_db, self.maximum_gain_db, self.peak_ceiling_db))
                or self.maximum_gain_db < 0 or not self.target_rms_db < self.peak_ceiling_db <= 0):
            raise ValueError("Invalid normalization policy")


def final_audit(candidate: Candidate, assessment: Assessment, silence: SilenceRange,
                blocked: tuple[SampleSpan, ...] = ()) -> None:
    if assessment.candidate_key != candidate.key or assessment.state != State.ACCEPTED:
        raise ValueError("Final exported interval has not been acoustically accepted")
    if not candidate.start_complete or not candidate.end_complete:
        raise ValueError("Incomplete acoustic endpoints")
    if any(silence.kind(b.start - a.end) == "hard_split" for a, b in zip(candidate.speech, candidate.speech[1:])):
        raise ValueError("Output crosses the user-defined hard silence boundary")
    if any(candidate.output.intersection(span) for span in blocked):
        raise ValueError("Final output crosses a forbidden event")


def normalize_clip(samples: np.ndarray, speech_mask: np.ndarray, policy: ExportPolicy):
    if (samples.ndim not in (1, 2) or samples.shape[0] == 0 or speech_mask.dtype != np.bool_
            or speech_mask.shape != samples.shape[:1] or not speech_mask.any()
            or not np.isfinite(samples).all()):
        raise ValueError("Invalid samples or speech mask for normalization")
    rms = float(np.sqrt(np.mean(np.square(samples[speech_mask].astype(np.float64)))))
    peak = float(np.max(np.abs(samples)))
    if rms <= 1e-10 or peak <= 1e-10:
        raise ValueError("Accepted segment contains no measurable speech")
    desired = 10 ** (policy.target_rms_db / 20) / rms
    # Only raise quiet audio, except attenuating an over-ceiling input.
    gain = min(max(1.0, desired), 10 ** (policy.maximum_gain_db / 20),
               10 ** (policy.peak_ceiling_db / 20) / peak)
    return (samples * gain).astype(np.float32), float(20 * np.log10(gain))


@dataclass(frozen=True)
class ExportItem:
    candidate: Candidate
    assessment: Assessment
    stem_path: Path
    alignment: ViewAlignment
    silence: SilenceRange
    blocked: tuple[SampleSpan, ...] = ()


def deliver(items: tuple[ExportItem, ...], destination: Path, policy: ExportPolicy,
            transcribe: Callable[[Path], str], *, cancelled: Callable[[], bool] = lambda: False,
            progress: Callable[[int, int, str], None] = lambda *args: None,
            video: Callable[[Candidate, Path], Path] | None = None) -> dict:
    if len({item.candidate.key for item in items}) != len(items):
        raise ValueError("Duplicate output candidate")
    for item in items:
        final_audit(item.candidate, item.assessment, item.silence, item.blocked)
        if (item.alignment.source.source_sha256 != item.candidate.source_sha256
                or item.alignment.source.sample_rate != item.candidate.sample_rate):
            raise ValueError("Export stem refers to another source")
        item.alignment.from_source(item.candidate.output)
    # Refuse overlap, even between differently named proposals of the same source.
    for i, item in enumerate(items):
        if any(item.candidate.source_sha256 == prior.candidate.source_sha256
               and item.candidate.output.intersection(prior.candidate.output) for prior in items[:i]):
            raise ValueError("Output candidates duplicate source audio")
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    ready, review = destination / "ready", destination / "needs_text_review"
    ready.mkdir()
    review.mkdir()
    rows = []
    report = {"schema": 1, "state": "running", "records": rows, "zip": None, "zip_error": None}

    def save() -> None:
        temporary = destination / "manifest.json.tmp"
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(destination / "manifest.json")

    try:
        save()
        for number, item in enumerate(items, 1):
            if cancelled():
                raise Cancelled("Export cancelled at a safe clip boundary")
            candidate = item.candidate
            progress(number - 1, len(items), "export_audio")
            record = {"candidate_key": candidate.key, "source_sha256": candidate.source_sha256,
                      "sample_rate": candidate.sample_rate, "span": asdict(candidate.output),
                      "policy_version": item.assessment.policy_version, "stt_status": "not_started"}
            rows.append(record)
            if candidate.output.length < policy.minimum_samples:
                record["stt_status"] = "export_length_policy"
                save()
                continue
            interval = item.alignment.from_source(candidate.output)
            with sf.SoundFile(str(item.stem_path)) as stream:
                if stream.samplerate != item.alignment.view_rate or len(stream) != item.alignment.view_samples:
                    raise ValueError("Stem geometry changed since analysis")
                stream.seek(interval.start)
                samples = stream.read(interval.length, dtype="float32", always_2d=True)
            if len(samples) != interval.length:
                raise ValueError("Truncated source stem")
            mask = np.zeros(len(samples), dtype=bool)
            for speech in candidate.speech:
                mapped = item.alignment.from_source(speech).intersection(interval)
                if mapped is not None:
                    mask[mapped.start - interval.start:mapped.end - interval.start] = True
            samples, gain = normalize_clip(samples, mask, policy)
            basename = f"{number:04d}_{candidate.source_sha256[:12]}_{candidate.output.start}-{candidate.output.end}"
            audio = review / f"{basename}.wav"
            sf.write(str(audio), samples, item.alignment.view_rate, subtype="PCM_24")
            record.update(audio=str(audio.relative_to(destination)), gain_db=gain)
            save()  # Audio survives a process interruption or STT failure.
            if cancelled():
                raise Cancelled("Cancelled before STT; saved audio is retained")
            progress(number - 1, len(items), "stt")
            try:
                text = transcribe(audio)
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("STT returned no usable text")
            except Cancelled:
                raise
            except Exception as error:
                record.update(stt_status="failed", stt_error=f"{type(error).__name__}: {error}")
                save()
                continue
            transcript = review / f"{basename}.txt"
            transcript.write_text(text.strip() + "\n", encoding="utf-8")
            audio = audio.replace(ready / audio.name)
            transcript = transcript.replace(ready / transcript.name)
            record.update(stt_status="ok", audio=str(audio.relative_to(destination)),
                          text=str(transcript.relative_to(destination)))
            if video is not None:
                # Video writer receives frozen source coordinates, never ASR timestamps.
                try:
                    result = Path(video(candidate, ready / f"{basename}.mp4")).resolve()
                    if result.parent != ready.resolve() or not result.is_file():
                        raise ValueError("Video writer returned an invalid output file")
                    record["video"] = str(result.relative_to(destination))
                except Exception as error:
                    record["video_error"] = f"{type(error).__name__}: {error}"
            save()
        report["state"] = "completed"
    except Cancelled:
        report["state"] = "cancelled"
    except Exception:
        report["state"] = "failed"
        save()
        raise
    save()
    if report["state"] == "completed" and any(row["stt_status"] == "ok" for row in rows):
        temporary = destination / "VoiceExtractor.zip.partial"
        try:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                # Only declared successful artifacts, not files coincidentally in a directory.
                for row in rows:
                    if row["stt_status"] == "ok":
                        for field in ("audio", "text", "video"):
                            if field in row:
                                archive.write(destination / row[field], row[field])
                archive.write(destination / "manifest.json", "manifest.json")
            archive_path = temporary.replace(destination / "VoiceExtractor.zip")
            report["zip"] = str(archive_path)
        except Exception as error:
            report["zip_error"] = f"{type(error).__name__}: {error}"
    save()
    progress(len(rows), len(items), report["state"])
    return report
