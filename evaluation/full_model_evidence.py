"""Read-only, interval-local evidence from three experimental speaker models.

These models are fallible.  Values returned here are *not* identity decisions and
must not by themselves certify audio for training or export.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_samples(start: float, end: float, frames: int, rate: int) -> tuple[int, int]:
    if not all(math.isfinite(value) for value in (start, end)):
        raise ValueError("Interval boundaries must be finite")
    if not (0 <= start < end <= frames / rate + 1e-7):
        raise ValueError(f"Interval {start}–{end} is outside 0–{frames / rate}s")
    left = round(start * rate)
    right = min(round(end * rate), frames)
    if left >= right:
        raise ValueError("Interval contains no samples")
    return left, right


def _read_first_channel(audio: sf.SoundFile, left: int, right: int) -> np.ndarray:
    audio.seek(left)
    values = audio.read(right - left, dtype="float32", always_2d=True)
    if len(values) != right - left:
        raise IOError("Audio file ended during interval read")
    # Match the existing probe's first-channel convention, never average stereo.
    return values[:, 0]


def _read_source_at_output_rate(
    audio: sf.SoundFile, left: int, right: int, output_rate: int,
) -> np.ndarray:
    """Resample a bounded source window on the whole-file resampler's phase.

    The probe resamples channel zero from time zero. Starting on a reduced
    source-rate period preserves that phase; 100 ms context protects the sinc
    filter at both cut points. Only the selected output samples are retained.
    """
    if audio.samplerate == output_rate:
        return _read_first_channel(audio, left, right)
    import torch
    import torchaudio

    source_rate = audio.samplerate
    period = source_rate // math.gcd(source_rate, output_rate)
    margin = math.ceil(source_rate * 0.1)
    input_left = max(0, math.floor((left * source_rate / output_rate - margin) / period) * period)
    input_right = min(
        len(audio), math.ceil(right * source_rate / output_rate) + margin,
    )
    original = _read_first_channel(audio, input_left, input_right)
    converted = torchaudio.functional.resample(
        torch.from_numpy(original.copy()), source_rate, output_rate,
    ).numpy()
    offset = input_left * output_rate // source_rate
    begin, finish = left - offset, right - offset
    if begin < 0 or finish > len(converted):
        raise IOError("Resampled source window did not cover the requested interval")
    return converted[begin:finish]


def _waveform_metrics(source: np.ndarray, output: np.ndarray) -> dict:
    if source.shape != output.shape or not source.size:
        raise ValueError("Evidence waveforms must have equal nonzero lengths")
    if not (np.isfinite(source).all() and np.isfinite(output).all()):
        raise ValueError("Model evidence contains non-finite audio")
    x = source.astype(np.float64, copy=False)
    y = output.astype(np.float64, copy=False)
    input_power = float(np.dot(x, x))
    output_power = float(np.dot(y, y))
    input_rms = math.sqrt(input_power / len(x))
    output_rms = math.sqrt(output_power / len(y))
    return {
        "sample_count": len(x),
        "input_rms": input_rms,
        "output_rms": output_rms,
        "output_over_input_db": 10 * math.log10(
            max(output_power, 1e-20) / max(input_power, 1e-20)
        ),
        "waveform_cosine": (
            float(np.dot(x, y) / math.sqrt(input_power * output_power))
            if input_power > 0 and output_power > 0 else None
        ),
    }


class PersonalVADCoverage:
    """Measure FSM target coverage after checking the report's source identity."""

    def __init__(self, report_path: Path, source_path: Path):
        self.report_path = Path(report_path)
        self.source_path = Path(source_path)
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        episode = report["episode"]
        expected_hash = episode["source_sha256"]
        self.source_sha256 = file_sha256(self.source_path)
        if self.source_sha256 != expected_hash:
            raise ValueError("Personal VAD report does not match the supplied source WAV")
        with sf.SoundFile(self.source_path) as audio:
            self.duration_seconds = len(audio) / audio.samplerate
        if abs(self.duration_seconds - float(episode["duration_seconds"])) > 0.001:
            raise ValueError("Personal VAD report has a different source duration")
        self.spans = sorted(
            (float(start), float(end)) for start, end in episode["target_fsm_segments"]
        )
        if any(start < 0 or start >= end or end > self.duration_seconds + 0.001
               for start, end in self.spans):
            raise ValueError("Personal VAD report contains an invalid FSM interval")
        if any(self.spans[index][1] > self.spans[index + 1][0]
               for index in range(len(self.spans) - 1)):
            raise ValueError("Personal VAD FSM intervals overlap")

    @classmethod
    def from_report(cls, report_path: Path, source_path: Path) -> PersonalVADCoverage:
        return cls(report_path, source_path)

    def evidence(self, start: float, end: float) -> dict:
        if not (math.isfinite(start) and math.isfinite(end)
                and 0 <= start < end <= self.duration_seconds + 1e-7):
            raise ValueError("Personal VAD interval is outside the source")
        overlap = sum(
            max(0.0, min(end, span_end) - max(start, span_start))
            for span_start, span_end in self.spans
        )
        return {
            "model": "personal_vad_2_fsm",
            "start": start,
            "end": end,
            "target_coverage_fraction": min(1.0, overlap / (end - start)),
            "target_overlap_seconds": overlap,
            "source_sha256": self.source_sha256,
            "decision": None,
        }


class WaveformComparison:
    """Stream exact intervals from source and globally normalized model WAVs."""

    def __init__(self, source_wav: Path, output_wav: Path, report_path: Path | None = None):
        self.source_wav = Path(source_wav)
        self.output_wav = Path(output_wav)
        self.report = None
        self.source_sha256 = file_sha256(self.source_wav)
        with sf.SoundFile(self.source_wav) as source, sf.SoundFile(self.output_wav) as output:
            if abs(len(source) / source.samplerate - len(output) / output.samplerate) > 1 / output.samplerate:
                raise ValueError("Source and model WAVs are not time-aligned")
            self.rate = source.samplerate
            self.output_rate = output.samplerate
            self.frames = len(output)
        if report_path is not None:
            self.report = json.loads(Path(report_path).read_text(encoding="utf-8"))
            if Path(self.report["source"]).resolve() != self.source_wav.resolve():
                raise ValueError("VoiceFilter report points to a different source WAV")
            if ("source_sha256" in self.report
                    and self.report["source_sha256"] != self.source_sha256):
                raise ValueError("VoiceFilter report source SHA-256 disagrees with WAV")
            if abs(self.frames / self.output_rate - float(self.report["duration_seconds"])) > 0.001:
                raise ValueError("VoiceFilter report duration disagrees with WAVs")
            if self.output_rate != int(self.report["sample_rate"]):
                raise ValueError("VoiceFilter report sample rate disagrees with WAVs")

    def evidence(self, start: float, end: float) -> dict:
        left, right = _checked_samples(start, end, self.frames, self.output_rate)
        with sf.SoundFile(self.source_wav) as source, sf.SoundFile(self.output_wav) as output:
            input_audio = _read_source_at_output_rate(
                source, left, right, self.output_rate,
            )
            output_audio = _read_first_channel(output, left, right)
        metrics = _waveform_metrics(input_audio, output_audio)
        playback_db = metrics.pop("output_over_input_db")
        result = {
            "model": "voicefilter_full_episode",
            "start": start,
            "end": end,
            "sample_rate": self.output_rate,
            "source_sample_rate": self.rate,
            "source_sha256": self.source_sha256,
            **metrics,
            "output_over_input_db_playback": playback_db,
            "gain_caveat": "Output WAV has global playback normalization; compare intervals only within the same run.",
            "decision": None,
        }
        if self.report is not None:
            # Probe pre-scales input by source_peak, then writes output after
            # playback_global_gain.  Undo both constants for the probe's gain.
            playback_gain = float(self.report["playback_global_gain"])
            source_peak = float(self.report["source_peak"])
            if playback_gain <= 0 or source_peak <= 0:
                raise ValueError("Invalid VoiceFilter report normalization constants")
            result["output_over_input_db_model_estimate"] = (
                playback_db - 20 * math.log10(playback_gain)
                + 20 * math.log10(source_peak)
            )
        return result


class SpExPlusEvaluator:
    """Run the local SpEx+ probe on one exact 8 kHz source interval."""

    def __init__(self, source_wav: Path, model, reference, device, checkpoint_path: Path | None = None):
        self.source_wav = Path(source_wav)
        self.model = model
        self.reference = reference
        self.device = device
        self.checkpoint_path = checkpoint_path
        with sf.SoundFile(self.source_wav) as audio:
            if audio.samplerate != 8000:
                raise ValueError("SpEx+ source must be an 8 kHz WAV")
            self.frames = len(audio)

    @classmethod
    def from_paths(cls, source_wav: Path, reference_wav: Path, device: str = "cuda") -> SpExPlusEvaluator:
        import torch
        from evaluation.probe_spexplus_local import load_audio, load_model

        torch_device = torch.device(device)
        if torch_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable for SpEx+; pass device='cpu'")
        model, checkpoint = load_model(torch_device)
        reference = load_audio(Path(reference_wav)).to(torch_device)
        return cls(source_wav, model, reference, torch_device, checkpoint)

    def evidence(self, start: float, end: float) -> dict:
        import torch
        from torch.nn import functional as F

        left, right = _checked_samples(start, end, self.frames, 8000)
        chunk_frames = 4 * 8000
        input_power = output_power = cross_power = 0.0
        peak_output = 0.0
        chunk_count = 0
        reference_length = torch.tensor([self.reference.numel()], device=self.device)
        with sf.SoundFile(self.source_wav) as audio:
            for chunk_left in range(left, right, chunk_frames):
                chunk_right = min(chunk_left + chunk_frames, right)
                source_np = _read_first_channel(audio, chunk_left, chunk_right)
                original = torch.from_numpy(source_np)
                model_input = original if original.numel() >= 800 else F.pad(
                    original, (0, 800 - original.numel())
                )
                with torch.inference_mode():
                    outputs = self.model(
                        model_input.unsqueeze(0).to(self.device),
                        self.reference.unsqueeze(0),
                        reference_length,
                    )
                    extracted = outputs[0][0].detach().float().cpu()
                if extracted.numel() < original.numel():
                    raise ValueError("SpEx+ output is shorter than a requested source chunk")
                output_np = extracted[:original.numel()].numpy()
                if not np.isfinite(output_np).all():
                    raise ValueError("SpEx+ returned non-finite output")
                source64 = source_np.astype(np.float64, copy=False)
                output64 = output_np.astype(np.float64, copy=False)
                input_power += float(np.dot(source64, source64))
                output_power += float(np.dot(output64, output64))
                cross_power += float(np.dot(source64, output64))
                peak_output = max(peak_output, float(np.max(np.abs(output_np))))
                chunk_count += 1
        sample_count = right - left
        metrics = {
            "sample_count": sample_count,
            "input_rms": math.sqrt(input_power / sample_count),
            "output_rms": math.sqrt(output_power / sample_count),
            "output_over_input_db": 10 * math.log10(
                max(output_power, 1e-20) / max(input_power, 1e-20)
            ),
            "waveform_cosine": (
                cross_power / math.sqrt(input_power * output_power)
                if input_power > 0 and output_power > 0 else None
            ),
        }
        return {
            "model": "spexplus",
            "analysis_mode": "independent_exact_candidate_interval",
            "chunk_seconds": 4.0,
            "chunk_count": chunk_count,
            "chunk_caveat": "Independent candidate-local chunks; model context resets every 4 seconds.",
            "start": start,
            "end": end,
            "sample_rate": 8000,
            **metrics,
            "extreme_output_amplitude": peak_output > 1.0,
            "amplitude_caveat": "Research checkpoint output may have extreme scale; not product audio.",
            "checkpoint": str(self.checkpoint_path) if self.checkpoint_path else None,
            "decision": None,
        }
