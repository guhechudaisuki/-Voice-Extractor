from __future__ import annotations

import gc
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import soundfile as sf
import torch
import torchaudio

from .config import FFMPEG, FFPROBE, UVR_MODEL, UVR_ROOT
from .types import TimeSpan
from .stem_stitching import write_stitched_stem


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"音频命令执行失败：{detail[-2000:]}")
    return result


def probe_duration(path: Path) -> float:
    try:
        info = sf.info(str(path))
        return info.frames / info.samplerate
    except (OSError, RuntimeError):
        pass
    result = _run(
        [
            str(FFPROBE),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def has_video_stream(path: Path) -> bool:
    """Return whether a media file contains a video stream."""

    result = _run(
        [
            str(FFPROBE),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(path),
        ]
    )
    return bool(result.stdout.strip())


def normalize_audio(source: Path, destination: Path, sample_rate: int = 44100, stereo: bool = True) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    channels = "2" if stereo else "1"
    _run(
        [
            str(FFMPEG),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            channels,
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]
    )
    return destination


def extract_audio_range(source: Path, destination: Path, start: float, end: float) -> Path:
    """Decode one bounded range for long-audio processing."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not math.isfinite(start) or not math.isfinite(end) or end <= max(0., start):
        raise ValueError("Invalid audio range")
    try:
        handle = sf.SoundFile(str(source))
    except (OSError, RuntimeError):
        handle = None
    if handle is not None:
        with handle:
            rate = handle.samplerate
            first = max(0, round(start * rate))
            last = min(len(handle), round(end * rate))
            if last <= first:
                raise ValueError("Audio range exceeds recording")
            handle.seek(first)
            data = handle.read(last - first, dtype="float32", always_2d=True)
        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        elif data.shape[1] > 2:
            data = np.repeat(data.mean(axis=1, keepdims=True), 2, axis=1)
        if rate != 44100:
            data = torchaudio.functional.resample(torch.from_numpy(data.T), rate, 44100).T.numpy()
        sf.write(str(destination), data, 44100, subtype="PCM_16")
        return destination
    _run(
        [
            str(FFMPEG),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{max(0.0, start):.3f}",
            "-i",
            str(source),
            "-t",
            f"{max(0.01, end - start):.3f}",
            "-vn",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]
    )
    return destination


def pad_for_separator(path: Path, minimum_seconds: float = 6.0) -> float:
    """Pad very short files so UVR's multi-band inverse STFT has valid edges."""
    info = sf.info(str(path))
    if info.frames >= int(round(minimum_seconds * info.samplerate)):
        return info.frames / info.samplerate
    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    original_duration = len(data) / sr
    minimum_frames = int(round(minimum_seconds * sr))
    if len(data) >= minimum_frames:
        return original_duration
    padding = np.zeros((minimum_frames - len(data), data.shape[1]), dtype=np.float32)
    sf.write(str(path), np.concatenate([data, padding], axis=0), sr, subtype="PCM_16")
    return original_duration


def trim_audio_in_place(path: Path, duration: float) -> Path:
    info = sf.info(str(path))
    if info.subtype == 'PCM_16' and 0 < info.frames <= int(round(duration * info.samplerate)):
        return path
    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    frames = max(1, min(len(data), int(round(duration * sr))))
    sf.write(str(path), data[:frames], sr, subtype="PCM_16")
    return path


def mute_spans(
    source: Path,
    destination: Path,
    spans: Iterable[TimeSpan],
    fade_seconds: float = 0.02,
) -> Path:
    """Silence selected ranges without changing the audio timeline."""

    data, sample_rate = sf.read(str(source), always_2d=True, dtype="float32")
    total_frames = len(data)
    fade_frames = max(0, int(round(fade_seconds * sample_rate)))
    for span in sorted(spans, key=lambda item: (item.start, item.end)):
        begin = max(0, min(total_frames, int(round(span.start * sample_rate))))
        end = max(begin, min(total_frames, int(round(span.end * sample_rate))))
        if end <= begin:
            continue
        if fade_frames:
            left = max(0, begin - fade_frames)
            if begin > left:
                data[left:begin] *= np.linspace(1.0, 0.0, begin - left, endpoint=False)[:, None]
            right = min(total_frames, end + fade_frames)
            if right > end:
                data[end:right] *= np.linspace(0.0, 1.0, right - end, endpoint=False)[:, None]
        data[begin:end] = 0.0
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destination), data, sample_rate, subtype="PCM_16")
    return destination


def load_mono(path: Path, sample_rate: int = 16000) -> torch.Tensor:
    waveform, source_rate = torchaudio.load(str(path))
    waveform = waveform.mean(dim=0, keepdim=True)
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    return waveform[0].contiguous().float()


def is_effectively_silent(path: Path, threshold_db: float = -55.0) -> bool:
    wav = load_mono(path, 16000)
    if wav.numel() == 0:
        return True
    rms = torch.sqrt(torch.mean(wav.square()) + 1e-12)
    db = 20.0 * torch.log10(rms + 1e-12)
    return float(db) < threshold_db


def write_clip(
    source: Path,
    destination: Path,
    start: float,
    end: float,
    sample_rate: int | None = None,
    normalize_level: bool = False,
    target_rms_db: float = -20.0,
    max_gain_db: float = 18.0,
    peak_db: float = -1.0,
) -> Path:
    with sf.SoundFile(str(source), mode="r") as handle:
        sr = handle.samplerate
        begin = max(0, int(round(start * sr)))
        finish = min(len(handle), int(round(end * sr)))
        if finish <= begin:
            raise ValueError(f"无效裁切范围：{start:.3f}-{end:.3f}")
        handle.seek(begin)
        data = handle.read(finish - begin, dtype="float32", always_2d=True)
    clip = data
    if sample_rate is not None and sample_rate != sr:
        tensor = torch.from_numpy(clip.T)
        tensor = torchaudio.functional.resample(tensor, sr, sample_rate)
        clip = tensor.T.numpy()
        sr = sample_rate
    if normalize_level:
        # Training clips benefit from a consistent floor, but normal clips
        # should not be attenuated. Cap gain by both a conservative maximum and
        # the available peak headroom to avoid amplifying separator artifacts.
        finite = np.nan_to_num(clip.astype(np.float32, copy=False))
        rms = float(np.sqrt(np.mean(np.square(finite), dtype=np.float64)))
        if rms > 1e-5:
            current_db = 20.0 * math.log10(max(rms, 1e-8))
            gain_db = min(max_gain_db, max(0.0, target_rms_db - current_db))
            peak = float(np.max(np.abs(finite)))
            if peak > 1e-8:
                peak_headroom_db = peak_db - 20.0 * math.log10(peak)
                gain_db = min(gain_db, max(0.0, peak_headroom_db))
            if gain_db > 0.0:
                finite = finite * (10.0 ** (gain_db / 20.0))
            clip = np.clip(finite, -1.0, 1.0)
    # 15ms 半余弦淡入淡出：消除裁切边界的截断爆音（De-click）。
    fade = max(1, int(round(0.015 * sr)))
    if clip.shape[0] > 2 * fade:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, fade, dtype=np.float32))
        clip[:fade] *= ramp[:, None]
        clip[-fade:] *= ramp[::-1][:, None]
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destination), clip, sr, subtype="PCM_16")
    return destination


def write_video_clip(
    source: Path,
    destination: Path,
    start: float,
    end: float,
) -> Path:
    """Render an accurately bounded clip from a source video."""

    duration = max(0.01, end - start)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            str(FFMPEG),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-ss",
            f"{max(0.0, start):.3f}",
            "-t",
            f"{duration:.3f}",
            "-map",
            "0:v:0?",
            "-map",
            "0:a:0?",
            "-sn",
            "-dn",
            "-c:v",
            "mpeg4",
            "-q:v",
            "3",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-shortest",
            str(destination),
        ]
    )
    return destination


def speech_ratio(spans: Iterable[TimeSpan], start: float, end: float) -> float:
    duration = max(1e-6, end - start)
    covered = 0.0
    for span in spans:
        covered += max(0.0, min(end, span.end) - max(start, span.start))
    return min(1.0, covered / duration)


class UVR5Separator:
    """Loads GPT-SoVITS' HP2 model without modifying its installation."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = device if device == "cuda" and torch.cuda.is_available() else "cpu"

    def separate(self, source: Path, output: Path, progress: Callable[[float, str], None] | None = None) -> Path:
        return self.separate_many([(source, output)], progress=progress)[0]

    @staticmethod
    def _chunk_count(duration: float, chunk_seconds: float, overlap_seconds: float) -> int:
        if duration <= chunk_seconds:
            return 1
        step = chunk_seconds - overlap_seconds
        return 1 + int(math.ceil((duration - chunk_seconds) / step))

    def separate_many(
        self,
        items: list[tuple[Path, Path]],
        progress: Callable[[float, str], None] | None = None,
        chunk_seconds: float = 60.0,
        overlap_seconds: float = 2.0,
    ) -> list[Path]:
        if not items:
            return []
        if chunk_seconds <= 0:
            raise ValueError("UVR 分块时长必须大于 0 秒")
        if overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
            raise ValueError("UVR 重叠时长必须大于等于 0 且小于分块时长")
        progress = progress or (lambda _value, _message: None)
        for source, _ in items:
            if is_effectively_silent(source):
                raise ValueError(f"音频没有可分析的有效信号：{source.name}")
        uvr_path = str(UVR_ROOT)
        if uvr_path not in sys.path:
            sys.path.insert(0, uvr_path)
        from .model_pool import MODEL_POOL, resource_key
        from .uvr_memory import MemoryUVR
        progress(0.0, "UVR 分离：正在准备常驻模型")
        engine = MODEL_POOL.get((resource_key('uvr-hp2', UVR_MODEL), self.device),
            lambda: MemoryUVR(UVR_MODEL, self.device), device=self.device)
        durations = [probe_duration(source) for source, _ in items]
        total = sum(self._chunk_count(value, chunk_seconds, overlap_seconds) for value in durations)
        completed = 0
        results = []
        for item_index, ((source, output), duration) in enumerate(zip(items, durations), 1):
            count = self._chunk_count(duration, chunk_seconds, overlap_seconds)
            step = chunk_seconds if count == 1 else chunk_seconds - overlap_seconds
            def chunks():
                nonlocal completed
                with sf.SoundFile(str(source)) as stream:
                    for index in range(count):
                        start, end = index * step, min(duration, index * step + chunk_seconds)
                        stream.seek(round(start * stream.samplerate))
                        samples = stream.read(round(end * stream.samplerate) - round(start * stream.samplerate),
                                              dtype='float32', always_2d=True)
                        minimum = 6 * stream.samplerate
                        if len(samples) < minimum:
                            samples = np.pad(samples, ((0, minimum - len(samples)), (0, 0)))
                        progress(completed / total, f"UVR 分离：文件 {item_index}/{len(items)}，块 {index+1}/{count}")
                        stem, rate = engine.separate(samples, stream.samplerate)
                        expected = round((end - start) * rate)
                        stem = stem[:expected]
                        if len(stem) < expected:
                            stem = np.pad(stem, ((0, expected - len(stem)), (0, 0)))
                        completed += 1
                        progress(completed / total, "UVR 分离：分块完成")
                        yield start, stem, rate
            write_stitched_stem(chunks(), output, duration, [i * step for i in range(count)])
            results.append(output)
        return results
