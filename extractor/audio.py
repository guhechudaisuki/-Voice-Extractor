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
    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    original_duration = len(data) / sr
    minimum_frames = int(round(minimum_seconds * sr))
    if len(data) >= minimum_frames:
        return original_duration
    padding = np.zeros((minimum_frames - len(data), data.shape[1]), dtype=np.float32)
    sf.write(str(path), np.concatenate([data, padding], axis=0), sr, subtype="PCM_16")
    return original_duration


def trim_audio_in_place(path: Path, duration: float) -> Path:
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
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destination), clip, sr, subtype="PCM_16")
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
        from vr import AudioPre

        progress(0.0, "UVR 分离：正在加载模型")
        pre = AudioPre(
            agg=10,
            model_path=str(UVR_MODEL),
            device=torch.device(self.device),
            is_half=self.device == "cuda",
            tta=False,
        )
        durations = [probe_duration(source) for source, _ in items]
        total_chunks = sum(self._chunk_count(value, chunk_seconds, overlap_seconds) for value in durations)
        completed_chunks = 0
        results: list[Path] = []
        active_chunk_roots: list[Path] = []
        try:
            for item_index, ((source, output), duration) in enumerate(zip(items, durations), start=1):
                output.parent.mkdir(parents=True, exist_ok=True)
                chunk_count = self._chunk_count(duration, chunk_seconds, overlap_seconds)
                step = chunk_seconds if chunk_count == 1 else chunk_seconds - overlap_seconds
                chunk_root = output.parent / f".chunks_{item_index:03d}"
                shutil.rmtree(chunk_root, ignore_errors=True)
                chunk_root.mkdir(parents=True, exist_ok=True)
                active_chunk_roots.append(chunk_root)
                pieces: list[tuple[float, np.ndarray, int]] = []
                for chunk_index in range(chunk_count):
                    start = chunk_index * step
                    end = min(duration, start + chunk_seconds)
                    if end <= start:
                        continue
                    label = f"文件 {item_index}/{len(items)}，块 {chunk_index + 1}/{chunk_count}"
                    base_progress = completed_chunks / max(1, total_chunks)
                    progress(base_progress, f"UVR 分离：{label}，正在准备")
                    source_chunk = chunk_root / f"input_{chunk_index:04d}.wav"
                    extract_audio_range(source, source_chunk, start, end)
                    actual_duration = pad_for_separator(source_chunk, minimum_seconds=6.0)
                    progress(
                        (completed_chunks + 0.05) / max(1, total_chunks),
                        f"UVR 分离：{label}，已解码，开始运行模型",
                    )
                    generated = chunk_root / f"vocal_{source_chunk.name}_10.wav"
                    pre._path_audio_(
                        str(source_chunk),
                        ins_root=None,
                        vocal_root=str(chunk_root),
                        format="wav",
                        is_hp3=False,
                    )
                    progress(
                        (completed_chunks + 0.90) / max(1, total_chunks),
                        f"UVR 分离：{label}，模型完成，正在整理",
                    )
                    if not generated.exists():
                        raise RuntimeError(f"UVR5 未生成预期文件：{generated}")
                    stem_data, stem_rate = sf.read(str(generated), always_2d=True, dtype="float32")
                    if not np.isfinite(stem_data).all():
                        raise RuntimeError(f"UVR5 输出包含 NaN/Inf：{source.name}")
                    expected_frames = max(1, int(round(actual_duration * stem_rate)))
                    stem_data = np.nan_to_num(stem_data[:expected_frames])
                    if len(stem_data) < expected_frames:
                        stem_data = np.pad(stem_data, ((0, expected_frames - len(stem_data)), (0, 0)))
                    pieces.append((start, stem_data, stem_rate))
                    completed_chunks += 1
                    progress(
                        completed_chunks / max(1, total_chunks),
                        f"UVR 分离：{label}，处理完成",
                    )

                if not pieces:
                    raise RuntimeError(f"UVR5 未产生有效分块：{source.name}")
                sample_rate = pieces[0][2]
                total_frames = max(1, int(round(duration * sample_rate)))
                channels = pieces[0][1].shape[1]
                mixed = np.zeros((total_frames, channels), dtype=np.float64)
                weights = np.zeros((total_frames, 1), dtype=np.float64)
                placements: list[tuple[int, int, np.ndarray]] = []
                for start, piece, piece_rate in pieces:
                    if piece_rate != sample_rate:
                        raise RuntimeError("UVR5 分块采样率不一致")
                    if piece.shape[1] != channels:
                        raise RuntimeError("UVR5 分块声道数不一致")
                    begin = max(0, int(round(start * sample_rate)))
                    finish = min(total_frames, begin + len(piece))
                    if finish <= begin:
                        continue
                    length = finish - begin
                    placements.append((begin, finish, piece[:length]))

                for piece_index, (begin, finish, piece) in enumerate(placements):
                    length = finish - begin
                    blend = np.ones((length, 1), dtype=np.float64)
                    if piece_index > 0:
                        previous_finish = placements[piece_index - 1][1]
                        fade_frames = min(length, max(0, previous_finish - begin))
                        if fade_frames:
                            blend[:fade_frames, 0] *= (
                                np.arange(1, fade_frames + 1, dtype=np.float64) / (fade_frames + 1)
                            )
                    if piece_index + 1 < len(placements):
                        next_begin = placements[piece_index + 1][0]
                        fade_frames = min(length, max(0, finish - next_begin))
                        if fade_frames:
                            blend[-fade_frames:, 0] *= (
                                np.arange(fade_frames, 0, -1, dtype=np.float64) / (fade_frames + 1)
                            )
                    mixed[begin:finish] += piece * blend
                    weights[begin:finish] += blend
                if np.any(weights <= 0.0):
                    raise RuntimeError("UVR5 分块拼接出现未覆盖的音频区间")
                mixed = (mixed / weights).astype(np.float32)
                sf.write(str(output), mixed, sample_rate, subtype="PCM_16")
                shutil.rmtree(chunk_root, ignore_errors=True)
                active_chunk_roots.remove(chunk_root)
                results.append(output)
                progress(
                    completed_chunks / max(1, total_chunks),
                    f"UVR 分离：文件 {item_index}/{len(items)}，分块拼接完成",
                )
            return results
        finally:
            for chunk_root in active_chunk_roots:
                shutil.rmtree(chunk_root, ignore_errors=True)
            del pre
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
