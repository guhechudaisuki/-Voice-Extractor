from __future__ import annotations

import csv
import gc
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

from .audio import load_mono
from .config import OVERLAP_MODEL, PANNS_LABELS, PANNS_MODEL, VENDOR_ROOT
from .types import TimeSpan
from .model_pool import is_memory_error


def _merge_spans(spans: list[TimeSpan], gap: float = 0.08) -> list[TimeSpan]:
    if not spans:
        return []
    output = [min(spans, key=lambda item: item.start)]
    for span in sorted(spans, key=lambda item: (item.start, item.end))[1:]:
        previous = output[-1]
        if span.start <= previous.end + gap:
            output[-1] = TimeSpan(previous.start, max(previous.end, span.end))
        else:
            output.append(span)
    return output


def _subtract_spans(
    source: list[TimeSpan],
    removed: list[TimeSpan],
    minimum_seconds: float = 0.30,
) -> list[TimeSpan]:
    removed = _merge_spans(removed)
    output: list[TimeSpan] = []
    for span in source:
        cursors = [span.start]
        ends: list[float] = []
        for cut in removed:
            if cut.end <= span.start or cut.start >= span.end:
                continue
            ends.append(max(span.start, cut.start))
            cursors.append(min(span.end, cut.end))
        ends.append(span.end)
        for start, end in zip(cursors, ends):
            if end - start >= minimum_seconds:
                output.append(TimeSpan(start, end))
    return output


@dataclass
class OverlapDecision:
    rejected: bool
    score: float
    peak: float
    active_fraction: float


class OverlapDetector:
    SAMPLE_RATE = 16000
    WINDOW_SAMPLES = 160000

    def __init__(self, device: str = 'cuda', batch_size: int = 16) -> None:
        self.device = device
        self.batch_size = max(1, int(batch_size))
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, min(4, (torch.get_num_threads() or 1)))
        self.options = options
        providers = ['CPUExecutionProvider']
        if device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
        try:
            self.session = ort.InferenceSession(str(OVERLAP_MODEL), sess_options=options, providers=providers)
        except Exception:
            if providers == ['CPUExecutionProvider']:
                raise
            self._cpu_session()
        self.device = ('cuda' if 'CUDAExecutionProvider' in self.session.get_providers() else 'cpu')
        # Some distributed pyannote graphs have a fixed batch dimension.
        shape = self.session.get_inputs()[0].shape
        if isinstance(shape[0], int):
            self.batch_size = 1

    def _cpu_session(self):
        self.session = ort.InferenceSession(str(OVERLAP_MODEL), sess_options=self.options,
                                           providers=['CPUExecutionProvider'])
        self.device = 'cpu'

    def _infer_windows(self, waveform, plans):
        offset = 0
        while offset < len(plans):
            batch = plans[offset:offset + self.batch_size]
            waves, ratios = [], []
            for start, end in batch:
                part = waveform[round(start * self.SAMPLE_RATE):round(end * self.SAMPLE_RATE)]
                ratios.append(min(1., len(part) / self.WINDOW_SAMPLES))
                waves.append(np.pad(part[:self.WINDOW_SAMPLES],
                                    (0, max(0, self.WINDOW_SAMPLES - len(part)))))
            try:
                values = self.session.run(None, {'x': np.stack(waves)[:, None].astype(np.float32)})[0]
            except Exception as error:
                if self.device != 'cuda' or not is_memory_error(error):
                    raise
                values = None
            if values is None:
                if self.batch_size > 1:
                    self.batch_size = max(1, self.batch_size // 2)
                else:
                    self._cpu_session()
                continue
            for plan, output, ratio in zip(batch, values, ratios):
                probabilities = self._softmax(output)
                yield plan, probabilities[:max(1, round(len(probabilities) * ratio))]
            offset += len(batch)

    @staticmethod
    def _softmax(value: np.ndarray) -> np.ndarray:
        value = value - value.max(axis=-1, keepdims=True)
        exp = np.exp(value)
        return exp / exp.sum(axis=-1, keepdims=True)

    def analyze(self, path: Path, threshold: float = 0.35) -> OverlapDecision:
        return self.analyze_waveform(load_mono(path, self.SAMPLE_RATE), threshold)

    def analyze_waveform(self, waveform: torch.Tensor, threshold: float = 0.35) -> OverlapDecision:
        waveform = waveform.detach().float().cpu().numpy()
        duration_samples = len(waveform)
        hop = self.WINDOW_SAMPLES // 2
        scores: list[np.ndarray] = []
        plans = []
        for start in range(0, max(1, duration_samples), hop):
            plans.append((start / self.SAMPLE_RATE,
                          min(duration_samples, start + self.WINDOW_SAMPLES) / self.SAMPLE_RATE))
            if start + self.WINDOW_SAMPLES >= duration_samples:
                break
        for _, probabilities in self._infer_windows(waveform, plans):
            speech_probability = 1.0 - probabilities[:, 0]
            overlap_probability = probabilities[:, 4:7].sum(axis=1)
            active = speech_probability >= 0.20
            if active.any():
                scores.append(overlap_probability[active])

        if not scores:
            return OverlapDecision(False, 0.0, 0.0, 0.0)
        values = np.concatenate(scores)
        score = float(np.quantile(values, 0.95))
        peak = float(values.max())
        active_fraction = float((values >= threshold).mean())
        rejected = (score >= threshold and active_fraction >= 0.02) or peak >= max(0.68, threshold + 0.25)
        return OverlapDecision(rejected, round(score, 5), round(peak, 5), round(active_fraction, 5))

    def clean_spans(
        self,
        path: Path,
        spans: list[TimeSpan],
        threshold: float = 0.35,
        progress: Callable[[float, str], None] | None = None,
        minimum_overlap_seconds: float = 0.10,
        padding: float = 0.06,
    ) -> tuple[list[TimeSpan], list[TimeSpan]]:
        """Remove only frame ranges with simultaneous speakers."""

        progress = progress or (lambda _value, _message: None)
        if not spans:
            return [], []
        waveform = load_mono(path, self.SAMPLE_RATE).numpy()
        plans: list[tuple[float, float]] = []
        hop_seconds = 8.0
        for span in spans:
            cursor = span.start
            while cursor < span.end:
                end = min(span.end, cursor + 10.0)
                plans.append((cursor, end))
                if end >= span.end:
                    break
                cursor += hop_seconds

        dirty: list[TimeSpan] = []
        frame_shift = 270 / self.SAMPLE_RATE
        frame_offset = 991 / (2 * self.SAMPLE_RATE)
        progress(0.0, f"多人重叠扫描 0/{len(plans)}")
        for plan_index, ((start, end), probabilities) in enumerate(self._infer_windows(waveform, plans), 1):
            values = probabilities[:, 4:7].sum(axis=1)
            active = values >= threshold
            run_start: int | None = None
            for frame_index in range(len(active) + 1):
                is_active = frame_index < len(active) and bool(active[frame_index])
                if is_active and run_start is None:
                    run_start = frame_index
                elif not is_active and run_start is not None:
                    overlap_start = start + frame_offset + run_start * frame_shift
                    overlap_end = start + frame_offset + frame_index * frame_shift
                    if overlap_end - overlap_start >= minimum_overlap_seconds:
                        dirty.append(
                            TimeSpan(
                                max(start, overlap_start - padding),
                                min(end, overlap_end + padding),
                            )
                        )
                    run_start = None
            progress(plan_index / len(plans), f"多人重叠扫描 {plan_index}/{len(plans)}")
        dirty = _merge_spans(dirty, gap=0.12)
        return _subtract_spans(spans, dirty), dirty


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, value: torch.Tensor, pool_size: tuple[int, int]) -> torch.Tensor:
        value = F.relu_(self.bn1(self.conv1(value)))
        value = F.relu_(self.bn2(self.conv2(value)))
        return F.avg_pool2d(value, kernel_size=pool_size)


class Cnn10(nn.Module):
    def __init__(self, classes_num: int = 527) -> None:
        super().__init__()
        vendor = str(VENDOR_ROOT)
        if vendor not in sys.path:
            sys.path.insert(0, vendor)
        from torchlibrosa.stft import LogmelFilterBank, Spectrogram

        self.spectrogram_extractor = Spectrogram(
            n_fft=1024,
            hop_length=320,
            win_length=1024,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )
        self.logmel_extractor = LogmelFilterBank(
            sr=32000,
            n_fft=1024,
            n_mels=64,
            fmin=50,
            fmax=14000,
            ref=1.0,
            amin=1e-10,
            top_db=None,
            freeze_parameters=True,
        )
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.fc1 = nn.Linear(512, 512)
        self.fc_audioset = nn.Linear(512, classes_num)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Squared STFT magnitudes overflow FP16 before the logarithm. Keep the
        # acoustic frontend in FP32; convolutional inference can use autocast.
        with torch.autocast('cuda', enabled=False):
            value = self.spectrogram_extractor(waveform.float())
            value = self.logmel_extractor(value)
        value = value.transpose(1, 3)
        value = self.bn0(value)
        value = value.transpose(1, 3)
        value = F.dropout(self.conv_block1(value, (2, 2)), p=0.2, training=False)
        value = F.dropout(self.conv_block2(value, (2, 2)), p=0.2, training=False)
        value = F.dropout(self.conv_block3(value, (2, 2)), p=0.2, training=False)
        value = F.dropout(self.conv_block4(value, (2, 2)), p=0.2, training=False)
        value = torch.mean(value, dim=3)
        maximum, _ = torch.max(value, dim=2)
        value = maximum + torch.mean(value, dim=2)
        value = F.relu_(self.fc1(value))
        return torch.sigmoid(self.fc_audioset(value))


@dataclass
class SingingDecision:
    rejected: bool
    singing_score: float
    speech_score: float
    top_labels: list[tuple[str, float]]


class SingingDetector:
    SINGING_LABELS = {
        "Singing",
        "Choir",
        "Male singing",
        "Female singing",
        "Child singing",
        "Synthetic singing",
        "Rapping",
        "Vocal music",
        "A capella",
    }
    SPEECH_LABELS = {
        "Speech",
        "Male speech, man speaking",
        "Female speech, woman speaking",
        "Child speech, kid speaking",
    }

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        with PANNS_LABELS.open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.labels = [row["display_name"] for row in rows]
        self.singing_indices = [index for index, label in enumerate(self.labels) if label in self.SINGING_LABELS]
        self.speech_indices = [index for index, label in enumerate(self.labels) if label in self.SPEECH_LABELS]
        model = Cnn10(len(self.labels))
        checkpoint = torch.load(PANNS_MODEL, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        self.model = model.eval()
        try:
            self.model.to(self.device)
        except RuntimeError as error:
            if not is_memory_error(error):
                raise
            self.device = torch.device('cpu')
            self.model.to(self.device)

    def _scores(self, batch):
        try:
            with torch.inference_mode(), torch.autocast('cuda', enabled=self.device.type == 'cuda'):
                scores = self.model(batch.to(self.device)).float().cpu()
            if not bool(torch.isfinite(scores).all()):
                with torch.inference_mode(), torch.autocast('cuda', enabled=False):
                    scores = self.model(batch.float().to(self.device)).float().cpu()
                if not bool(torch.isfinite(scores).all()):
                    raise ValueError('Nonfinite singing evidence; refusing to classify audio')
            return scores
        except RuntimeError as error:
            if self.device.type != 'cuda' or not is_memory_error(error):
                raise
        if len(batch) > 1:
            middle = len(batch) // 2
            return torch.cat((self._scores(batch[:middle]), self._scores(batch[middle:])))
        self.model.cpu().float()
        self.device = torch.device('cpu')
        torch.cuda.empty_cache()
        return self._scores(batch)

    def close(self) -> None:
        if hasattr(self, "model"):
            del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def analyze(self, path: Path, threshold: float = 0.22) -> SingingDecision:
        return self.analyze_waveform(load_mono(path, 32000), threshold)

    def analyze_waveform(self, waveform: torch.Tensor, threshold: float = 0.22) -> SingingDecision:
        minimum = 32000
        if waveform.numel() < minimum:
            waveform = F.pad(waveform, (0, minimum - waveform.numel()))
        scores = self._scores(waveform.unsqueeze(0))[0]
        singing = float(scores[self.singing_indices].max())
        speech = float(scores[self.speech_indices].max())
        top = torch.topk(scores, k=6)
        top_labels = [(self.labels[int(index)], round(float(value), 5)) for value, index in zip(top.values, top.indices)]
        rejected = singing >= threshold and singing >= speech * 0.80
        return SingingDecision(rejected, round(singing, 5), round(speech, 5), top_labels)

    def clean_spans(
        self,
        path: Path,
        spans: list[TimeSpan],
        threshold: float = 0.22,
        progress: Callable[[float, str], None] | None = None,
        window_seconds: float = 3.0,
        hop_seconds: float = 1.5,
    ) -> tuple[list[TimeSpan], list[TimeSpan]]:
        """Remove local singing windows while preserving nearby dialogue."""

        progress = progress or (lambda _value, _message: None)
        if not spans:
            return [], []
        waveform = load_mono(path, 32000)
        plans: list[tuple[float, float]] = []
        for span in spans:
            if span.duration <= window_seconds:
                plans.append((span.start, span.end))
                continue
            cursor = span.start
            while cursor + window_seconds < span.end:
                plans.append((cursor, cursor + window_seconds))
                cursor += hop_seconds
            plans.append((max(span.start, span.end - window_seconds), span.end))

        dirty = self._singing_windows(
            waveform, plans, window_seconds, threshold,
            lambda value, message: progress(0.65 * value, message),
        )
        regions = _merge_spans(dirty, gap=0.0)
        local_plans: list[tuple[float, float]] = []
        for region in regions:
            cursor = region.start
            while cursor + 0.8 < region.end:
                local_plans.append((cursor, cursor + 0.8))
                cursor += 0.2
            local_plans.append((max(region.start, region.end - 0.8), region.end))
        local_dirty = self._singing_windows(
            waveform, local_plans, 0.8, threshold,
            lambda value, message: progress(0.65 + 0.35 * value, message),
        )
        dirty = []
        for region in regions:
            confirmed = [part for part in local_dirty
                         if region.start <= part.start and part.end <= region.end]
            # Short context can miss sustained singing. Preserve the coarse
            # finding when local windows cannot resolve it at all.
            dirty.extend(confirmed or [region])
        dirty = _merge_spans(dirty, gap=0.03)
        progress(1.0, f"歌声扫描完成：{len(dirty)} 段局部污染")
        return _subtract_spans(spans, dirty), dirty

    def _singing_windows(
        self,
        waveform: torch.Tensor,
        plans: list[tuple[float, float]],
        window_seconds: float,
        threshold: float,
        progress: Callable[[float, str], None],
    ) -> list[TimeSpan]:
        target_samples = max(32000, int(round(window_seconds * 32000)))
        dirty: list[TimeSpan] = []
        batch_size = 16
        for offset in range(0, len(plans), batch_size):
            batch_plans = plans[offset : offset + batch_size]
            parts: list[torch.Tensor] = []
            for start, end in batch_plans:
                part = waveform[round(start * 32000):round(end * 32000)]
                part = F.pad(part, (0, max(0, target_samples - part.numel())))
                parts.append(part[:target_samples])
            scores = self._scores(torch.stack(parts))
            for (start, end), values in zip(batch_plans, scores):
                singing = float(values[self.singing_indices].max())
                speech = float(values[self.speech_indices].max())
                if singing >= threshold and singing >= speech * 0.80:
                    dirty.append(TimeSpan(start, end))
            completed = min(len(plans), offset + len(batch_plans))
            progress(completed / len(plans), f"歌声扫描 {completed}/{len(plans)}")
        return dirty
