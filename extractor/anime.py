from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .audio import load_mono
from .config import ANIME_CHAR_MODEL, ANIME_VA_MODEL


SAMPLE_RATE = 16_000
N_FFT = 400
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 80


@dataclass(frozen=True)
class AnimeIdentityProfile:
    target_char: torch.Tensor
    target_va: torch.Tensor
    exclusion_char: tuple[torch.Tensor, ...]
    exclusion_va: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class AnimeIdentityDecision:
    char_target_max: float
    char_exclusion_max: float
    char_margin: float
    va_target_max: float
    va_exclusion_max: float
    va_margin: float


def _speechbrain_fbank(waveform: torch.Tensor) -> torch.Tensor:
    """Reproduce the fbank used when the local anime ECAPA models were exported."""

    value = waveform.detach().float().reshape(1, -1)
    maximum = torch.max(torch.abs(value))
    if maximum > 1.0:
        value = value / maximum
    value = value * 32768.0

    window = torch.hamming_window(WIN_LENGTH, dtype=value.dtype)
    stft = torch.stft(
        value,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        center=True,
        pad_mode="constant",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spectrogram = stft.real.square() + stft.imag.square()
    spectrogram = spectrogram.transpose(1, 2)

    mel_min = 2595.0 * math.log10(1.0 + 0.0 / 700.0)
    mel_max = 2595.0 * math.log10(1.0 + 8000.0 / 700.0)
    mel = torch.linspace(mel_min, mel_max, N_MELS + 2, dtype=value.dtype)
    hz = 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)
    band = hz[1:-1] - hz[:-2]
    central = hz[1:-1]
    frequencies = torch.linspace(0.0, SAMPLE_RATE // 2, N_FFT // 2 + 1, dtype=value.dtype)
    slope = (frequencies.unsqueeze(0) - central.unsqueeze(1)) / band.unsqueeze(1)
    filters = torch.maximum(
        torch.zeros(1, dtype=value.dtype),
        torch.minimum(slope + 1.0, -slope + 1.0),
    ).transpose(0, 1)
    fbanks = torch.matmul(spectrogram, filters)
    fbanks = 10.0 * torch.log10(torch.clamp(fbanks, min=1e-10))
    floor = fbanks.amax(dim=(-2, -1), keepdim=True) - 80.0
    return torch.maximum(fbanks, floor)


class AnimeIdentityVerifier:
    """Optional local anime-voice identity scorer for conservative recovery.

    The scorer is intentionally separate from the primary ERes2Net/CAM++ gate.
    It is used only when both local model files and user-supplied exclusion
    references are available, so a missing optional asset never changes the
    normal extraction path.
    """

    MARGIN_FLOOR = 0.15

    def __init__(self) -> None:
        import onnxruntime

        self._ort = onnxruntime
        self.char = onnxruntime.InferenceSession(
            str(ANIME_CHAR_MODEL),
            providers=["CPUExecutionProvider"],
        )
        self.va = onnxruntime.InferenceSession(
            str(ANIME_VA_MODEL),
            providers=["CPUExecutionProvider"],
        )
        self.char_input = self.char.get_inputs()[0].name
        self.va_input = self.va.get_inputs()[0].name

    @classmethod
    def available(cls) -> bool:
        return ANIME_CHAR_MODEL.is_file() and ANIME_VA_MODEL.is_file()

    def close(self) -> None:
        self.char = None
        self.va = None

    @staticmethod
    def _prepare(waveform: torch.Tensor) -> np.ndarray:
        value = waveform.detach().float().flatten()
        if value.numel() < SAMPLE_RATE // 2:
            value = F.pad(value, (0, SAMPLE_RATE // 2 - value.numel()))
        return _speechbrain_fbank(value).numpy().astype(np.float32, copy=False)

    def _embed(self, session, input_name: str, waveform: torch.Tensor) -> torch.Tensor:
        features = self._prepare(waveform)
        output = session.run(None, {input_name: features})[0][0]
        return F.normalize(torch.from_numpy(np.asarray(output)).float(), dim=0)

    def embed_waveforms(self, waveforms: Iterable[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        char_values: list[torch.Tensor] = []
        va_values: list[torch.Tensor] = []
        for waveform in waveforms:
            char_values.append(self._embed(self.char, self.char_input, waveform))
            va_values.append(self._embed(self.va, self.va_input, waveform))
        if not char_values:
            empty = torch.empty((0, 192))
            return empty, empty
        return torch.stack(char_values), torch.stack(va_values)

    def build_profile(
        self,
        target_references: Iterable[Path],
        exclusion_groups: Iterable[Iterable[Path]],
    ) -> AnimeIdentityProfile:
        target_paths = [Path(path) for path in target_references if path]
        groups = [[Path(path) for path in group if path] for group in exclusion_groups]
        groups = [group for group in groups if group]
        target_char, target_va = self.embed_waveforms(
            load_mono(path, SAMPLE_RATE) for path in target_paths
        )
        exclusion_char: list[torch.Tensor] = []
        exclusion_va: list[torch.Tensor] = []
        for group in groups:
            char, va = self.embed_waveforms(load_mono(path, SAMPLE_RATE) for path in group)
            exclusion_char.append(char)
            exclusion_va.append(va)
        return AnimeIdentityProfile(
            target_char=target_char,
            target_va=target_va,
            exclusion_char=tuple(exclusion_char),
            exclusion_va=tuple(exclusion_va),
        )

    def score(self, waveform: torch.Tensor, profile: AnimeIdentityProfile) -> AnimeIdentityDecision:
        char, va = self.embed_waveforms([waveform])
        char_target = float((char[0] @ profile.target_char.T).max())
        va_target = float((va[0] @ profile.target_va.T).max())
        char_exclusion = max(
            (float((char[0] @ group.T).max()) for group in profile.exclusion_char),
            default=-1.0,
        )
        va_exclusion = max(
            (float((va[0] @ group.T).max()) for group in profile.exclusion_va),
            default=-1.0,
        )
        return AnimeIdentityDecision(
            char_target_max=round(char_target, 5),
            char_exclusion_max=round(char_exclusion, 5),
            char_margin=round(char_target - char_exclusion, 5),
            va_target_max=round(va_target, 5),
            va_exclusion_max=round(va_exclusion, 5),
            va_margin=round(va_target - va_exclusion, 5),
        )
