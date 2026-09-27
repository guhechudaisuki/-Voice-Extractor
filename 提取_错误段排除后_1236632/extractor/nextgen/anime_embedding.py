"""Inference-only adapter for locally installed anime speaker ONNX models.

The encoder is reference-conditioned at use time: the same frozen model encodes
user references and candidate speech. Nothing here trains or identifies a
character by name. The filterbank intentionally matches the preprocessing of
the existing anime_speaker_embedding checkpoints; an ordinary torchaudio mel
filterbank would change the model input distribution.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


_MODEL_FILES = {
    "char": "anime-speaker-char/anime_speaker_char_ecapa.onnx",
    "va": "anime-speaker-va/anime_speaker_va_ecapa.onnx",
}


def speechbrain_compatible_fbank(waveform: torch.Tensor) -> torch.Tensor:
    """Return the frozen anime model's 16 kHz, 80-bin features [frames, bins].

    The model was trained with SpeechBrain's default 25/10 ms Hamming STFT,
    triangular HTK-mel power filters, per-utterance 80 dB floor, and the
    checkpoint's unusual x32768 waveform scale. Resampling must happen before
    this function; accepting a rate here would risk a silent mismatch.
    """
    samples = torch.as_tensor(waveform, dtype=torch.float32, device="cpu")
    if samples.ndim != 1 or samples.numel() < 3200 or not torch.isfinite(samples).all():
        raise ValueError("Expected at least 0.2 s of finite mono audio at 16 kHz")
    if float(samples.square().mean().sqrt()) < 1e-5:
        raise ValueError("Silent audio cannot provide speaker identity evidence")
    peak = samples.abs().amax()
    if peak > 1.0:
        samples = samples / peak
    samples = samples * 32768.0

    spectrum = torch.stft(
        samples.unsqueeze(0), n_fft=400, hop_length=160, win_length=400,
        window=torch.hamming_window(400), center=True, pad_mode="constant",
        normalized=False, onesided=True, return_complex=True,
    )
    power = (spectrum.real.square() + spectrum.imag.square()).transpose(1, 2)

    mel_edges = torch.linspace(0.0, 2595.0 * np.log10(1.0 + 8000.0 / 700.0), 82)
    hz_edges = 700.0 * (torch.pow(10.0, mel_edges / 2595.0) - 1.0)
    centers = hz_edges[1:-1, None]
    left_bands = (hz_edges[1:-1] - hz_edges[:-2])[:, None]
    freqs = torch.linspace(0.0, 8000.0, 201)[None, :]
    slope = (freqs - centers) / left_bands
    filters = torch.minimum(slope + 1.0, 1.0 - slope).clamp_min_(0.0).transpose(0, 1)

    features = 10.0 * torch.log10((power @ filters).clamp_min_(1e-10))
    features = torch.maximum(features, features.amax(dim=(-2, -1), keepdim=True) - 80.0)
    return features[0].contiguous()


class AnimeSpeakerOnnx:
    """Frozen char/va embeddings; no network access and no checkpoint writes."""

    def __init__(self, models_root: str | Path, variant: str):
        if variant not in _MODEL_FILES:
            raise ValueError(f"Unknown anime speaker variant: {variant}")
        path = Path(models_root) / _MODEL_FILES[variant]
        if not path.is_file():
            raise FileNotFoundError(path)

        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"],
        )
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs) != 1 or inputs[0].name != "features"
                or inputs[0].shape[-1] != 80 or len(outputs) != 1
                or outputs[0].shape[-1] != 192):
            raise ValueError("Unexpected anime speaker ONNX input/output contract")
        self.variant = variant
        self.path = path

    def encode(self, waveform: torch.Tensor, *, sample_rate: int) -> np.ndarray:
        if sample_rate != 16000:
            raise ValueError("Anime speaker encoder requires resampled 16 kHz audio")
        features = speechbrain_compatible_fbank(waveform)
        result = self.session.run(None, {"features": features.numpy()[None]})[0]
        vector = np.asarray(result, dtype=np.float32).reshape(-1)
        if vector.shape != (192,) or not np.isfinite(vector).all():
            raise ValueError("Anime speaker ONNX returned an invalid embedding")
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8:
            raise ValueError("Anime speaker ONNX returned a zero embedding")
        return vector / norm
