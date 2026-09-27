"""Local-only, frozen WavLM speaker TDNN features before statistics pooling.

Frame cells describe decision sampling, not identity accuracy. The TDNN's
convolution support is wider; transformer attention also sees the entire input
context. Edges without cells remain missing evidence, never repeated/padded
scores. No third-party module is modified and no downloads are initiated.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch

from .timeline import SampleSpan, ViewAlignment, require_digest


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrameGeometry:
    sample_rate: int
    hop: int
    convolution_support: int

    def __post_init__(self) -> None:
        if any(type(v) is not int or v <= 0 for v in (
            self.sample_rate, self.hop, self.convolution_support,
        )) or self.convolution_support < self.hop:
            raise ValueError("Invalid feature geometry")

    @classmethod
    def from_config(cls, config, sample_rate: int = 16000) -> FrameGeometry:
        hop, support = 1, 1
        if len(config.conv_kernel) != len(config.conv_stride):
            raise ValueError("Convolution configuration lengths differ")
        if len(config.tdnn_kernel) != len(config.tdnn_dilation):
            raise ValueError("TDNN configuration lengths differ")
        for kernel, stride in zip(config.conv_kernel, config.conv_stride):
            if kernel < 1 or stride < 1:
                raise ValueError("Invalid convolution configuration")
            support += (kernel - 1) * hop
            hop *= stride
        for kernel, dilation in zip(config.tdnn_kernel, config.tdnn_dilation):
            if kernel < 1 or dilation < 1:
                raise ValueError("Invalid TDNN configuration")
            support += (kernel - 1) * dilation * hop
        return cls(sample_rate, hop, support)

    def count(self, samples: int) -> int:
        return max(0, (samples - self.convolution_support) // self.hop + 1)

    def cells(self, count: int, offset: int = 0) -> tuple[SampleSpan, ...]:
        if type(count) is not int or count < 0 or type(offset) is not int or offset < 0:
            raise ValueError("Invalid frame count/offset")
        left = offset + (self.convolution_support - self.hop) // 2
        return tuple(SampleSpan(left + i * self.hop, left + (i + 1) * self.hop)
                     for i in range(count))


@dataclass(frozen=True)
class FeatureSequence:
    values: torch.Tensor  # [T,D], no pooled sentence embedding
    cells: tuple[SampleSpan, ...]  # source timeline
    source_digest: str
    backbone_digest: str

    def __post_init__(self) -> None:
        require_digest(self.source_digest)
        require_digest(self.backbone_digest)
        if (self.values.ndim != 2 or not self.cells or self.values.shape[0] != len(self.cells)
                or self.values.shape[1] < 1 or not torch.isfinite(self.values).all()):
            raise ValueError("Invalid time-local features")
        if any(a.end > b.start for a, b in zip(self.cells, self.cells[1:])):
            raise ValueError("Feature cells overlap or are unordered")


def require_paired(raw: FeatureSequence, stem: FeatureSequence) -> None:
    if (raw.cells != stem.cells or raw.source_digest != stem.source_digest
            or raw.backbone_digest != stem.backbone_digest or raw.values.shape != stem.values.shape):
        raise ValueError("Raw/stem features must have the same source, encoder and verified time cells")


class WavLMSpeakerFeatures:
    preprocessing_version = "wavlm-tdnn-unpooled-v1"

    def __init__(self, directory: str | Path, device: str = "cpu"):
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

        directory = Path(directory).resolve(strict=True)
        weights = sorted(directory.glob("*.safetensors")) or sorted(directory.glob("pytorch_model*.bin"))
        config = directory / "config.json"
        preprocessing = directory / "preprocessor_config.json"
        if not weights or not config.is_file() or not preprocessing.is_file():
            raise ValueError("A complete local encoder checkpoint is required")
        payload = [(p.name, file_digest(p)) for p in (config, preprocessing, *weights)]
        self.digest = hashlib.sha256(json.dumps(
            [self.preprocessing_version, payload], sort_keys=True,
        ).encode()).hexdigest()
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(str(directory), local_files_only=True)
        if self.processor.sampling_rate != 16000:
            raise ValueError("This encoder contract expects 16 kHz input")
        self.model = WavLMForXVector.from_pretrained(str(directory), local_files_only=True).to(device).eval()
        self.model.requires_grad_(False)
        self.geometry = FrameGeometry.from_config(self.model.config)
        self.device = torch.device(device)

    @torch.inference_mode()
    def encode(self, waveform: torch.Tensor, alignment: ViewAlignment,
               view_span: SampleSpan) -> FeatureSequence:
        if (waveform.ndim != 1 or waveform.numel() != view_span.length
                or not torch.isfinite(waveform).all()):
            raise ValueError("Expected finite mono samples for the exact view interval")
        if alignment.view_rate != self.geometry.sample_rate or not alignment.verified:
            raise ValueError("Audio must be resampled and alignment verified before encoding")
        alignment.to_source(view_span)
        count = self.geometry.count(waveform.numel())
        if count == 0:
            raise ValueError("Too little real context for TDNN; do not pad or duplicate short voices")
        data = self.processor(waveform.detach().cpu().float().numpy(), sampling_rate=16000,
                              return_tensors="pt")
        encoded = self.model.wavlm(data.input_values.to(self.device),
                                   output_hidden_states=self.model.config.use_weighted_layer_sum,
                                   return_dict=True)
        if self.model.config.use_weighted_layer_sum:
            layers = torch.stack(encoded.hidden_states, dim=1)
            weights = self.model.layer_weights.softmax(dim=0)[None, :, None, None]
            features = (layers * weights).sum(dim=1)
        else:
            features = encoded.last_hidden_state
        features = self.model.projector(features)
        for layer in self.model.tdnn:
            features = layer(features)
        if features.shape[1] != count:
            raise RuntimeError("Encoder output contradicts the recorded convolution geometry")
        cells = self.geometry.cells(count, view_span.start)
        # Shared rounded edges retain a partition even for 44.1 -> 16 kHz.
        # This differs deliberately from outward rounding of whole audio cuts.
        def edge(sample: int) -> int:
            numerator = (sample - alignment.delay_samples) * alignment.source.sample_rate
            return (numerator + alignment.view_rate // 2) // alignment.view_rate
        mapped = tuple(SampleSpan(edge(row.start), edge(row.end)) for row in cells)
        for cell in mapped:
            alignment.source.validate(cell)
        return FeatureSequence(features[0].detach().cpu(), mapped,
                               alignment.source.source_sha256, self.digest)
