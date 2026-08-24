from __future__ import annotations

import gc
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Literal, Sequence

import torch
import torch.nn.functional as F
import torchaudio

from .config import CAMPLUS_MODEL, SV_CODE, SV_MODEL, WAVLM_SV_MODEL, WESPEAKER_MODEL
from .audio import load_mono
from .types import TimeSpan


SpeakerMatchTier = Literal[
    "short_strong",
    "strong",
    "balanced",
    "tertiary",
    "recall",
    "weak",
    "rejected",
]


@dataclass
class SpeakerProfile:
    embeddings: torch.Tensor
    centroid: torch.Tensor
    reference_scores: list[float]
    suggested_threshold: float
    reference_floor: float
    calibration_base: float
    reference_indexes: tuple[int, ...] = ()


@dataclass
class SpeakerDecision:
    accepted: bool
    score: float
    window_min_score: float
    window_p20_score: float
    vote_ratio: float
    window_vote_ratio: float
    window_scores: list[float]
    reference_median_score: float = 0.0
    reference_max_score: float = 0.0
    reference_spread: float = 0.0
    embedding: torch.Tensor | None = field(default=None, repr=False, compare=False)


@dataclass
class SpeakerMatchDecision:
    accepted: bool
    primary: SpeakerDecision
    match_mode: str
    secondary: SpeakerDecision | None = None
    tier: SpeakerMatchTier = "rejected"
    merge_only: bool = False
    paired_reference_median: float = 0.0
    diagnostics: dict[str, float | str | bool] = field(default_factory=dict)


@dataclass
class SpeakerMatchProfile:
    primary: SpeakerProfile
    reference_paths: list[Path]
    base_threshold: float


@dataclass
class CAMPlusProfile:
    embeddings: torch.Tensor
    centroid: torch.Tensor
    reference_scores: list[float]
    reference_indexes: tuple[int, ...] = ()


@dataclass
class WavLMProfile:
    embeddings: torch.Tensor
    centroid: torch.Tensor
    reference_scores: list[float]
    acceptance_floor: float
    reference_indexes: tuple[int, ...] = ()


@dataclass
class WavLMDecision:
    score: float
    reference_median_score: float
    reference_max_score: float
    window_min_score: float
    window_p20_score: float
    window_scores: list[float]


@dataclass
class WeSpeakerProfile:
    embeddings: torch.Tensor
    centroid: torch.Tensor
    reference_scores: list[float]
    reference_indexes: tuple[int, ...] = ()


@dataclass
class WeSpeakerDecision:
    score: float
    reference_max_score: float
    window_min_score: float
    window_p20_score: float
    window_scores: list[float]


@dataclass
class ExclusionSpeakerProfile:
    label: str
    primary: SpeakerProfile
    secondary: CAMPlusProfile
    reference_paths: list[Path] = field(default_factory=list)
    quaternary: WeSpeakerProfile | None = None


@dataclass(frozen=True)
class SpeakerBoundary:
    """A locally detected change of speaker.

    ``primary_similarity`` and ``secondary_similarity`` are cosine scores
    between the audio immediately before and after ``time``.  They are kept
    in the result so callers can inspect borderline decisions without having
    to run the expensive embedding pass again.
    """

    time: float
    primary_similarity: float
    secondary_similarity: float | None
    confidence: float
    primary_drop: float = 0.0
    secondary_drop: float = 0.0
    # A boundary returned by the multi-scale audit carries the number of
    # independent context sizes that observed it.  Ordinary one-pass callers
    # keep the default value of one.
    scale_votes: int = 1
    scale_contexts: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "time": round(float(self.time), 5),
            "primary_similarity": round(float(self.primary_similarity), 5),
            "secondary_similarity": (
                None
                if self.secondary_similarity is None
                else round(float(self.secondary_similarity), 5)
            ),
            "confidence": round(float(self.confidence), 5),
            "primary_drop": round(float(self.primary_drop), 5),
            "secondary_drop": round(float(self.secondary_drop), 5),
            "scale_votes": int(self.scale_votes),
            "scale_contexts": [round(float(value), 3) for value in self.scale_contexts],
        }


@dataclass(frozen=True)
class SpeakerSplitResult:
    """Detailed result for diagnostics; ``split_speaker_spans`` returns only spans."""

    spans: tuple[TimeSpan, ...]
    boundaries: tuple[SpeakerBoundary, ...]


class SpeakerVerifier:
    def __init__(self, device: str = "cuda") -> None:
        self.device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        code_path = str(SV_CODE)
        if code_path not in sys.path:
            sys.path.insert(0, code_path)
        from ERes2NetV2 import ERes2NetV2

        self.kaldi = __import__("kaldi")
        state = torch.load(SV_MODEL, map_location="cpu", weights_only=False)
        model = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
        model.load_state_dict(state)
        self.model = model.eval().to(self.device)

    def close(self) -> None:
        if hasattr(self, "model"):
            del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _embeddings_from_waveforms(
        self,
        waveforms: Sequence[torch.Tensor],
        batch_size: int = 48,
        progress: Callable[[int, int], None] | None = None,
    ) -> torch.Tensor:
        if not waveforms:
            return torch.empty((0, 0))
        prepared: list[torch.Tensor] = []
        groups: dict[int, list[int]] = {}
        for index, waveform in enumerate(waveforms):
            value = waveform.detach().float().cpu().flatten()
            if value.numel() < 8000:
                value = F.pad(value, (0, 8000 - value.numel()))
            prepared.append(value)
            groups.setdefault(value.numel(), []).append(index)

        output: list[torch.Tensor | None] = [None] * len(prepared)
        completed = 0
        for indexes in groups.values():
            for offset in range(0, len(indexes), max(1, batch_size)):
                batch_indexes = indexes[offset : offset + max(1, batch_size)]
                features = torch.stack(
                    [
                        self.kaldi.fbank(
                            prepared[index].unsqueeze(0),
                            num_mel_bins=80,
                            sample_frequency=16000,
                            dither=0,
                        )
                        for index in batch_indexes
                    ]
                )
                with torch.inference_mode():
                    embeddings = self.model(features.to(self.device))
                    embeddings = F.normalize(embeddings.float(), p=2, dim=1).cpu()
                for index, embedding in zip(batch_indexes, embeddings):
                    output[index] = embedding
                completed += len(batch_indexes)
                if progress is not None:
                    progress(completed, len(prepared))
        return torch.stack([value for value in output if value is not None])

    def _embedding_from_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        return self._embeddings_from_waveforms([waveform], batch_size=1)[0]

    def embedding_from_file(self, path: Path) -> torch.Tensor:
        return self._embedding_from_waveform(load_mono(path, 16000))

    def build_profile(self, reference_paths: list[Path], base_threshold: float) -> SpeakerProfile:
        if not reference_paths:
            raise ValueError("至少需要一段参考音频")
        raw = torch.stack([self.embedding_from_file(path) for path in reference_paths])
        reference_indexes = torch.arange(len(raw))
        keep = self._reference_core_mask(raw, floor=0.50)
        if int(keep.sum()) >= 2:
            raw = raw[keep]
            reference_indexes = reference_indexes[keep]
        centroid = F.normalize(raw.mean(dim=0), dim=0)
        scores = raw @ centroid

        low_reference = float(torch.quantile(scores, 0.10)) if len(scores) > 1 else float(scores[0])
        # Cross-language dialogue, emotional delivery and vocal separation can
        # lower target scores substantially. Calibrate from reference cohesion
        # while letting the UI strictness add only a small conservative bias.
        reference_floor = min(0.68, max(0.52, low_reference - 0.22))
        calibration_base = min(0.70, max(0.66, low_reference - 0.14))
        suggested = max(base_threshold, calibration_base)
        return SpeakerProfile(
            embeddings=raw,
            centroid=centroid,
            reference_scores=[round(float(value), 5) for value in scores],
            suggested_threshold=round(suggested, 4),
            reference_floor=round(reference_floor, 4),
            calibration_base=round(calibration_base, 4),
            reference_indexes=tuple(int(value) for value in reference_indexes.tolist()),
        )

    @staticmethod
    def _reference_core_mask(raw: torch.Tensor, floor: float) -> torch.Tensor:
        """Keep references that agree with the rest of the reference set.

        Centroid filtering is self-reinforcing: one bad clip moves the centroid
        toward itself and makes good clips look less similar.  A leave-one-out
        pairwise median is robust to that failure and gives both speaker models
        the same definition of a usable reference core.
        """

        count = int(raw.shape[0])
        if count < 4:
            return torch.ones(count, dtype=torch.bool)
        pairwise = raw @ raw.T
        diagonal = torch.eye(count, dtype=torch.bool, device=raw.device)
        pairwise = pairwise.masked_fill(diagonal, float("nan"))
        cohesion = torch.nanmedian(pairwise, dim=1).values
        center = torch.median(cohesion)
        mad = torch.median(torch.abs(cohesion - center))
        cutoff = max(float(floor), float(center - max(0.08, 3.0 * float(mad))))
        keep = cohesion >= cutoff
        if int(keep.sum()) < 2:
            _, indexes = torch.topk(cohesion, k=min(2, count))
            keep = torch.zeros(count, dtype=torch.bool, device=raw.device)
            keep[indexes] = True
        return keep

    def verify(
        self,
        candidate_path: Path,
        profile: SpeakerProfile,
        threshold: float,
        window_seconds: float = 1.8,
        hop_seconds: float = 0.9,
    ) -> SpeakerDecision:
        return self.verify_waveform(
            load_mono(candidate_path, 16000),
            profile,
            threshold,
            window_seconds=window_seconds,
            hop_seconds=hop_seconds,
        )

    def verify_waveform(
        self,
        waveform: torch.Tensor,
        profile: SpeakerProfile,
        threshold: float,
        window_seconds: float = 1.8,
        hop_seconds: float = 0.9,
    ) -> SpeakerDecision:
        whole = self._embedding_from_waveform(waveform)
        score = float(whole @ profile.centroid)
        reference_scores = profile.embeddings @ whole
        reference_median = float(reference_scores.median())
        reference_max = float(reference_scores.max())
        vote_floor = threshold - 0.10
        vote_ratio = float((reference_scores >= vote_floor).float().mean())

        window_size = int(window_seconds * 16000)
        hop = int(hop_seconds * 16000)
        windows: list[torch.Tensor] = []
        if waveform.numel() <= window_size:
            windows = [waveform]
        else:
            starts = list(range(0, max(1, waveform.numel() - window_size + 1), hop))
            last = waveform.numel() - window_size
            if not starts or starts[-1] != last:
                starts.append(last)
            for start in starts:
                part = waveform[start : start + window_size]
                rms = torch.sqrt(torch.mean(part.square()) + 1e-12)
                if float(20 * torch.log10(rms + 1e-12)) > -48.0:
                    windows.append(part)
        if not windows:
            windows = [waveform]

        if len(windows) == 1 and windows[0].numel() == waveform.numel():
            window_embeddings = whole.unsqueeze(0)
        else:
            window_embeddings = self._embeddings_from_waveforms(windows)
        window_scores = [float(value) for value in window_embeddings @ profile.centroid]
        window_tensor = torch.tensor(window_scores)
        minimum = float(window_tensor.min())
        p20 = float(torch.quantile(window_tensor, 0.20))
        window_vote_ratio = float((window_tensor >= threshold - 0.12).float().mean())
        accepted = (
            score >= threshold
            and p20 >= threshold - 0.18
            and window_vote_ratio >= 0.55
            and vote_ratio >= 0.40
        )
        return SpeakerDecision(
            accepted=accepted,
            score=round(score, 5),
            window_min_score=round(minimum, 5),
            window_p20_score=round(p20, 5),
            vote_ratio=round(vote_ratio, 5),
            window_vote_ratio=round(window_vote_ratio, 5),
            window_scores=[round(value, 5) for value in window_scores],
            reference_median_score=round(reference_median, 5),
            reference_max_score=round(reference_max, 5),
            reference_spread=round(reference_max - reference_median, 5),
            embedding=whole,
        )


class CAMPlusVerifier:
    """Independent VoxCeleb verifier used only for conservative border cases."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        from funasr.models.campplus.model import CAMPPlus

        model = CAMPPlus(feat_dim=80, embedding_size=512)
        state = torch.load(CAMPLUS_MODEL, map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        self.model = model.eval().to(self.device)

    def close(self) -> None:
        if hasattr(self, "model"):
            del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _embeddings_from_waveforms(
        self,
        waveforms: Sequence[torch.Tensor],
        batch_size: int = 64,
        progress: Callable[[int, int], None] | None = None,
    ) -> torch.Tensor:
        if not waveforms:
            return torch.empty((0, 0))
        prepared: list[torch.Tensor] = []
        groups: dict[int, list[int]] = {}
        for index, waveform in enumerate(waveforms):
            value = waveform.detach().float().cpu().flatten()
            if value.numel() < 8000:
                value = F.pad(value, (0, 8000 - value.numel()))
            prepared.append(value)
            groups.setdefault(value.numel(), []).append(index)

        output: list[torch.Tensor | None] = [None] * len(prepared)
        completed = 0
        for indexes in groups.values():
            for offset in range(0, len(indexes), max(1, batch_size)):
                batch_indexes = indexes[offset : offset + max(1, batch_size)]
                features = []
                for index in batch_indexes:
                    feature = torchaudio.compliance.kaldi.fbank(
                        prepared[index].unsqueeze(0),
                        num_mel_bins=80,
                        sample_frequency=16000,
                    )
                    features.append(feature - feature.mean(dim=0, keepdim=True))
                with torch.inference_mode():
                    embeddings = self.model(torch.stack(features).to(self.device))
                    embeddings = F.normalize(embeddings.float(), p=2, dim=1).cpu()
                for index, embedding in zip(batch_indexes, embeddings):
                    output[index] = embedding
                completed += len(batch_indexes)
                if progress is not None:
                    progress(completed, len(prepared))
        return torch.stack([value for value in output if value is not None])

    def _embedding_from_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        return self._embeddings_from_waveforms([waveform], batch_size=1)[0]

    def embedding_from_file(self, path: Path) -> torch.Tensor:
        return self._embedding_from_waveform(load_mono(path, 16000))

    def build_profile(
        self,
        reference_paths: list[Path],
        reference_indexes: Sequence[int] | None = None,
    ) -> CAMPlusProfile:
        all_raw = torch.stack([self.embedding_from_file(path) for path in reference_paths])
        if reference_indexes is None:
            selected = SpeakerVerifier._reference_core_mask(all_raw, floor=0.35)
            reference_indexes_tensor = torch.arange(len(all_raw))[selected]
        else:
            reference_indexes_tensor = torch.tensor(
                list(reference_indexes), dtype=torch.long
            )
            if reference_indexes_tensor.numel() == 0:
                raise ValueError("至少需要一段有效参考音频")
            selected = reference_indexes_tensor
        raw = all_raw[selected]
        centroid = F.normalize(raw.mean(dim=0), dim=0)
        scores = raw @ centroid
        return CAMPlusProfile(
            embeddings=raw,
            centroid=centroid,
            reference_scores=[round(float(value), 5) for value in scores],
            reference_indexes=tuple(int(value) for value in reference_indexes_tensor.tolist()),
        )

    def verify(
        self,
        candidate_path: Path,
        profile: CAMPlusProfile,
        threshold: float,
        window_seconds: float = 1.8,
        hop_seconds: float = 0.9,
    ) -> SpeakerDecision:
        return self.verify_waveform(
            load_mono(candidate_path, 16000),
            profile,
            threshold,
            window_seconds=window_seconds,
            hop_seconds=hop_seconds,
        )

    def verify_waveform(
        self,
        waveform: torch.Tensor,
        profile: CAMPlusProfile,
        threshold: float,
        window_seconds: float = 1.8,
        hop_seconds: float = 0.9,
    ) -> SpeakerDecision:
        whole = self._embedding_from_waveform(waveform)
        score = float(whole @ profile.centroid)
        reference_scores = profile.embeddings @ whole
        reference_median = float(reference_scores.median())
        reference_max = float(reference_scores.max())
        vote_ratio = float((reference_scores >= threshold - 0.08).float().mean())

        window_size = int(window_seconds * 16000)
        hop = int(hop_seconds * 16000)
        if waveform.numel() <= window_size:
            windows = [waveform]
        else:
            starts = list(range(0, max(1, waveform.numel() - window_size + 1), hop))
            last = waveform.numel() - window_size
            if not starts or starts[-1] != last:
                starts.append(last)
            windows = []
            for start in starts:
                part = waveform[start : start + window_size]
                rms = torch.sqrt(torch.mean(part.square()) + 1e-12)
                if float(20 * torch.log10(rms + 1e-12)) > -48.0:
                    windows.append(part)
        if not windows:
            windows = [waveform]

        if len(windows) == 1 and windows[0].numel() == waveform.numel():
            window_embeddings = whole.unsqueeze(0)
        else:
            window_embeddings = self._embeddings_from_waveforms(windows)
        window_scores = [float(value) for value in window_embeddings @ profile.centroid]
        window_tensor = torch.tensor(window_scores)
        minimum = float(window_tensor.min())
        p20 = float(torch.quantile(window_tensor, 0.20))
        window_vote_ratio = float((window_tensor >= threshold - 0.06).float().mean())
        accepted = (
            len(window_scores) >= 2
            and score >= threshold
            and p20 >= threshold - 0.04
            and window_vote_ratio >= 0.75
            and vote_ratio >= 0.40
        )
        return SpeakerDecision(
            accepted=accepted,
            score=round(score, 5),
            window_min_score=round(minimum, 5),
            window_p20_score=round(p20, 5),
            vote_ratio=round(vote_ratio, 5),
            window_vote_ratio=round(window_vote_ratio, 5),
            window_scores=[round(value, 5) for value in window_scores],
            reference_median_score=round(reference_median, 5),
            reference_max_score=round(reference_max, 5),
            reference_spread=round(reference_max - reference_median, 5),
            embedding=whole,
        )


class WavLMSpeakerVerifier:
    """WavLM x-vector verifier used only for dual-model disagreements."""

    def __init__(self, device: str = "cuda") -> None:
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self.device = torch.device(
            "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            str(WAVLM_SV_MODEL), local_files_only=True
        )
        self.model = WavLMForXVector.from_pretrained(
            str(WAVLM_SV_MODEL), local_files_only=True
        ).eval().to(self.device)

    def close(self) -> None:
        if hasattr(self, "model"):
            del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _embeddings_from_waveforms(
        self,
        waveforms: Sequence[torch.Tensor],
        batch_size: int = 16,
        progress: Callable[[int, int], None] | None = None,
    ) -> torch.Tensor:
        if not waveforms:
            return torch.empty((0, 0))
        prepared: list[torch.Tensor] = []
        for waveform in waveforms:
            value = waveform.detach().float().cpu().flatten()
            if value.numel() < 8000:
                value = F.pad(value, (0, 8000 - value.numel()))
            prepared.append(value)

        output: list[torch.Tensor] = []
        completed = 0
        for offset in range(0, len(prepared), max(1, batch_size)):
            batch = prepared[offset : offset + max(1, batch_size)]
            inputs = self.feature_extractor(
                [value.numpy() for value in batch],
                sampling_rate=16000,
                padding=True,
                return_tensors="pt",
            )
            attention_mask = inputs.get("attention_mask")
            with torch.inference_mode():
                embeddings = self.model(
                    input_values=inputs["input_values"].to(self.device),
                    attention_mask=(
                        attention_mask.to(self.device)
                        if attention_mask is not None
                        else None
                    ),
                ).embeddings
                embeddings = F.normalize(embeddings.float(), p=2, dim=1).cpu()
            output.extend(embeddings)
            completed += len(batch)
            if progress is not None:
                progress(completed, len(prepared))
        return torch.stack(output)

    def _embedding_from_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        return self._embeddings_from_waveforms([waveform], batch_size=1)[0]

    def build_profile(
        self,
        reference_paths: list[Path],
        reference_indexes: Sequence[int] | None = None,
    ) -> WavLMProfile:
        all_raw = self._embeddings_from_waveforms(
            [load_mono(path, 16000) for path in reference_paths]
        )
        if reference_indexes is None:
            selected = SpeakerVerifier._reference_core_mask(all_raw, floor=0.72)
            indexes = torch.arange(len(all_raw))[selected]
        else:
            indexes = torch.tensor(list(reference_indexes), dtype=torch.long)
            if indexes.numel() == 0:
                raise ValueError("至少需要一段有效参考音频")
        raw = all_raw[indexes]
        centroid = F.normalize(raw.mean(dim=0), p=2, dim=0)
        scores = raw @ centroid
        low_reference = (
            float(torch.quantile(scores, 0.10)) if len(scores) > 1 else float(scores[0])
        )
        # WavLM x-vector scores are tightly clustered near one for clean
        # references. Keep a bounded channel/emotion allowance for separated
        # episode audio; final rescue still requires CAM++ dominance.
        acceptance_floor = max(0.82, min(0.90, low_reference - 0.10))
        return WavLMProfile(
            embeddings=raw,
            centroid=centroid,
            reference_scores=[round(float(value), 5) for value in scores],
            acceptance_floor=round(acceptance_floor, 5),
            reference_indexes=tuple(int(value) for value in indexes.tolist()),
        )

    def verify_waveform(
        self,
        waveform: torch.Tensor,
        profile: WavLMProfile,
        window_seconds: float = 1.8,
        hop_seconds: float = 0.9,
    ) -> WavLMDecision:
        whole = self._embedding_from_waveform(waveform)
        score = float(whole @ profile.centroid)
        reference_scores = profile.embeddings @ whole

        window_size = int(window_seconds * 16000)
        hop = int(hop_seconds * 16000)
        if waveform.numel() <= window_size:
            windows = [waveform]
        else:
            starts = list(range(0, waveform.numel() - window_size + 1, hop))
            last = waveform.numel() - window_size
            if not starts or starts[-1] != last:
                starts.append(last)
            windows = []
            for start in starts:
                part = waveform[start : start + window_size]
                rms = torch.sqrt(torch.mean(part.square()) + 1e-12)
                if float(20 * torch.log10(rms + 1e-12)) > -48.0:
                    windows.append(part)
        if not windows:
            windows = [waveform]
        window_embeddings = self._embeddings_from_waveforms(windows)
        window_scores = window_embeddings @ profile.centroid
        return WavLMDecision(
            score=round(score, 5),
            reference_median_score=round(float(reference_scores.median()), 5),
            reference_max_score=round(float(reference_scores.max()), 5),
            window_min_score=round(float(window_scores.min()), 5),
            window_p20_score=round(float(torch.quantile(window_scores, 0.20)), 5),
            window_scores=[round(float(value), 5) for value in window_scores],
        )


class WeSpeakerVerifier:
    """Deterministic CPU ONNX verifier used for residual speaker ambiguity."""

    SAMPLE_RATE = 16000
    WINDOW_SECONDS = 3.0
    HOP_SECONDS = 1.5

    def __init__(self) -> None:
        import onnxruntime

        # CPU execution is fast for the small ambiguity batch and avoids
        # provider-dependent CUDA kernel variation between identical runs.
        self.session = onnxruntime.InferenceSession(
            str(WESPEAKER_MODEL),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def close(self) -> None:
        if hasattr(self, "session"):
            del self.session
        gc.collect()

    @classmethod
    def _windows(cls, waveform: torch.Tensor) -> list[torch.Tensor]:
        value = waveform.detach().float().cpu().flatten()
        window = int(round(cls.WINDOW_SECONDS * cls.SAMPLE_RATE))
        hop = int(round(cls.HOP_SECONDS * cls.SAMPLE_RATE))
        if value.numel() <= window:
            return [value]
        starts = list(range(0, value.numel() - window + 1, hop))
        last = value.numel() - window
        if not starts or starts[-1] != last:
            starts.append(last)
        return [value[start : start + window] for start in starts]

    @classmethod
    def _fbank(cls, waveform: torch.Tensor) -> torch.Tensor:
        features = torchaudio.compliance.kaldi.fbank(
            (waveform * 32768.0).unsqueeze(0),
            num_mel_bins=80,
            frame_length=25.0,
            frame_shift=10.0,
            dither=0.0,
            sample_frequency=float(cls.SAMPLE_RATE),
            window_type="hamming",
            use_energy=False,
        )
        return features - features.mean(dim=0, keepdim=True)

    def _window_embeddings(
        self,
        waveforms: Sequence[torch.Tensor],
        batch_size: int = 64,
        progress: Callable[[int, int], None] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        features: list[torch.Tensor] = []
        ranges: list[tuple[int, int]] = []
        for waveform in waveforms:
            start = len(features)
            features.extend(self._fbank(window) for window in self._windows(waveform))
            ranges.append((start, len(features)))
        if not features:
            return torch.empty((0, 0)), ranges

        output: list[torch.Tensor | None] = [None] * len(features)
        groups: dict[int, list[int]] = {}
        for index, feature in enumerate(features):
            groups.setdefault(feature.shape[0], []).append(index)
        completed = 0
        for frame_count in sorted(groups):
            indexes = groups[frame_count]
            for offset in range(0, len(indexes), max(1, batch_size)):
                batch_indexes = indexes[offset : offset + max(1, batch_size)]
                batch = torch.stack([features[index] for index in batch_indexes]).numpy()
                embeddings = self.session.run(
                    [self.output_name],
                    {self.input_name: batch},
                )[0]
                values = F.normalize(torch.from_numpy(embeddings).float(), p=2, dim=1)
                for index, value in zip(batch_indexes, values):
                    output[index] = value
                completed += len(values)
                if progress is not None:
                    progress(completed, len(features))
        if any(value is None for value in output):
            raise RuntimeError("WeSpeaker 批量嵌入结果不完整")
        return torch.stack([value for value in output if value is not None]), ranges

    def embeddings_from_waveforms(
        self,
        waveforms: Sequence[torch.Tensor],
        batch_size: int = 64,
        progress: Callable[[int, int], None] | None = None,
    ) -> torch.Tensor:
        window_embeddings, ranges = self._window_embeddings(
            waveforms,
            batch_size=batch_size,
            progress=progress,
        )
        if not ranges:
            return torch.empty((0, 0))
        return torch.stack(
            [
                F.normalize(window_embeddings[start:end].mean(dim=0), p=2, dim=0)
                for start, end in ranges
            ]
        )

    def build_profile(
        self,
        reference_paths: list[Path],
        reference_indexes: Sequence[int] | None = None,
    ) -> WeSpeakerProfile:
        all_embeddings = self.embeddings_from_waveforms(
            [load_mono(path, self.SAMPLE_RATE) for path in reference_paths]
        )
        if reference_indexes is None:
            indexes = torch.arange(len(all_embeddings))
        else:
            indexes = torch.tensor(list(reference_indexes), dtype=torch.long)
        if indexes.numel() == 0:
            raise ValueError("至少需要一段有效参考音频")
        embeddings = all_embeddings[indexes]
        centroid = F.normalize(embeddings.mean(dim=0), p=2, dim=0)
        scores = embeddings @ centroid
        return WeSpeakerProfile(
            embeddings=embeddings,
            centroid=centroid,
            reference_scores=[round(float(value), 5) for value in scores],
            reference_indexes=tuple(int(value) for value in indexes.tolist()),
        )

    def verify_waveform(
        self,
        waveform: torch.Tensor,
        profile: WeSpeakerProfile,
    ) -> WeSpeakerDecision:
        window_embeddings, ranges = self._window_embeddings([waveform])
        start, end = ranges[0]
        windows = window_embeddings[start:end]
        whole = F.normalize(windows.mean(dim=0), p=2, dim=0)
        scores = windows @ profile.centroid
        return WeSpeakerDecision(
            score=round(float(whole @ profile.centroid), 5),
            reference_max_score=round(float((profile.embeddings @ whole).max()), 5),
            window_min_score=round(float(scores.min()), 5),
            window_p20_score=round(float(torch.quantile(scores, 0.20)), 5),
            window_scores=[round(float(value), 5) for value in scores],
        )


class DualSpeakerVerifier:
    """Classify target-speaker turns using ERes2Net and CAM++ consensus."""

    SHORT_MIN_DURATION = 0.55
    FULL_MIN_DURATION = 2.20
    BALANCED_MIN_DURATION = 3.00

    @staticmethod
    def _window_continuity(
        primary: SpeakerDecision,
        secondary: SpeakerDecision,
        duration: float,
        threshold: float,
    ) -> bool:
        """Reject turns containing a sustained low-evidence speaker window."""

        pairs = list(zip(primary.window_scores, secondary.window_scores))
        if not pairs or duration < 1.8:
            return True
        primary_floor = max(0.32, threshold - 0.35)
        secondary_floor = max(0.28, threshold - 0.38)
        low_windows = sum(
            primary_score < primary_floor and secondary_score < secondary_floor
            for primary_score, secondary_score in pairs
        )
        # One uncertain window can be a breath, overlap residue, or a quiet
        # phoneme. More than 20% low windows is usually a speaker switch.
        return low_windows / len(pairs) <= 0.20

    @staticmethod
    def _paired_reference_median(
        primary: SpeakerDecision,
        secondary: SpeakerDecision,
        primary_profile: SpeakerProfile,
        secondary_profile: CAMPlusProfile,
    ) -> float:
        """Pair scores from the same source reference before taking the median."""

        if primary.embedding is None or secondary.embedding is None:
            return (
                primary.reference_median_score + secondary.reference_median_score
            ) / 2.0

        primary_scores = primary_profile.embeddings @ primary.embedding
        secondary_scores = secondary_profile.embeddings @ secondary.embedding
        primary_indexes = primary_profile.reference_indexes or tuple(range(len(primary_scores)))
        secondary_indexes = secondary_profile.reference_indexes or tuple(
            range(len(secondary_scores))
        )
        secondary_by_index = {
            index: float(score)
            for index, score in zip(secondary_indexes, secondary_scores)
        }
        paired_scores = [
            (float(score) + secondary_by_index[index]) / 2.0
            for index, score in zip(primary_indexes, primary_scores)
            if index in secondary_by_index
        ]
        if not paired_scores:
            return (
                primary.reference_median_score + secondary.reference_median_score
            ) / 2.0
        return float(torch.tensor(paired_scores).median())

    @classmethod
    def _classify_match(
        cls,
        duration: float,
        primary: SpeakerDecision,
        secondary: SpeakerDecision,
        paired_reference_median: float,
        threshold: float = 0.70,
    ) -> SpeakerMatchTier:
        if not cls._window_continuity(primary, secondary, duration, threshold):
            return "rejected"
        short_strong = (
            cls.SHORT_MIN_DURATION <= duration < cls.FULL_MIN_DURATION
            and primary.score >= max(0.70, threshold)
            and secondary.score >= max(0.68, threshold - 0.02)
            and primary.reference_median_score >= 0.58
            and secondary.reference_median_score >= 0.56
            and paired_reference_median >= 0.58
            and primary.reference_spread <= 0.12
            and secondary.reference_spread <= 0.12
        )
        if short_strong:
            return "short_strong"

        strong = (
            duration >= cls.FULL_MIN_DURATION
            and primary.score >= max(0.58, threshold - 0.10)
            and secondary.score >= max(0.60, threshold - 0.08)
            and paired_reference_median >= max(0.50, threshold - 0.20)
            and primary.reference_spread <= 0.15
        )
        if strong:
            return "strong"

        balanced = (
            duration >= cls.BALANCED_MIN_DURATION
            and primary.score >= max(0.54, threshold - 0.14)
            and secondary.score >= max(0.54, threshold - 0.14)
            and primary.reference_median_score >= max(0.47, threshold - 0.23)
            and secondary.reference_median_score >= max(0.46, threshold - 0.24)
            and paired_reference_median >= max(0.48, threshold - 0.22)
            and primary.reference_spread <= 0.13
        )
        if balanced:
            return "balanced"

        # Recall tier: both independent models still need direct evidence from
        # the reference set, but this accepts natural cross-language/emotional
        # variation that misses the conservative balanced tier.  It is not a
        # global threshold drop because the reference medians, spreads and
        # local window evidence remain mandatory.
        recall = (
            duration >= cls.SHORT_MIN_DURATION
            and primary.score >= max(0.54, threshold - 0.16)
            and secondary.score >= max(0.46, threshold - 0.24)
            and paired_reference_median >= max(0.43, threshold - 0.27)
            and primary.reference_median_score >= 0.42
            and secondary.reference_median_score >= 0.36
            and primary.reference_max_score >= 0.50
            and secondary.reference_max_score >= 0.40
            and primary.reference_spread <= 0.20
            and secondary.reference_spread <= 0.22
            and max(primary.window_p20_score, secondary.window_p20_score) >= 0.34
            and max(primary.window_vote_ratio, secondary.window_vote_ratio) >= 0.25
        )
        if recall:
            return "recall"

        weak = (
            duration >= cls.FULL_MIN_DURATION
            and primary.score >= 0.62
            and secondary.score >= 0.40
            and primary.reference_median_score >= 0.52
            and paired_reference_median >= 0.47
            and primary.reference_spread <= 0.14
        )
        return "weak" if weak else "rejected"

    def __init__(
        self,
        device: str = "cuda",
        status: Callable[[str], None] | None = None,
    ) -> None:
        self.device = device
        self.status = status or (lambda _message: None)
        self.primary = SpeakerVerifier(device)
        self.secondary: CAMPlusVerifier | None = None
        self.secondary_profile: CAMPlusProfile | None = None
        self.tertiary: WavLMSpeakerVerifier | None = None
        self.tertiary_profile: WavLMProfile | None = None
        self.tertiary_target_spans: list[TimeSpan] = []
        self.quaternary: WeSpeakerVerifier | None = None
        self.quaternary_profile: WeSpeakerProfile | None = None
        # Boundary detection can run before a reference profile is built.  Keep
        # its CAM++ instance separate until ``_ensure_secondary`` can attach a
        # profile, then reuse the same model to avoid loading it twice.
        self.boundary_secondary: CAMPlusVerifier | None = None

    def close(self) -> None:
        self.primary.close()
        closed: set[int] = set()
        for verifier in (self.secondary, self.boundary_secondary):
            if verifier is not None and id(verifier) not in closed:
                verifier.close()
                closed.add(id(verifier))
        if self.tertiary is not None:
            self.tertiary.close()
        if self.quaternary is not None:
            self.quaternary.close()

    def build_profile(self, reference_paths: list[Path], base_threshold: float) -> SpeakerMatchProfile:
        return SpeakerMatchProfile(
            primary=self.primary.build_profile(reference_paths, base_threshold),
            reference_paths=list(reference_paths),
            base_threshold=base_threshold,
        )

    def build_exclusion_profiles(
        self,
        reference_groups: Sequence[Sequence[Path]],
        target_profile: SpeakerMatchProfile,
    ) -> list[ExclusionSpeakerProfile]:
        """Build anonymous per-role profiles used only as negative boundaries."""

        self._ensure_secondary(target_profile)
        assert self.secondary is not None
        output: list[ExclusionSpeakerProfile] = []
        for index, paths in enumerate(reference_groups, start=1):
            group = [Path(path) for path in paths if path]
            if not group:
                continue
            primary = self.primary.build_profile(group, target_profile.base_threshold)
            secondary = self.secondary.build_profile(
                group,
                reference_indexes=primary.reference_indexes,
            )
            output.append(
                ExclusionSpeakerProfile(
                    label=f"排除角色 {index}",
                    primary=primary,
                    secondary=secondary,
                    reference_paths=group,
                )
            )
        return output

    def exclusion_audit(
        self,
        match: SpeakerMatchDecision,
        target_profile: SpeakerMatchProfile,
        exclusion_profiles: Sequence[ExclusionSpeakerProfile],
        *,
        tertiary_recovery: bool = False,
    ) -> dict[str, float | str | bool] | None:
        """Reject when both speaker models prefer the same excluded role."""

        secondary = match.secondary
        if (
            not exclusion_profiles
            or match.primary.embedding is None
            or secondary is None
            or secondary.embedding is None
        ):
            return None
        target_secondary = self._ensure_secondary(target_profile)
        primary_embedding = match.primary.embedding
        secondary_embedding = secondary.embedding
        target_primary_score = float(primary_embedding @ target_profile.primary.centroid)
        target_secondary_score = float(secondary_embedding @ target_secondary.centroid)
        target_primary_max = float(
            (target_profile.primary.embeddings @ primary_embedding).max()
        )
        target_secondary_max = float(
            (target_secondary.embeddings @ secondary_embedding).max()
        )

        strongest: dict[str, float | str | bool] | None = None
        strongest_margin = float("inf")
        for exclusion in exclusion_profiles:
            negative_primary_score = float(primary_embedding @ exclusion.primary.centroid)
            negative_secondary_score = float(
                secondary_embedding @ exclusion.secondary.centroid
            )
            negative_primary_max = float(
                (exclusion.primary.embeddings @ primary_embedding).max()
            )
            negative_secondary_max = float(
                (exclusion.secondary.embeddings @ secondary_embedding).max()
            )
            primary_margin = target_primary_score - negative_primary_score
            secondary_margin = target_secondary_score - negative_secondary_score
            primary_direct_margin = target_primary_max - negative_primary_max
            secondary_direct_margin = target_secondary_max - negative_secondary_max
            primary_negative_vote = (
                primary_margin <= 0.02 and primary_direct_margin <= 0.02
            )
            secondary_negative_vote = (
                secondary_margin <= 0.02 and secondary_direct_margin <= 0.02
            )
            # A WavLM rescue is deliberately used only for dual-model
            # disagreements.  If one lightweight model still votes for a
            # supplied exclusion person, the other model must beat that person
            # by a clear margin in both centroid and direct-reference space.
            # This keeps genuine cross-model rescues while preventing WavLM
            # from overturning a near-tie against a known negative voice.
            tertiary_conflict = (tertiary_recovery or match.tier == "tertiary") and (
                (
                    primary_negative_vote
                    and (
                        secondary_margin < 0.15
                        or secondary_direct_margin < 0.15
                    )
                )
                or (
                    secondary_negative_vote
                    and (
                        primary_margin < 0.15
                        or primary_direct_margin < 0.15
                    )
                )
            )
            rejected = (
                primary_negative_vote and secondary_negative_vote
            ) or tertiary_conflict
            combined_margin = primary_margin + secondary_margin
            diagnostics: dict[str, float | str | bool] = {
                "excluded_role_rejected": rejected,
                "excluded_role": exclusion.label,
                "excluded_primary_margin": round(primary_margin, 5),
                "excluded_secondary_margin": round(secondary_margin, 5),
                "excluded_primary_direct_margin": round(primary_direct_margin, 5),
                "excluded_secondary_direct_margin": round(secondary_direct_margin, 5),
                "excluded_primary_vote": primary_negative_vote,
                "excluded_secondary_vote": secondary_negative_vote,
                "excluded_tertiary_conflict": tertiary_conflict,
            }
            if rejected:
                return diagnostics
            if combined_margin < strongest_margin:
                strongest_margin = combined_margin
                strongest = diagnostics
        return strongest

    def _ensure_secondary(self, profile: SpeakerMatchProfile) -> CAMPlusProfile:
        if self.secondary is None:
            self.status("边界候选出现：正在加载 CAM++ 二次声纹模型")
            self.secondary = self.boundary_secondary or CAMPlusVerifier(self.device)
            self.boundary_secondary = self.secondary
            # Reuse the primary model's retained source indexes. Filtering the
            # two models independently can pair different clips and makes an
            # extra, outlier reference reduce matching accuracy.
            self.secondary_profile = self.secondary.build_profile(
                profile.reference_paths,
                reference_indexes=profile.primary.reference_indexes,
            )
            self.status("CAM++ 二次声纹模型已就绪")
        assert self.secondary_profile is not None
        return self.secondary_profile

    def _ensure_tertiary(self, profile: SpeakerMatchProfile) -> WavLMProfile:
        if self.tertiary is None:
            self.status("检测到双模型歧义：正在加载 WavLM 第三声纹模型")
            self.tertiary = WavLMSpeakerVerifier(self.device)
            self.tertiary_profile = self.tertiary.build_profile(
                profile.reference_paths,
                reference_indexes=profile.primary.reference_indexes,
            )
            self.status("WavLM 第三声纹模型已就绪")
        assert self.tertiary_profile is not None
        return self.tertiary_profile

    def _tertiary_pair(
        self, profile: SpeakerMatchProfile
    ) -> tuple[WavLMSpeakerVerifier, WavLMProfile]:
        tertiary_profile = self._ensure_tertiary(profile)
        assert self.tertiary is not None
        return self.tertiary, tertiary_profile

    def quaternary_pair(
        self, profile: SpeakerMatchProfile
    ) -> tuple[WeSpeakerVerifier, WeSpeakerProfile]:
        if self.quaternary is None:
            self.status("正在加载 WeSpeaker 第四声纹裁决模型")
            self.quaternary = WeSpeakerVerifier()
            self.quaternary_profile = self.quaternary.build_profile(
                profile.reference_paths,
                reference_indexes=profile.primary.reference_indexes,
            )
            self.status("WeSpeaker 第四声纹裁决模型已就绪")
        assert self.quaternary_profile is not None
        return self.quaternary, self.quaternary_profile

    def multimodel_exclusion_audit(
        self,
        match: SpeakerMatchDecision,
        target_profile: SpeakerMatchProfile,
        fourth_embedding: torch.Tensor,
        fourth_target: WeSpeakerProfile,
        exclusion_profiles: Sequence[ExclusionSpeakerProfile],
    ) -> dict[str, float | str | bool] | None:
        """Veto late recovery when two of three models prefer one exclusion."""

        secondary = match.secondary
        if (
            not exclusion_profiles
            or match.primary.embedding is None
            or secondary is None
            or secondary.embedding is None
        ):
            return None
        assert self.quaternary is not None
        target_secondary = self._ensure_secondary(target_profile)
        primary_embedding = match.primary.embedding
        secondary_embedding = secondary.embedding
        target_scores = {
            "primary_centroid": float(
                primary_embedding @ target_profile.primary.centroid
            ),
            "primary_direct": float(
                (target_profile.primary.embeddings @ primary_embedding).max()
            ),
            "secondary_centroid": float(
                secondary_embedding @ target_secondary.centroid
            ),
            "secondary_direct": float(
                (target_secondary.embeddings @ secondary_embedding).max()
            ),
            "fourth_centroid": float(fourth_embedding @ fourth_target.centroid),
            "fourth_direct": float(
                (fourth_target.embeddings @ fourth_embedding).max()
            ),
        }
        strongest: dict[str, float | str | bool] | None = None
        strongest_margin = float("inf")
        all_models_clear = True
        for exclusion in exclusion_profiles:
            if exclusion.quaternary is None:
                exclusion.quaternary = self.quaternary.build_profile(
                    exclusion.reference_paths
                )
            negative_fourth = exclusion.quaternary
            margins = {
                "primary_centroid": target_scores["primary_centroid"]
                - float(primary_embedding @ exclusion.primary.centroid),
                "primary_direct": target_scores["primary_direct"]
                - float((exclusion.primary.embeddings @ primary_embedding).max()),
                "secondary_centroid": target_scores["secondary_centroid"]
                - float(secondary_embedding @ exclusion.secondary.centroid),
                "secondary_direct": target_scores["secondary_direct"]
                - float(
                    (exclusion.secondary.embeddings @ secondary_embedding).max()
                ),
                "fourth_centroid": target_scores["fourth_centroid"]
                - float(fourth_embedding @ negative_fourth.centroid),
                "fourth_direct": target_scores["fourth_direct"]
                - float((negative_fourth.embeddings @ fourth_embedding).max()),
            }
            primary_vote = (
                margins["primary_centroid"] <= 0.02
                and margins["primary_direct"] <= 0.02
            )
            secondary_vote = (
                margins["secondary_centroid"] <= 0.02
                and margins["secondary_direct"] <= 0.02
            )
            fourth_vote = (
                margins["fourth_centroid"] <= 0.02
                and margins["fourth_direct"] <= 0.02
            )
            vote_count = sum((primary_vote, secondary_vote, fourth_vote))
            role_all_models_clear = all(value >= 0.02 for value in margins.values())
            all_models_clear = all_models_clear and role_all_models_clear
            diagnostics: dict[str, float | str | bool] = {
                "excluded_role_rejected": vote_count >= 2,
                "excluded_all_models_clear": role_all_models_clear,
                "excluded_role": exclusion.label,
                "excluded_primary_margin": round(
                    margins["primary_centroid"], 5
                ),
                "excluded_secondary_margin": round(
                    margins["secondary_centroid"], 5
                ),
                "excluded_wespeaker_margin": round(
                    margins["fourth_centroid"], 5
                ),
                "excluded_primary_direct_margin": round(
                    margins["primary_direct"], 5
                ),
                "excluded_secondary_direct_margin": round(
                    margins["secondary_direct"], 5
                ),
                "excluded_wespeaker_direct_margin": round(
                    margins["fourth_direct"], 5
                ),
                "excluded_primary_vote": primary_vote,
                "excluded_secondary_vote": secondary_vote,
                "excluded_wespeaker_vote": fourth_vote,
                "excluded_multimodel_vote_count": vote_count,
            }
            if vote_count >= 2:
                return diagnostics
            total_margin = (
                margins["primary_centroid"]
                + margins["secondary_centroid"]
                + margins["fourth_centroid"]
            )
            if total_margin < strongest_margin:
                strongest_margin = total_margin
                strongest = diagnostics
        if strongest is not None:
            strongest["excluded_all_models_clear"] = all_models_clear
        return strongest

    def promote_with_tertiary(
        self,
        waveform: torch.Tensor,
        profile: SpeakerMatchProfile,
        match: SpeakerMatchDecision,
        duration: float,
    ) -> SpeakerMatchDecision | None:
        """Recover only CAM++-dominant turns independently supported by WavLM."""

        secondary = match.secondary
        if (
            secondary is None
            or match.accepted
            or match.tier not in {"recall", "rejected"}
            or not 2.20 <= duration <= 15.0
            or match.primary.score < 0.40
            or secondary.score < 0.53
            or secondary.score - match.primary.score < 0.03
            or match.primary.reference_max_score < 0.38
            or secondary.reference_max_score < 0.48
        ):
            return None
        tertiary_profile = self._ensure_tertiary(profile)
        assert self.tertiary is not None
        decision = self.tertiary.verify_waveform(waveform, tertiary_profile)
        floor = tertiary_profile.acceptance_floor
        if (
            decision.score < floor
            or decision.reference_max_score < floor
            or decision.window_p20_score < floor - 0.08
        ):
            return None
        diagnostics = {
            **match.diagnostics,
            "speaker_tier": "tertiary",
            "wavlm_rescue": True,
            "wavlm_score": decision.score,
            "wavlm_reference_median": decision.reference_median_score,
            "wavlm_reference_max": decision.reference_max_score,
            "wavlm_window_p20": decision.window_p20_score,
            "wavlm_acceptance_floor": floor,
        }
        return SpeakerMatchDecision(
            accepted=True,
            primary=match.primary,
            match_mode="tertiary",
            secondary=secondary,
            tier="tertiary",
            merge_only=False,
            paired_reference_median=match.paired_reference_median,
            diagnostics=diagnostics,
        )

    def promote_local_with_tertiary(
        self,
        waveform: torch.Tensor,
        profile: SpeakerMatchProfile,
        match: SpeakerMatchDecision,
        duration: float,
    ) -> SpeakerMatchDecision | None:
        """Confirm a boundary fragment without weakening the global speaker gate.

        Local change detection can split one target utterance into sub-two-second
        phonetic fragments.  Their ERes2Net/CAM++ whole-clip scores are less
        stable than a complete sentence, so WavLM may confirm them only when
        both lightweight models still provide minimum direct support.  Callers
        must separately apply any supplied exclusion-speaker profiles before
        using this recovery path.
        """

        secondary = match.secondary
        if (
            secondary is None
            or match.accepted
            or not 0.75 <= duration <= 8.0
            or match.primary.score < 0.48
            or secondary.score < 0.44
            or match.primary.reference_max_score < 0.43
            or secondary.reference_max_score < 0.40
        ):
            return None
        tertiary_profile = self._ensure_tertiary(profile)
        assert self.tertiary is not None
        window_seconds = min(1.2, max(0.8, duration))
        hop_seconds = min(0.45, max(0.25, duration / 2.0))
        decision = self.tertiary.verify_waveform(
            waveform,
            tertiary_profile,
            window_seconds=window_seconds,
            hop_seconds=hop_seconds,
        )
        floor = tertiary_profile.acceptance_floor
        if (
            decision.score < floor
            or decision.reference_max_score < floor
            or decision.window_p20_score < floor - 0.05
        ):
            return None
        diagnostics = {
            **match.diagnostics,
            "speaker_tier": "tertiary",
            "wavlm_local_rescue": True,
            "wavlm_score": decision.score,
            "wavlm_reference_median": decision.reference_median_score,
            "wavlm_reference_max": decision.reference_max_score,
            "wavlm_window_p20": decision.window_p20_score,
            "wavlm_acceptance_floor": floor,
        }
        return SpeakerMatchDecision(
            accepted=True,
            primary=match.primary,
            match_mode="tertiary",
            secondary=secondary,
            tier="tertiary",
            merge_only=False,
            paired_reference_median=match.paired_reference_median,
            diagnostics=diagnostics,
        )

    def promote_contrastive_edge_with_tertiary(
        self,
        core_waveform: torch.Tensor,
        residual_waveform: torch.Tensor,
        profile: SpeakerMatchProfile,
        core_match: SpeakerMatchDecision,
        residual_match: SpeakerMatchDecision,
        core_duration: float,
        residual_duration: float,
    ) -> SpeakerMatchDecision | None:
        """Confirm a target core only when its short edge is clearly another voice."""

        core_secondary = core_match.secondary
        residual_secondary = residual_match.secondary
        if (
            core_secondary is None
            or residual_secondary is None
            or residual_match.accepted
            or not 0.25 <= residual_duration <= 0.90
            or residual_match.primary.reference_max_score > 0.42
            or residual_secondary.reference_max_score > 0.35
        ):
            return None

        primary_score_margin = (
            core_match.primary.score - residual_match.primary.score
        )
        primary_direct_margin = (
            core_match.primary.reference_max_score
            - residual_match.primary.reference_max_score
        )
        secondary_score_margin = core_secondary.score - residual_secondary.score
        secondary_direct_margin = (
            core_secondary.reference_max_score
            - residual_secondary.reference_max_score
        )
        if min(
            primary_score_margin,
            primary_direct_margin,
            secondary_score_margin,
            secondary_direct_margin,
        ) < 0.15:
            return None

        promoted = self.promote_local_with_tertiary(
            core_waveform,
            profile,
            core_match,
            core_duration,
        )
        if promoted is None:
            return None

        assert self.tertiary is not None
        tertiary_profile = self._ensure_tertiary(profile)
        residual_window = min(0.80, max(0.40, residual_duration))
        residual_hop = min(0.30, max(0.20, residual_duration / 2.0))
        residual_tertiary = self.tertiary.verify_waveform(
            residual_waveform,
            tertiary_profile,
            window_seconds=residual_window,
            hop_seconds=residual_hop,
        )
        floor = tertiary_profile.acceptance_floor
        core_tertiary_score = float(promoted.diagnostics["wavlm_score"])
        core_tertiary_direct = float(
            promoted.diagnostics["wavlm_reference_max"]
        )
        if (
            residual_tertiary.reference_max_score > floor - 0.15
            or core_tertiary_score - residual_tertiary.score < 0.25
            or core_tertiary_direct - residual_tertiary.reference_max_score < 0.25
        ):
            return None

        diagnostics = {
            **promoted.diagnostics,
            "contrastive_edge_tertiary_passed": True,
            "contrastive_primary_score_margin": round(primary_score_margin, 5),
            "contrastive_primary_direct_margin": round(primary_direct_margin, 5),
            "contrastive_secondary_score_margin": round(secondary_score_margin, 5),
            "contrastive_secondary_direct_margin": round(
                secondary_direct_margin, 5
            ),
            "contrastive_residual_eres_score": residual_match.primary.score,
            "contrastive_residual_eres_reference_max": (
                residual_match.primary.reference_max_score
            ),
            "contrastive_residual_camplus_score": residual_secondary.score,
            "contrastive_residual_camplus_reference_max": (
                residual_secondary.reference_max_score
            ),
            "contrastive_residual_wavlm_score": residual_tertiary.score,
            "contrastive_residual_wavlm_reference_max": (
                residual_tertiary.reference_max_score
            ),
            "contrastive_wavlm_score_margin": round(
                core_tertiary_score - residual_tertiary.score, 5
            ),
            "contrastive_wavlm_direct_margin": round(
                core_tertiary_direct - residual_tertiary.reference_max_score,
                5,
            ),
        }
        return SpeakerMatchDecision(
            accepted=True,
            primary=promoted.primary,
            match_mode="tertiary",
            secondary=promoted.secondary,
            tier="tertiary",
            merge_only=False,
            paired_reference_median=promoted.paired_reference_median,
            diagnostics=diagnostics,
        )

    def verify(
        self,
        candidate_path: Path,
        profile: SpeakerMatchProfile,
        threshold: float,
        duration: float,
        window_seconds: float = 1.8,
        hop_seconds: float = 0.9,
    ) -> SpeakerMatchDecision:
        return self.verify_waveform(
            load_mono(candidate_path, 16000),
            profile,
            threshold,
            duration=duration,
            window_seconds=window_seconds,
            hop_seconds=hop_seconds,
        )

    def verify_waveform(
        self,
        waveform: torch.Tensor,
        profile: SpeakerMatchProfile,
        threshold: float,
        duration: float,
        window_seconds: float = 1.8,
        hop_seconds: float = 0.9,
    ) -> SpeakerMatchDecision:
        primary = self.primary.verify_waveform(
            waveform,
            profile.primary,
            threshold,
            window_seconds=window_seconds,
            hop_seconds=hop_seconds,
        )
        secondary_profile = self._ensure_secondary(profile)
        assert self.secondary is not None
        secondary_threshold = max(0.56, min(0.62, threshold - 0.12))
        secondary = self.secondary.verify_waveform(
            waveform,
            secondary_profile,
            secondary_threshold,
            window_seconds=window_seconds,
            hop_seconds=hop_seconds,
        )
        paired_reference_median = self._paired_reference_median(
            primary,
            secondary,
            profile.primary,
            secondary_profile,
        )
        tier = self._classify_match(
            duration,
            primary,
            secondary,
            paired_reference_median,
            threshold,
        )
        # Recall is diagnostic-only. Weak candidates must not enter a training
        # dataset merely because an episode-local centroid happened to agree.
        accepted = tier in {"short_strong", "strong", "balanced"}
        diagnostics: dict[str, float | str | bool] = {
            "duration": round(float(duration), 5),
            "speaker_tier": tier,
            "merge_only": tier == "weak",
            "eres_score": primary.score,
            "camplus_score": secondary.score,
            "eres_reference_median": primary.reference_median_score,
            "eres_reference_max": primary.reference_max_score,
            "eres_reference_spread": primary.reference_spread,
            "camplus_reference_median": secondary.reference_median_score,
            "camplus_reference_max": secondary.reference_max_score,
            "camplus_reference_spread": secondary.reference_spread,
            "paired_reference_median": round(paired_reference_median, 5),
        }
        return SpeakerMatchDecision(
            accepted=accepted,
            primary=primary,
            match_mode=tier,
            secondary=secondary,
            tier=tier,
            merge_only=tier == "weak",
            paired_reference_median=round(paired_reference_median, 5),
            diagnostics=diagnostics,
        )

    def embedding_pair(self, candidate_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized embeddings from both verifier models."""

        waveform = load_mono(candidate_path, 16000)
        if self.secondary is None:
            raise RuntimeError("CAM++ 声纹模型尚未建立参考 profile")
        return (
            self.primary._embedding_from_waveform(waveform),
            self.secondary._embedding_from_waveform(waveform),
        )

    def split_speaker_spans(
        self,
        audio_path: Path,
        spans: Sequence[TimeSpan],
        *,
        progress: Callable[[float, str], None] | None = None,
        **kwargs: float,
    ) -> list[TimeSpan]:
        """Split VAD spans at local speaker changes before transcription."""

        splitter = LocalSpeakerTurnSplitter(
            self.primary,
            secondary_factory=self._boundary_secondary_factory,
        )
        return splitter.split_speaker_spans(audio_path, spans, progress=progress, **kwargs)

    def locate_target_spans(
        self,
        audio_path: Path,
        spans: Sequence[TimeSpan],
        profile: SpeakerMatchProfile,
        *,
        progress: Callable[[float, str], None] | None = None,
        **kwargs: float,
    ) -> list[TimeSpan]:
        """Find continuous target-speaker regions before transcription."""

        secondary_profile = self._ensure_secondary(profile)
        assert self.secondary is not None
        locator = TargetSpeakerSpanLocator(
            self.primary,
            self.secondary,
            profile.primary,
            secondary_profile,
            tertiary_factory=lambda: self._tertiary_pair(profile),
        )
        located = locator.locate(audio_path, spans, progress=progress, **kwargs)
        self.tertiary_target_spans = list(locator.tertiary_target_spans)
        return located

    def _boundary_secondary_factory(self) -> CAMPlusVerifier:
        if self.boundary_secondary is None:
            self.status("检测到疑似换人：正在加载 CAM++ 边界模型")
            self.boundary_secondary = self.secondary or CAMPlusVerifier(self.device)
            self.status("CAM++ 边界模型已就绪")
        return self.boundary_secondary


class TargetSpeakerSpanLocator:
    """Classify local windows as target/non-target and form target turns.

    This is deliberately narrower than generic diarization: the application
    only needs boundaries where the target speaker starts or stops. Acoustic
    changes inside one target-speaker turn therefore do not create output
    fragments unless the local target evidence itself disappears.
    """

    SAMPLE_RATE = 16000

    def __init__(
        self,
        primary: SpeakerVerifier,
        secondary: CAMPlusVerifier,
        primary_profile: SpeakerProfile,
        secondary_profile: CAMPlusProfile,
        tertiary_factory: Callable[
            [], tuple[WavLMSpeakerVerifier, WavLMProfile]
        ]
        | None = None,
    ) -> None:
        self.primary = primary
        self.secondary = secondary
        self.primary_profile = primary_profile
        self.secondary_profile = secondary_profile
        self.tertiary_factory = tertiary_factory
        self.tertiary_target_spans: list[TimeSpan] = []

    @staticmethod
    def _normalize_spans(spans: Sequence[TimeSpan], duration: float) -> list[TimeSpan]:
        output: list[TimeSpan] = []
        for span in sorted(spans, key=lambda item: (item.start, item.end)):
            start = max(0.0, min(duration, float(span.start)))
            end = max(start, min(duration, float(span.end)))
            if end > start:
                output.append(TimeSpan(start, end))
        return output

    @staticmethod
    def _smooth(states: list[bool]) -> list[bool]:
        if len(states) < 3:
            return states
        output = list(states)
        for index in range(1, len(states) - 1):
            if states[index - 1] == states[index + 1] != states[index]:
                output[index] = states[index - 1]
        return output

    def locate(
        self,
        audio_path: Path,
        spans: Sequence[TimeSpan],
        *,
        window_seconds: float = 1.80,
        hop_seconds: float = 0.45,
        primary_floor: float = 0.54,
        secondary_floor: float = 0.46,
        strong_primary: float = 0.64,
        strong_secondary: float = 0.60,
        minimum_target_seconds: float = 1.0,
        bridge_seconds: float = 0.65,
        progress: Callable[[float, str], None] | None = None,
    ) -> list[TimeSpan]:
        if window_seconds <= 0 or hop_seconds <= 0 or hop_seconds > window_seconds:
            raise ValueError("目标声纹滑窗参数无效")
        progress = progress or (lambda _value, _message: None)
        self.tertiary_target_spans = []
        waveform = load_mono(audio_path, self.SAMPLE_RATE)
        duration = waveform.numel() / self.SAMPLE_RATE
        normalized = self._normalize_spans(spans, duration)

        plans: list[tuple[int, float, float]] = []
        for span_index, span in enumerate(normalized):
            if span.duration <= window_seconds:
                plans.append((span_index, span.start, span.end))
                continue
            starts: list[float] = []
            cursor = span.start
            while cursor + window_seconds < span.end:
                starts.append(cursor)
                cursor += hop_seconds
            last = max(span.start, span.end - window_seconds)
            if not starts or abs(starts[-1] - last) > 1e-6:
                starts.append(last)
            plans.extend((span_index, start, min(span.end, start + window_seconds)) for start in starts)
        if not plans:
            progress(1.0, "目标声纹定位完成：没有讲话窗口")
            return []

        windows = [
            waveform[int(round(start * self.SAMPLE_RATE)) : int(round(end * self.SAMPLE_RATE))]
            for _span_index, start, end in plans
        ]
        progress(0.0, f"目标声纹定位：ERes2Net 0/{len(windows)}")
        primary_embeddings = self.primary._embeddings_from_waveforms(
            windows,
            progress=lambda completed, total: progress(
                0.45 * completed / max(1, total),
                f"目标声纹定位：ERes2Net {completed}/{total}",
            ),
        )
        progress(0.45, f"目标声纹定位：CAM++ 0/{len(windows)}")
        secondary_embeddings = self.secondary._embeddings_from_waveforms(
            windows,
            progress=lambda completed, total: progress(
                0.45 + 0.45 * completed / max(1, total),
                f"目标声纹定位：CAM++ {completed}/{total}",
            ),
        )

        primary_scores = primary_embeddings @ self.primary_profile.centroid
        secondary_scores = secondary_embeddings @ self.secondary_profile.centroid
        primary_reference = primary_embeddings @ self.primary_profile.embeddings.T
        secondary_reference = secondary_embeddings @ self.secondary_profile.embeddings.T
        primary_max = primary_reference.max(dim=1).values
        secondary_max = secondary_reference.max(dim=1).values

        states: list[bool] = []
        ambiguous_indexes: list[int] = []
        for index, (p_score, s_score, p_max, s_max) in enumerate(
            zip(primary_scores, secondary_scores, primary_max, secondary_max)
        ):
            p_value = float(p_score)
            s_value = float(s_score)
            p_reference = float(p_max)
            s_reference = float(s_max)
            balanced = (
                p_value >= primary_floor
                and s_value >= secondary_floor
                and p_reference >= primary_floor - 0.02
                and s_reference >= secondary_floor - 0.02
            )
            # CAM++ is the main protection against ERes2Net confusing a
            # similar voice; ERes2Net must still provide minimum support.
            strong = s_value >= strong_secondary and p_value >= primary_floor
            standard = balanced or strong
            states.append(standard)
            if (
                not standard
                and self.tertiary_factory is not None
                and p_value >= 0.38
                and s_value >= 0.46
                and s_value - p_value >= 0.02
                and s_reference >= 0.44
            ):
                ambiguous_indexes.append(index)

        standard_counts: dict[int, int] = {}
        plan_counts: dict[int, int] = {}
        for plan_index, (span_index, _start, _end) in enumerate(plans):
            plan_counts[span_index] = plan_counts.get(span_index, 0) + 1
            if states[plan_index]:
                standard_counts[span_index] = standard_counts.get(span_index, 0) + 1
        # WavLM may repair a genuinely missed island, but it must not fill
        # holes inside an already well-supported target region. Filling those
        # holes can bridge another speaker or expand a correct sentence edge.
        ambiguous_indexes = [
            index
            for index in ambiguous_indexes
            if standard_counts.get(plans[index][0], 0) <= 1
            and standard_counts.get(plans[index][0], 0)
            / max(1, plan_counts.get(plans[index][0], 1))
            <= 0.25
        ]

        tertiary_indexes: set[int] = set()
        if ambiguous_indexes and self.tertiary_factory is not None:
            progress(0.90, f"双模型歧义窗口：WavLM 复核 0/{len(ambiguous_indexes)}")
            tertiary, tertiary_profile = self.tertiary_factory()
            tertiary_embeddings = tertiary._embeddings_from_waveforms(
                [windows[index] for index in ambiguous_indexes],
                progress=lambda completed, total: progress(
                    0.90 + 0.10 * completed / max(1, total),
                    f"双模型歧义窗口：WavLM 复核 {completed}/{total}",
                ),
            )
            tertiary_scores = tertiary_embeddings @ tertiary_profile.centroid
            tertiary_max = (
                tertiary_embeddings @ tertiary_profile.embeddings.T
            ).max(dim=1).values
            floor = tertiary_profile.acceptance_floor
            for index, wavlm_score, wavlm_max in zip(
                ambiguous_indexes, tertiary_scores, tertiary_max
            ):
                p_value = float(primary_scores[index])
                s_value = float(secondary_scores[index])
                s_reference = float(secondary_max[index])
                if (
                    p_value >= 0.40
                    and s_value >= 0.50
                    and s_value - p_value >= 0.03
                    and s_reference >= 0.46
                    and float(wavlm_score) >= floor - 0.05
                    and float(wavlm_max) >= floor
                ):
                    states[index] = True
                    tertiary_indexes.add(index)
        else:
            progress(1.0, "目标声纹定位：没有需要第三模型复核的窗口")

        by_span: dict[int, list[int]] = {}
        for plan_index, (span_index, _start, _end) in enumerate(plans):
            by_span.setdefault(span_index, []).append(plan_index)

        target_spans: list[TimeSpan] = []
        tertiary_target_spans: list[TimeSpan] = []
        half = window_seconds / 2.0
        for span_index, indexes in by_span.items():
            span = normalized[span_index]
            local_states = self._smooth([states[index] for index in indexes])
            intervals: list[TimeSpan] = []
            for plan_index, state in zip(indexes, local_states):
                if not state:
                    continue
                _item_span, start, end = plans[plan_index]
                center = (start + end) / 2.0
                interval = TimeSpan(
                    max(span.start, center - half),
                    min(span.end, center + half),
                )
                if intervals and interval.start <= intervals[-1].end + bridge_seconds:
                    intervals[-1] = TimeSpan(intervals[-1].start, max(intervals[-1].end, interval.end))
                else:
                    intervals.append(interval)
            target_spans.extend(item for item in intervals if item.duration >= minimum_target_seconds)
            tertiary_intervals: list[TimeSpan] = []
            for plan_index in indexes:
                if plan_index not in tertiary_indexes:
                    continue
                _item_span, start, end = plans[plan_index]
                center = (start + end) / 2.0
                interval = TimeSpan(
                    max(span.start, center - half),
                    min(span.end, center + half),
                )
                if (
                    tertiary_intervals
                    and interval.start <= tertiary_intervals[-1].end + bridge_seconds
                ):
                    tertiary_intervals[-1] = TimeSpan(
                        tertiary_intervals[-1].start,
                        max(tertiary_intervals[-1].end, interval.end),
                    )
                else:
                    tertiary_intervals.append(interval)
            tertiary_target_spans.extend(tertiary_intervals)
        self.tertiary_target_spans = tertiary_target_spans
        progress(1.0, f"目标声纹定位完成：找到 {len(target_spans)} 个目标人物回合")
        return target_spans


class LocalSpeakerTurnSplitter:
    """Detect local speaker changes without assigning global speaker clusters.

    The splitter compares audio immediately to the left and right of each
    scan point.  ERes2Net proposes candidates; CAM++ independently confirms
    them.  Confirmed nearby candidates are collapsed into one boundary and
    each input VAD span is returned as continuous single-speaker ``TimeSpan``
    objects.  It deliberately does not label or globally cluster speakers.

    Parameters are conservative defaults for dialogue after vocal separation.
    ``scan_hop_seconds`` controls boundary precision, while
    ``context_seconds`` controls how much speech each model sees on either
    side.  Near VAD edges the context may be as short as 0.4 seconds; both
    embedding front ends pad those windows before inference.
    """

    SAMPLE_RATE = 16000

    def __init__(
        self,
        primary: SpeakerVerifier,
        secondary: CAMPlusVerifier | None = None,
        secondary_factory: Callable[[], CAMPlusVerifier] | None = None,
    ) -> None:
        self.primary = primary
        self.secondary = secondary
        self.secondary_factory = secondary_factory

    def _ensure_secondary(self) -> CAMPlusVerifier:
        if self.secondary is None:
            if self.secondary_factory is not None:
                self.secondary = self.secondary_factory()
            else:
                device = "cuda" if self.primary.device.type == "cuda" else "cpu"
                self.secondary = CAMPlusVerifier(device)
        return self.secondary

    @staticmethod
    def _normalize_spans(spans: Iterable[TimeSpan], duration: float) -> list[TimeSpan]:
        normalized: list[TimeSpan] = []
        for span in sorted(spans, key=lambda item: (item.start, item.end)):
            start = max(0.0, min(float(span.start), duration))
            end = max(start, min(float(span.end), duration))
            if end > start:
                normalized.append(TimeSpan(start, end))
        return normalized

    @staticmethod
    def _rms_db(waveform: torch.Tensor) -> float:
        rms = torch.sqrt(torch.mean(waveform.square()) + 1e-12)
        return float(20.0 * torch.log10(rms + 1e-12))

    @staticmethod
    def _embedding_similarity(
        verifier: SpeakerVerifier | CAMPlusVerifier,
        waveform: torch.Tensor,
        left_start: int,
        boundary: int,
        right_end: int,
    ) -> float:
        left = verifier._embedding_from_waveform(waveform[left_start:boundary])
        right = verifier._embedding_from_waveform(waveform[boundary:right_end])
        return float(left @ right)

    @staticmethod
    def _confidence(primary_similarity: float, secondary_similarity: float) -> float:
        primary_strength = max(0.0, min(1.0, (0.72 - primary_similarity) / 0.42))
        secondary_strength = max(0.0, min(1.0, (0.60 - secondary_similarity) / 0.38))
        return round((primary_strength + secondary_strength) / 2.0, 5)

    @staticmethod
    def _is_confirmed_change(
        primary_similarity: float,
        secondary_similarity: float,
        primary_drop: float,
        secondary_drop: float,
        primary_threshold: float,
        secondary_threshold: float,
        minimum_similarity_drop: float,
    ) -> bool:
        """Require low cross-model similarity plus corroborated local drop."""

        low_cross_similarity = (
            primary_similarity <= primary_threshold
            and secondary_similarity <= secondary_threshold
        )
        # A prosody change can make one embedding dip even when the speaker
        # has not changed.  Require the second model to corroborate at least
        # part of the drop instead of accepting any non-negative value.
        corroborated_drop = (
            primary_drop >= minimum_similarity_drop
            and secondary_drop >= minimum_similarity_drop * 0.25
        ) or (
            secondary_drop >= minimum_similarity_drop
            and primary_drop >= minimum_similarity_drop * 0.25
        )
        return low_cross_similarity and corroborated_drop

    @staticmethod
    def _collapse_candidates(
        candidates: list[SpeakerBoundary],
        minimum_separation_seconds: float,
    ) -> list[SpeakerBoundary]:
        if not candidates:
            return []
        groups: list[list[SpeakerBoundary]] = [[candidates[0]]]
        for candidate in candidates[1:]:
            if candidate.time - groups[-1][-1].time <= minimum_separation_seconds:
                groups[-1].append(candidate)
            else:
                groups.append([candidate])
        output: list[SpeakerBoundary] = []
        for group in groups:
            representative = min(
                group,
                key=lambda item: (
                    (item.primary_similarity + (item.secondary_similarity or 1.0)) / 2.0,
                    -item.confidence,
                ),
            )
            output.append(
                SpeakerBoundary(
                    # Use the strongest local observation instead of the
                    # arithmetic center; asymmetric speech around a turn can
                    # otherwise move the cut noticeably toward one speaker.
                    time=representative.time,
                    primary_similarity=representative.primary_similarity,
                    secondary_similarity=representative.secondary_similarity,
                    confidence=max(item.confidence for item in group),
                    primary_drop=representative.primary_drop,
                    secondary_drop=representative.secondary_drop,
                )
            )
        return output

    def detect_speaker_boundaries(
        self,
        audio_path: Path,
        spans: Sequence[TimeSpan],
        *,
        context_seconds: float = 1.20,
        scan_hop_seconds: float = 0.25,
        primary_threshold: float = 0.68,
        secondary_threshold: float = 0.52,
        primary_candidate_threshold: float = 0.74,
        minimum_similarity_drop: float = 0.06,
        minimum_separation_seconds: float = 0.55,
        silence_db: float = -48.0,
        progress: Callable[[float, str], None] | None = None,
    ) -> list[SpeakerBoundary]:
        """Return locally confirmed speaker-change boundaries.

        A boundary is accepted only when both embedding models have low
        cross-boundary similarity and at least one model sees a meaningful
        local drop while the other corroborates part of that drop. Low
        similarity alone is never enough to split a turn.
        """

        if context_seconds < 0.30:
            raise ValueError("说话人边界上下文至少需要 0.3 秒")
        if scan_hop_seconds <= 0.0:
            raise ValueError("说话人边界扫描步长必须大于 0 秒")
        if primary_candidate_threshold < primary_threshold:
            raise ValueError("ERes2Net 候选阈值不能低于确认阈值")
        if minimum_separation_seconds < 0.0:
            raise ValueError("换人边界最小间隔不能小于 0 秒")
        progress = progress or (lambda _value, _message: None)
        waveform = load_mono(audio_path, self.SAMPLE_RATE)
        duration = waveform.numel() / self.SAMPLE_RATE
        normalized = self._normalize_spans(spans, duration)
        context = max(1, int(round(context_seconds * self.SAMPLE_RATE)))
        hop = max(1, int(round(scan_hop_seconds * self.SAMPLE_RATE)))

        plans: list[tuple[TimeSpan, int]] = []
        for span in normalized:
            span_start = int(round(span.start * self.SAMPLE_RATE))
            span_end = int(round(span.end * self.SAMPLE_RATE))
            first = span_start + context
            last = span_end - context
            if last < first:
                continue
            positions = list(range(first, last + 1, hop))
            if positions and positions[-1] != last:
                positions.append(last)
            plans.extend((span, position) for position in positions)
        if not plans:
            progress(1.0, "说话人边界检测完成：没有足够长的讲话段")
            return []

        active_plans: list[tuple[int, int, int]] = []
        primary_ranges: list[tuple[int, int]] = []
        for _span, position in plans:
            left_start = position - context
            right_end = position + context
            if min(
                self._rms_db(waveform[left_start:position]),
                self._rms_db(waveform[position:right_end]),
            ) <= silence_db:
                continue
            active_plans.append((position, left_start, right_end))
            quarter = max(1, context // 2)
            primary_ranges.extend(
                [
                    (left_start, position),
                    (position, right_end),
                    (left_start, position - quarter),
                    (position - quarter, position),
                    (position, position + quarter),
                    (position + quarter, right_end),
                ]
            )
        unique_primary_ranges = list(dict.fromkeys(primary_ranges))
        progress(0.0, f"说话人边界：ERes2Net 批量提取 0/{len(unique_primary_ranges)}")
        primary_embeddings = self.primary._embeddings_from_waveforms(
            [waveform[start:end] for start, end in unique_primary_ranges],
            progress=lambda completed, total: progress(
                0.55 * completed / max(1, total),
                f"说话人边界：ERes2Net 批量提取 {completed}/{total}",
            ),
        )
        primary_cache = {
            key: embedding for key, embedding in zip(unique_primary_ranges, primary_embeddings)
        }
        primary_records: list[tuple[int, float, float, float]] = []
        for position, left_start, right_end in active_plans:
            quarter = max(1, context // 2)
            cross = float(primary_cache[(left_start, position)] @ primary_cache[(position, right_end)])
            if cross > primary_candidate_threshold:
                continue
            left_internal = float(
                primary_cache[(left_start, position - quarter)]
                @ primary_cache[(position - quarter, position)]
            )
            right_internal = float(
                primary_cache[(position, position + quarter)]
                @ primary_cache[(position + quarter, right_end)]
            )
            local_baseline = min(left_internal, right_internal)
            primary_records.append((position, cross, local_baseline, local_baseline - cross))

        if not primary_records:
            progress(1.0, "说话人边界检测完成：未发现换人")
            return []

        secondary = self._ensure_secondary()
        total = len(primary_records)
        secondary_ranges: list[tuple[int, int]] = []
        for position, _primary_score, _primary_baseline, _primary_drop in primary_records:
            left_start = position - context
            right_end = position + context
            quarter = max(1, context // 2)
            secondary_ranges.extend(
                [
                    (left_start, position),
                    (position, right_end),
                    (left_start, position - quarter),
                    (position - quarter, position),
                    (position, position + quarter),
                    (position + quarter, right_end),
                ]
            )
        unique_secondary_ranges = list(dict.fromkeys(secondary_ranges))
        progress(0.55, f"说话人边界：CAM++ 批量核验 0/{len(unique_secondary_ranges)}")
        secondary_embeddings = secondary._embeddings_from_waveforms(
            [waveform[start:end] for start, end in unique_secondary_ranges],
            progress=lambda completed, count: progress(
                0.55 + 0.45 * completed / max(1, count),
                f"说话人边界：CAM++ 批量核验 {completed}/{count}",
            ),
        )
        secondary_cache = {
            key: embedding for key, embedding in zip(unique_secondary_ranges, secondary_embeddings)
        }
        confirmed: list[SpeakerBoundary] = []
        for position, primary_score, primary_baseline, primary_drop in primary_records:
            left_start = position - context
            right_end = position + context
            secondary_score = float(
                secondary_cache[(left_start, position)]
                @ secondary_cache[(position, right_end)]
            )
            quarter = max(1, context // 2)
            secondary_left = float(
                secondary_cache[(left_start, position - quarter)]
                @ secondary_cache[(position - quarter, position)]
            )
            secondary_right = float(
                secondary_cache[(position, position + quarter)]
                @ secondary_cache[(position + quarter, right_end)]
            )
            secondary_baseline = min(secondary_left, secondary_right)
            secondary_drop = secondary_baseline - secondary_score
            if self._is_confirmed_change(
                primary_score,
                secondary_score,
                primary_drop,
                secondary_drop,
                primary_threshold,
                secondary_threshold,
                minimum_similarity_drop,
            ):
                confirmed.append(
                    SpeakerBoundary(
                        time=position / self.SAMPLE_RATE,
                        primary_similarity=round(primary_score, 5),
                        secondary_similarity=round(secondary_score, 5),
                        confidence=self._confidence(primary_score, secondary_score),
                        primary_drop=round(primary_baseline - primary_score, 5),
                        secondary_drop=round(secondary_baseline - secondary_score, 5),
                    )
                )
        output = self._collapse_candidates(confirmed, minimum_separation_seconds)
        progress(1.0, f"说话人边界检测完成：确认 {len(output)} 个换人点")
        return output

    @staticmethod
    def _cluster_multiscale_boundaries(
        detections: Sequence[tuple[float, SpeakerBoundary]],
        *,
        cluster_seconds: float,
        minimum_context_votes: int,
    ) -> list[SpeakerBoundary]:
        """Collapse nearby observations and retain only cross-scale changes.

        A single context can mistake an accent, breath, or short phoneme for a
        change of speaker.  Distinct context sizes are treated as independent
        observations; repeated hits from one scale do not increase confidence.
        """

        if not detections:
            return []
        ordered = sorted(detections, key=lambda item: item[1].time)
        groups: list[list[tuple[float, SpeakerBoundary]]] = [[ordered[0]]]
        for context, boundary in ordered[1:]:
            if boundary.time - groups[-1][-1][1].time <= cluster_seconds:
                groups[-1].append((context, boundary))
            else:
                groups.append([(context, boundary)])

        output: list[SpeakerBoundary] = []
        for group in groups:
            contexts = sorted({round(float(context), 3) for context, _ in group})
            if len(contexts) < minimum_context_votes:
                continue
            representative = min(
                (boundary for _context, boundary in group),
                key=lambda item: (
                    item.primary_similarity + (item.secondary_similarity or 1.0),
                    -item.confidence,
                ),
            )
            output.append(
                SpeakerBoundary(
                    time=representative.time,
                    primary_similarity=representative.primary_similarity,
                    secondary_similarity=representative.secondary_similarity,
                    confidence=max(item.confidence for _context, item in group),
                    primary_drop=representative.primary_drop,
                    secondary_drop=representative.secondary_drop,
                    scale_votes=len(contexts),
                    scale_contexts=tuple(contexts),
                )
            )
        return output

    def detect_multiscale_speaker_boundaries(
        self,
        audio_path: Path,
        spans: Sequence[TimeSpan],
        *,
        contexts: Sequence[float] = (0.70, 0.90, 1.10),
        cluster_seconds: float = 0.40,
        minimum_context_votes: int = 2,
        scan_hop_seconds: float = 0.10,
        primary_threshold: float = 0.68,
        secondary_threshold: float = 0.52,
        primary_candidate_threshold: float = 0.78,
        minimum_similarity_drop: float = 0.06,
        minimum_separation_seconds: float = 0.20,
        progress: Callable[[float, str], None] | None = None,
    ) -> list[SpeakerBoundary]:
        """Find speaker changes that survive multiple context sizes.

        The method is deliberately an audit, not a new speaker clustering
        model.  A boundary must be observed by at least two distinct context
        sizes before callers may split or veto a candidate around it.
        """

        if not contexts:
            return []
        if cluster_seconds < 0.0:
            raise ValueError("多尺度边界聚类窗口不能小于 0 秒")
        if minimum_context_votes < 1:
            raise ValueError("多尺度边界至少需要一个上下文尺度")
        progress = progress or (lambda _value, _message: None)
        unique_contexts = tuple(sorted({float(value) for value in contexts}))
        detections: list[tuple[float, SpeakerBoundary]] = []
        # A short-context pass is an inexpensive screen.  Longer contexts are
        # only evaluated for spans where that screen found a possible change;
        # this keeps the default full-episode path close to the old runtime
        # while retaining cross-scale confirmation on genuinely suspicious
        # turns.
        screening_context = unique_contexts[len(unique_contexts) // 2]
        screen_boundaries = self.detect_speaker_boundaries(
            audio_path,
            spans,
            context_seconds=screening_context,
            scan_hop_seconds=scan_hop_seconds,
            primary_threshold=primary_threshold,
            secondary_threshold=secondary_threshold,
            primary_candidate_threshold=primary_candidate_threshold,
            minimum_similarity_drop=minimum_similarity_drop,
            minimum_separation_seconds=minimum_separation_seconds,
            progress=lambda value, message: progress(
                0.45 * value,
                f"多尺度换人审计 {screening_context:.2f}s：{message}",
            ),
        )
        detections.extend(
            (screening_context, boundary) for boundary in screen_boundaries
        )
        if not screen_boundaries:
            progress(1.0, "多尺度换人审计完成：筛查未发现候选边界")
            return []
        suspicious_spans = [
            span
            for span in spans
            if any(span.start < boundary.time < span.end for boundary in screen_boundaries)
        ]
        remaining_contexts = [
            context for context in unique_contexts if context != screening_context
        ]
        for index, context in enumerate(remaining_contexts, start=1):
            progress(
                0.45 + 0.55 * (index - 1) / max(1, len(remaining_contexts)),
                f"多尺度换人审计：上下文 {context:.2f}s",
            )
            boundaries = self.detect_speaker_boundaries(
                audio_path,
                suspicious_spans,
                context_seconds=context,
                scan_hop_seconds=scan_hop_seconds,
                primary_threshold=primary_threshold,
                secondary_threshold=secondary_threshold,
                primary_candidate_threshold=primary_candidate_threshold,
                minimum_similarity_drop=minimum_similarity_drop,
                minimum_separation_seconds=minimum_separation_seconds,
                progress=lambda value, message, context=context: progress(
                    0.45
                    + 0.55 * ((index - 1) + value)
                    / max(1, len(remaining_contexts)),
                    f"多尺度换人审计 {context:.2f}s：{message}",
                ),
            )
            detections.extend((context, boundary) for boundary in boundaries)
        output = self._cluster_multiscale_boundaries(
            detections,
            cluster_seconds=cluster_seconds,
            minimum_context_votes=minimum_context_votes,
        )
        progress(1.0, f"多尺度换人审计完成：确认 {len(output)} 个稳定边界")
        return output

    def analyze(
        self,
        audio_path: Path,
        spans: Sequence[TimeSpan],
        *,
        minimum_turn_seconds: float = 0.30,
        progress: Callable[[float, str], None] | None = None,
        **boundary_kwargs: float,
    ) -> SpeakerSplitResult:
        """Return split spans and boundary diagnostics in one result."""

        if minimum_turn_seconds < 0.0:
            raise ValueError("最短说话回合不能小于 0 秒")
        waveform = load_mono(audio_path, self.SAMPLE_RATE)
        duration = waveform.numel() / self.SAMPLE_RATE
        normalized = self._normalize_spans(spans, duration)
        boundaries = self.detect_speaker_boundaries(
            audio_path,
            normalized,
            progress=progress,
            **boundary_kwargs,
        )
        turns: list[TimeSpan] = []
        for span in normalized:
            cuts = [
                boundary.time
                for boundary in boundaries
                if span.start + minimum_turn_seconds <= boundary.time <= span.end - minimum_turn_seconds
            ]
            cursor = span.start
            span_turns: list[TimeSpan] = []
            for cut in cuts:
                if cut - cursor >= minimum_turn_seconds:
                    span_turns.append(TimeSpan(cursor, cut))
                cursor = cut
            if span.end - cursor >= minimum_turn_seconds:
                span_turns.append(TimeSpan(cursor, span.end))
            elif span_turns:
                previous = span_turns[-1]
                span_turns[-1] = TimeSpan(previous.start, span.end)
            elif span.duration >= minimum_turn_seconds:
                span_turns.append(span)
            turns.extend(span_turns)
        return SpeakerSplitResult(tuple(turns), tuple(boundaries))

    def split_speaker_spans(
        self,
        audio_path: Path,
        spans: Sequence[TimeSpan],
        *,
        minimum_turn_seconds: float = 0.30,
        progress: Callable[[float, str], None] | None = None,
        **boundary_kwargs: float,
    ) -> list[TimeSpan]:
        """Return VAD spans split into continuous local-speaker turns."""

        return list(
            self.analyze(
                audio_path,
                spans,
                minimum_turn_seconds=minimum_turn_seconds,
                progress=progress,
                **boundary_kwargs,
            ).spans
        )

    # Short alias for callers that already make the speaker context explicit.
    split_spans = split_speaker_spans
