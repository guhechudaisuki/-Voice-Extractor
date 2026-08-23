from __future__ import annotations

import gc
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Literal, Sequence

import torch
import torch.nn.functional as F
import torchaudio

from .config import CAMPLUS_MODEL, SV_CODE, SV_MODEL
from .audio import load_mono
from .types import TimeSpan


SpeakerMatchTier = Literal[
    "short_strong",
    "strong",
    "balanced",
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

    def to_dict(self) -> dict[str, float | None]:
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

    def build_profile(self, reference_paths: list[Path], base_threshold: float) -> SpeakerMatchProfile:
        return SpeakerMatchProfile(
            primary=self.primary.build_profile(reference_paths, base_threshold),
            reference_paths=list(reference_paths),
            base_threshold=base_threshold,
        )

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
        )
        return locator.locate(audio_path, spans, progress=progress, **kwargs)

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
    ) -> None:
        self.primary = primary
        self.secondary = secondary
        self.primary_profile = primary_profile
        self.secondary_profile = secondary_profile

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
                0.50 * completed / max(1, total),
                f"目标声纹定位：ERes2Net {completed}/{total}",
            ),
        )
        progress(0.50, f"目标声纹定位：CAM++ 0/{len(windows)}")
        secondary_embeddings = self.secondary._embeddings_from_waveforms(
            windows,
            progress=lambda completed, total: progress(
                0.50 + 0.50 * completed / max(1, total),
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
        for p_score, s_score, p_max, s_max in zip(
            primary_scores,
            secondary_scores,
            primary_max,
            secondary_max,
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
            states.append(balanced or strong)

        by_span: dict[int, list[int]] = {}
        for plan_index, (span_index, _start, _end) in enumerate(plans):
            by_span.setdefault(span_index, []).append(plan_index)

        target_spans: list[TimeSpan] = []
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
