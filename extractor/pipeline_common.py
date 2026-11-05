from __future__ import annotations

import csv
import io
import json
import logging
import math
import os
import re
import shutil
import time
import uuid
import zipfile
from dataclasses import dataclass, asdict
from contextlib import ExitStack
from pathlib import Path
from typing import Callable, Iterable

import torch

from .audio import (
    UVR5Separator,
    load_mono,
    mute_spans,
    normalize_audio,
    pad_for_separator,
    probe_duration,
    speech_ratio,
    trim_audio_in_place,
    write_clip,
    has_video_stream,
    write_video_clip,
)
from .candidate_lattice import propose_raw_vad_subturns, reserves_audio
from .config import ASSET_ROOT, OUTPUT_ROOT, WORK_ROOT, WAVLM_SV_MODEL, ensure_local_assets
from .filters import OverlapDetector, SingingDetector
from .model_pool import MODEL_POOL, pooled_encoder
from .config import PANNS_MODEL, OVERLAP_MODEL
from .extraction_guards import (
    VocalTractGeometryGuard,
    PitchRegisterGuard, SubtitleJoinGuard, carve_tainted_spans, subtitle_atomic_spans,
)
from .speaker import (
    CAMPlusProfile,
    DualSpeakerVerifier,
    ExclusionSpeakerProfile,
    LocalSpeakerTurnSplitter,
    SpeakerBoundary,
    SpeakerMatchDecision,
    SpeakerMatchProfile,
    SpeakerProfile,
)
from .transcription import FunASRTools, WhisperSegmenter
from .acoustic_qc import (
    detect_speaker_change_valleys, evaluate_clip_acoustics,
    snap_crop_boundaries_to_valley, split_overlong_sentences,
)
from .media_cache import (media_cache_hit, media_cache_dir,
                           save_media_cache, load_media_cache, is_wave_file)
from .subtitles import (
    SubtitleGuide, clean_dialogue_text, discover_subtitle_for_target, validate_subtitle_bindings,
)
from .subtitle_assistance import (
    restore_subtitle_sentences, retry_subtitle_vad, split_at_subtitle_pauses,
)
from .types import BatchPipelineResult, CandidateSentence, PipelineResult, TimeSpan
from .recall import MIN_TARGET_SECONDS, recover_raw_vad, annotate_raw_recovery
from .sentence_continuity import has_exact_target_certificate, merge_certified_contiguous

LOGGER = logging.getLogger(__name__)


def _cache_waveforms_ready(*paths: Path) -> bool:
    """Cache only complete RIFF/WAVE inputs."""
    return all(is_wave_file(path) for path in paths)


@dataclass
class PipelineOptions:
    cluster_identity_mode: str = "enforce"
    cluster_target_margin: float = 0.06
    cluster_keep_narration: bool = True
    cluster_embedding_cache: bool = True
    cluster_require_dominant: bool = False
    cluster_bootstrap: bool = False
    cluster_bootstrap_max_rounds: int = 3
    # WavLM supplies an independent channel with its own embedding cache.
    cluster_wavlm_channel: bool = True
    # WeSpeaker-ResNet34-LM 第四通道仲裁：救援标记候选进 Tier A 前的
    # 异构否决票（大间隔模型与其余三模型相关性低）。
    cluster_wespeaker_arbitration: bool = True
    # Original audio can confirm a voice obscured by separator artifacts.
    cluster_raw_crosscheck: bool = True
    # Share independently confirmed references across a batch while retaining
    # the user references as the identity anchor.
    cluster_cross_episode_pool: bool = True
    speaker_threshold: float = 0.68
    # Nearby speech remains separate until same-speaker verification permits
    # a join. The upper limit is always a hard silence boundary.
    silence_min_seconds: float = 0.20
    silence_split_seconds: float = 0.85
    silence_max_seconds: float | None = None
    overlap_threshold: float = 0.35
    singing_threshold: float = 0.22
    # Keep short turns long enough to reach speaker verification.  The final
    # export gate remains at min_output_seconds, so this is not a quality gate.
    min_sentence_seconds: float = 0.35
    min_output_seconds: float = 0.85
    max_sentence_seconds: float = 45.0
    pad_seconds: float = 0.10
    keep_rejected: bool = False
    use_singing_detector: bool = True
    use_overlap_detector: bool = True
    # UVR remains authoritative.  The time-aligned original channel may only
    # rescue a stem rejection after singing/overlap gates have passed.
    use_raw_speaker_rescue: bool = True
    export_all_sentences: bool = False
    # Optional extra validation after each track's continuous decoder.
    # Disabled: preserve the direct union of the two selected timelines.
    branch_validation: bool = False
    export_video_clips: bool = False
    export_sample_rate: int = 16000
    cleanup_work: bool = False
    # Research switch only.  Keep the shipped one-click rules unchanged until
    # short-scene and independent human reviews establish a real improvement.
    experimental_internal_reassembly: bool = False
    experimental_adjacent_domain_rescue: bool = False
    # The island-level identity review blocks whole turns whose local
    # islands carry other-speaker evidence. It abstains without exclusion
    # references or anime models.
    experimental_final_island_consensus: bool = True
    # Time spans the user marked as wrong while listening. Any accepted turn
    # overlapping one is deleted before STT.
    user_excluded_spans: tuple[tuple[float, float], ...] = ()

    use_shot_boundaries: bool = True

    def __post_init__(self) -> None:
        if self.export_sample_rate not in {16000, 24000, 32000, 44100, 48000}:
            raise ValueError('导出采样率必须为 16000、24000、32000、44100 或 48000')
        if self.cluster_identity_mode not in {"off", "audit", "enforce"}:
            raise ValueError("cluster_identity_mode 必须为 off、audit 或 enforce")
        if not 0.0 <= self.cluster_target_margin <= 1.0:
            raise ValueError("cluster_target_margin 必须在 0–1 之间")
        if type(self.cluster_bootstrap_max_rounds) is not int or not 1 <= self.cluster_bootstrap_max_rounds <= 10:
            raise ValueError("cluster_bootstrap_max_rounds 必须为 1–10 的整数")
        if self.cluster_bootstrap and self.cluster_identity_mode == "off":
            raise ValueError("自动扩充参考需要开启簇级判决")
        if self.cluster_bootstrap and self.export_all_sentences:
            raise ValueError("自动扩充参考不能与输出全部单句同时开启，请选择一种提取方式")
        if self.silence_max_seconds is None:
            self.silence_max_seconds = float(self.silence_split_seconds)
        else:
            self.silence_split_seconds = float(self.silence_max_seconds)
        self.silence_min_seconds = float(self.silence_min_seconds)
        if self.silence_min_seconds < 0.0:
            raise ValueError("静音下限不能小于 0 秒")
        if self.silence_split_seconds < self.silence_min_seconds:
            raise ValueError("静音下限不能大于静音上限")
        self.user_excluded_spans = tuple(
            (float(span[0]), float(span[1]))
            for span in self.user_excluded_spans
            if float(span[1]) > float(span[0])
        )


ProgressCallback = Callable[[float, str], None]


def _noop_progress(value: float, message: str) -> None:
    LOGGER.info("%.0f%% %s", value * 100, message)


def _safe_name(value: str, fallback: str = "sentence") -> str:
    value = re.sub(r"[^\w\-\u3040-\u30ff\u3400-\u9fff]+", "_", value, flags=re.UNICODE).strip("_.")
    return value[:80] or fallback


def _language_from_text(text: str, current: str) -> str:
    # Whisper can report Chinese for a short Japanese sentence even while its
    # own transcript contains kana.  Prefer the observed script before routing
    # to the Chinese-only refinement model, which would overwrite that text.
    if re.search(r"[\u3040-\u30ff]", text):
        return "ja"
    if current in {"zh", "ja", "en", "ko"}:
        return current
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    return current or "auto"


def _srt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _manifest_record(candidate: CandidateSentence) -> dict:
    record = candidate.to_dict()
    quality = candidate.diagnostics.get("acoustic_qc", {})
    record.update({name: quality.get(name, "") for name in ("quality_grade", "snr_db")})
    return record


def _record_acoustic_quality(candidate: CandidateSentence, audio_path: Path) -> None:
    # Inspect the actual export, including normalization/resampling/clipping.
    import soundfile as sf
    wave, rate = sf.read(str(audio_path), dtype="float32")
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    candidate.diagnostics["acoustic_qc"] = evaluate_clip_acoustics(
        wave, rate, candidate.speaker_score, candidate.window_min_score)
