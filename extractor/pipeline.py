"""Public extraction API composed from source-scoped stage components."""
from __future__ import annotations
from . import pipeline_common as common
from .pipeline_common import (ASSET_ROOT, BatchPipelineResult, CAMPlusProfile, Callable, CandidateSentence, DualSpeakerVerifier, ExclusionSpeakerProfile, ExitStack, FunASRTools, Iterable, LOGGER, LocalSpeakerTurnSplitter, MIN_TARGET_SECONDS, MODEL_POOL, OUTPUT_ROOT, OVERLAP_MODEL, OverlapDetector, PANNS_MODEL, Path, PipelineOptions, PipelineResult, PitchRegisterGuard, ProgressCallback, SingingDetector, SpeakerBoundary, SpeakerMatchDecision, SpeakerMatchProfile, SpeakerProfile, SubtitleGuide, SubtitleJoinGuard, TimeSpan, UVR5Separator, VocalTractGeometryGuard, WAVLM_SV_MODEL, WORK_ROOT, WhisperSegmenter, _cache_waveforms_ready, _language_from_text, _manifest_record, _noop_progress, _record_acoustic_quality, _safe_name, _srt_time, annotate_raw_recovery, asdict, carve_tainted_spans, clean_dialogue_text, csv, dataclass, detect_speaker_change_valleys, discover_subtitle_for_target, ensure_local_assets, evaluate_clip_acoustics, has_exact_target_certificate, has_video_stream, io, is_wave_file, json, load_media_cache, load_mono, logging, math, media_cache_dir, media_cache_hit, merge_certified_contiguous, mute_spans, normalize_audio, os, pad_for_separator, pooled_encoder, probe_duration, propose_raw_vad_subturns, re, recover_raw_vad, reserves_audio, restore_subtitle_sentences, retry_subtitle_vad, save_media_cache, shutil, snap_crop_boundaries_to_valley, speech_ratio, split_at_subtitle_pauses, split_overlong_sentences, subtitle_atomic_spans, time, torch, trim_audio_in_place, uuid, validate_subtitle_bindings, write_clip, write_video_clip, zipfile)
from .stages.s01_media import S01MediaMethods
from .stages.s03_guards import S03GuardsMethods
from .stages.identity_support import IdentitySupportMethods
from .stages.s06_transcription import S06TranscriptionMethods
from .stages.s07_export import S07ExportMethods
from .stages.sentence_repair import SentenceRepairMethods
from .stages.sentence_assembly import SentenceAssemblyMethods
from .stages.local_islands import LocalIslandsMethods
from .stages.candidate_recovery import CandidateRecoveryMethods
from .stages.cluster_recovery import ClusterRecoveryMethods
from .stages.multimodel_recovery import MultimodelRecoveryMethods
from .stages.runner import RunnerMethods


class ExtractionPipeline(S01MediaMethods, S03GuardsMethods, IdentitySupportMethods, S06TranscriptionMethods, S07ExportMethods, SentenceRepairMethods, SentenceAssemblyMethods, LocalIslandsMethods, CandidateRecoveryMethods, ClusterRecoveryMethods, MultimodelRecoveryMethods, RunnerMethods):
    """Reference-conditioned, stem-authoritative extraction. State lasts one task."""

    def __init__(self, options: common.PipelineOptions | None = None, device: str = "cuda") -> None:
        common.ensure_local_assets()
        self.options = options or common.PipelineOptions()
        self.device = device if device == "cuda" and common.torch.cuda.is_available() else "cpu"
        self._raw_target_waveform: common.torch.Tensor | None = None
        self._raw_blocked_spans: tuple[common.TimeSpan, ...] = ()
        self._subtitle_guide: common.SubtitleGuide | None = None
        self._stage_audit: dict | None = None
        self._cluster_report: dict | None = None
        self._cluster_accepted_spans: list | None = None
        self._batch_bank: list = []
        self._verified_fallback_registry: dict[int, list[common.CandidateSentence]] = {}
        self._verified_fallback_anchors: list[common.CandidateSentence] = []
        self._tail_scorer_cache: tuple | str | None = None
        self._local_score_cache = None
        self._reference_sha256: list[str] = []
        self._local_identity_reject_threshold: float | None = None
        self._local_identity_model_sha256 = ""
        self._pitch_guard: common.PitchRegisterGuard | None = None
        self._subtitle_join_guard: common.SubtitleJoinGuard | None = None
        self._speaker_change_points: tuple[float, ...] = ()
        self._pitch_span_cache: dict[tuple[float, float], dict] = {}

    def _trace_spans(
        self,
        step: str,
        spans: common.Iterable[common.TimeSpan | common.CandidateSentence],
    ) -> None:
        """Keep timestamped evidence for offline recall auditing."""
        if self._stage_audit is None:
            return
        items = sorted(
            (round(float(span.start), 5), round(float(span.end), 5))
            for span in spans
            if span.end > span.start
        )
        self._stage_audit["stages"][step] = {"count": len(items), "spans": items}
