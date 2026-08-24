from __future__ import annotations

import csv
import gc
import json
import logging
import re
import shutil
import time
import uuid
import zipfile
from dataclasses import dataclass, asdict
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
from .config import OUTPUT_ROOT, WORK_ROOT, ensure_local_assets
from .filters import OverlapDetector, SingingDetector
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
from .types import BatchPipelineResult, CandidateSentence, PipelineResult, TimeSpan

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineOptions:
    speaker_threshold: float = 0.68
    # Silence gaps in [silence_min_seconds, silence_split_seconds] remain
    # separate until the two-side speaker check explicitly joins them. Gaps
    # below the lower bound are treated as one continuous speech island; gaps
    # above the upper bound are hard sentence boundaries.
    silence_min_seconds: float = 0.20
    silence_split_seconds: float = 0.85
    # Public name used by the UI and manifests. ``silence_split_seconds`` is
    # retained as the internal/backward-compatible upper-bound name.
    silence_max_seconds: float | None = None
    overlap_threshold: float = 0.35
    singing_threshold: float = 0.22
    # Keep short turns long enough to reach speaker verification.  The final
    # export gate remains at min_output_seconds, so this is not a quality gate.
    min_sentence_seconds: float = 0.55
    min_output_seconds: float = 1.20
    max_sentence_seconds: float = 45.0
    pad_seconds: float = 0.10
    keep_rejected: bool = False
    use_singing_detector: bool = True
    use_overlap_detector: bool = True
    # Export every cleaned silence-delimited sentence instead of applying the
    # target-speaker gate. This is an explicit manual-review mode.
    export_all_sentences: bool = False
    # When the source is a video, also export matching original-media clips.
    export_video_clips: bool = False
    cleanup_work: bool = False

    def __post_init__(self) -> None:
        if self.silence_max_seconds is None:
            self.silence_max_seconds = float(self.silence_split_seconds)
        else:
            self.silence_split_seconds = float(self.silence_max_seconds)
        self.silence_min_seconds = float(self.silence_min_seconds)
        if self.silence_min_seconds < 0.0:
            raise ValueError("静音下限不能小于 0 秒")
        if self.silence_split_seconds < self.silence_min_seconds:
            raise ValueError("静音下限不能大于静音上限")


ProgressCallback = Callable[[float, str], None]


def _noop_progress(value: float, message: str) -> None:
    LOGGER.info("%.0f%% %s", value * 100, message)


def _safe_name(value: str, fallback: str = "sentence") -> str:
    value = re.sub(r"[^\w\-\u3040-\u30ff\u3400-\u9fff]+", "_", value, flags=re.UNICODE).strip("_.")
    return value[:80] or fallback


def _language_from_text(text: str, current: str) -> str:
    if current in {"zh", "ja", "en", "ko"}:
        return current
    if re.search(r"[\u3040-\u30ff]", text):
        return "ja"
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


class ExtractionPipeline:
    """Strict reference-speaker sentence extractor.

    Analysis is performed on the separated vocal stem. Accepted clips are
    written from that stem, so background music and unrelated silent gaps are
    never included in the required output.
    """

    def __init__(self, options: PipelineOptions | None = None, device: str = "cuda") -> None:
        ensure_local_assets()
        self.options = options or PipelineOptions()
        self.device = device if device == "cuda" and torch.cuda.is_available() else "cpu"

    @staticmethod
    def _needs_final_multimodel_exclusion(
        candidate: CandidateSentence,
        exclusion: dict[str, float | str | bool] | None,
        *,
        require_third_model: bool = False,
    ) -> bool:
        """Decide when exclusion evidence needs an independent third vote.

        The previous gate only opened for recall/low-coverage candidates.  A
        long locator recovery could therefore contain a speaker switch and
        never reach WeSpeaker, even when one of the supplied exclusion roles
        already matched it.  Boundary risk is now part of the decision: a
        candidate with locally detected boundaries is audited regardless of its
        aggregate target score.  A third-model vote is still not a veto by
        itself for a clean ordinary sentence; it becomes decisive when it
        agrees with either base model or the candidate has an unresolved local
        boundary.
        """

        if not exclusion:
            return False
        primary_vote = bool(exclusion.get("excluded_primary_vote"))
        secondary_vote = bool(exclusion.get("excluded_secondary_vote"))
        third_vote = bool(exclusion.get("excluded_wespeaker_vote"))
        coverage = float(candidate.diagnostics.get("target_coverage", 1.0))
        tier = str(candidate.diagnostics.get("speaker_tier", ""))
        locator_boundaries = int(
            candidate.diagnostics.get("locator_boundary_count", 0) or 0
        )
        local_boundaries = int(
            candidate.diagnostics.get("multi_model_internal_boundary_count", 0)
            or candidate.diagnostics.get("multi_model_anchor_boundary_count", 0)
            or 0
        )
        boundary_risk = bool(
            candidate.diagnostics.get("local_boundary_recovery")
            or locator_boundaries >= 2
            or local_boundaries >= 1
        )
        if require_third_model:
            return third_vote and (primary_vote or secondary_vote or boundary_risk)
        if not (primary_vote or secondary_vote):
            return boundary_risk and third_vote
        if coverage <= 0.05 and primary_vote:
            return True
        return (
            (tier == "recall" and coverage < 0.60)
            or (boundary_risk and coverage < 0.95)
            or boundary_risk
        )

    @staticmethod
    def _is_structural_boundary(boundary: SpeakerBoundary) -> bool:
        """Return whether a local boundary is strong enough to block recovery.

        Cross-scale agreement is the primary signal.  The low-similarity rule
        remains as a compatibility fallback for boundaries produced by older
        one-pass diagnostics.
        """

        return bool(
            getattr(boundary, "scale_votes", 1) >= 2
            or (
                boundary.primary_similarity <= 0.15
                and boundary.secondary_similarity is not None
                and boundary.secondary_similarity <= 0.05
            )
        )

    @staticmethod
    def _is_reliable_boundary_record(record: object) -> bool:
        """Apply the same boundary rule to serialized recovery diagnostics."""

        if not isinstance(record, dict):
            return False
        try:
            scale_votes = int(record.get("scale_votes", 1) or 1)
            primary = float(record.get("primary_similarity", 1.0))
            secondary_value = record.get("secondary_similarity")
            secondary = (
                float(secondary_value)
                if secondary_value is not None
                else 1.0
            )
            confidence = float(record.get("confidence", 0.0))
            primary_drop = float(record.get("primary_drop", 0.0))
            secondary_drop = float(record.get("secondary_drop", 0.0))
        except (TypeError, ValueError):
            return False
        if scale_votes >= 2 or (primary <= 0.15 and secondary <= 0.05):
            return True
        # Older one-scale diagnostics do not carry scale_votes.  Preserve
        # only a corroborated high-confidence cut; a low score from one model
        # alone must not split an otherwise continuous utterance.
        return bool(
            confidence >= 0.80
            and primary <= 0.48
            and secondary <= 0.18
            and primary_drop >= 0.06
            and secondary_drop >= 0.02
        )

    @classmethod
    def _candidate_boundary_times(
        cls,
        candidate: CandidateSentence,
    ) -> list[float]:
        """Collect reliable serialized cuts that fall inside one candidate.

        Recovery diagnostics have been written by both the old one-pass
        boundary detector and the newer multi-scale detector.  Normalizing
        them here keeps later recovery code from treating a low-confidence
        diagnostic as a hard cut, while still allowing old cached jobs to be
        audited with the current rules.
        """

        diagnostics = candidate.diagnostics
        values: list[float] = []
        for key in (
            "recovery_boundary_details",
            "multi_model_internal_boundaries",
            "multi_model_anchor_boundaries",
        ):
            records = diagnostics.get(key, [])
            if not isinstance(records, (list, tuple)):
                continue
            for record in records:
                if not cls._is_reliable_boundary_record(record):
                    continue
                try:
                    values.append(float(record["time"]))
                except (KeyError, TypeError, ValueError):
                    continue
        # A few early diagnostics stored only numeric recovery times.
        records = diagnostics.get("recovery_boundaries", [])
        if isinstance(records, (list, tuple)):
            for record in records:
                if isinstance(record, (int, float)):
                    values.append(float(record))
        return sorted(
            {
                round(value, 5)
                for value in values
                if candidate.start + 0.35 <= value <= candidate.end - 0.35
            }
        )

    @staticmethod
    def _anime_recovery_base_gate(diagnostics: dict[str, object]) -> bool:
        """Require independent episode evidence before anime-score recovery."""

        if not bool(diagnostics.get("multi_model_base_consensus")):
            return False
        if int(diagnostics.get("multi_model_common_anchor_count", 0) or 0) < 1:
            return False
        if not bool(
            diagnostics.get("multi_model_wespeaker_support")
            or diagnostics.get("multi_model_fourth_model_rescue")
        ):
            return False
        if float(diagnostics.get("multi_model_continuity_ratio", 1.0) or 0.0) < 0.70:
            return False
        return not any(
            float(diagnostics.get(name, 1.0) or 0.0) <= 0.0
            for name in (
                "excluded_primary_direct_margin",
                "excluded_secondary_direct_margin",
            )
        )

    def _job_paths(self, job_id: str) -> dict[str, Path]:
        root = WORK_ROOT / job_id
        return {
            "root": root,
            "normalized_refs": root / "references_normalized",
            "reference_stems": root / "reference_stems",
            "reference_clips": root / "reference_voice_clips",
            "negative_refs": root / "negative_references_normalized",
            "negative_stems": root / "negative_reference_stems",
            "negative_clips": root / "negative_reference_voice_clips",
            "normalized_target": root / "target_normalized.wav",
            "singing_removed_target": root / "target_singing_removed.wav",
            "stems": root / "stems",
            "candidate_clips": root / "candidate_clips",
            "output": OUTPUT_ROOT / job_id,
        }

    def _prepare_references(
        self,
        references: list[Path],
        paths: dict[str, Path],
        progress: ProgressCallback,
    ) -> tuple[list[Path], list[float]]:
        paths["normalized_refs"].mkdir(parents=True, exist_ok=True)
        normalized: list[Path] = []
        durations: list[float] = []
        for index, reference in enumerate(references, start=1):
            destination = paths["normalized_refs"] / f"reference_{index:03d}.wav"
            normalize_audio(reference, destination, sample_rate=44100, stereo=True)
            durations.append(pad_for_separator(destination))
            normalized.append(destination)
            progress(0.04 + 0.08 * index / max(1, len(references)), f"规范化参考音频 {index}/{len(references)}")
        return normalized, durations

    def _prepare_negative_references(
        self,
        reference_groups: list[list[Path]],
        paths: dict[str, Path],
        progress: ProgressCallback,
    ) -> tuple[list[list[Path]], list[list[float]]]:
        normalized_groups: list[list[Path]] = []
        duration_groups: list[list[float]] = []
        total = sum(len(group) for group in reference_groups)
        completed = 0
        for group_index, group in enumerate(reference_groups, start=1):
            normalized: list[Path] = []
            durations: list[float] = []
            group_dir = paths["negative_refs"] / f"role_{group_index:03d}"
            group_dir.mkdir(parents=True, exist_ok=True)
            for reference_index, reference in enumerate(group, start=1):
                destination = group_dir / f"reference_{reference_index:03d}.wav"
                normalize_audio(reference, destination, sample_rate=44100, stereo=True)
                durations.append(pad_for_separator(destination))
                normalized.append(destination)
                completed += 1
                progress(
                    0.12,
                    f"规范化排除角色 {group_index}：{reference_index}/{len(group)} "
                    f"（总计 {completed}/{max(1, total)}）",
                )
            normalized_groups.append(normalized)
            duration_groups.append(durations)
        return normalized_groups, duration_groups

    @staticmethod
    def _merge_vad_spans(
        spans: list[TimeSpan],
        gap: float = 0.18,
        blocked: Iterable[TimeSpan] = (),
    ) -> list[TimeSpan]:
        if not spans:
            return []
        spans = sorted(spans, key=lambda item: item.start)
        blocked = tuple(blocked)
        merged: list[TimeSpan] = [spans[0]]
        for span in spans[1:]:
            previous = merged[-1]
            gap_start = previous.end
            gap_end = span.start
            crosses_blocked = any(
                min(gap_end, dirty.end) - max(gap_start, dirty.start) > 0.01
                for dirty in blocked
            )
            if span.start <= previous.end + gap and not crosses_blocked:
                merged[-1] = TimeSpan(previous.start, max(previous.end, span.end))
            else:
                merged.append(span)
        return merged

    def _make_reference_clips(
        self,
        reference_stems: list[Path],
        spans_by_path: dict[Path, list[TimeSpan]],
        paths: dict[str, Path],
        output_dir: Path | None = None,
        prefix: str = "reference",
    ) -> list[Path]:
        output: list[Path] = []
        output_dir = output_dir or paths["reference_clips"]
        output_dir.mkdir(parents=True, exist_ok=True)
        for reference_index, stem in enumerate(reference_stems, start=1):
            spans = self._merge_vad_spans(spans_by_path.get(stem, []), gap=0.35)
            clip_index = 0
            for span in spans:
                start = max(0.0, span.start - 0.05)
                end = min(probe_duration(stem), span.end + 0.05)
                cursor = start
                while end - cursor >= 0.8:
                    clip_end = min(end, cursor + 8.0)
                    if clip_end - cursor < 0.8:
                        break
                    clip_index += 1
                    destination = output_dir / (
                        f"{prefix}_{reference_index:03d}_{clip_index:03d}.wav"
                    )
                    write_clip(stem, destination, cursor, clip_end, sample_rate=16000)
                    output.append(destination)
                    cursor = clip_end
            if clip_index == 0:
                # Short references are still useful; the verifier pads them safely.
                destination = output_dir / f"{prefix}_{reference_index:03d}_001.wav"
                write_clip(stem, destination, 0.0, probe_duration(stem), sample_rate=16000)
                output.append(destination)
        if not output:
            raise ValueError("参考音频中未检测到有效讲话")
        return output

    @staticmethod
    def _waveform_span(waveform: torch.Tensor, span: TimeSpan, sample_rate: int = 16000) -> torch.Tensor:
        start = max(0, int(round(span.start * sample_rate)))
        end = min(waveform.numel(), int(round(span.end * sample_rate)))
        return waveform[start:end]

    @staticmethod
    def _join_transcript_text(left: str, right: str) -> str:
        """Join adjacent ASR fragments without inserting spaces in CJK text."""

        left = (left or "").strip()
        right = (right or "").strip()
        if not left:
            return right
        if not right:
            return left
        cjk = r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]"
        return f"{left}{right}" if re.search(cjk, left[-1:] + right[:1]) else f"{left} {right}"

    @classmethod
    def _coalesce_short_sentences(
        cls,
        sentences: list[CandidateSentence],
        minimum_seconds: float,
        max_gap: float = 0.65,
    ) -> list[CandidateSentence]:
        """Merge tiny ASR fragments inside one already verified speaker turn."""

        output = list(sorted(sentences, key=lambda item: (item.start, item.end)))
        index = 0
        while index < len(output):
            current = output[index]
            if current.duration >= minimum_seconds:
                index += 1
                continue
            if index + 1 < len(output):
                following = output[index + 1]
                if following.start - current.end <= max_gap:
                    current.end = max(current.end, following.end)
                    current.whisper_text = cls._join_transcript_text(
                        current.whisper_text, following.whisper_text
                    )
                    current.text = cls._join_transcript_text(current.text, following.text)
                    current.diagnostics["stt_short_merged"] = True
                    current.diagnostics["stt_merged_span"] = [following.start, following.end]
                    del output[index + 1]
                    continue
            if index > 0:
                previous = output[index - 1]
                if current.start - previous.end <= max_gap:
                    previous.end = max(previous.end, current.end)
                    previous.whisper_text = cls._join_transcript_text(
                        previous.whisper_text, current.whisper_text
                    )
                    previous.text = cls._join_transcript_text(previous.text, current.text)
                    previous.diagnostics["stt_short_merged"] = True
                    previous.diagnostics["stt_merged_span"] = [current.start, current.end]
                    del output[index]
                    index = max(0, index - 1)
                    continue
            index += 1
        return output

    @staticmethod
    def _merge_adjacent_target_turns(
        turns: list[CandidateSentence],
        minimum_seconds: float,
        max_gap: float = 0.25,
    ) -> list[CandidateSentence]:
        """Rejoin target turns split by an over-sensitive local boundary."""

        output = list(sorted(turns, key=lambda item: (item.start, item.end)))
        index = 0
        while index + 1 < len(output):
            left, right = output[index], output[index + 1]
            gap = max(0.0, right.start - left.end)
            left_index = left.diagnostics.get("speaker_turn_index")
            right_index = right.diagnostics.get("speaker_turn_index")
            adjacent_turns = (
                isinstance(left_index, int)
                and isinstance(right_index, int)
                and right_index == left_index + 1
            )
            if (
                gap <= max_gap
                and adjacent_turns
                and left.diagnostics.get("speech_block_index")
                == right.diagnostics.get("speech_block_index")
            ):
                left.end = max(left.end, right.end)
                left.diagnostics["speaker_turns_merged"] = True
                left.diagnostics["merged_turn_span"] = [right.start, right.end]
                # Keep conservative metadata when two verified turns are joined.
                left.speaker_score = min(left.speaker_score, right.speaker_score)
                left.window_p20_score = min(left.window_p20_score, right.window_p20_score)
                left.speaker_vote_ratio = min(left.speaker_vote_ratio, right.speaker_vote_ratio)
                left.window_vote_ratio = min(left.window_vote_ratio, right.window_vote_ratio)
                del output[index + 1]
                continue
            index += 1
        return output

    @staticmethod
    def _target_coverage(span: TimeSpan, target_spans: list[TimeSpan]) -> float:
        if span.duration <= 0.0 or not target_spans:
            return 0.0
        covered = sum(
            max(0.0, min(span.end, target.end) - max(span.start, target.start))
            for target in target_spans
        )
        return min(1.0, covered / span.duration)

    @staticmethod
    def _same_speech_block(
        left: TimeSpan,
        right: TimeSpan,
        blocks: Iterable[TimeSpan],
    ) -> bool:
        """Return whether two adjacent turns belong to one VAD speech island."""

        if right.start - left.end > 0.02:
            return False
        return any(
            block.start <= left.start + 0.02
            and right.end <= block.end + 0.02
            for block in blocks
        )

    @staticmethod
    def _target_side_recovery_support(
        match: SpeakerMatchDecision,
        duration: float,
        threshold: float,
    ) -> bool:
        """Recognize a short target-like side without exporting it alone.

        A local splitter can cut a low-energy phoneme below the complete-turn
        gate.  The side may still be useful when joining it to an adjacent
        verified target turn, but it must carry independent support from both
        speaker models and must not be treated as a standalone sentence.
        """

        secondary = match.secondary
        if secondary is None or not 0.75 <= duration <= 4.0:
            return False
        if match.tier not in {"weak", "recall"}:
            return False
        return (
            match.primary.score >= max(0.54, threshold - 0.15)
            and secondary.score >= max(0.52, threshold - 0.16)
            and match.primary.reference_max_score >= 0.50
            and secondary.reference_max_score >= 0.48
            and match.paired_reference_median >= 0.47
        )

    @staticmethod
    def _short_edge_recovery_support(
        match: SpeakerMatchDecision | None,
        duration: float,
        threshold: float,
    ) -> bool:
        """Return whether a sub-sentence fragment may support a join.

        VAD and local speaker cuts can leave a 0.x-second phoneme on its own.
        Such a fragment is never an output sentence.  It is useful only when
        both speaker models provide direct reference evidence and a later
        whole-span verification accepts the joined sentence.
        """

        if match is None or match.secondary is None:
            return False
        if not 0.20 <= duration < 1.20:
            return False
        primary = match.primary
        secondary = match.secondary
        if duration < 0.75:
            local_window_score = min(
                primary.window_p20_score,
                secondary.window_p20_score,
            )
            local_window_vote = min(
                primary.window_vote_ratio,
                secondary.window_vote_ratio,
            )
        else:
            local_window_score = max(
                primary.window_p20_score,
                secondary.window_p20_score,
            )
            local_window_vote = max(
                primary.window_vote_ratio,
                secondary.window_vote_ratio,
            )
        return (
            primary.score >= max(0.52, threshold - 0.16)
            and secondary.score >= max(0.48, threshold - 0.20)
            and primary.reference_max_score >= 0.48
            and secondary.reference_max_score >= 0.40
            and match.paired_reference_median >= 0.44
            and local_window_score >= 0.34
            and local_window_vote >= 0.25
        )

    @staticmethod
    def _short_edge_can_join(
        candidate: CandidateSentence,
        complete_edge_seconds: float,
    ) -> bool:
        """Require explicit evidence before joining a short rejected side."""

        if candidate.duration < complete_edge_seconds:
            return bool(candidate.diagnostics.get("short_edge_evidence"))
        return bool(
            candidate.diagnostics.get("short_edge_evidence")
            or candidate.diagnostics.get("speaker_tier") == "recall"
            or (
                float(candidate.diagnostics.get("eres_score", 0.0)) >= 0.50
                and float(candidate.diagnostics.get("camplus_score", 0.0)) >= 0.50
                and float(
                    candidate.diagnostics.get("paired_reference_median", 0.0)
                )
                >= 0.40
            )
        )

    def _verify_short_edge_candidate(
        self,
        candidate: CandidateSentence,
        verifier: DualSpeakerVerifier,
        waveform: torch.Tensor,
        profile: SpeakerMatchProfile,
        exclusion_profiles: list[ExclusionSpeakerProfile],
        threshold: float,
    ) -> SpeakerMatchDecision | None:
        """Score a short fragment as join evidence without accepting it."""

        if candidate.duration < 0.20:
            return None
        candidate.diagnostics["short_edge_pending"] = False
        match = self._verify_speaker_span(
            verifier,
            waveform,
            TimeSpan(candidate.start, candidate.end),
            profile,
            threshold,
        )
        self._apply_speaker_match(candidate, match, profile, threshold)
        supported = self._short_edge_recovery_support(
            match,
            candidate.duration,
            threshold,
        )
        candidate.diagnostics["short_edge_evidence"] = supported
        candidate.diagnostics["short_edge_duration"] = round(candidate.duration, 5)
        exclusion = verifier.exclusion_audit(
            match,
            profile,
            exclusion_profiles,
        )
        if exclusion is not None:
            candidate.diagnostics.update(exclusion)
        if exclusion and exclusion.get("excluded_role_rejected"):
            candidate.reject_reason = (
                f"接近排除角色 {exclusion['excluded_role']}，已按排除角色删除"
            )
            candidate.diagnostics["short_edge_evidence"] = False
        else:
            # Keep this reason even when the match is strong enough to support
            # a join.  The caller must never place the fragment in output by
            # itself; _merge_verified_target_turns owns the whole-span gate.
            candidate.reject_reason = "说话回合过短"
        return match

    def _reconcile_target_split_spans(
        self,
        split_result: list[TimeSpan],
        clean_spans: list[TimeSpan],
        target_spans: list[TimeSpan],
        verifier: DualSpeakerVerifier,
        profile: SpeakerMatchProfile,
        waveform: torch.Tensor,
        exclusion_profiles: list[ExclusionSpeakerProfile],
        threshold: float,
    ) -> tuple[list[TimeSpan], int]:
        """Undo a false local cut when a complete target island is verified.

        The generic change detector is intentionally cautious, but short
        phonemes and prosody can still create a cut inside one target sentence.
        A cut is reversible only inside one VAD island, when the independent
        target locator covers both sides and the complete joined span passes
        the ordinary dual-model identity gate again.  This does not join across
        positive silence or use a lower global threshold.
        """

        if len(split_result) < 2 or not target_spans:
            return list(split_result), 0
        ordered = sorted(split_result, key=lambda item: (item.start, item.end))
        output: list[TimeSpan] = []
        merged_count = 0
        index = 0
        side_match_cache: dict[tuple[float, float], SpeakerMatchDecision] = {}

        def side_match(span: TimeSpan) -> SpeakerMatchDecision:
            key = (round(span.start, 3), round(span.end, 3))
            match = side_match_cache.get(key)
            if match is None:
                match = self._verify_speaker_span(
                    verifier, waveform, span, profile, threshold
                )
                side_match_cache[key] = match
            return match

        while index < len(ordered):
            current = ordered[index]
            while index + 1 < len(ordered):
                following = ordered[index + 1]
                if not self._same_speech_block(current, following, clean_spans):
                    break
                joined = TimeSpan(current.start, following.end)
                if joined.duration > min(20.0, self.options.max_sentence_seconds):
                    break
                joined_coverage = self._target_coverage(joined, target_spans)
                if joined_coverage < 0.55:
                    break
                left_match = side_match(current)
                right_match = side_match(following)
                # Locator coverage only says that a sliding window touched a
                # side. It must never overrule the independent side identity
                # check, otherwise a similar neighboring speaker can make an
                # entire VAD island look like one target turn.
                left_supported = left_match.accepted or self._target_side_recovery_support(
                    left_match, current.duration, threshold
                )
                right_supported = right_match.accepted or self._target_side_recovery_support(
                    right_match, following.duration, threshold
                )
                if not left_supported or not right_supported:
                    break
                left_exclusion = verifier.exclusion_audit(
                    left_match, profile, exclusion_profiles
                )
                right_exclusion = verifier.exclusion_audit(
                    right_match, profile, exclusion_profiles
                )
                if (
                    left_exclusion
                    and left_exclusion.get("excluded_role_rejected")
                ) or (
                    right_exclusion
                    and right_exclusion.get("excluded_role_rejected")
                ):
                    break
                match = self._verify_speaker_span(
                    verifier,
                    waveform,
                    joined,
                    profile,
                    threshold,
                )
                exclusion = verifier.exclusion_audit(
                    match,
                    profile,
                    exclusion_profiles,
                )
                if not match.accepted or (
                    exclusion and exclusion.get("excluded_role_rejected")
                ):
                    break
                current = joined
                index += 1
                merged_count += 1
            output.append(current)
            index += 1
        return output, merged_count

    @staticmethod
    def _candidate_source_span(candidate: CandidateSentence) -> TimeSpan | None:
        """Recover the original VAD/speaker span for a recovery candidate."""

        diagnostics = candidate.diagnostics
        pairs = (
            ("locator_source_start", "locator_source_end"),
            ("long_turn_parent_start", "long_turn_parent_end"),
            ("original_turn_start", "original_turn_end"),
            ("recovery_parent_start", "recovery_parent_end"),
        )
        for start_name, end_name in pairs:
            try:
                start = float(diagnostics[start_name])
                end = float(diagnostics[end_name])
            except (KeyError, TypeError, ValueError):
                continue
            if end > start:
                return TimeSpan(start, end)
        return None

    @staticmethod
    def _local_target_quality(match: SpeakerMatchDecision | None) -> float:
        """Score a local window without turning it into a global threshold."""

        if match is None or match.secondary is None:
            return -1.0
        primary = float(match.primary.score)
        secondary = float(match.secondary.score)
        direct = min(
            float(match.primary.reference_max_score),
            float(match.secondary.reference_max_score),
        )
        return 0.45 * primary + 0.45 * secondary + 0.10 * direct

    def _audit_risky_target_edges(
        self,
        accepted_turns: list[CandidateSentence],
        verifier: DualSpeakerVerifier,
        profile: SpeakerMatchProfile,
        audio_path: Path,
        waveform: torch.Tensor,
        exclusion_profiles: list[ExclusionSpeakerProfile],
        target_spans: list[TimeSpan],
        threshold: float,
        progress: ProgressCallback,
    ) -> int:
        """Trim locator/repartition candidates whose edges are another voice.

        A long VAD island can contain a short reply from a different speaker.
        The 1.8 second target locator may still mark one broad region as
        positive, and a whole-span embedding then hides that contamination.
        This audit is restricted to candidates whose source turn had incomplete
        target coverage.  It uses short multi-scale boundaries only to propose
        cuts, then requires a target-quality gain on one side and a complete
        re-verification of the retained core.
        """

        if not accepted_turns:
            return 0
        splitter = LocalSpeakerTurnSplitter(
            verifier.primary,
            secondary=verifier.secondary,
        )
        # Edge audits revisit the same split points from both directions. Keep
        # the expensive dual-model embedding result per time span so a long
        # batch does not appear to stall while recomputing identical windows.
        match_cache: dict[tuple[float, float], SpeakerMatchDecision] = {}

        def cached_match(span: TimeSpan) -> SpeakerMatchDecision:
            key = (round(span.start, 3), round(span.end, 3))
            match = match_cache.get(key)
            if match is None:
                match = self._verify_speaker_span(
                    verifier, waveform, span, profile, threshold
                )
                match_cache[key] = match
            return match

        contrastive_cache: dict[
            tuple[float, float, float, float], SpeakerMatchDecision | None
        ] = {}

        def contrastive_edge_match(
            retained: TimeSpan,
            residual: TimeSpan,
            retained_match: SpeakerMatchDecision,
            residual_match: SpeakerMatchDecision,
        ) -> SpeakerMatchDecision | None:
            if retained_match.accepted:
                return retained_match
            if not 0.25 <= residual.duration <= 0.90:
                return retained_match
            key = (
                round(retained.start, 3),
                round(retained.end, 3),
                round(residual.start, 3),
                round(residual.end, 3),
            )
            if key not in contrastive_cache:
                contrastive_cache[key] = (
                    verifier.promote_contrastive_edge_with_tertiary(
                        self._waveform_span(waveform, retained),
                        self._waveform_span(waveform, residual),
                        profile,
                        retained_match,
                        residual_match,
                        retained.duration,
                        residual.duration,
                    )
                )
            return contrastive_cache[key]

        replacements: list[tuple[CandidateSentence, CandidateSentence]] = []
        for candidate in list(accepted_turns):
            diagnostics = candidate.diagnostics
            # The same-turn pass has already validated both the recovered edge
            # and the complete source span. Re-auditing its interior with
            # shorter windows would mistake prosody changes for new speakers
            # and cut the sentence back to the old incomplete core.
            if diagnostics.get("same_turn_edge_recovery"):
                continue
            if not (
                diagnostics.get("target_locator_recovery")
                or diagnostics.get("long_turn_repartition")
            ):
                continue
            source = self._candidate_source_span(candidate)
            minimum_retained_seconds = max(1.20, self.options.min_output_seconds)
            if source is None or candidate.duration < minimum_retained_seconds:
                continue
            source_coverage = diagnostics.get("locator_source_target_coverage")
            if source_coverage is None:
                source_coverage = diagnostics.get("long_turn_parent_coverage", 1.0)
            try:
                source_coverage = float(source_coverage)
            except (TypeError, ValueError):
                source_coverage = 1.0
            if source_coverage >= 0.90:
                continue

            audit_span = TimeSpan(
                max(source.start, candidate.start - 0.05),
                min(source.end, candidate.end + 0.05),
            )
            if audit_span.duration < max(1.80, minimum_retained_seconds + 0.25):
                continue
            short_candidate = candidate.duration < 2.20
            boundaries = splitter.detect_multiscale_speaker_boundaries(
                audio_path,
                [audit_span],
                # The boundary detector requires at least 0.30 s of context;
                # keep the short-candidate audit within that contract.
                contexts=(0.30, 0.40, 0.55) if short_candidate else (0.30, 0.40, 0.55, 0.70),
                cluster_seconds=0.20,
                minimum_context_votes=2,
                scan_hop_seconds=0.05,
                primary_candidate_threshold=0.72,
                minimum_similarity_drop=0.04,
                minimum_separation_seconds=0.15,
                progress=lambda value, message: progress(
                    0.80,
                    f"局部边缘纯度审计：{message}",
                ),
            )
            internal = sorted(
                {
                    round(boundary.time, 3): boundary
                    for boundary in boundaries
                    if candidate.start + (0.08 if short_candidate else 0.15)
                    <= boundary.time
                    <= candidate.end - (0.08 if short_candidate else 0.15)
                }.values(),
                key=lambda item: item.time,
            )
            if not internal:
                continue

            current = TimeSpan(candidate.start, candidate.end)
            current_match = cached_match(current)
            changed = False
            audit_details: list[dict[str, float | str]] = []
            edge_margin = 0.08 if short_candidate else 0.15
            # A local boundary is evidence about an edge, not permission to
            # repeatedly carve the middle out of a sentence.  Audit each edge
            # at most once and prefer the boundary nearest the original edge;
            # this keeps a short contaminating reply out without deleting a
            # low-energy phoneme later in the same target sentence.
            for edge_kind in ("head", "tail"):
                options: list[
                    tuple[
                        float,
                        str,
                        TimeSpan,
                        SpeakerMatchDecision,
                        SpeakerMatchDecision,
                        SpeakerBoundary,
                    ]
                ] = []
                for boundary in internal:
                    cut = float(boundary.time)
                    if not current.start + edge_margin < cut < current.end - edge_margin:
                        continue
                    left = TimeSpan(current.start, cut)
                    right = TimeSpan(cut, current.end)
                    left_match = cached_match(left)
                    right_match = cached_match(right)
                    left_quality = self._local_target_quality(left_match)
                    right_quality = self._local_target_quality(right_match)
                    left_exclusion = verifier.exclusion_audit(
                        left_match,
                        profile,
                        exclusion_profiles,
                        tertiary_recovery=left_match.tier == "tertiary",
                    )
                    right_exclusion = verifier.exclusion_audit(
                        right_match,
                        profile,
                        exclusion_profiles,
                        tertiary_recovery=right_match.tier == "tertiary",
                    )
                    left_negative = bool(
                        left_exclusion
                        and left_exclusion.get("excluded_role_rejected")
                    )
                    right_negative = bool(
                        right_exclusion
                        and right_exclusion.get("excluded_role_rejected")
                    )
                    if boundary.scale_votes < 3:
                        continue
                    if (
                        right.duration >= minimum_retained_seconds
                        and right_quality - left_quality >= 0.10
                        and right_quality >= 0.52
                        and (
                            left_negative
                            or left_quality <= (0.35 if left.duration < 0.40 else 0.50)
                        )
                    ) and edge_kind == "head":
                        options.append(
                            (
                                right_quality - left_quality,
                                "head",
                                right,
                                right_match,
                                left_match,
                                boundary,
                            )
                        )
                    if (
                        left.duration >= minimum_retained_seconds
                        and left_quality - right_quality >= 0.10
                        and left_quality >= 0.52
                        and (
                            right_negative
                            or right_quality <= (0.35 if right.duration < 0.40 else 0.50)
                        )
                    ) and edge_kind == "tail":
                        options.append(
                            (
                                left_quality - right_quality,
                                "tail",
                                left,
                                left_match,
                                right_match,
                                boundary,
                            )
                        )
                if not options:
                    continue
                if edge_kind == "head":
                    # Earliest accepted cut is the conservative head trim.
                    # A larger quality gain is only a tie-breaker.
                    selected = min(
                        options,
                        key=lambda item: (
                            item[2].start - candidate.start,
                            -item[0],
                        ),
                    )
                else:
                    # Latest accepted cut is the conservative tail trim.
                    selected = min(
                        options,
                        key=lambda item: (
                            candidate.end - item[2].end,
                            -item[0],
                        ),
                    )
                (
                    gain,
                    edge,
                    retained,
                    retained_match,
                    residual_match,
                    boundary,
                ) = selected
                if (
                    retained.duration < minimum_retained_seconds
                    or retained.duration >= current.duration - 0.05
                ):
                    continue
                contrastive = contrastive_edge_match(
                    retained,
                    (
                        TimeSpan(current.start, boundary.time)
                        if edge == "tail"
                        else TimeSpan(boundary.time, current.end)
                    ),
                    retained_match,
                    residual_match,
                )
                if contrastive is None:
                    continue
                retained_match = contrastive
                audit_details.append(
                    {
                        "edge": edge,
                        "retained_start": retained.start,
                        "retained_end": retained.end,
                        "quality_gain": round(gain, 5),
                        "boundary": round(boundary.time, 5),
                        "boundary_scale_votes": boundary.scale_votes,
                        "contrastive_edge": contrastive is not None
                        and bool(
                            retained_match.diagnostics.get(
                                "contrastive_edge_tertiary_passed"
                            )
                        ),
                    }
                )
                current = retained
                current_match = retained_match
                changed = True

            if not changed:
                continue
            if not current_match.accepted:
                promoted = verifier.promote_local_with_tertiary(
                    self._waveform_span(waveform, current),
                    profile,
                    current_match,
                    current.duration,
                )
                if promoted is not None:
                    current_match = promoted
            if not current_match.accepted:
                continue
            exclusion = verifier.exclusion_audit(
                current_match,
                profile,
                exclusion_profiles,
                tertiary_recovery=current_match.tier == "tertiary",
            )
            if exclusion and exclusion.get("excluded_role_rejected"):
                continue
            replacement = CandidateSentence(current.start, current.end, "")
            replacement.diagnostics.update(candidate.diagnostics)
            self._apply_speaker_match(replacement, current_match, profile, threshold)
            replacement.diagnostics.update(
                {
                    "edge_purity_trim": True,
                    "edge_purity_source_start": candidate.start,
                    "edge_purity_source_end": candidate.end,
                    "edge_purity_source_target_coverage": round(source_coverage, 5),
                    "edge_purity_boundaries": [boundary.to_dict() for boundary in internal],
                    "edge_purity_steps": audit_details,
                    "target_coverage": round(
                        self._target_coverage(current, target_spans),
                        5,
                    ),
                }
            )
            if exclusion is not None:
                replacement.diagnostics.update(exclusion)
            replacements.append((candidate, replacement))

        for original, replacement in replacements:
            for index, current in enumerate(accepted_turns):
                if current is original:
                    accepted_turns[index] = replacement
                    break
        if replacements:
            progress(0.80, f"局部边缘纯度审计：修正 {len(replacements)} 个候选")
        return len(replacements)

    @staticmethod
    def _partition_tainted_spans(
        spans: list[TimeSpan],
        tainted: list[TimeSpan],
        *,
        minimum_overlap_seconds: float,
        minimum_fraction: float = 0.0,
    ) -> tuple[list[TimeSpan], list[TimeSpan]]:
        """Drop a whole silence-delimited utterance if a forbidden event occurs.

        Cutting only the overlap/song frames leaves sentence fragments.  The
        utterance is therefore atomic at this stage: one confirmed forbidden
        range rejects the whole item.
        """

        clean: list[TimeSpan] = []
        rejected: list[TimeSpan] = []
        for span in spans:
            overlap = sum(
                max(0.0, min(span.end, dirty.end) - max(span.start, dirty.start))
                for dirty in tainted
            )
            fraction = overlap / max(1e-6, span.duration)
            if overlap >= minimum_overlap_seconds or fraction >= minimum_fraction > 0.0:
                rejected.append(span)
            else:
                clean.append(span)
        return clean, rejected

    @staticmethod
    def _snap_target_spans_to_speech(
        target_spans: list[TimeSpan],
        speech_spans: list[TimeSpan],
        minimum_overlap_seconds: float = 0.10,
    ) -> list[TimeSpan]:
        """Return complete speaker units touched by target-locator windows.

        Each unit stays independent.  Joining the first and last touched unit
        here can silently cross a confirmed speaker boundary and export a
        sequential two-speaker conversation as one target sentence.
        """

        snapped: dict[tuple[float, float], TimeSpan] = {}
        ordered_speech = sorted(speech_spans, key=lambda item: (item.start, item.end))
        for target in target_spans:
            for speech in ordered_speech:
                overlap = min(target.end, speech.end) - max(target.start, speech.start)
                if overlap >= min(minimum_overlap_seconds, speech.duration * 0.25):
                    snapped[(round(speech.start, 5), round(speech.end, 5))] = speech
        return sorted(snapped.values(), key=lambda item: (item.start, item.end))

    def _merge_short_silence_same_speaker(
        self,
        spans: list[TimeSpan],
        verifier: DualSpeakerVerifier,
        profile: SpeakerMatchProfile,
        waveform: torch.Tensor,
        progress: ProgressCallback,
        *,
        maximum_silence_seconds: float = 0.85,
        primary_same_floor: float = 0.76,
        secondary_same_floor: float = 0.64,
        forbidden_joins: Iterable[TimeSpan] = (),
    ) -> list[TimeSpan]:
        """Merge a short silent gap only when both speaker models agree.

        The returned span covers the original gap, so the short silence is
        retained in exported training audio exactly as requested.
        """

        spans = sorted(spans, key=lambda item: (item.start, item.end))
        if len(spans) < 2:
            return spans
        parts = [self._waveform_span(waveform, span) for span in spans]
        progress(0.0, f"短静音声纹核验：ERes2Net 0/{len(parts)}")
        primary_embeddings = verifier.primary._embeddings_from_waveforms(
            parts,
            progress=lambda completed, total: progress(
                0.50 * completed / max(1, total),
                f"短静音声纹核验：ERes2Net {completed}/{total}",
            ),
        )
        verifier._ensure_secondary(profile)
        assert verifier.secondary is not None
        progress(0.50, f"短静音声纹核验：CAM++ 0/{len(parts)}")
        secondary_embeddings = verifier.secondary._embeddings_from_waveforms(
            parts,
            progress=lambda completed, total: progress(
                0.50 + 0.50 * completed / max(1, total),
                f"短静音声纹核验：CAM++ {completed}/{total}",
            ),
        )

        output: list[TimeSpan] = [spans[0]]
        for index in range(1, len(spans)):
            previous_source = spans[index - 1]
            current = spans[index]
            gap = max(0.0, current.start - previous_source.end)
            primary_similarity = float(primary_embeddings[index - 1] @ primary_embeddings[index])
            secondary_similarity = float(
                secondary_embeddings[index - 1] @ secondary_embeddings[index]
            )
            blocked_join = gap > 0.0 and any(
                min(current.start, blocked.end) - max(previous_source.end, blocked.start)
                > 0.01
                and not (
                    blocked.start <= previous_source.start + 0.02
                    and blocked.end >= current.end - 0.02
                )
                for blocked in forbidden_joins
            )
            same_speaker = (
                gap <= maximum_silence_seconds
                and not blocked_join
                and primary_similarity >= primary_same_floor
                and secondary_similarity >= secondary_same_floor
            )
            if same_speaker:
                output[-1] = TimeSpan(output[-1].start, current.end)
            else:
                output.append(current)
        progress(1.0, f"短静音声纹核验完成：{len(spans)} 段合并为 {len(output)} 段")
        return output

    @staticmethod
    def _apply_speaker_match(
        candidate: CandidateSentence,
        match,
        profile: SpeakerMatchProfile,
        threshold: float,
    ) -> None:
        decision = match.primary
        candidate.speaker_score = decision.score
        candidate.window_min_score = decision.window_min_score
        candidate.window_p20_score = decision.window_p20_score
        candidate.speaker_vote_ratio = decision.vote_ratio
        candidate.window_vote_ratio = decision.window_vote_ratio
        candidate.speaker_threshold = threshold
        candidate.diagnostics.update(
            {
                "speaker_vote_ratio": decision.vote_ratio,
                "speaker_window_p20": decision.window_p20_score,
                "speaker_window_vote_ratio": decision.window_vote_ratio,
                "speaker_threshold": threshold,
                "speaker_reference_floor": profile.primary.reference_floor,
                "speaker_calibration_base": profile.primary.calibration_base,
                "speaker_match_mode": match.match_mode,
                "speaker_tier": match.tier,
                "eres_reference_median": decision.reference_median_score,
                "eres_reference_max": decision.reference_max_score,
                "eres_reference_spread": decision.reference_spread,
                "paired_reference_median": match.paired_reference_median,
            }
        )
        candidate.diagnostics.update(match.diagnostics)
        if match.secondary is not None:
            candidate.diagnostics.update(
                {
                    "camplus_score": match.secondary.score,
                    "camplus_window_p20": match.secondary.window_p20_score,
                    "camplus_window_vote_ratio": match.secondary.window_vote_ratio,
                    "camplus_reference_vote_ratio": match.secondary.vote_ratio,
                    "camplus_reference_median": match.secondary.reference_median_score,
                    "camplus_reference_max": match.secondary.reference_max_score,
                    "camplus_reference_spread": match.secondary.reference_spread,
                }
            )

    def _verify_speaker_span(
        self,
        verifier: DualSpeakerVerifier,
        waveform: torch.Tensor,
        span: TimeSpan,
        profile: SpeakerMatchProfile,
        threshold: float,
    ) -> SpeakerMatchDecision:
        return verifier.verify_waveform(
            self._waveform_span(waveform, span),
            profile,
            threshold,
            duration=span.duration,
            window_seconds=min(1.8, max(1.0, span.duration)),
            hop_seconds=min(0.9, max(0.5, span.duration / 2)),
        )

    def _recover_same_turn_edges(
        self,
        accepted_turns: list[CandidateSentence],
        turn_spans: list[TimeSpan],
        stable_boundaries: list[SpeakerBoundary],
        verifier: DualSpeakerVerifier,
        profile: SpeakerMatchProfile,
        waveform: torch.Tensor,
        target_spans: list[TimeSpan],
        exclusion_profiles: list[ExclusionSpeakerProfile],
        threshold: float,
        forbidden_joins: Iterable[TimeSpan],
        progress: ProgressCallback,
    ) -> int:
        """Restore a short clipped edge inside an otherwise stable turn.

        Target locators can stop at the last high-energy phoneme even when the
        original VAD/speaker turn continues for a short tail.  The extension
        is considered only when no multi-scale boundary occurs anywhere in the
        source turn and the complete source turn passes verification again.
        This prevents the old locator recovery from re-installing a mixed turn.
        """

        if not accepted_turns or not turn_spans:
            return 0
        structural = [
            boundary
            for boundary in stable_boundaries
            if self._is_structural_boundary(boundary)
        ]
        recovered: list[CandidateSentence] = []
        for candidate in list(accepted_turns):
            if candidate.diagnostics.get("local_boundary_recovery"):
                continue
            try:
                turn_index = int(candidate.diagnostics.get("speaker_turn_index", -1))
            except (TypeError, ValueError):
                turn_index = -1
            source = turn_spans[turn_index] if 0 <= turn_index < len(turn_spans) else None
            if source is None or not (
                source.start <= candidate.start + 0.05
                and candidate.end <= source.end + 0.05
            ):
                containing = [
                    span
                    for span in turn_spans
                    if span.start <= candidate.start + 0.05
                    and candidate.end <= span.end + 0.05
                ]
                if not containing:
                    continue
                source = min(containing, key=lambda item: item.duration)
            if (
                source.start > candidate.start + 0.05
                or source.end < candidate.end - 0.05
            ):
                continue
            extension = source.duration - candidate.duration
            # Low-energy sentence edges can be substantially longer than the
            # old fixed 1.25 s allowance.  The extension is still bounded and
            # must pass an independent side check below; this is not a global
            # threshold relaxation.
            maximum_extension = max(
                1.25,
                min(2.40, self.options.silence_split_seconds * 2.5),
            )
            if not 0.05 <= extension <= maximum_extension:
                continue
            if source.duration > min(20.0, self.options.max_sentence_seconds):
                continue
            if any(source.start + 0.30 < boundary.time < source.end - 0.30 for boundary in structural):
                continue
            if any(
                min(source.end, blocked.end) - max(source.start, blocked.start) > 0.01
                for blocked in forbidden_joins
            ):
                continue
            coverage = self._target_coverage(source, target_spans)
            if target_spans and coverage < 0.55:
                continue

            # Audit both missing sides independently.  The previous code chose
            # either the head or tail, so a candidate clipped on both ends
            # could never recover the second valid side.
            edge_results: list[tuple[str, TimeSpan, float]] = []
            edge_details: list[dict[str, object]] = []
            if candidate.start > source.start + 0.05:
                edge_results.append(
                    ("head", TimeSpan(source.start, candidate.start), 0.0)
                )
            if source.end > candidate.end + 0.05:
                edge_results.append(
                    ("tail", TimeSpan(candidate.end, source.end), 0.0)
                )
            if not edge_results:
                continue

            supported_edges: list[tuple[str, TimeSpan, float]] = []
            for edge_kind, edge_span, _ in edge_results:
                edge_match = self._verify_speaker_span(
                    verifier,
                    waveform,
                    edge_span,
                    profile,
                    threshold,
                )
                edge_coverage = self._target_coverage(edge_span, target_spans)
                # Locator coverage is only a hint. It cannot promote an edge
                # whose independent speaker identity check failed.
                edge_supported = (
                    edge_match.accepted
                    or self._target_side_recovery_support(
                        edge_match,
                        edge_span.duration,
                        threshold,
                    )
                    or self._short_edge_recovery_support(
                        edge_match,
                        edge_span.duration,
                        threshold,
                    )
                )
                edge_exclusion = verifier.exclusion_audit(
                    edge_match,
                    profile,
                    exclusion_profiles,
                )
                if edge_exclusion and edge_exclusion.get("excluded_role_rejected"):
                    edge_supported = False
                edge_details.append(
                    {
                        "edge": edge_kind,
                        "start": edge_span.start,
                        "end": edge_span.end,
                        "duration": round(edge_span.duration, 5),
                        "target_coverage": round(edge_coverage, 5),
                        "supported": edge_supported,
                        "excluded": bool(
                            edge_exclusion
                            and edge_exclusion.get("excluded_role_rejected")
                        ),
                    }
                )
                if edge_supported:
                    supported_edges.append(
                        (edge_kind, edge_span, edge_coverage)
                    )
            if not supported_edges:
                continue

            expanded_start = min(
                [candidate.start]
                + [edge.start for _kind, edge, _coverage in supported_edges]
            )
            expanded_end = max(
                [candidate.end]
                + [edge.end for _kind, edge, _coverage in supported_edges]
            )
            expanded_span = TimeSpan(expanded_start, expanded_end)
            expanded_extension = expanded_span.duration - candidate.duration
            if not 0.05 <= expanded_extension <= maximum_extension:
                continue
            expanded_coverage = self._target_coverage(expanded_span, target_spans)
            match = self._verify_speaker_span(
                verifier,
                waveform,
                expanded_span,
                profile,
                threshold,
            )
            if not match.accepted:
                continue
            exclusion = verifier.exclusion_audit(
                match,
                profile,
                exclusion_profiles,
            )
            if exclusion and exclusion.get("excluded_role_rejected"):
                continue

            expanded = CandidateSentence(expanded_span.start, expanded_span.end, "")
            expanded.diagnostics.update(candidate.diagnostics)
            self._apply_speaker_match(expanded, match, profile, threshold)
            expanded.diagnostics.update(
                {
                    "same_turn_edge_recovery": True,
                    "same_turn_source_start": source.start,
                    "same_turn_source_end": source.end,
                    "same_turn_extension_seconds": round(expanded_extension, 5),
                    "same_turn_edge_sides": [
                        kind for kind, _edge, _edge_coverage in supported_edges
                    ],
                    "same_turn_edge_details": edge_details,
                    "same_turn_edge_start": min(
                        edge.start for _kind, edge, _coverage in supported_edges
                    ),
                    "same_turn_edge_end": max(
                        edge.end for _kind, edge, _coverage in supported_edges
                    ),
                    "same_turn_edge_coverage": round(
                        max(coverage, expanded_coverage), 5
                    ),
                    "target_coverage": round(expanded_coverage, 5),
                }
            )
            if exclusion is not None:
                expanded.diagnostics.update(exclusion)
            recovered.append(expanded)

        if not recovered:
            return 0
        for expanded in recovered:
            for existing in list(accepted_turns):
                if (
                    existing.start >= expanded.start - 0.05
                    and existing.end <= expanded.end + 0.05
                ):
                    accepted_turns.remove(existing)
            accepted_turns.append(expanded)
        progress(
            0.76,
            f"同一说话回合短尾复核：恢复 {len(recovered)} 个完整回合",
        )
        return len(recovered)

    def _merge_verified_target_turns(
        self,
        accepted_turns: list[CandidateSentence],
        rejected: list[CandidateSentence],
        verifier: DualSpeakerVerifier,
        profile: SpeakerMatchProfile,
        waveform: torch.Tensor,
        threshold: float,
        target_spans: list[TimeSpan],
        vad_spans: list[TimeSpan],
        forbidden_joins: Iterable[TimeSpan],
        progress: ProgressCallback,
        exclusion_profiles: Iterable[ExclusionSpeakerProfile] = (),
    ) -> int:
        """Join short-silence fragments only around a formally accepted core.

        Edge fragments never become output by themselves.  They may only repair
        a clipped edge when they are adjacent to a strict target turn,
        both speaker models regard the neighboring audio as the same voice, and
        the complete joined sentence passes formal verification again.
        """

        if not accepted_turns:
            return 0
        accepted_ids = {id(candidate) for candidate in accepted_turns}
        # Do not run speaker models for every tiny VAD island in an episode.
        # Only fragments adjacent to a verified target turn can repair that
        # turn, so score those pending edges on demand here.
        pending_edges = [
            candidate
            for candidate in rejected
            if candidate.reject_reason == "说话回合过短"
            and candidate.diagnostics.get("short_edge_pending")
            and any(
                0.0 <= candidate.start - accepted.end <= self.options.silence_split_seconds
                or 0.0 <= accepted.start - candidate.end <= self.options.silence_split_seconds
                for accepted in accepted_turns
            )
        ]
        for candidate in pending_edges:
            self._verify_short_edge_candidate(
                candidate,
                verifier,
                waveform,
                profile,
                list(exclusion_profiles),
                threshold,
            )
        possible_edges = [
            candidate
            for candidate in rejected
            if candidate.reject_reason in {"声纹匹配不足", "说话回合过短"}
            and self._short_edge_can_join(
                candidate,
                max(1.20, self.options.min_output_seconds),
            )
            and not any(
                min(candidate.end, accepted.end) - max(candidate.start, accepted.start)
                > 0.10
                for accepted in accepted_turns
            )
        ]
        edge_by_id: dict[int, CandidateSentence] = {}
        for accepted in accepted_turns:
            left = [
                candidate
                for candidate in possible_edges
                if 0.0
                <= accepted.start - candidate.end
                <= self.options.silence_split_seconds
            ]
            right = [
                candidate
                for candidate in possible_edges
                if 0.0
                <= candidate.start - accepted.end
                <= self.options.silence_split_seconds
            ]
            if left:
                candidate = max(left, key=lambda item: item.end)
                edge_by_id[id(candidate)] = candidate
            if right:
                candidate = min(right, key=lambda item: item.start)
                edge_by_id[id(candidate)] = candidate
        recall_edges = list(edge_by_id.values())
        pool = sorted(
            [*accepted_turns, *recall_edges],
            key=lambda item: (item.start, item.end),
        )
        if len(pool) < 2:
            return 0
        merged_spans = self._merge_short_silence_same_speaker(
            [TimeSpan(candidate.start, candidate.end) for candidate in pool],
            verifier,
            profile,
            waveform,
            progress,
            maximum_silence_seconds=self.options.silence_split_seconds,
            forbidden_joins=forbidden_joins,
        )

        output: list[CandidateSentence] = []
        used_accepted_ids: set[int] = set()
        used_recall_ids: set[int] = set()
        merged_count = 0
        for merged_span in merged_spans:
            members = [
                candidate
                for candidate in pool
                if merged_span.start - 0.02 <= candidate.start
                and candidate.end <= merged_span.end + 0.02
            ]
            strict_members = [
                candidate for candidate in members if id(candidate) in accepted_ids
            ]
            if not strict_members:
                continue
            if len(members) == 1:
                output.append(strict_members[0])
                used_accepted_ids.add(id(strict_members[0]))
                continue
            coverage = self._target_coverage(merged_span, target_spans)
            if (
                merged_span.duration > min(20.0, self.options.max_sentence_seconds)
                or coverage < 0.30
            ):
                output.extend(strict_members)
                used_accepted_ids.update(id(candidate) for candidate in strict_members)
                continue
            match = self._verify_speaker_span(
                verifier,
                waveform,
                merged_span,
                profile,
                threshold,
            )
            merged_exclusion = verifier.exclusion_audit(
                match,
                profile,
                list(exclusion_profiles),
            )
            if not match.accepted:
                output.extend(strict_members)
                used_accepted_ids.update(id(candidate) for candidate in strict_members)
                continue
            if merged_exclusion and merged_exclusion.get("excluded_role_rejected"):
                output.extend(strict_members)
                used_accepted_ids.update(id(candidate) for candidate in strict_members)
                continue

            merged = CandidateSentence(merged_span.start, merged_span.end, "")
            self._apply_speaker_match(merged, match, profile, threshold)
            merged.diagnostics.update(
                {
                    "post_target_silence_merge": True,
                    "merged_strict_target_count": len(strict_members),
                    "merged_recall_edge_count": len(members) - len(strict_members),
                    "target_coverage": round(coverage, 5),
                    "speech_ratio": round(
                        speech_ratio(vad_spans, merged_span.start, merged_span.end),
                        5,
                    ),
                }
            )
            merged.diagnostics["speech_seconds"] = round(
                merged.duration * merged.diagnostics["speech_ratio"],
                5,
            )
            output.append(merged)
            used_accepted_ids.update(id(candidate) for candidate in strict_members)
            used_recall_ids.update(
                id(candidate)
                for candidate in members
                if id(candidate) not in accepted_ids
            )
            merged_count += 1

        output.extend(
            candidate
            for candidate in accepted_turns
            if id(candidate) not in used_accepted_ids
        )
        accepted_turns[:] = sorted(
            {
                (round(candidate.start, 5), round(candidate.end, 5)): candidate
                for candidate in output
            }.values(),
            key=lambda item: (item.start, item.end),
        )
        if used_recall_ids:
            rejected[:] = [
                candidate for candidate in rejected if id(candidate) not in used_recall_ids
            ]
        return merged_count

    def _repartition_long_target_turns(
        self,
        accepted_turns: list[CandidateSentence],
        rejected: list[CandidateSentence],
        verifier: DualSpeakerVerifier,
        profile: SpeakerMatchProfile,
        waveform: torch.Tensor,
        target_spans: list[TimeSpan],
        exclusion_profiles: list[ExclusionSpeakerProfile],
        threshold: float,
        progress: ProgressCallback,
    ) -> int:
        """Re-score target evidence inside a long, low-coverage turn.

        A whole-turn embedding can be strong after averaging several speakers.
        When the locator covers only part of such a turn, the original span is
        therefore not exportable.  We form groups from the locator's positive
        regions, verify each group independently, and discard the mixed parent.
        The larger bridge is used only inside this recovery pass: it repairs
        locator holes within one continuous VAD island and never joins two
        silence islands globally.
        """

        if not accepted_turns or not target_spans:
            return 0
        evidence_bridge = max(1.50, self.options.silence_split_seconds)
        replaced = 0
        for parent in list(accepted_turns):
            coverage = float(parent.diagnostics.get("target_coverage", 1.0))
            if parent.duration < 8.0 or coverage >= 0.90:
                continue
            evidence = sorted(
                {
                    (round(max(parent.start, item.start), 5),
                     round(min(parent.end, item.end), 5))
                    for item in target_spans
                    if min(parent.end, item.end) - max(parent.start, item.start) >= 0.35
                }
            )
            if not evidence:
                continue
            # First identify at least one independently matching locator
            # interval. Unsupported intervals before that anchor are usually
            # the other speaker's lead-in and must not be averaged into the
            # recovered target. Intervals after an anchor may fill a short
            # locator hole and are rechecked on the complete group below.
            supported_indexes: list[int] = []
            for index, (start, end) in enumerate(evidence):
                item = TimeSpan(start, end)
                if item.duration < 0.75:
                    continue
                item_match = self._verify_speaker_span(
                    verifier, waveform, item, profile, threshold
                )
                if item_match.accepted:
                    supported_indexes.append(index)
            if not supported_indexes:
                parent.reject_reason = "疑似混合说话人，保守舍弃"
                accepted_turns.remove(parent)
                rejected.append(parent)
                continue
            groups: list[TimeSpan] = []
            anchor = supported_indexes[0]
            start, end = evidence[anchor]
            for index in range(anchor + 1, len(evidence)):
                next_start, next_end = evidence[index]
                if next_start - end > evidence_bridge:
                    groups.append(TimeSpan(start, end))
                    start, end = next_start, next_end
                else:
                    end = max(end, next_end)
            groups.append(TimeSpan(start, end))

            replacements: list[CandidateSentence] = []
            for group in groups:
                if group.duration < max(2.20, self.options.min_output_seconds):
                    continue
                group_coverage = self._target_coverage(group, target_spans)
                if group_coverage < 0.70:
                    continue
                group_match = self._verify_speaker_span(
                    verifier, waveform, group, profile, threshold
                )
                if not group_match.accepted and group.duration <= 15.0:
                    group_match = (
                        verifier.promote_with_tertiary(
                            self._waveform_span(waveform, group),
                            profile,
                            group_match,
                            group.duration,
                        )
                        or group_match
                    )
                if not group_match.accepted:
                    continue
                exclusion = verifier.exclusion_audit(
                    group_match, profile, exclusion_profiles,
                    tertiary_recovery=group_match.tier == "tertiary",
                )
                if exclusion and exclusion.get("excluded_role_rejected"):
                    continue
                replacement = CandidateSentence(group.start, group.end, "")
                self._apply_speaker_match(replacement, group_match, profile, threshold)
                replacement.diagnostics.update(
                    {
                        "long_turn_repartition": True,
                        "long_turn_parent_start": parent.start,
                        "long_turn_parent_end": parent.end,
                        "long_turn_parent_coverage": round(coverage, 5),
                        "target_coverage": round(group_coverage, 5),
                    }
                )
                if exclusion is not None:
                    replacement.diagnostics.update(exclusion)
                replacements.append(replacement)

            if not replacements:
                parent.reject_reason = "疑似混合说话人，保守舍弃"
                accepted_turns.remove(parent)
                rejected.append(parent)
                continue
            accepted_turns.remove(parent)
            parent.reject_reason = "疑似混合说话人，保守舍弃"
            rejected.append(parent)
            accepted_turns.extend(replacements)
            replaced += len(replacements)

        if replaced:
            accepted_turns.sort(key=lambda item: (item.start, item.end))
            progress(0.80, f"长回合递归重分配：恢复 {replaced} 个独立目标段")
        return replaced

    @staticmethod
    def _prune_weak_recovery_candidates(
        accepted_turns: list[CandidateSentence],
        rejected: list[CandidateSentence],
    ) -> int:
        """Remove recovery promotions that lack a complete target region."""

        removed = 0
        retained: list[CandidateSentence] = []
        for candidate in accepted_turns:
            diagnostics = candidate.diagnostics
            coverage = float(diagnostics.get("target_coverage", 1.0) or 0.0)
            locator_fragment = (
                bool(diagnostics.get("target_locator_recovery"))
                and not bool(diagnostics.get("post_target_silence_merge"))
                and candidate.duration < 8.0
                and coverage < 0.90
            )
            weak_cluster_fragment = (
                bool(diagnostics.get("multi_model_target_recovery"))
                and diagnostics.get("speaker_tier") == "rejected"
                and candidate.duration < 8.0
                and coverage < 0.85
            )
            if locator_fragment or weak_cluster_fragment:
                candidate.reject_reason = "局部目标候选未连接到完整句"
                rejected.append(candidate)
                removed += 1
            else:
                retained.append(candidate)
        accepted_turns[:] = retained
        return removed

    def _build_adaptive_speaker_profile(
        self,
        accepted_turns: list[CandidateSentence],
        verifier: DualSpeakerVerifier,
        profile: SpeakerMatchProfile,
        waveform: torch.Tensor,
        threshold: float,
    ) -> tuple[SpeakerMatchProfile, CAMPlusProfile, float, float] | None:
        """Add only strict, domain-matched target turns as temporary anchors."""

        anchors = [
            candidate
            for candidate in accepted_turns
            if 2.20 <= candidate.duration <= 15.0
            and (
                candidate.duration <= 4.50
                or candidate.diagnostics.get("post_target_silence_merge")
            )
            and candidate.diagnostics.get("speaker_tier")
            in {"short_strong", "strong", "balanced"}
            and float(candidate.diagnostics.get("target_coverage", 0.0)) >= 0.90
            and float(candidate.diagnostics.get("speech_ratio", 1.0)) >= 0.80
            and int(candidate.diagnostics.get("merged_recall_edge_count", 0)) == 0
        ]
        if len(anchors) < 3:
            return None

        primary_embeddings: list[torch.Tensor] = []
        secondary_embeddings: list[torch.Tensor] = []
        primary_original_scores: list[float] = []
        secondary_original_scores: list[float] = []
        for candidate in anchors:
            match = self._verify_speaker_span(
                verifier,
                waveform,
                TimeSpan(candidate.start, candidate.end),
                profile,
                threshold,
            )
            if (
                not match.accepted
                or match.primary.embedding is None
                or match.secondary is None
                or match.secondary.embedding is None
            ):
                continue
            primary_embeddings.append(match.primary.embedding)
            secondary_embeddings.append(match.secondary.embedding)
            primary_original_scores.append(match.primary.score)
            secondary_original_scores.append(match.secondary.score)
        if len(primary_embeddings) < 3:
            return None

        secondary_profile = verifier._ensure_secondary(profile)
        primary_all = torch.cat(
            [profile.primary.embeddings, torch.stack(primary_embeddings)],
            dim=0,
        )
        secondary_all = torch.cat(
            [secondary_profile.embeddings, torch.stack(secondary_embeddings)],
            dim=0,
        )
        primary_centroid = torch.nn.functional.normalize(primary_all.mean(dim=0), dim=0)
        secondary_centroid = torch.nn.functional.normalize(secondary_all.mean(dim=0), dim=0)
        primary_anchor_gains = (
            torch.stack(primary_embeddings) @ primary_centroid
            - torch.tensor(primary_original_scores)
        )
        secondary_anchor_gains = (
            torch.stack(secondary_embeddings) @ secondary_centroid
            - torch.tensor(secondary_original_scores)
        )
        primary_gain_floor = max(
            0.10,
            min(0.14, float(torch.quantile(primary_anchor_gains, 0.10))),
        )
        secondary_gain_floor = max(
            0.08,
            min(0.11, float(torch.quantile(secondary_anchor_gains, 0.10))),
        )
        original_indexes = list(
            profile.primary.reference_indexes
            or tuple(range(len(profile.primary.embeddings)))
        )
        indexes = tuple(
            [*original_indexes, *(10_000 + index for index in range(len(primary_embeddings)))]
        )
        adaptive_primary = SpeakerProfile(
            embeddings=primary_all,
            centroid=primary_centroid,
            reference_scores=[float(value) for value in primary_all @ primary_centroid],
            suggested_threshold=profile.primary.suggested_threshold,
            reference_floor=profile.primary.reference_floor,
            calibration_base=profile.primary.calibration_base,
            reference_indexes=indexes,
        )
        adaptive_secondary = CAMPlusProfile(
            embeddings=secondary_all,
            centroid=secondary_centroid,
            reference_scores=[float(value) for value in secondary_all @ secondary_centroid],
            reference_indexes=indexes,
        )
        return (
            SpeakerMatchProfile(
                primary=adaptive_primary,
                reference_paths=profile.reference_paths,
                base_threshold=profile.base_threshold,
            ),
            adaptive_secondary,
            round(primary_gain_floor, 5),
            round(secondary_gain_floor, 5),
        )

    @staticmethod
    def _needs_boundary_recovery(match: SpeakerMatchDecision, duration: float) -> bool:
        """Select near-target turns and audit every long accepted turn."""

        secondary = match.secondary
        if secondary is None or duration < 2.50:
            return False
        if not match.accepted:
            return (
                duration >= 3.0
                and match.primary.score >= 0.50
                and secondary.score >= 0.50
                and match.primary.reference_median_score >= 0.40
                and secondary.reference_median_score >= 0.40
            )
        # A whole-turn embedding can hide a second speaker in the middle. The
        # local pass is mandatory for longer accepted turns.
        return True

    def _recover_target_segments(
        self,
        span: TimeSpan,
        boundaries: list[SpeakerBoundary],
        verifier: DualSpeakerVerifier,
        waveform: torch.Tensor,
        profile: SpeakerMatchProfile,
        exclusion_profiles: list[ExclusionSpeakerProfile],
        threshold: float,
        progress: ProgressCallback,
        recovery_index: int,
        recovery_count: int,
    ) -> tuple[list[tuple[CandidateSentence, SpeakerMatchDecision]], list[CandidateSentence]] | None:
        """Split a suspicious turn at every confirmed boundary and rescore it.

        The old recovery pass only kept a target suffix. That leaves an already
        accepted prefix (or an additional speaker later in the turn) in the
        exported file. Every resulting continuous segment is now verified on
        its own; unverified pieces are rejected instead of being joined back.
        """

        minimum = self.options.min_sentence_seconds
        candidates = sorted(
            {
                round(boundary.time, 3)
                for boundary in boundaries
                if boundary.confidence >= 0.90
                and span.start + minimum <= boundary.time <= span.end - minimum
            }
        )
        if not candidates:
            return None

        cuts = [span.start, *candidates, span.end]
        parts = [TimeSpan(start, end) for start, end in zip(cuts, cuts[1:])]
        accepted: list[tuple[CandidateSentence, SpeakerMatchDecision]] = []
        edge_supported: list[tuple[CandidateSentence, SpeakerMatchDecision]] = []
        rejected: list[CandidateSentence] = []
        for part_index, part in enumerate(parts, start=1):
            match: SpeakerMatchDecision | None = None
            candidate = CandidateSentence(part.start, part.end, "")
            candidate.speaker_threshold = threshold
            candidate.diagnostics.update(
                {
                    "local_boundary_recovery": True,
                    "original_turn_start": span.start,
                    "original_turn_end": span.end,
                    "recovery_part_index": part_index,
                    "recovery_part_count": len(parts),
                    "recovery_boundaries": candidates,
                    "recovery_boundary_details": [
                        boundary.to_dict() for boundary in boundaries
                    ],
                }
            )
            if part.duration < minimum:
                # A boundary can isolate a low-energy target phoneme.  Score
                # it as non-exportable edge evidence so the later merge pass
                # can reunite it with an adjacent complete target turn.
                match = self._verify_short_edge_candidate(
                    candidate,
                    verifier,
                    waveform,
                    profile,
                    exclusion_profiles,
                    threshold,
                )
                if match is None:
                    candidate.reject_reason = "说话回合过短"
            else:
                match = self._verify_speaker_span(
                    verifier, waveform, part, profile, threshold
                )
                original_match_tier = match.tier
                self._apply_speaker_match(candidate, match, profile, threshold)
                exclusion = verifier.exclusion_audit(
                    match,
                    profile,
                    exclusion_profiles,
                )
                if exclusion is not None:
                    candidate.diagnostics.update(exclusion)
                if exclusion and exclusion.get("excluded_role_rejected"):
                    candidate.reject_reason = (
                        f"更接近{exclusion['excluded_role']}，已按排除角色删除"
                    )
                if not match.accepted:
                    rescued_match = None
                    if not candidate.reject_reason and (
                        match.tier == "recall" or exclusion_profiles
                    ):
                        rescued_match = verifier.promote_local_with_tertiary(
                            self._waveform_span(waveform, part),
                            profile,
                            match,
                            part.duration,
                        )
                    if rescued_match is not None:
                        match = rescued_match
                        self._apply_speaker_match(
                            candidate,
                            match,
                            profile,
                            threshold,
                        )
                        candidate.diagnostics["local_edge_only"] = (
                            original_match_tier == "rejected"
                        )
                    elif self._target_side_recovery_support(
                        match,
                        part.duration,
                        threshold,
                    ):
                        # Keep a recall-quality side available for one joined
                        # sentence, but never expose it as an independent clip.
                        candidate.diagnostics["target_side_recovery_evidence"] = True
                        candidate.reject_reason = "声纹匹配不足"
                    elif not candidate.reject_reason:
                        candidate.reject_reason = "声纹匹配不足"
            progress(
                0.72
                + 0.055
                * (
                    (recovery_index - 1) + part_index / max(1, len(parts))
                )
                / max(1, recovery_count),
                f"局部换人复核 {recovery_index}/{recovery_count}，"
                f"片段 {part_index}/{len(parts)}",
            )
            if candidate.reject_reason:
                if (
                    candidate.diagnostics.get("target_side_recovery_evidence")
                    and match is not None
                ):
                    edge_supported.append((candidate, match))
                else:
                    rejected.append(candidate)
            else:
                assert match is not None
                accepted.append((candidate, match))

        # Rejoin consecutive locally confirmed target pieces.  They were split
        # only to audit the change points; exporting each phonetic fragment
        # separately would recreate the incomplete-sentence problem.
        merged_accepted: list[tuple[CandidateSentence, SpeakerMatchDecision]] = []
        merge_pool = sorted(
            [*accepted, *edge_supported],
            key=lambda item: (item[0].start, item[0].end),
        )
        accepted_ids = {id(candidate) for candidate, _match in accepted}
        cursor = 0
        while cursor < len(merge_pool):
            group = [merge_pool[cursor]]
            cursor += 1
            while (
                cursor < len(merge_pool)
                and merge_pool[cursor][0].start - group[-1][0].end <= 0.02
            ):
                group.append(merge_pool[cursor])
                cursor += 1
            strict_group = [item for item in group if id(item[0]) in accepted_ids]
            if not strict_group:
                rejected.extend(item[0] for item in group)
                continue
            if len(group) == 1:
                merged_accepted.extend(strict_group)
                rejected.extend(
                    item[0] for item in group if id(item[0]) not in accepted_ids
                )
                continue
            joined_span = TimeSpan(group[0][0].start, group[-1][0].end)
            joined_match = self._verify_speaker_span(
                verifier,
                waveform,
                joined_span,
                profile,
                threshold,
            )
            joined_exclusion = verifier.exclusion_audit(
                joined_match,
                profile,
                exclusion_profiles,
            )
            if (
                not joined_match.accepted
                and not (
                    joined_exclusion
                    and joined_exclusion.get("excluded_role_rejected")
                )
            ):
                joined_match = (
                    verifier.promote_local_with_tertiary(
                        self._waveform_span(waveform, joined_span),
                        profile,
                        joined_match,
                        joined_span.duration,
                    )
                    or joined_match
                )
            if joined_match.accepted and not (
                joined_exclusion
                and joined_exclusion.get("excluded_role_rejected")
            ):
                joined = CandidateSentence(joined_span.start, joined_span.end, "")
                self._apply_speaker_match(joined, joined_match, profile, threshold)
                joined.diagnostics.update(
                    {
                        "local_boundary_recovery": True,
                        "original_turn_start": span.start,
                        "original_turn_end": span.end,
                        "local_target_parts_merged": len(group),
                        "local_target_side_evidence_merged": sum(
                            id(item[0]) not in accepted_ids for item in group
                        ),
                        "recovery_boundaries": candidates,
                        "recovery_boundary_details": [
                            boundary.to_dict() for boundary in boundaries
                        ],
                    }
                )
                if joined_exclusion is not None:
                    joined.diagnostics.update(joined_exclusion)
                merged_accepted.append((joined, joined_match))
            else:
                merged_accepted.extend(strict_group)
                rejected.extend(
                    item[0] for item in group if id(item[0]) not in accepted_ids
                )
        accepted = merged_accepted

        if not accepted:
            # A confirmed boundary with no independently matching piece is
            # safer to discard as a mixed turn than to export the original span.
            discarded = CandidateSentence(span.start, span.end, "")
            discarded.reject_reason = "疑似混合说话人，保守舍弃"
            discarded.diagnostics.update(
                {
                    "local_boundary_recovery": True,
                    "recovery_boundaries": candidates,
                    "recovery_boundary_details": [
                        boundary.to_dict() for boundary in boundaries
                    ],
                }
            )
            return [], [discarded]
        return accepted, rejected

    @staticmethod
    def _promote_confirmed_recall_candidates(
        scored_turns: list[
            tuple[TimeSpan, CandidateSentence, SpeakerMatchDecision | None]
        ],
        accepted_turns: list[CandidateSentence],
        rejected: list[CandidateSentence],
    ) -> int:
        """Promote only recall turns independently confirmed window-by-window.

        This is intentionally separate from adaptive/local-seed recall.  It
        handles target utterances elsewhere in the episode, but requires the
        target locator to cover almost the entire turn and both independent
        speaker models to support every local window.  A similar-sounding
        whole-turn embedding alone can never pass this gate.
        """

        match_by_id = {
            id(candidate): match
            for _span, candidate, match in scored_turns
            if match is not None
        }
        retained: list[CandidateSentence] = []
        promoted = 0
        for candidate in rejected:
            if candidate.reject_reason != "声纹匹配不足":
                retained.append(candidate)
                continue
            match = match_by_id.get(id(candidate))
            secondary = match.secondary if match is not None else None
            if match is None or secondary is None or match.tier != "recall":
                retained.append(candidate)
                continue

            coverage = float(candidate.diagnostics.get("target_coverage", 0.0))
            independent_window_consensus = (
                candidate.window_vote_ratio >= 0.75
                and secondary.window_vote_ratio >= 0.75
                and candidate.window_p20_score >= 0.50
                and secondary.window_p20_score >= 0.46
            )
            direct_reference_evidence = (
                match.primary.score >= 0.58
                and secondary.score >= 0.50
                and match.primary.reference_median_score >= 0.50
                and secondary.reference_median_score >= 0.44
                and match.paired_reference_median >= 0.49
            )
            if (
                coverage < 0.90
                or not independent_window_consensus
                or not direct_reference_evidence
            ):
                retained.append(candidate)
                continue

            candidate.reject_reason = ""
            candidate.diagnostics.update(
                {
                    "confirmed_recall": True,
                    "confirmed_recall_coverage_floor": 0.90,
                    "confirmed_recall_window_vote_floor": 0.75,
                }
            )
            accepted_turns.append(candidate)
            promoted += 1

        rejected[:] = retained
        return promoted

    @staticmethod
    def _promote_global_target_cluster(
        scored_turns: list[
            tuple[TimeSpan, CandidateSentence, SpeakerMatchDecision | None]
        ],
        accepted_turns: list[CandidateSentence],
        rejected: list[CandidateSentence],
        extra_seed_items: Iterable[tuple[CandidateSentence, SpeakerMatchDecision]] = (),
    ) -> int:
        """Recover target turns belonging to the episode-wide target cluster.

        Reference recordings can differ from an episode in channel, emotion,
        and vocal effort.  Strong turns already found inside the episode are a
        cleaner domain-matched representation.  Both embedding spaces must
        independently agree with that cluster and the target-window locator
        must cover nearly the full turn.
        """

        accepted_ids = {id(candidate) for candidate in accepted_turns}
        match_by_id = {
            id(candidate): match
            for _span, candidate, match in scored_turns
            if match is not None
        }
        seed_items = [
            (candidate, match)
            for _span, candidate, match in scored_turns
            if id(candidate) in accepted_ids
            and match is not None
            and match.accepted
            and match.secondary is not None
            and match.primary.embedding is not None
            and match.secondary.embedding is not None
        ]
        seed_items.extend(
            (candidate, match)
            for candidate, match in extra_seed_items
            if match.accepted
            and match.secondary is not None
            and match.primary.embedding is not None
            and match.secondary.embedding is not None
        )
        if len(seed_items) < 3:
            return 0

        primary_centroid = torch.nn.functional.normalize(
            torch.stack([match.primary.embedding for _candidate, match in seed_items]).mean(dim=0),
            dim=0,
        )
        secondary_centroid = torch.nn.functional.normalize(
            torch.stack([match.secondary.embedding for _candidate, match in seed_items]).mean(dim=0),
            dim=0,
        )

        retained: list[CandidateSentence] = []
        promoted = 0
        for candidate in rejected:
            if candidate.reject_reason != "声纹匹配不足":
                retained.append(candidate)
                continue
            match = match_by_id.get(id(candidate))
            secondary = match.secondary if match is not None else None
            if (
                match is None
                or secondary is None
                or match.tier != "recall"
                or match.primary.embedding is None
                or secondary.embedding is None
            ):
                retained.append(candidate)
                continue

            coverage = float(candidate.diagnostics.get("target_coverage", 0.0))
            primary_cluster_score = float(match.primary.embedding @ primary_centroid)
            secondary_cluster_score = float(secondary.embedding @ secondary_centroid)
            cluster_consensus = (
                primary_cluster_score >= 0.78
                and secondary_cluster_score >= 0.72
            )
            direct_evidence = (
                match.primary.score >= 0.58
                and secondary.score >= 0.47
                and match.paired_reference_median >= 0.47
                and match.primary.reference_max_score >= 0.54
                and secondary.reference_max_score >= 0.44
            )
            if coverage < 0.85 or not cluster_consensus or not direct_evidence:
                retained.append(candidate)
                continue

            candidate.reject_reason = ""
            candidate.diagnostics.update(
                {
                    "global_target_cluster": True,
                    "global_target_primary_score": round(primary_cluster_score, 5),
                    "global_target_secondary_score": round(secondary_cluster_score, 5),
                }
            )
            accepted_turns.append(candidate)
            promoted += 1

        rejected[:] = retained
        return promoted

    def _build_contrastive_edge_candidates(
        self,
        scored_turns: list[
            tuple[TimeSpan, CandidateSentence, SpeakerMatchDecision | None]
        ],
        accepted_turns: list[CandidateSentence],
        verifier: DualSpeakerVerifier,
        profile: SpeakerMatchProfile,
        waveform: torch.Tensor,
        target_spans: list[TimeSpan],
        effective_threshold: float,
    ) -> list[tuple[TimeSpan, CandidateSentence, SpeakerMatchDecision]]:
        """Recover locator cores with a short, decisively non-target edge.

        The normal boundary scanner needs enough audio on both sides of a cut.
        A rapid reply shorter than that context can therefore remain attached
        to an otherwise complete target utterance.  This path trusts neither
        locator coverage nor a single verifier: the core and residual must be
        separated by one locator edge and show a strong three-model contrast.
        WeSpeaker performs a fourth independent check in the later consensus
        pass before any candidate can be exported.
        """

        if not target_spans:
            return []
        occupied = [TimeSpan(item.start, item.end) for item in accepted_turns]
        output: list[tuple[TimeSpan, CandidateSentence, SpeakerMatchDecision]] = []
        seen: set[tuple[float, float]] = set()
        for source_span, source, _source_match in scored_turns:
            source_coverage = float(
                source.diagnostics.get(
                    "target_coverage",
                    self._target_coverage(source_span, target_spans),
                )
            )
            if (
                source.reject_reason != "声纹匹配不足"
                or not 0.65 <= source_coverage <= 0.92
                or source_span.duration > min(12.0, self.options.max_sentence_seconds)
            ):
                continue

            clipped = sorted(
                (
                    TimeSpan(
                        max(source_span.start, target.start),
                        min(source_span.end, target.end),
                    )
                    for target in target_spans
                    if min(source_span.end, target.end)
                    - max(source_span.start, target.start)
                    >= 0.10
                ),
                key=lambda item: (item.start, item.end),
            )
            merged: list[TimeSpan] = []
            for item in clipped:
                if merged and item.start <= merged[-1].end + 0.05:
                    merged[-1] = TimeSpan(
                        merged[-1].start,
                        max(merged[-1].end, item.end),
                    )
                else:
                    merged.append(item)
            if len(merged) != 1:
                continue

            located = merged[0]
            shares_start = abs(located.start - source_span.start) <= 0.05
            shares_end = abs(located.end - source_span.end) <= 0.05
            if shares_start == shares_end:
                continue
            if shares_start:
                core_span = TimeSpan(source_span.start, located.end)
                residual_span = TimeSpan(located.end, source_span.end)
                edge = "tail"
            else:
                core_span = TimeSpan(located.start, source_span.end)
                residual_span = TimeSpan(source_span.start, located.start)
                edge = "head"
            if (
                core_span.duration < max(1.20, self.options.min_output_seconds)
                or not 0.25 <= residual_span.duration <= 0.90
                or self._target_coverage(core_span, target_spans) < 0.98
                or self._target_coverage(
                    core_span, verifier.tertiary_target_spans
                )
                < 0.35
                or any(
                    min(core_span.end, existing.end)
                    - max(core_span.start, existing.start)
                    > 0.10
                    for existing in occupied
                )
            ):
                continue
            key = (round(core_span.start, 5), round(core_span.end, 5))
            if key in seen:
                continue

            core_match = self._verify_speaker_span(
                verifier,
                waveform,
                core_span,
                profile,
                effective_threshold,
            )
            residual_match = self._verify_speaker_span(
                verifier,
                waveform,
                residual_span,
                profile,
                effective_threshold,
            )
            promoted_match = verifier.promote_contrastive_edge_with_tertiary(
                self._waveform_span(waveform, core_span),
                self._waveform_span(waveform, residual_span),
                profile,
                core_match,
                residual_match,
                core_span.duration,
                residual_span.duration,
            )
            if promoted_match is None:
                continue

            candidate = CandidateSentence(core_span.start, core_span.end, "")
            self._apply_speaker_match(
                candidate,
                promoted_match,
                profile,
                effective_threshold,
            )
            candidate.reject_reason = "声纹匹配不足"
            candidate.diagnostics.update(
                {
                    "contrastive_edge_trim": True,
                    "contrastive_edge": edge,
                    "original_turn_start": source_span.start,
                    "original_turn_end": source_span.end,
                    "contrastive_residual_start": residual_span.start,
                    "contrastive_residual_end": residual_span.end,
                    "target_coverage": round(
                        self._target_coverage(core_span, target_spans), 5
                    ),
                    "tertiary_target_coverage": round(
                        self._target_coverage(
                            core_span, verifier.tertiary_target_spans
                        ),
                        5,
                    ),
                }
            )
            for name in ("speaker_turn_index", "speech_block_index"):
                if name in source.diagnostics:
                    candidate.diagnostics[name] = source.diagnostics[name]
            output.append((core_span, candidate, promoted_match))
            seen.add(key)
        return output

    def _promote_multimodel_target_subclusters(
        self,
        scored_turns: list[
            tuple[TimeSpan, CandidateSentence, SpeakerMatchDecision | None]
        ],
        accepted_turns: list[CandidateSentence],
        rejected: list[CandidateSentence],
        verifier: DualSpeakerVerifier,
        profile: SpeakerMatchProfile,
        audio_path: Path,
        waveform: torch.Tensor,
        target_spans: list[TimeSpan],
        exclusion_profiles: list[ExclusionSpeakerProfile],
        progress: ProgressCallback,
    ) -> int:
        """Recover clean speaker-only rejects with deterministic model votes.

        Accepted episode turns remain separate target prototypes instead of
        being averaged into one centroid. Both base models need support from
        two prototypes, and all three models must agree on at least one of the
        same prototypes. Scores from unrelated models are never averaged.
        """

        raw_anchors = sorted(
            {
                (round(item.start, 5), round(item.end, 5)): item
                for item in accepted_turns
                if item.duration >= max(1.80, self.options.min_output_seconds)
                and not item.diagnostics.get("local_edge_only")
            }.values(),
            key=lambda item: (item.start, item.end),
        )
        matches = {
            id(candidate): match
            for _span, candidate, match in scored_turns
            if match is not None
            and match.secondary is not None
            and match.primary.embedding is not None
            and match.secondary.embedding is not None
        }
        # A mixed original turn can fail whole-clip verification even when
        # one side is a clean target sentence.  Feed stable recovery pieces
        # into the same anchor-consensus path instead of lowering the global
        # speaker threshold for the mixed turn.
        recovery_pieces: list[CandidateSentence] = []
        recovery_sources: dict[
            tuple[float, float], tuple[CandidateSentence, list[float]]
        ] = {}
        low_coverage_parent_ids: set[int] = set()
        for parent in list(rejected):
            if parent.reject_reason not in {
                "声纹匹配不足",
                "疑似混合说话人，目标声纹不连续",
                "疑似混合说话人，保守舍弃",
            }:
                continue
            coverage = float(parent.diagnostics.get("target_coverage", 1.0) or 0.0)
            is_split_piece = bool(parent.diagnostics.get("recovery_subsegment"))
            stable_times = self._candidate_boundary_times(parent)
            continuity_ratio = float(
                parent.diagnostics.get("multi_model_continuity_ratio", 1.0) or 1.0
            )
            mixed_evidence = bool(
                stable_times
                or int(
                    parent.diagnostics.get(
                        "multi_model_internal_boundary_count", 0
                    )
                    or 0
                )
                > 0
                or continuity_ratio < 0.70
            )
            if coverage < 0.85 and mixed_evidence and not is_split_piece:
                low_coverage_parent_ids.add(id(parent))
            source_start = float(
                parent.diagnostics.get("original_turn_start", parent.start)
            )
            source_end = float(
                parent.diagnostics.get("original_turn_end", parent.end)
            )
            if stable_times:
                recovery_sources[(round(source_start, 5), round(source_end, 5))] = (
                    parent,
                    stable_times,
                )
        for parent, stable_times in recovery_sources.values():
            source_start = float(parent.diagnostics.get("original_turn_start", parent.start))
            source_end = float(parent.diagnostics.get("original_turn_end", parent.end))
            points = [source_start, *stable_times, source_end]
            for part_index, (start, end) in enumerate(zip(points, points[1:]), start=1):
                part = TimeSpan(start, end)
                if part.duration < max(1.80, self.options.min_sentence_seconds):
                    continue
                if any(
                    abs(existing.start - part.start) <= 0.02
                    and abs(existing.end - part.end) <= 0.02
                    for existing in [*accepted_turns, *rejected]
                ):
                    continue
                piece = CandidateSentence(part.start, part.end, "")
                piece.diagnostics.update(parent.diagnostics)
                piece.diagnostics.update(
                    {
                        "recovery_subsegment": True,
                        "recovery_parent_start": source_start,
                        "recovery_parent_end": source_end,
                        "recovery_subsegment_index": part_index,
                        "recovery_subsegment_count": len(points) - 1,
                        "recovery_parent_target_coverage": round(
                            float(parent.diagnostics.get("target_coverage", 0.0) or 0.0),
                            5,
                        ),
                        "target_coverage": round(
                            self._target_coverage(part, target_spans), 5
                        ),
                    }
                )
                match = self._verify_speaker_span(
                    verifier,
                    waveform,
                    part,
                    profile,
                    profile.primary.suggested_threshold,
                )
                self._apply_speaker_match(
                    piece,
                    match,
                    profile,
                    profile.primary.suggested_threshold,
                )
                piece.reject_reason = "声纹匹配不足"
                recovery_pieces.append(piece)
                matches[id(piece)] = match
        if recovery_pieces:
            rejected.extend(recovery_pieces)
        boundary_clean_residuals = [
            candidate
            for candidate in rejected
            if candidate.reject_reason == "疑似混合说话人，目标声纹不连续"
            and max(1.80, self.options.min_sentence_seconds)
            <= candidate.duration
            <= min(15.0, self.options.max_sentence_seconds)
            and not any(
                candidate.start + 0.05
                < float(boundary)
                < candidate.end - 0.05
                for boundary in candidate.diagnostics.get(
                    "recovery_boundaries", []
                )
            )
        ]
        for candidate in boundary_clean_residuals:
            if id(candidate) not in matches:
                matches[id(candidate)] = self._verify_speaker_span(
                    verifier,
                    waveform,
                    TimeSpan(candidate.start, candidate.end),
                    profile,
                    profile.primary.suggested_threshold,
                )
        eligible = [
            candidate
            for candidate in rejected
            if candidate.reject_reason
            in {"声纹匹配不足", "疑似混合说话人，目标声纹不连续"}
            and max(1.80, self.options.min_sentence_seconds)
            <= candidate.duration
            <= min(15.0, self.options.max_sentence_seconds)
            and id(candidate) in matches
            and id(candidate) not in low_coverage_parent_ids
            and (
                not candidate.diagnostics.get("recovery_subsegment")
                or float(candidate.diagnostics.get("target_coverage", 0.0) or 0.0)
                >= 0.60
            )
        ]
        if not raw_anchors:
            return 0

        assert verifier.secondary is not None
        purity_splitter = LocalSpeakerTurnSplitter(
            verifier.primary,
            secondary=verifier.secondary,
        )
        anchor_boundaries = purity_splitter.detect_speaker_boundaries(
            audio_path,
            [TimeSpan(item.start, item.end) for item in raw_anchors],
            context_seconds=0.90,
            scan_hop_seconds=0.10,
            primary_candidate_threshold=0.78,
            minimum_similarity_drop=0.06,
            minimum_separation_seconds=0.30,
            progress=lambda value, message: progress(
                0.795,
                f"目标锚点换人复核：{message}",
            ),
        )
        anchors: list[CandidateSentence] = []
        anchor_matches: list[SpeakerMatchDecision] = []
        for anchor in raw_anchors:
            anchor_internal_boundaries = [
                boundary
                for boundary in anchor_boundaries
                if anchor.start < boundary.time < anchor.end
            ]
            anchor.diagnostics["multi_model_anchor_boundary_count"] = len(
                anchor_internal_boundaries
            )
            anchor.diagnostics["multi_model_anchor_boundaries"] = [
                boundary.to_dict() for boundary in anchor_internal_boundaries
            ]
            decisive_boundary = any(
                boundary.primary_similarity <= 0.15
                and boundary.secondary_similarity is not None
                and boundary.secondary_similarity <= 0.05
                for boundary in anchor_internal_boundaries
            )
            anchor.diagnostics["multi_model_anchor_decisive_boundary"] = (
                decisive_boundary
            )
            if decisive_boundary:
                anchor.reject_reason = "多模型复核确认内部换人"
                anchor.diagnostics["structural_hard_reject"] = True
                if anchor in accepted_turns:
                    accepted_turns.remove(anchor)
                if anchor not in rejected:
                    rejected.append(anchor)
                continue
            anchor_match = self._verify_speaker_span(
                verifier,
                waveform,
                TimeSpan(anchor.start, anchor.end),
                profile,
                profile.primary.suggested_threshold,
            )
            if (
                not anchor_match.accepted
                or anchor_match.secondary is None
                or anchor_match.primary.embedding is None
                or anchor_match.secondary.embedding is None
            ):
                continue
            anchor_exclusion = verifier.exclusion_audit(
                anchor_match,
                profile,
                exclusion_profiles,
            )
            if anchor_exclusion and anchor_exclusion.get("excluded_role_rejected"):
                continue
            anchors.append(anchor)
            anchor_matches.append(anchor_match)
        if len(anchors) < 2:
            return 0

        anchor_waveforms = [
            self._waveform_span(waveform, TimeSpan(item.start, item.end))
            for item in anchors
        ]
        primary_anchors = torch.stack(
            [match.primary.embedding for match in anchor_matches]
        )
        secondary_anchors = torch.stack(
            [match.secondary.embedding for match in anchor_matches]
        )
        fourth, fourth_reference = verifier.quaternary_pair(profile)
        fourth_waveforms = [
            *anchor_waveforms,
            *[
                self._waveform_span(
                    waveform,
                    TimeSpan(candidate.start, candidate.end),
                )
                for candidate in eligible
            ],
            *[
                self._waveform_span(
                    waveform,
                    TimeSpan(
                        float(candidate.diagnostics["contrastive_residual_start"]),
                        float(candidate.diagnostics["contrastive_residual_end"]),
                    ),
                )
                for candidate in eligible
                if candidate.diagnostics.get("contrastive_edge_trim")
            ],
        ]
        progress(0.795, "多模型声纹裁决：正在准备 WeSpeaker 批量复核")
        fourth_embeddings = fourth.embeddings_from_waveforms(
            fourth_waveforms,
            progress=lambda completed, total: progress(
                0.795,
                f"多模型声纹裁决：WeSpeaker {completed}/{total}",
            ),
        )
        unfiltered_anchor_count = len(anchors)
        fourth_anchors = fourth_embeddings[:unfiltered_anchor_count]
        candidate_end = unfiltered_anchor_count + len(eligible)
        fourth_candidates = fourth_embeddings[
            unfiltered_anchor_count:candidate_end
        ]
        contrastive_candidates = [
            candidate
            for candidate in eligible
            if candidate.diagnostics.get("contrastive_edge_trim")
        ]
        fourth_residuals = {
            id(candidate): embedding
            for candidate, embedding in zip(
                contrastive_candidates,
                fourth_embeddings[candidate_end:],
            )
        }
        fourth_anchor_reference = fourth_anchors @ fourth_reference.embeddings.T
        fourth_anchor_keep = fourth_anchor_reference.max(dim=1).values >= 0.30
        for anchor, score, keep in zip(
            anchors,
            fourth_anchor_reference.max(dim=1).values,
            fourth_anchor_keep,
        ):
            anchor.diagnostics["wespeaker_anchor_reference_max"] = round(
                float(score), 5
            )
            anchor.diagnostics["wespeaker_anchor_reference_valid"] = bool(keep)
        keep_indexes = fourth_anchor_keep.nonzero().flatten()
        if keep_indexes.numel() < 2:
            return 0
        anchors = [anchors[int(index)] for index in keep_indexes]
        primary_anchors = primary_anchors[keep_indexes]
        secondary_anchors = secondary_anchors[keep_indexes]
        fourth_anchors = fourth_anchors[keep_indexes]

        def continuity_parts(candidate: CandidateSentence) -> list[torch.Tensor]:
            value = self._waveform_span(
                waveform,
                TimeSpan(candidate.start, candidate.end),
            )
            size = round(1.20 * 16000)
            hop = round(0.60 * 16000)
            if value.numel() <= size:
                return [value]
            starts = list(range(0, value.numel() - size + 1, hop))
            last = value.numel() - size
            if starts[-1] != last:
                starts.append(last)
            return [value[start : start + size] for start in starts]

        anchor_window_waveforms: list[torch.Tensor] = []
        anchor_window_ranges: list[tuple[int, int]] = []
        for anchor in anchors:
            start_index = len(anchor_window_waveforms)
            anchor_window_waveforms.extend(continuity_parts(anchor))
            anchor_window_ranges.append(
                (start_index, len(anchor_window_waveforms))
            )
        anchor_primary_windows = verifier.primary._embeddings_from_waveforms(
            anchor_window_waveforms
        )
        anchor_secondary_windows = verifier.secondary._embeddings_from_waveforms(
            anchor_window_waveforms
        )
        anchor_fourth_windows = fourth.embeddings_from_waveforms(
            anchor_window_waveforms
        )
        continuity_anchor_indexes = list(range(len(anchors)))
        while len(continuity_anchor_indexes) >= 2:
            removed_indexes: list[int] = []
            for anchor_index in continuity_anchor_indexes:
                anchor = anchors[anchor_index]
                start_index, end_index = anchor_window_ranges[anchor_index]
                comparison_indexes = [
                    index
                    for index in continuity_anchor_indexes
                    if index != anchor_index
                ]
                common_windows = (
                    (
                        anchor_primary_windows[start_index:end_index]
                        @ primary_anchors[comparison_indexes].T
                        >= 0.38
                    )
                    & (
                        anchor_secondary_windows[start_index:end_index]
                        @ secondary_anchors[comparison_indexes].T
                        >= 0.30
                    )
                    & (
                        anchor_fourth_windows[start_index:end_index]
                        @ fourth_anchors[comparison_indexes].T
                        >= 0.40
                    )
                ).any(dim=1)
                continuity_ratio = float(common_windows.float().mean())
                anchor.diagnostics["multi_model_anchor_continuity"] = round(
                    continuity_ratio, 5
                )
                decisive_boundary = any(
                    int(boundary.get("scale_votes", 1)) >= 2
                    or (
                        float(boundary.get("primary_similarity", 1.0)) <= 0.15
                        and boundary.get("secondary_similarity") is not None
                        and float(boundary["secondary_similarity"]) <= 0.05
                    )
                    for boundary in anchor.diagnostics.get(
                        "multi_model_anchor_boundaries", []
                    )
                )
                anchor.diagnostics["multi_model_anchor_decisive_boundary"] = (
                    decisive_boundary
                )
                if continuity_ratio < 0.70 or decisive_boundary:
                    removed_indexes.append(anchor_index)
            if not removed_indexes:
                break
            for anchor_index in removed_indexes:
                anchor = anchors[anchor_index]
                # A low episode-cluster continuity score makes this a poor
                # recovery prototype, but is not sufficient to discard a
                # directly verified target turn. Decisive speaker boundaries
                # were already hard-rejected before profile construction.
                anchor.diagnostics["multi_model_anchor_excluded"] = True
            removed_set = set(removed_indexes)
            continuity_anchor_indexes = [
                index
                for index in continuity_anchor_indexes
                if index not in removed_set
            ]
        if len(continuity_anchor_indexes) < 2:
            return 0
        continuity_anchor_indexes_tensor = torch.tensor(
            continuity_anchor_indexes,
            dtype=torch.long,
        )
        anchors = [anchors[index] for index in continuity_anchor_indexes]
        primary_anchors = primary_anchors[continuity_anchor_indexes_tensor]
        secondary_anchors = secondary_anchors[continuity_anchor_indexes_tensor]
        fourth_anchors = fourth_anchors[continuity_anchor_indexes_tensor]
        if not eligible:
            return 0

        provisional: list[CandidateSentence] = []
        for candidate, fourth_embedding in zip(eligible, fourth_candidates):
            boundary_piece = bool(candidate.diagnostics.get("recovery_subsegment"))
            match = matches[id(candidate)]
            assert match.secondary is not None
            assert match.primary.embedding is not None
            assert match.secondary.embedding is not None
            primary_top = torch.topk(
                match.primary.embedding @ primary_anchors.T,
                k=2,
            ).values
            secondary_top = torch.topk(
                match.secondary.embedding @ secondary_anchors.T,
                k=2,
            ).values
            fourth_top = torch.topk(
                fourth_embedding @ fourth_anchors.T,
                k=2,
            ).values
            primary_second = float(primary_top[1])
            secondary_second = float(secondary_top[1])
            fourth_first = float(fourth_top[0])
            fourth_second = float(fourth_top[1])
            common_anchor_mask = (
                (match.primary.embedding @ primary_anchors.T >= 0.64)
                & (match.secondary.embedding @ secondary_anchors.T >= 0.58)
                & (
                    fourth_embedding @ fourth_anchors.T
                    >= (0.45 if boundary_piece else 0.60)
                )
            )
            common_anchor_count = int(common_anchor_mask.sum())
            fourth_direct = float(
                (fourth_reference.embeddings @ fourth_embedding).max()
            )
            direct_reference_evidence = (
                match.primary.reference_max_score >= 0.42
                or match.secondary.reference_max_score >= 0.42
                or fourth_direct >= 0.42
            )
            base_consensus = (
                primary_second >= 0.64
                and secondary_second >= 0.58
            )
            boundary_consensus = (
                boundary_piece
                and primary_second >= 0.64
                and secondary_second >= 0.58
                and fourth_first >= 0.45
                and common_anchor_count >= 1
            )
            wespeaker_dominant_consensus = (
                float(primary_top[0]) >= 0.66
                and float(secondary_top[0]) >= 0.64
                and fourth_second >= 0.70
                and common_anchor_count >= 1
            )
            fourth_model_rescue = (
                float(primary_top[0]) >= 0.60
                and float(secondary_top[0]) >= 0.56
                and fourth_first >= 0.72
                and fourth_second >= 0.68
            )
            base_consensus = (
                base_consensus
                or boundary_consensus
                or wespeaker_dominant_consensus
                or fourth_model_rescue
            )
            contrastive_edge = bool(
                candidate.diagnostics.get("contrastive_edge_trim")
            )
            fourth_support = fourth_first >= (
                0.60
                if contrastive_edge
                else (0.45 if boundary_piece else 0.64)
            )
            contrastive_fourth_passed = not contrastive_edge
            if contrastive_edge:
                residual_fourth = fourth_residuals[id(candidate)]
                residual_fourth_score = float(
                    residual_fourth @ fourth_reference.centroid
                )
                residual_fourth_direct = float(
                    (fourth_reference.embeddings @ residual_fourth).max()
                )
                fourth_score_margin = float(
                    fourth_embedding @ fourth_reference.centroid
                ) - residual_fourth_score
                fourth_direct_margin = fourth_direct - residual_fourth_direct
                contrastive_fourth_passed = (
                    fourth_direct >= 0.55
                    and residual_fourth_direct <= 0.35
                    and fourth_score_margin >= 0.20
                    and fourth_direct_margin >= 0.20
                )
                candidate.diagnostics.update(
                    {
                        "contrastive_residual_wespeaker_score": round(
                            residual_fourth_score, 5
                        ),
                        "contrastive_residual_wespeaker_reference_max": round(
                            residual_fourth_direct, 5
                        ),
                        "contrastive_wespeaker_score_margin": round(
                            fourth_score_margin, 5
                        ),
                        "contrastive_wespeaker_direct_margin": round(
                            fourth_direct_margin, 5
                        ),
                        "contrastive_edge_fourth_passed": (
                            contrastive_fourth_passed
                        ),
                    }
                )
            candidate.diagnostics.update(
                {
                    "multi_model_consensus_checked": True,
                    "episode_eres_top1": round(float(primary_top[0]), 5),
                    "episode_eres_top2": round(primary_second, 5),
                    "episode_camplus_top1": round(float(secondary_top[0]), 5),
                    "episode_camplus_top2": round(secondary_second, 5),
                    "episode_wespeaker_top1": round(fourth_first, 5),
                    "episode_wespeaker_top2": round(fourth_second, 5),
                    "multi_model_base_consensus": base_consensus,
                    "multi_model_boundary_consensus": boundary_consensus,
                    "multi_model_wespeaker_dominant_consensus": (
                        wespeaker_dominant_consensus
                    ),
                    "multi_model_fourth_model_rescue": fourth_model_rescue,
                    "multi_model_wespeaker_support": fourth_support,
                    "multi_model_common_anchor_count": common_anchor_count,
                    "multi_model_direct_reference": direct_reference_evidence,
                }
            )
            if (
                not base_consensus
                or not fourth_support
                or (common_anchor_count < 1 and not fourth_model_rescue)
                or not direct_reference_evidence
                or not contrastive_fourth_passed
            ):
                continue

            boundary_residual = (
                candidate.reject_reason
                == "疑似混合说话人，目标声纹不连续"
            ) or boundary_piece
            if boundary_residual or contrastive_edge:
                exclusion = verifier.multimodel_exclusion_audit(
                    match,
                    profile,
                    fourth_embedding,
                    fourth_reference,
                    exclusion_profiles,
                )
            else:
                exclusion = verifier.exclusion_audit(
                    match,
                    profile,
                    exclusion_profiles,
                )
            if exclusion is not None:
                candidate.diagnostics.update(exclusion)
            if exclusion and exclusion.get("excluded_role_rejected"):
                candidate.reject_reason = (
                    f"更接近{exclusion['excluded_role']}，已按排除角色删除"
                )
                continue
            if boundary_residual and exclusion:
                exclusion_votes = int(
                    exclusion.get("excluded_multimodel_vote_count", 0) or 0
                )
                boundary_target_consensus = bool(
                    boundary_piece
                    and candidate.diagnostics.get("multi_model_boundary_consensus")
                    and int(candidate.diagnostics.get("multi_model_common_anchor_count", 0))
                    >= 2
                )
                if not exclusion.get("excluded_all_models_clear", False) and not (
                    boundary_target_consensus and exclusion_votes < 2
                ):
                    candidate.reject_reason = "边界残片未通过三模型排除人物净胜复核"
                    candidate.diagnostics["structural_hard_reject"] = True
                    continue
                if boundary_target_consensus and exclusion_votes == 1:
                    candidate.diagnostics["excluded_ambiguous_target_consensus"] = True

            overlapping = [
                existing
                for existing in accepted_turns
                if min(candidate.end, existing.end)
                - max(candidate.start, existing.start)
                > 0.10
            ]
            if overlapping and not all(
                candidate.start <= existing.start + 0.25
                and candidate.end >= existing.end - 0.25
                for existing in overlapping
            ):
                continue
            provisional.append(candidate)

        continuity_ready: list[CandidateSentence] = []
        continuity_waveforms: list[torch.Tensor] = []
        continuity_ranges: list[tuple[int, int]] = []
        for candidate in provisional:
            start_index = len(continuity_waveforms)
            continuity_waveforms.extend(continuity_parts(candidate))
            continuity_ranges.append((start_index, len(continuity_waveforms)))

        if continuity_waveforms:
            progress(0.795, "多模型候选首尾连续性复核")
            primary_windows = verifier.primary._embeddings_from_waveforms(
                continuity_waveforms
            )
            secondary_windows = verifier.secondary._embeddings_from_waveforms(
                continuity_waveforms
            )
            fourth_windows = fourth.embeddings_from_waveforms(
                continuity_waveforms
            )
            for candidate, (start_index, end_index) in zip(
                provisional,
                continuity_ranges,
            ):
                boundary_piece = bool(candidate.diagnostics.get("recovery_subsegment"))
                common_windows = (
                    (
                        primary_windows[start_index:end_index]
                        @ primary_anchors.T
                        >= 0.38
                    )
                    & (
                        secondary_windows[start_index:end_index]
                        @ secondary_anchors.T
                        >= 0.30
                    )
                    & (
                        fourth_windows[start_index:end_index]
                        @ fourth_anchors.T
                        >= (0.32 if boundary_piece else 0.40)
                    )
                ).any(dim=1)
                continuity_ratio = float(common_windows.float().mean())
                candidate.diagnostics.update(
                    {
                        "multi_model_continuity_ratio": round(
                            continuity_ratio, 5
                        ),
                        "multi_model_continuity_windows": len(common_windows),
                        "multi_model_continuity_passed": int(
                            common_windows.sum()
                        ),
                    }
                )
                edge_tolerant = (
                    continuity_ratio >= 0.55
                    and int(
                        candidate.diagnostics.get(
                            "multi_model_common_anchor_count", 0
                        )
                    )
                    >= 1
                    and float(candidate.diagnostics.get("episode_eres_top2", 0.0))
                    >= 0.72
                    and float(
                        candidate.diagnostics.get("episode_camplus_top2", 0.0)
                    )
                    >= 0.60
                    and float(
                        candidate.diagnostics.get("episode_wespeaker_top1", 0.0)
                    )
                    >= 0.65
                )
                fourth_model_edge_tolerant = (
                    continuity_ratio >= 0.55
                    and bool(
                        candidate.diagnostics.get(
                            "multi_model_fourth_model_rescue"
                        )
                    )
                    and float(
                        candidate.diagnostics.get("episode_wespeaker_top2", 0.0)
                    )
                    >= 0.68
                )
                edge_tolerant = edge_tolerant or fourth_model_edge_tolerant
                if boundary_piece:
                    edge_tolerant = edge_tolerant or (
                        continuity_ratio >= 0.55
                        and int(
                            candidate.diagnostics.get(
                                "multi_model_common_anchor_count", 0
                            )
                        )
                        >= 1
                        and float(candidate.diagnostics.get("episode_eres_top2", 0.0))
                        >= 0.64
                        and float(
                            candidate.diagnostics.get("episode_camplus_top2", 0.0)
                        )
                        >= 0.56
                    )
                candidate.diagnostics["multi_model_edge_tolerant"] = edge_tolerant
                if continuity_ratio < 0.70 and not edge_tolerant:
                    candidate.reject_reason = "多模型复核显示目标声纹不连续"
                    candidate.diagnostics["structural_hard_reject"] = True
                    continue
                continuity_ready.append(candidate)

        internal_boundaries: list[SpeakerBoundary] = []
        if continuity_ready:
            internal_boundaries = purity_splitter.detect_multiscale_speaker_boundaries(
                audio_path,
                [TimeSpan(item.start, item.end) for item in continuity_ready],
                scan_hop_seconds=0.10,
                primary_candidate_threshold=0.78,
                minimum_similarity_drop=0.06,
                minimum_separation_seconds=0.30,
                progress=lambda value, message: progress(
                    0.795,
                    f"多模型候选换人细扫：{message}",
                ),
            )

        promoted_ids: set[int] = set()
        promoted = 0
        for candidate in continuity_ready:
            boundaries = [
                boundary
                for boundary in internal_boundaries
                if candidate.start < boundary.time < candidate.end
            ]
            candidate.diagnostics["multi_model_internal_boundary_count"] = len(
                boundaries
            )
            candidate.diagnostics["multi_model_internal_boundaries"] = [
                boundary.to_dict() for boundary in boundaries
            ]
            decisive_boundary = any(
                self._is_structural_boundary(boundary)
                for boundary in boundaries
            )
            candidate.diagnostics["multi_model_decisive_boundary"] = (
                decisive_boundary
            )
            low_coverage_boundary = bool(
                boundaries
                and float(candidate.diagnostics.get("target_coverage", 1.0) or 0.0)
                < 0.85
            )
            if boundaries and (
                low_coverage_boundary
                or not candidate.diagnostics.get("multi_model_edge_tolerant")
                or decisive_boundary
            ):
                candidate.reject_reason = "多模型复核确认内部换人"
                candidate.diagnostics["structural_hard_reject"] = True
                continue
            overlapping = [
                existing
                for existing in accepted_turns
                if min(candidate.end, existing.end)
                - max(candidate.start, existing.start)
                > 0.10
            ]
            for existing in overlapping:
                accepted_turns.remove(existing)
            candidate.reject_reason = ""
            candidate.diagnostics["multi_model_target_recovery"] = True
            accepted_turns.append(candidate)
            promoted_ids.add(id(candidate))
            promoted += 1

        if promoted_ids:
            rejected[:] = [
                candidate
                for candidate in rejected
                if id(candidate) not in promoted_ids
            ]
        return promoted

    @staticmethod
    def _expand_recall_candidates(
        scored_turns: list[
            tuple[TimeSpan, CandidateSentence, SpeakerMatchDecision | None]
        ],
        accepted_turns: list[CandidateSentence],
        rejected: list[CandidateSentence],
        profile: SpeakerMatchProfile,
        extra_seed_items: Iterable[tuple[CandidateSentence, SpeakerMatchDecision]] = (),
    ) -> int:
        """Recover target turns that are acoustically close to confirmed seeds.

        A fixed lower threshold would admit other speakers globally. Instead,
        use accepted turns from the same silence-delimited speech block as
        local seeds and require both verifier embeddings to stay close to that
        block centroid. Only candidates rejected for speaker mismatch are
        eligible; overlap, singing, short-turn and confirmed mixed-speaker
        decisions stay hard rejects.
        """

        accepted_ids = {id(candidate) for candidate in accepted_turns}
        extra_seed_items = list(extra_seed_items)
        seed_blocks = {
            int(candidate.diagnostics["speech_block_index"])
            for candidate in accepted_turns
            if "speech_block_index" in candidate.diagnostics
        }
        seed_items = [
            (candidate, match)
            for _span, candidate, match in scored_turns
            if id(candidate) in accepted_ids
            and match is not None
            and match.secondary is not None
            and match.primary.embedding is not None
            and match.secondary.embedding is not None
        ]
        seed_items.extend(
            (candidate, match)
            for candidate, match in extra_seed_items
            if match.secondary is not None
            and match.primary.embedding is not None
            and match.secondary.embedding is not None
        )
        seed_blocks.update(
            int(candidate.diagnostics["speech_block_index"])
            for candidate, _match in extra_seed_items
            if isinstance(candidate.diagnostics.get("speech_block_index"), int)
        )
        if len(seed_items) < 3:
            return 0

        primary_seeds = torch.stack(
            [match.primary.embedding for _candidate, match in seed_items]
        )
        secondary_seeds = torch.stack(
            [match.secondary.embedding for _candidate, match in seed_items]
        )
        primary_centroid = torch.nn.functional.normalize(primary_seeds.mean(dim=0), dim=0)
        secondary_centroid = torch.nn.functional.normalize(secondary_seeds.mean(dim=0), dim=0)
        primary_seed_scores = primary_seeds @ primary_centroid
        secondary_seed_scores = secondary_seeds @ secondary_centroid

        # The floors follow the observed target-speaker variation, with hard
        # lower bounds so a weak seed set cannot turn into an open gate.
        # Keep a hard floor, but allow substantially more natural variation
        # than the first adaptive pass. The candidate still needs direct
        # evidence from both speaker models below, so this is not a global
        # threshold reduction.
        primary_floor = max(
            0.64,
            float(torch.quantile(primary_seed_scores, 0.10)) - 0.18,
        )
        secondary_floor = max(
            0.58,
            float(torch.quantile(secondary_seed_scores, 0.10)) - 0.18,
        )
        candidate_by_id = {
            id(candidate): match
            for _span, candidate, match in scored_turns
            if match is not None
        }
        retained_rejected: list[CandidateSentence] = []
        promoted = 0
        for candidate in rejected:
            # Keep all non-speaker rejects permanently excluded.
            if candidate.reject_reason != "声纹匹配不足":
                retained_rejected.append(candidate)
                continue
            match = candidate_by_id.get(id(candidate))
            candidate_block = candidate.diagnostics.get("speech_block_index")
            if (
                not isinstance(candidate_block, int)
                or candidate_block not in seed_blocks
            ):
                retained_rejected.append(candidate)
                continue
            # Coverage comes from the independent sliding-window locator.  A
            # low-coverage turn can contain another speaker even when its
            # whole-turn embedding looks close, so keep the recall gate high.
            if float(candidate.diagnostics.get("target_coverage", 0.0)) < 0.85:
                retained_rejected.append(candidate)
                continue
            if (
                match is None
                or match.secondary is None
                or match.primary.embedding is None
                or match.secondary.embedding is None
            ):
                retained_rejected.append(candidate)
                continue
            # A rejected dual-model decision is not eligible for recall. Only
            # the explicit weak tier, which has strong primary/reference
            # evidence, may enter this block-local recovery pass.
            if match.tier not in {"weak", "recall"}:
                retained_rejected.append(candidate)
                continue

            primary_seed_score = float(match.primary.embedding @ primary_centroid)
            secondary_seed_score = float(match.secondary.embedding @ secondary_centroid)
            primary_reference_max = float(
                candidate.diagnostics.get("eres_reference_max", 0.0)
            )
            secondary_reference_max = float(
                candidate.diagnostics.get("camplus_reference_max", 0.0)
            )
            reference_floor = float(profile.primary.reference_floor)
            reference_evidence = (
                primary_reference_max >= max(0.52, reference_floor - 0.11)
                and secondary_reference_max >= max(0.48, reference_floor - 0.13)
            )
            window_evidence = (
                candidate.window_p20_score >= primary_floor - 0.20
                and match.secondary.window_p20_score >= secondary_floor - 0.20
                and candidate.window_vote_ratio >= 0.50
                and match.secondary.window_vote_ratio >= 0.50
            )
            direct_evidence = (
                match.primary.score >= 0.54
                and match.secondary.score >= 0.50
                and match.paired_reference_median
                >= max(0.46, reference_floor - 0.14)
            )
            if (
                primary_seed_score < primary_floor
                or secondary_seed_score < secondary_floor
                or not reference_evidence
                or not window_evidence
                or not direct_evidence
            ):
                retained_rejected.append(candidate)
                continue

            candidate.reject_reason = ""
            candidate.diagnostics.update(
                {
                    "adaptive_recall": True,
                    "adaptive_recall_primary_seed": round(primary_seed_score, 5),
                    "adaptive_recall_secondary_seed": round(secondary_seed_score, 5),
                    "adaptive_recall_primary_floor": round(primary_floor, 5),
                    "adaptive_recall_secondary_floor": round(secondary_floor, 5),
                }
            )
            accepted_turns.append(candidate)
            promoted += 1

        rejected[:] = retained_rejected
        return promoted

    def _promote_anime_identity_candidates(
        self,
        scored_turns: list[
            tuple[TimeSpan, CandidateSentence, SpeakerMatchDecision | None]
        ],
        accepted_turns: list[CandidateSentence],
        rejected: list[CandidateSentence],
        profile: SpeakerMatchProfile,
        waveform: torch.Tensor,
        reference_clips: list[Path],
        negative_reference_clips: list[list[Path]],
        progress: ProgressCallback,
    ) -> int:
        """Recover a narrow class of complete target turns missed by SV models.

        This is an optional independent identity check, not a lower speaker
        threshold. It requires both anime ECAPA variants to beat every supplied
        exclusion role by a margin, while ERes2Net/CAM++ still provide minimum
        acoustic support and the target locator covers the whole turn.
        """

        if not negative_reference_clips:
            return 0
        try:
            from .anime import AnimeIdentityVerifier
        except Exception:
            return 0
        if not AnimeIdentityVerifier.available():
            return 0

        eligible: list[tuple[CandidateSentence, SpeakerMatchDecision]] = []
        match_by_id = {
            id(candidate): match
            for _span, candidate, match in scored_turns
            if match is not None
        }
        for candidate in rejected:
            if candidate.reject_reason != "声纹匹配不足":
                continue
            match = match_by_id.get(id(candidate))
            secondary = match.secondary if match is not None else None
            if match is None or secondary is None:
                continue
            diagnostics = candidate.diagnostics
            # The anime-domain scorer is an auxiliary identity check. It may
            # recover a borderline target turn only after the ordinary
            # episode-level multi-model evidence has established the same
            # target cluster. It must never overturn a base-model rejection
            # when there is no common anchor, no WeSpeaker support, or a
            # supplied exclusion role wins a direct-reference comparison.
            if (
                candidate.duration < max(1.20, self.options.min_output_seconds)
                or candidate.duration > min(15.0, self.options.max_sentence_seconds)
                or float(diagnostics.get("target_coverage", 0.0)) < 0.90
                or match.primary.score < 0.50
                or secondary.score < 0.40
                or match.primary.reference_max_score < 0.48
                or secondary.reference_max_score < 0.38
                or match.paired_reference_median < 0.38
                or not self._anime_recovery_base_gate(diagnostics)
            ):
                continue
            eligible.append((candidate, match))
        if not eligible:
            return 0

        scorer = None
        try:
            progress(0.80, f"动漫声纹独立恢复：准备复核 {len(eligible)} 个完整候选")
            scorer = AnimeIdentityVerifier()
            anime_profile = scorer.build_profile(reference_clips, negative_reference_clips)
            promoted_ids: set[int] = set()
            promoted = 0
            for index, (candidate, _match) in enumerate(eligible, start=1):
                decision = scorer.score(
                    self._waveform_span(waveform, TimeSpan(candidate.start, candidate.end)),
                    anime_profile,
                )
                candidate.diagnostics.update(
                    {
                        "anime_identity_checked": True,
                        "anime_char_target_max": decision.char_target_max,
                        "anime_char_exclusion_max": decision.char_exclusion_max,
                        "anime_char_margin": decision.char_margin,
                        "anime_va_target_max": decision.va_target_max,
                        "anime_va_exclusion_max": decision.va_exclusion_max,
                        "anime_va_margin": decision.va_margin,
                        "anime_identity_margin_floor": scorer.MARGIN_FLOOR,
                    }
                )
                if (
                    decision.char_margin <= scorer.MARGIN_FLOOR
                    or decision.va_margin <= scorer.MARGIN_FLOOR
                ):
                    progress(0.80, f"动漫声纹独立恢复：复核 {index}/{len(eligible)}")
                    continue
                candidate.reject_reason = ""
                candidate.diagnostics["anime_identity_recovery"] = True
                overlapping = [
                    existing
                    for existing in accepted_turns
                    if min(candidate.end, existing.end)
                    - max(candidate.start, existing.start)
                    > 0.10
                ]
                if overlapping and not all(
                    candidate.start <= existing.start + 0.25
                    and candidate.end >= existing.end - 0.25
                    for existing in overlapping
                ):
                    continue
                for existing in overlapping:
                    accepted_turns.remove(existing)
                accepted_turns.append(candidate)
                promoted_ids.add(id(candidate))
                promoted += 1
                progress(0.80, f"动漫声纹独立恢复：复核 {index}/{len(eligible)}，恢复 1 段")
            rejected[:] = [
                candidate for candidate in rejected if id(candidate) not in promoted_ids
            ]
            return promoted
        finally:
            if scorer is not None:
                scorer.close()

    def _write_outputs(
        self,
        accepted: list[CandidateSentence],
        rejected: list[CandidateSentence],
        stem: Path,
        target_name: str,
        paths: dict[str, Path],
        progress: ProgressCallback,
        *,
        create_archive: bool = True,
        source_media: Path | None = None,
    ) -> tuple[Path, Path, Path]:
        output_dir = paths["output"]
        output_dir.mkdir(parents=True, exist_ok=True)
        audio_dir = output_dir / "audio"
        text_dir = output_dir / "text"
        video_dir = output_dir / "video"
        audio_dir.mkdir(exist_ok=True)
        text_dir.mkdir(exist_ok=True)
        video_source = None
        if self.options.export_video_clips and source_media is not None:
            try:
                if has_video_stream(source_media):
                    video_source = source_media
                    video_dir.mkdir(exist_ok=True)
            except (OSError, RuntimeError) as exc:
                LOGGER.warning("视频流检测失败，跳过视频片段：%s", exc)
        for index, candidate in enumerate(accepted, start=1):
            stem_name = f"{index:04d}_{_safe_name(candidate.text or candidate.whisper_text)}"
            audio_path = audio_dir / f"{stem_name}.wav"
            text_path = text_dir / f"{stem_name}.txt"
            write_clip(
                stem,
                audio_path,
                candidate.start,
                candidate.end,
                sample_rate=16000,
                normalize_level=True,
            )
            text_path.write_text(candidate.text or candidate.whisper_text, encoding="utf-8")
            candidate.audio_file = str(audio_path.relative_to(output_dir))
            candidate.text_file = str(text_path.relative_to(output_dir))
            if video_source is not None:
                video_path = video_dir / f"{stem_name}.mp4"
                write_video_clip(video_source, video_path, candidate.start, candidate.end)
                candidate.video_file = str(video_path.relative_to(output_dir))
            progress(0.94 + 0.05 * index / max(1, len(accepted)), f"导出句子 {index}/{len(accepted)}")

        if self.options.keep_rejected and rejected:
            rejected_dir = output_dir / "rejected_audio"
            rejected_dir.mkdir(exist_ok=True)
            for index, candidate in enumerate(rejected, start=1):
                rejected_path = rejected_dir / f"{index:04d}_{_safe_name(candidate.reject_reason, 'rejected')}.wav"
                write_clip(stem, rejected_path, candidate.start, candidate.end, sample_rate=16000)

        accepted_records = [candidate.to_dict() for candidate in accepted]
        rejected_records = [candidate.to_dict() for candidate in rejected]
        records = [*accepted_records, *rejected_records]
        reject_summary: dict[str, int] = {}
        for candidate in rejected:
            reason = candidate.reject_reason or "未知原因"
            reject_summary[reason] = reject_summary.get(reason, 0) + 1
        manifest = output_dir / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "target": target_name,
                    "accepted_count": len(accepted),
                    "rejected_count": len(rejected),
                    "reject_summary": reject_summary,
                    "options": asdict(self.options),
                    "sentences": records,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        csv_path = output_dir / "manifest.csv"
        fields = [
            "audio_file",
            "text_file",
            "video_file",
            "text",
            "language",
            "start",
            "end",
            "speaker_score",
            "window_min_score",
            "window_p20_score",
            "speaker_vote_ratio",
            "window_vote_ratio",
            "speaker_threshold",
            "overlap_score",
            "singing_score",
            "accepted",
            "reject_reason",
        ]
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)

        srt_path = output_dir / "transcript.srt"
        srt_parts: list[str] = []
        for index, candidate in enumerate(accepted, start=1):
            srt_parts.append(
                f"{index}\n{_srt_time(candidate.start)} --> {_srt_time(candidate.end)}\n{candidate.text or candidate.whisper_text}\n"
            )
        srt_path.write_text("\n".join(srt_parts), encoding="utf-8")

        archive_path = OUTPUT_ROOT / f"{output_dir.name}.zip"
        if not create_archive:
            return manifest, srt_path, archive_path
        # Never expose a partially written archive to Explorer or the GUI.
        # Windows may try to open the path as soon as it appears, so build the
        # ZIP beside the final path and publish it only after ZipFile closes.
        partial_archive = archive_path.with_name(archive_path.name + ".part")
        partial_archive.unlink(missing_ok=True)
        with zipfile.ZipFile(partial_archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(output_dir.rglob("*")):
                if file.is_file():
                    archive.write(file, file.relative_to(output_dir))
        partial_archive.replace(archive_path)
        return manifest, srt_path, archive_path

    def run(
        self,
        references: Iterable[str | Path],
        target: str | Path,
        negative_references: Iterable[Iterable[str | Path]] | None = None,
        progress: ProgressCallback | None = None,
        job_id: str | None = None,
        create_archive: bool = True,
    ) -> PipelineResult:
        progress = progress or _noop_progress
        references = [Path(path) for path in references if path]
        negative_reference_groups = [
            [Path(path) for path in group if path]
            for group in (negative_references or [])
        ]
        negative_reference_groups = [
            group for group in negative_reference_groups if group
        ]
        target = Path(target)
        if not references:
            raise ValueError("请至少提供一段参考音频")
        if not target.exists():
            raise FileNotFoundError(target)
        job_id = job_id or time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        paths = self._job_paths(job_id)
        paths["root"].mkdir(parents=True, exist_ok=True)
        singing_detector: SingingDetector | None = None
        try:
            progress(0.01, "检查本地模型")
            normalized_refs, reference_durations = self._prepare_references(references, paths, progress)
            normalized_negative_groups, negative_duration_groups = (
                self._prepare_negative_references(
                    negative_reference_groups,
                    paths,
                    progress,
                )
            )
            target_normalized = normalize_audio(target, paths["normalized_target"], sample_rate=44100, stereo=True)
            target_duration = probe_duration(target_normalized)
            target_for_separator = target_normalized
            pre_singing_spans: list[TimeSpan] = []
            if self.options.use_singing_detector:
                singing_detector = SingingDetector("cpu")
                progress(0.12, "UVR 前检测人类歌声（纯器乐不会删除）")
                _non_singing, pre_singing_spans = singing_detector.clean_spans(
                    target_normalized,
                    [TimeSpan(0.0, target_duration)],
                    self.options.singing_threshold,
                    progress=lambda value, message: progress(0.12 + 0.04 * value, message),
                )
                if pre_singing_spans:
                    target_for_separator = mute_spans(
                        target_normalized,
                        paths["singing_removed_target"],
                        pre_singing_spans,
                    )
                    progress(
                        0.16,
                        f"UVR 前已静音移除 {len(pre_singing_spans)} 个有人演唱区间，时间轴保持不变",
                    )
                else:
                    progress(0.16, "UVR 前未检测到有人演唱；纯音乐保持原样")
            target_duration = pad_for_separator(target_for_separator)

            progress(0.16, "提取参考与目标人声：准备 UVR 分块")
            separator = UVR5Separator(self.device)
            separation_items = [
                (path, paths["reference_stems"] / f"reference_{index:03d}_vocals.wav")
                for index, path in enumerate(normalized_refs, start=1)
            ]
            negative_item_count = 0
            for group_index, group in enumerate(normalized_negative_groups, start=1):
                group_dir = paths["negative_stems"] / f"role_{group_index:03d}"
                for reference_index, path in enumerate(group, start=1):
                    separation_items.append(
                        (
                            path,
                            group_dir / f"reference_{reference_index:03d}_vocals.wav",
                        )
                    )
                    negative_item_count += 1
            separation_items.append((target_for_separator, paths["stems"] / "target_vocals.wav"))
            try:
                separated = separator.separate_many(
                    separation_items,
                    progress=lambda value, message: progress(0.16 + 0.24 * value, message),
                )
            except ValueError as exc:
                if "有效信号" not in str(exc):
                    raise
                # A fully silent target has no sentence by definition. Return
                # the same empty artifact shape as a music-only target.
                output_dir = paths["output"]
                output_dir.mkdir(parents=True, exist_ok=True)
                manifest, transcript, archive = self._write_outputs(
                    [],
                    [],
                    target_normalized,
                    target.name,
                    paths,
                    progress,
                    create_archive=create_archive,
                )
                progress(1.0, "完成：目标音频没有人声")
                return PipelineResult(job_id, output_dir, archive, [], [], manifest, transcript)
            reference_stems = separated[: len(normalized_refs)]
            negative_flat_stems = separated[
                len(normalized_refs) : len(normalized_refs) + negative_item_count
            ]
            stem = separated[-1]
            for reference_stem, original_duration in zip(reference_stems, reference_durations):
                trim_audio_in_place(reference_stem, original_duration)
            negative_stem_groups: list[list[Path]] = []
            negative_cursor = 0
            for durations in negative_duration_groups:
                count = len(durations)
                group_stems = negative_flat_stems[
                    negative_cursor : negative_cursor + count
                ]
                negative_cursor += count
                for reference_stem, original_duration in zip(group_stems, durations):
                    trim_audio_in_place(reference_stem, original_duration)
                negative_stem_groups.append(group_stems)
            trim_audio_in_place(stem, target_duration)

            progress(0.40, "UVR 完成：按每一段静音检测最小讲话片段")
            vad_tools = FunASRTools(self.device)
            vad_map = vad_tools.vad_many(
                [
                    *reference_stems,
                    *(item for group in negative_stem_groups for item in group),
                    stem,
                ],
                progress=lambda value, message: progress(0.40 + 0.08 * value, message),
            )
            reference_clips = self._make_reference_clips(reference_stems, vad_map, paths)
            negative_reference_clips = [
                self._make_reference_clips(
                    group,
                    vad_map,
                    paths,
                    output_dir=paths["negative_clips"] / f"role_{group_index:03d}",
                    prefix="negative",
                )
                for group_index, group in enumerate(negative_stem_groups, start=1)
            ]
            # The lower bound has a distinct meaning from the upper bound:
            # sub-threshold VAD gaps are treated as detector jitter inside one
            # speech island; only gaps in [min, max] remain eligible for the
            # later dual-model same-speaker join. Gaps above max stay hard
            # sentence boundaries.
            vad_spans = self._merge_vad_spans(
                vad_map.get(stem, []),
                gap=max(
                    0.0,
                    min(
                        self.options.silence_min_seconds,
                        self.options.silence_split_seconds,
                    ),
                ),
            )
            if not vad_spans:
                manifest, transcript, archive = self._write_outputs(
                    [],
                    [],
                    stem,
                    target.name,
                    paths,
                    progress,
                    create_archive=create_archive,
                )
                progress(1.0, "完成：目标音频未检测到讲话")
                return PipelineResult(
                    job_id,
                    paths["output"],
                    archive,
                    [],
                    [],
                    manifest,
                    transcript,
                )

            accepted_turns: list[CandidateSentence] = []
            rejected: list[CandidateSentence] = []
            overlap_detector = OverlapDetector() if self.options.use_overlap_detector else None
            verifier = DualSpeakerVerifier(
                self.device,
                status=lambda message: progress(0.49, message),
            )
            try:
                profile = verifier.build_profile(reference_clips, self.options.speaker_threshold)
                exclusion_profiles: list[ExclusionSpeakerProfile] = []
                if negative_reference_clips:
                    progress(
                        0.49,
                        f"建立 {len(negative_reference_clips)} 个排除角色声纹边界",
                    )
                    exclusion_profiles = verifier.build_exclusion_profiles(
                        negative_reference_clips,
                        profile,
                    )
                target_waveform = load_mono(stem, 16000)
                # Filter each smallest silence-delimited island before any
                # joining.  Otherwise one overlap elsewhere in a long merged
                # block incorrectly deletes a clean target utterance.
                clean_atomic_spans = list(vad_spans)
                blocked_join_spans: list[TimeSpan] = []
                if singing_detector is not None:
                    _post_clean, residual_singing = singing_detector.clean_spans(
                        stem,
                        clean_atomic_spans,
                        self.options.singing_threshold,
                        progress=lambda value, message: progress(0.49 + 0.03 * value, message),
                    )
                    # Pre-UVR human singing has already been muted.  Include its
                    # original ranges here so any VAD edge residue rejects the
                    # complete utterance instead of exporting a fragment.
                    singing_evidence = [*pre_singing_spans, *residual_singing]
                    clean_atomic_spans, singing_blocks = self._partition_tainted_spans(
                        clean_atomic_spans,
                        singing_evidence,
                        minimum_overlap_seconds=0.20,
                        minimum_fraction=0.15,
                    )
                    blocked_join_spans.extend(singing_blocks)
                    rejected.extend(
                        CandidateSentence(
                            span.start,
                            span.end,
                            "",
                            reject_reason="检测到有人唱歌",
                            singing_score=self.options.singing_threshold,
                        )
                        for span in singing_blocks
                    )

                if overlap_detector is not None and clean_atomic_spans:
                    _overlap_clean, overlap_evidence = overlap_detector.clean_spans(
                        stem,
                        clean_atomic_spans,
                        self.options.overlap_threshold,
                        progress=lambda value, message: progress(0.52 + 0.03 * value, message),
                    )
                    clean_atomic_spans, overlap_blocks = self._partition_tainted_spans(
                        clean_atomic_spans,
                        overlap_evidence,
                        minimum_overlap_seconds=0.08,
                        minimum_fraction=0.04,
                    )
                    blocked_join_spans.extend(overlap_blocks)
                    rejected.extend(
                        CandidateSentence(
                            span.start,
                            span.end,
                            "",
                            reject_reason="检测到多人同时发声",
                            overlap_score=self.options.overlap_threshold,
                        )
                        for span in overlap_blocks
                    )

                progress(
                    0.55,
                    f"最小语音岛过滤完成：剩余 {len(clean_atomic_spans)} 段；"
                    f"舍弃歌声 {sum(item.reject_reason == '检测到有人唱歌' for item in rejected)} 段、"
                    f"多人重叠 {sum(item.reject_reason == '检测到多人同时发声' for item in rejected)} 段",
                )
                if not clean_atomic_spans:
                    manifest, transcript, archive = self._write_outputs(
                        [],
                        rejected,
                        stem,
                        target.name,
                        paths,
                        progress,
                        create_archive=create_archive,
                    )
                    progress(1.0, "完成：过滤后没有单人讲话")
                    return PipelineResult(
                        job_id,
                        paths["output"],
                        archive,
                        [],
                        rejected,
                        manifest,
                        transcript,
                    )

                if self.options.export_all_sentences:
                    # Manual-review mode deliberately stops before any target
                    # speaker decision.  The audio has already passed the
                    # human-singing and overlap filters above; each remaining
                    # lower-bound-aware silence island is transcribed and
                    # exported so the user can select the identity by ear.
                    progress(
                        0.64,
                        f"全句模式：按静音切分保留 {len(clean_atomic_spans)} 段，跳过目标声纹筛选",
                    )
                    all_candidates = [
                        CandidateSentence(
                            span.start,
                            span.end,
                            "",
                            diagnostics={
                                "all_sentence_export": True,
                                "speaker_gate_bypassed": True,
                                "audio_boundary_source": "silence",
                                "speech_ratio": round(
                                    speech_ratio(vad_spans, span.start, span.end),
                                    5,
                                ),
                            },
                        )
                        for span in clean_atomic_spans
                    ]
                    segmenter = WhisperSegmenter(self.device)
                    transcribed = segmenter.transcribe_spans(
                        stem,
                        [TimeSpan(item.start, item.end) for item in all_candidates],
                        progress=lambda value, message: progress(
                            0.64 + 0.20 * value, message
                        ),
                    )
                    by_span: dict[int, list[CandidateSentence]] = {}
                    for fragment in transcribed:
                        span_index = int(
                            fragment.diagnostics.get("transcription_span_index", -1)
                        )
                        by_span.setdefault(span_index, []).append(fragment)
                    accepted_all: list[CandidateSentence] = []
                    for index, candidate in enumerate(all_candidates):
                        fragments = sorted(
                            by_span.get(index, []),
                            key=lambda item: (item.start, item.end),
                        )
                        text = ""
                        languages: list[str] = []
                        for fragment in fragments:
                            text = self._join_transcript_text(
                                text, fragment.whisper_text
                            )
                            if fragment.language and fragment.language != "auto":
                                languages.append(fragment.language)
                        text = text.strip()
                        if not text:
                            candidate.reject_reason = "STT 未返回有效文本"
                            rejected.append(candidate)
                            continue
                        candidate.whisper_text = text
                        candidate.text = text
                        candidate.language = _language_from_text(
                            text,
                            max(set(languages), key=languages.count)
                            if languages
                            else "auto",
                        )
                        candidate.accepted = True
                        candidate.diagnostics.update(
                            {
                                "stt_fragment_count": len(fragments),
                                "stt_multi_fragment_merged": len(fragments) > 1,
                            }
                        )
                        accepted_all.append(candidate)
                    manifest, transcript, archive = self._write_outputs(
                        accepted_all,
                        rejected,
                        stem,
                        target.name,
                        paths,
                        progress,
                        create_archive=create_archive,
                        source_media=target,
                    )
                    progress(
                        1.0,
                        f"全句模式完成：输出 {len(accepted_all)} 段，STT 无文本 {len(rejected)} 段",
                    )
                    return PipelineResult(
                        job_id,
                        paths["output"],
                        archive,
                        accepted_all,
                        rejected,
                        manifest,
                        transcript,
                    )

                # Keep every silence-delimited island independent until target
                # identity has been verified. A short gap is evidence for the
                # later same-speaker join, not permission to contaminate the
                # first speaker embedding with the following island. This is
                # important for short replies: a preceding breath/noise tail
                # can otherwise lower the target score of the next sentence.
                # Singing and overlap filtering above have already removed the
                # islands that must remain hard no-join gaps.
                clean_spans = list(clean_atomic_spans)
                progress(
                    0.63,
                    f"静音切分完成：{len(clean_spans)} 个独立语音岛；"
                    f"短于 {self.options.silence_split_seconds:.2f} 秒的间隔留给身份确认后合并",
                )
                progress(
                    0.63,
                    f"保留 {len(clean_spans)} 个静音分隔语音岛，先独立核验目标人物",
                )
                progress(0.63, f"开始检查 {len(clean_spans)} 个讲话块内部是否换人")
                split_result = verifier.split_speaker_spans(
                    stem,
                    clean_spans,
                    # Preserve short local sides as edge evidence. The old
                    # 0.85s floor folded a short reply back into the previous
                    # speaker, making a mixed sentence look like one target
                    # turn. These sides remain non-exportable until a joined
                    # whole-span verification succeeds.
                    minimum_turn_seconds=0.30,
                    minimum_edge_seconds=0.20,
                    context_seconds=1.20,
                    scan_hop_seconds=0.30,
                    minimum_similarity_drop=0.08,
                    minimum_separation_seconds=0.70,
                    progress=lambda value, message: progress(0.63 + 0.07 * value, message),
                )
                target_spans = verifier.locate_target_spans(
                    stem,
                    clean_spans,
                    profile,
                    window_seconds=1.80,
                    hop_seconds=0.45,
                    primary_floor=0.50,
                    secondary_floor=0.42,
                    strong_primary=0.62,
                    strong_secondary=0.54,
                    minimum_target_seconds=0.85,
                    bridge_seconds=0.55,
                    progress=lambda value, message: progress(0.70 + 0.03 * value, message),
                )
                effective_threshold = max(
                    self.options.speaker_threshold,
                    profile.primary.suggested_threshold,
                )
                split_result, reconciled_split_count = (
                    self._reconcile_target_split_spans(
                        split_result,
                        clean_spans,
                        target_spans,
                        verifier,
                        profile,
                        target_waveform,
                        exclusion_profiles,
                        effective_threshold,
                    )
                )
                if reconciled_split_count:
                    progress(
                        0.73,
                        f"目标覆盖复核：恢复 {reconciled_split_count} 个误切边界",
                    )
                progress(0.73, f"换人切分完成：得到 {len(split_result)} 个独立说话回合")
                scored_turns: list[
                    tuple[TimeSpan, CandidateSentence, SpeakerMatchDecision | None]
                ] = []
                for index, span in enumerate(split_result, start=1):
                    candidate = CandidateSentence(span.start, span.end, "")
                    candidate.speaker_threshold = effective_threshold
                    candidate.diagnostics["speaker_turn_index"] = index - 1
                    candidate.diagnostics["speech_block_index"] = next(
                        (
                            block_index
                            for block_index, block in enumerate(clean_spans)
                            if block.start <= span.start + 0.02
                            and span.end - 0.02 <= block.end
                        ),
                        -1,
                    )
                    candidate.diagnostics["target_coverage"] = round(
                        self._target_coverage(span, target_spans),
                        5,
                    )
                    candidate.diagnostics["speech_ratio"] = round(
                        speech_ratio(vad_spans, span.start, span.end),
                        5,
                    )
                    candidate.diagnostics["speech_seconds"] = round(
                        candidate.duration * candidate.diagnostics["speech_ratio"],
                        5,
                    )
                    match: SpeakerMatchDecision | None = None
                    if (
                        candidate.diagnostics["speech_seconds"] < 0.20
                        or candidate.diagnostics["speech_ratio"] < 0.18
                    ):
                        candidate.reject_reason = "有效讲话不足或接近空段"
                    elif candidate.duration < self.options.min_sentence_seconds:
                        # Keep a short VAD island available as edge evidence.
                        # It is scored lazily only if a neighboring complete
                        # target turn makes a join plausible.
                        candidate.reject_reason = "说话回合过短"
                        candidate.diagnostics["short_edge_pending"] = True
                    elif candidate.duration > self.options.max_sentence_seconds:
                        candidate.reject_reason = "说话回合过长"
                    else:
                        match = self._verify_speaker_span(
                            verifier,
                            target_waveform,
                            span,
                            profile,
                            effective_threshold,
                        )
                        self._apply_speaker_match(candidate, match, profile, effective_threshold)
                        if not match.accepted:
                            candidate.reject_reason = "声纹匹配不足"
                        elif target_spans and candidate.diagnostics["target_coverage"] < 0.60:
                            candidate.reject_reason = "疑似混合说话人，目标声纹不连续"
                    scored_turns.append((span, candidate, match))
                    progress(
                        0.73 + 0.04 * index / max(1, len(split_result)),
                        f"初步匹配目标人物 {index}/{len(split_result)}",
                    )

                recovery_turns = [
                    (span, candidate, match)
                    for span, candidate, match in scored_turns
                    if match is not None
                    and (
                        self._needs_boundary_recovery(match, candidate.duration)
                        or (
                            candidate.duration >= 3.0
                            and float(candidate.diagnostics.get("target_coverage", 1.0)) < 0.90
                            and match.primary.score >= 0.46
                            and match.secondary is not None
                            and match.secondary.score >= 0.46
                        )
                    )
                ]
                recovery_boundaries: list[SpeakerBoundary] = []
                if recovery_turns:
                    progress(
                        0.77,
                        f"发现 {len(recovery_turns)} 个疑似混合回合，正在局部补切",
                    )
                    assert verifier.secondary is not None
                    recovery_splitter = LocalSpeakerTurnSplitter(
                        verifier.primary,
                        secondary=verifier.secondary,
                    )
                    screen_boundaries = recovery_splitter.detect_speaker_boundaries(
                        stem,
                        [item[0] for item in recovery_turns],
                        context_seconds=0.60,
                        scan_hop_seconds=0.10,
                        primary_candidate_threshold=0.78,
                        minimum_similarity_drop=0.06,
                        minimum_separation_seconds=0.30,
                        progress=lambda value, message: progress(
                            0.77 + 0.01 * value, message
                        ),
                    )
                    suspicious_turns = [
                        item[0]
                        for item in recovery_turns
                        if float(item[1].diagnostics.get("target_coverage", 1.0)) < 0.90
                        or any(
                            item[0].start < boundary.time < item[0].end
                            for boundary in screen_boundaries
                        )
                    ]
                    if suspicious_turns:
                        recovery_boundaries = recovery_splitter.detect_multiscale_speaker_boundaries(
                            stem,
                            suspicious_turns,
                            scan_hop_seconds=0.10,
                            primary_candidate_threshold=0.78,
                            minimum_similarity_drop=0.06,
                            minimum_separation_seconds=0.30,
                            progress=lambda value, message: progress(
                                0.78 + 0.01 * value, message
                            ),
                        )

                recovery_lookup = {id(candidate): index for index, (_span, candidate, _match) in enumerate(recovery_turns, start=1)}
                local_exclusion_spans: list[TimeSpan] = []
                for span, candidate, match in scored_turns:
                    recovery_index = recovery_lookup.get(id(candidate))
                    if recovery_index is not None and match is not None:
                        local_boundaries = [
                            boundary
                            for boundary in recovery_boundaries
                            if span.start < boundary.time < span.end
                        ]
                        recovered = self._recover_target_segments(
                            span,
                            local_boundaries,
                            verifier,
                            target_waveform,
                            profile,
                            exclusion_profiles,
                            effective_threshold,
                            progress,
                            recovery_index,
                            len(recovery_turns),
                        )
                        if recovered is not None:
                            accepted_candidates, discarded = recovered
                            local_exclusion_spans.extend(
                                TimeSpan(item.start, item.end)
                                for item in discarded
                                if item.diagnostics.get("excluded_role_rejected")
                                and item.duration >= 1.60
                            )
                            for recovered_candidate, recovered_match in accepted_candidates:
                                for key in ("speech_block_index", "speaker_turn_index"):
                                    if key in candidate.diagnostics:
                                        recovered_candidate.diagnostics[key] = candidate.diagnostics[key]
                                recovered_candidate.diagnostics["target_coverage"] = round(
                                    self._target_coverage(
                                        TimeSpan(recovered_candidate.start, recovered_candidate.end),
                                        target_spans,
                                    ),
                                    5,
                                )
                                recovered_candidate.diagnostics["speech_ratio"] = round(
                                    speech_ratio(
                                        vad_spans,
                                        recovered_candidate.start,
                                        recovered_candidate.end,
                                    ),
                                    5,
                                )
                                recovered_candidate.diagnostics["speech_seconds"] = round(
                                    recovered_candidate.duration
                                    * recovered_candidate.diagnostics["speech_ratio"],
                                    5,
                                )
                                if (
                                    recovered_candidate.diagnostics["speech_seconds"] < 0.35
                                    or recovered_candidate.diagnostics["speech_ratio"] < 0.18
                                ):
                                    recovered_candidate.reject_reason = "有效讲话不足或接近空段"
                                    discarded.append(recovered_candidate)
                                elif (
                                    target_spans
                                    and recovered_candidate.diagnostics["target_coverage"] < 0.60
                                ):
                                    recovered_candidate.reject_reason = "疑似混合说话人，目标声纹不连续"
                                    discarded.append(recovered_candidate)
                                else:
                                    accepted_turns.append(recovered_candidate)
                            rejected.extend(discarded)
                            continue
                        if match.accepted:
                            # No confirmed local boundary means this verified
                            # speech block is continuous, so keep it intact.
                            candidate.diagnostics["boundary_audit"] = "clean"

                    if candidate.reject_reason:
                        rejected.append(candidate)
                    else:
                        accepted_turns.append(candidate)

                def install_recovered_target(candidate: CandidateSentence) -> bool:
                    overlapping = [
                        existing
                        for existing in accepted_turns
                        if min(candidate.end, existing.end)
                        - max(candidate.start, existing.start)
                        > 0.10
                    ]
                    if overlapping and not all(
                        candidate.start <= existing.start + 0.25
                        and candidate.end >= existing.end - 0.25
                        for existing in overlapping
                    ):
                        return False
                    for existing in overlapping:
                        accepted_turns.remove(existing)
                    accepted_turns.append(candidate)
                    return True

                # Target-specific recovery: the locator is better at finding a
                # known speaker inside a long sequential-speaker block than the
                # generic change detector. Snap its windows back to complete
                # VAD islands, then require a formal dual-model match.
                locator_units: list[TimeSpan] = []
                for turn in split_result:
                    cuts = [
                        boundary.time
                        for boundary in recovery_boundaries
                        if boundary.confidence >= 0.90
                        and turn.start + 0.35 <= boundary.time <= turn.end - 0.35
                    ]
                    points = [turn.start, *sorted(set(cuts)), turn.end]
                    locator_units.extend(
                        TimeSpan(start, end)
                        for start, end in zip(points, points[1:])
                        if end - start >= 0.35
                    )
                locator_bases = self._snap_target_spans_to_speech(
                    target_spans,
                    locator_units,
                )
                # Also score the locator's own continuous regions.  A long VAD
                # island may contain the target only in the middle, while a
                # locally detected prosody boundary can split one true sentence
                # into pieces too short for reliable verification.  Clamp every
                # locator region to a generic speaker turn and snap only nearby
                # edges to precise local boundaries.
                raw_locator_bases: list[TimeSpan] = []
                boundary_suppressed_keys: set[tuple[float, float]] = set()
                for target_region in target_spans:
                    for turn in split_result:
                        start = max(target_region.start, turn.start)
                        end = min(target_region.end, turn.end)
                        if end - start < self.options.min_sentence_seconds:
                            continue
                        internal_boundaries = [
                            boundary
                            for boundary in recovery_boundaries
                            if boundary.confidence >= 0.90
                            and start < boundary.time < end
                        ]
                        structural_boundaries = [
                            boundary
                            for boundary in internal_boundaries
                            if self._is_structural_boundary(boundary)
                        ]
                        # A stable boundary is a hard limit for locator
                        # recovery.  The individual sides can still be
                        # recovered by the later anchor-consensus pass.
                        if structural_boundaries:
                            continue
                        # Emotional/prosodic changes can create several false
                        # local speaker boundaries inside one compact target
                        # region. Add one boundary-suppressed whole candidate;
                        # the normal fine-grained candidates remain available
                        # and the whole candidate must pass formal verification.
                        if len(internal_boundaries) >= 2 and end - start <= 4.0:
                            whole_start = start
                            start_edges = [
                                boundary.time
                                for boundary in recovery_boundaries
                                if boundary.confidence >= 0.90
                                and start - 0.05 <= boundary.time <= start + 0.75
                            ]
                            if start_edges:
                                whole_start = min(start_edges)
                            whole_end = end
                            if 0.0 <= turn.end - end <= 1.25:
                                whole_end = turn.end
                            else:
                                end_edges = [
                                    boundary.time
                                    for boundary in recovery_boundaries
                                    if boundary.confidence >= 0.90
                                    and end - 0.75 <= boundary.time <= end + 0.05
                                ]
                                if end_edges:
                                    whole_end = max(end_edges)
                            whole = TimeSpan(whole_start, whole_end)
                            if (
                                whole.duration >= 2.20
                                and whole.duration <= min(
                                    12.0, self.options.max_sentence_seconds
                                )
                                and self._target_coverage(whole, target_spans) >= 0.55
                            ):
                                raw_locator_bases.append(whole)
                                boundary_suppressed_keys.add(
                                    (round(whole.start, 5), round(whole.end, 5))
                                )
                        nearby_start = [
                            boundary.time
                            for boundary in recovery_boundaries
                            if start <= boundary.time < end
                            and abs(boundary.time - start) <= 0.75
                        ]
                        nearby_end = [
                            boundary.time
                            for boundary in recovery_boundaries
                            if start < boundary.time <= end
                            and abs(boundary.time - end) <= 0.75
                        ]
                        if nearby_start:
                            start = min(nearby_start)
                        elif start - turn.start <= 0.35:
                            start = turn.start
                        if nearby_end:
                            end = max(nearby_end)
                        elif turn.end - end <= 0.35:
                            end = turn.end
                        if end - start >= self.options.min_sentence_seconds:
                            raw_locator_bases.append(TimeSpan(start, end))
                locator_bases = sorted(
                    {
                        (round(item.start, 5), round(item.end, 5)): item
                        for item in [*raw_locator_bases, *locator_bases]
                    }.values(),
                    key=lambda item: (item.start, item.duration),
                )

                def source_turn_for_locator(locator: TimeSpan) -> TimeSpan:
                    """Bind a locator candidate to its actual split turn.

                    Locator candidates are assembled in a nested target/turn
                    loop.  Reusing that loop variable after the loop silently
                    attached every candidate to the last turn in the episode,
                    which disabled the later edge audit for mixed utterances.
                    """

                    containing = [
                        turn
                        for turn in split_result
                        if turn.start <= locator.start + 0.05
                        and locator.end <= turn.end + 0.05
                    ]
                    if containing:
                        return min(containing, key=lambda item: item.duration)
                    return locator

                locator_recovered = 0
                locator_rejected: list[
                    tuple[TimeSpan, SpeakerMatchDecision, float]
                ] = []
                for recovery_index, base in enumerate(locator_bases, start=1):
                    source_turn = source_turn_for_locator(base)
                    base_key = (round(base.start, 5), round(base.end, 5))
                    recovered_span = base
                    edge_expanded = False
                    # A short locally confirmed target fragment may sit directly
                    # outside the locator region because its dual-model score was
                    # diluted by duration.  Absorb only WavLM-confirmed fragments
                    # from the same original speaker turn, then verify the whole
                    # expanded sentence again below.
                    changed = True
                    while changed:
                        changed = False
                        for edge in accepted_turns:
                            diagnostics = edge.diagnostics
                            if not diagnostics.get("wavlm_local_rescue"):
                                continue
                            original_start = float(
                                diagnostics.get("original_turn_start", edge.start)
                            )
                            original_end = float(
                                diagnostics.get("original_turn_end", edge.end)
                            )
                            same_original_turn = (
                                original_start <= recovered_span.start + 0.05
                                and original_end >= recovered_span.end - 0.05
                            )
                            if not same_original_turn:
                                continue
                            if 0.0 <= recovered_span.start - edge.end <= 0.03:
                                recovered_span = TimeSpan(edge.start, recovered_span.end)
                                edge_expanded = True
                                changed = True
                                break
                            if 0.0 <= edge.start - recovered_span.end <= 0.03:
                                recovered_span = TimeSpan(recovered_span.start, edge.end)
                                edge_expanded = True
                                changed = True
                                break

                    if (
                        recovered_span.duration < self.options.min_sentence_seconds
                        or recovered_span.duration
                        > min(20.0, self.options.max_sentence_seconds)
                    ):
                        continue
                    local_negative_overlap = sum(
                        max(
                            0.0,
                            min(recovered_span.end, negative.end)
                            - max(recovered_span.start, negative.start),
                        )
                        for negative in local_exclusion_spans
                    )
                    if local_negative_overlap >= 0.10:
                        # Never let an averaged whole-turn embedding overwrite a
                        # local segment that two speaker models assigned to the
                        # same user-supplied exclusion person.
                        continue
                    coverage = self._target_coverage(recovered_span, target_spans)
                    if coverage < 0.55:
                        continue
                    if any(
                        self._is_structural_boundary(boundary)
                        for boundary in recovery_boundaries
                        if recovered_span.start < boundary.time < recovered_span.end
                    ):
                        # Do not reinstall a whole target region over a stable
                        # change point.  The split pieces are handled below.
                        continue
                    boundary_suppressed = base_key in boundary_suppressed_keys
                    recovered_match = self._verify_speaker_span(
                        verifier,
                        target_waveform,
                        recovered_span,
                        profile,
                        effective_threshold,
                    )
                    if not recovered_match.accepted:
                        tertiary_match = None
                        tertiary_coverage = self._target_coverage(
                            recovered_span,
                            verifier.tertiary_target_spans,
                        )
                        secondary = recovered_match.secondary
                        strong_cam_disagreement = (
                            secondary is not None
                            and recovered_match.primary.score >= 0.50
                            and secondary.score >= 0.60
                            and secondary.score - recovered_match.primary.score >= 0.05
                        )
                        minimum_tertiary_coverage = (
                            0.55 if boundary_suppressed else 0.80
                        )
                        if coverage >= minimum_tertiary_coverage and (
                            boundary_suppressed
                            or tertiary_coverage >= 0.35
                            or strong_cam_disagreement
                        ):
                            tertiary_match = verifier.promote_with_tertiary(
                                self._waveform_span(target_waveform, recovered_span),
                                profile,
                                recovered_match,
                                recovered_span.duration,
                            )
                        if tertiary_match is None and edge_expanded:
                            tertiary_match = verifier.promote_local_with_tertiary(
                                self._waveform_span(target_waveform, recovered_span),
                                profile,
                                recovered_match,
                                recovered_span.duration,
                            )
                        if tertiary_match is None:
                            if boundary_suppressed:
                                continue
                            locator_rejected.append(
                                (recovered_span, recovered_match, coverage)
                            )
                            continue
                        recovered_match = tertiary_match
                    recovered_candidate = CandidateSentence(
                        recovered_span.start,
                        recovered_span.end,
                        "",
                    )
                    self._apply_speaker_match(
                        recovered_candidate,
                        recovered_match,
                        profile,
                        effective_threshold,
                    )
                    recovered_candidate.diagnostics.update(
                        {
                            "target_locator_recovery": True,
                            "tertiary_locator_recovery": (
                                recovered_match.tier == "tertiary"
                            ),
                            "locator_source_start": source_turn.start,
                            "locator_source_end": source_turn.end,
                            "locator_source_target_coverage": round(
                                self._target_coverage(source_turn, target_spans),
                                5,
                            ),
                            "boundary_suppressed_recovery": boundary_suppressed,
                            "locator_edge_expansion": edge_expanded,
                            "tertiary_target_coverage": round(
                                self._target_coverage(
                                    recovered_span,
                                    verifier.tertiary_target_spans,
                                ),
                                5,
                            ),
                            "target_coverage": round(
                                self._target_coverage(recovered_span, target_spans),
                                5,
                            ),
                            "speech_ratio": round(
                                speech_ratio(
                                    vad_spans,
                                    recovered_span.start,
                                    recovered_span.end,
                                ),
                                5,
                            ),
                            "locator_recovery_index": recovery_index,
                            "locator_boundary_count": sum(
                                recovered_span.start < boundary.time < recovered_span.end
                                for boundary in recovery_boundaries
                            ),
                        }
                    )
                    recovered_candidate.diagnostics["speech_seconds"] = round(
                        recovered_candidate.duration
                        * recovered_candidate.diagnostics["speech_ratio"],
                        5,
                    )
                    recovered_exclusion = verifier.exclusion_audit(
                        recovered_match,
                        profile,
                        exclusion_profiles,
                        tertiary_recovery=(recovered_match.tier == "tertiary"),
                    )
                    if recovered_exclusion is not None:
                        recovered_candidate.diagnostics.update(recovered_exclusion)
                    if (
                        recovered_exclusion
                        and recovered_exclusion.get("excluded_role_rejected")
                    ):
                        recovered_candidate.reject_reason = (
                            f"更接近{recovered_exclusion['excluded_role']}，"
                            "已按排除角色删除"
                        )
                        rejected.append(recovered_candidate)
                        continue
                    if install_recovered_target(recovered_candidate):
                        locator_recovered += 1

                if locator_recovered:
                    progress(
                        0.795,
                        f"目标人物完整句恢复：定位恢复 {locator_recovered} 段",
                    )

                # The locator may have replaced a complete verified turn with
                # a high-confidence core. Recheck the original local turn now,
                # after locator recovery, so a short same-speaker tail can be
                # restored without allowing a stable speaker boundary through.
                self._recover_same_turn_edges(
                    accepted_turns,
                    list(split_result),
                    recovery_boundaries,
                    verifier,
                    profile,
                    target_waveform,
                    target_spans,
                    exclusion_profiles,
                    effective_threshold,
                    blocked_join_spans,
                    progress,
                )

                # A rescued local edge is normally only evidence for a nearby
                # complete sentence.  Keep it as a standalone output only when
                # it is long enough to be a real utterance and both independent
                # speaker models (or the explicit tertiary rescue) support the
                # whole span.  This recovers a clipped target tail while still
                # rejecting the common 0.x-second wrong-speaker residue.
                unconnected_edges: list[CandidateSentence] = []
                for candidate in accepted_turns:
                    if not candidate.diagnostics.get("local_edge_only"):
                        continue
                    diagnostics = candidate.diagnostics
                    primary_score = float(diagnostics.get("eres_score", 0.0) or 0.0)
                    secondary_score = float(
                        diagnostics.get("camplus_score", 0.0) or 0.0
                    )
                    target_coverage = float(
                        diagnostics.get("target_coverage", 0.0) or 0.0
                    )
                    standalone = (
                        candidate.duration >= max(1.20, self.options.min_output_seconds)
                        and target_coverage >= 0.90
                        and primary_score >= 0.48
                        and secondary_score >= 0.48
                        and diagnostics.get("speaker_tier")
                        in {"tertiary", "strong", "balanced"}
                        and not diagnostics.get("excluded_role_rejected")
                    )
                    if standalone:
                        diagnostics["standalone_local_edge"] = True
                    else:
                        unconnected_edges.append(candidate)
                if unconnected_edges:
                    unconnected_ids = {id(candidate) for candidate in unconnected_edges}
                    accepted_turns = [
                        candidate
                        for candidate in accepted_turns
                        if id(candidate) not in unconnected_ids
                    ]
                    for candidate in unconnected_edges:
                        candidate.reject_reason = "局部目标候选未连接到完整句"
                    rejected.extend(unconnected_edges)

                contrastive_edges = self._build_contrastive_edge_candidates(
                    scored_turns,
                    accepted_turns,
                    verifier,
                    profile,
                    target_waveform,
                    target_spans,
                    effective_threshold,
                )
                if contrastive_edges:
                    scored_turns.extend(contrastive_edges)
                    rejected.extend(
                        candidate
                        for _span, candidate, _match in contrastive_edges
                    )
                    progress(
                        0.795,
                        f"短边换人对比复核：提出 {len(contrastive_edges)} 个完整核心",
                    )

                consensus_recovered = self._promote_multimodel_target_subclusters(
                    scored_turns,
                    accepted_turns,
                    rejected,
                    verifier,
                    profile,
                    stem,
                    target_waveform,
                    target_spans,
                    exclusion_profiles,
                    progress,
                )
                if consensus_recovered:
                    progress(
                        0.795,
                        f"多模型目标子簇恢复：新增 {consensus_recovered} 个完整回合",
                    )

                anime_recovered = self._promote_anime_identity_candidates(
                    scored_turns,
                    accepted_turns,
                    rejected,
                    profile,
                    target_waveform,
                    reference_clips,
                    negative_reference_clips,
                    progress,
                )
                if anime_recovered:
                    progress(
                        0.80,
                        f"独立身份复核恢复：新增 {anime_recovered} 个完整回合",
                    )

                target_merges = self._merge_verified_target_turns(
                    accepted_turns,
                    rejected,
                    verifier,
                    profile,
                    target_waveform,
                    effective_threshold,
                    target_spans,
                    vad_spans,
                    blocked_join_spans,
                    progress=lambda value, message: progress(
                        0.795 + 0.005 * value,
                        message,
                    ),
                    exclusion_profiles=exclusion_profiles,
                )
                if target_merges:
                    progress(
                        0.80,
                        f"目标人物短静音合并完成：形成 {target_merges} 个完整句",
                    )

                repartitioned = self._repartition_long_target_turns(
                    accepted_turns,
                    rejected,
                    verifier,
                    profile,
                    target_waveform,
                    target_spans,
                    exclusion_profiles,
                    effective_threshold,
                    progress,
                )
                if repartitioned:
                    progress(
                        0.80,
                        f"长回合身份复核完成：保留 {len(accepted_turns)} 个独立目标段",
                    )
                edge_audited = self._audit_risky_target_edges(
                    accepted_turns,
                    verifier,
                    profile,
                    stem,
                    target_waveform,
                    exclusion_profiles,
                    target_spans,
                    effective_threshold,
                    progress,
                )
                if edge_audited:
                    progress(
                        0.80,
                        f"目标边缘纯度复核完成：修正 {edge_audited} 个回合",
                    )
                pruned_recoveries = self._prune_weak_recovery_candidates(
                    accepted_turns,
                    rejected,
                )
                if pruned_recoveries:
                    progress(
                        0.80,
                        f"局部恢复完整性复核：舍弃 {pruned_recoveries} 个不完整候选",
                    )

                adaptive_recovered = 0
                adaptive_profiles = self._build_adaptive_speaker_profile(
                    accepted_turns,
                    verifier,
                    profile,
                    target_waveform,
                    effective_threshold,
                )
                if adaptive_profiles is not None:
                    (
                        adaptive_profile,
                        adaptive_secondary,
                        adaptive_primary_gain_floor,
                        adaptive_secondary_gain_floor,
                    ) = adaptive_profiles
                    original_secondary = verifier.secondary_profile
                    verifier.secondary_profile = adaptive_secondary
                    try:
                        for base, original_match, coverage in locator_rejected:
                            if (
                                original_match.tier != "recall"
                                or coverage < 0.80
                                or base.duration < 2.20
                                or any(
                                    min(base.end, existing.end)
                                    - max(base.start, existing.start)
                                    > 0.10
                                    for existing in accepted_turns
                                )
                            ):
                                continue
                            adaptive_match = self._verify_speaker_span(
                                verifier,
                                target_waveform,
                                base,
                                adaptive_profile,
                                effective_threshold,
                            )
                            if not adaptive_match.accepted:
                                continue
                            assert original_match.secondary is not None
                            assert adaptive_match.secondary is not None
                            primary_gain = (
                                adaptive_match.primary.score
                                - original_match.primary.score
                            )
                            secondary_gain = (
                                adaptive_match.secondary.score
                                - original_match.secondary.score
                            )
                            if (
                                primary_gain < adaptive_primary_gain_floor
                                or secondary_gain < adaptive_secondary_gain_floor
                            ):
                                continue
                            recovered = CandidateSentence(base.start, base.end, "")
                            self._apply_speaker_match(
                                recovered,
                                adaptive_match,
                                adaptive_profile,
                                effective_threshold,
                            )
                            recovered.diagnostics.update(
                                {
                                    "adaptive_profile_recovery": True,
                                    "original_speaker_tier": original_match.tier,
                                    "original_eres_score": original_match.primary.score,
                                    "original_camplus_score": (
                                        original_match.secondary.score
                                        if original_match.secondary is not None
                                        else 0.0
                                    ),
                                    "adaptive_primary_gain": round(primary_gain, 5),
                                    "adaptive_secondary_gain": round(secondary_gain, 5),
                                    "adaptive_primary_gain_floor": (
                                        adaptive_primary_gain_floor
                                    ),
                                    "adaptive_secondary_gain_floor": (
                                        adaptive_secondary_gain_floor
                                    ),
                                    "target_coverage": round(coverage, 5),
                                    "locator_boundary_count": sum(
                                        base.start < boundary.time < base.end
                                        for boundary in recovery_boundaries
                                    ),
                                    "speech_ratio": round(
                                        speech_ratio(vad_spans, base.start, base.end),
                                        5,
                                    ),
                                }
                            )
                            recovered.diagnostics["speech_seconds"] = round(
                                recovered.duration
                                * recovered.diagnostics["speech_ratio"],
                                5,
                            )
                            if install_recovered_target(recovered):
                                adaptive_recovered += 1
                    finally:
                        verifier.secondary_profile = original_secondary
                if adaptive_recovered:
                    progress(
                        0.80,
                        f"域内严格声纹复核恢复 {adaptive_recovered} 个目标回合",
                    )

                if exclusion_profiles and accepted_turns:
                    progress(
                        0.80,
                        f"排除角色最终复核：0/{len(accepted_turns)}",
                    )
                    retained_turns: list[CandidateSentence] = []
                    fourth_verifier = None
                    fourth_profile = None
                    for audit_index, candidate in enumerate(accepted_turns, start=1):
                        audit_match = self._verify_speaker_span(
                            verifier,
                            target_waveform,
                            TimeSpan(candidate.start, candidate.end),
                            profile,
                            effective_threshold,
                        )
                        exclusion = verifier.exclusion_audit(
                            audit_match,
                            profile,
                            exclusion_profiles,
                            tertiary_recovery=(
                                candidate.diagnostics.get("speaker_tier") == "tertiary"
                                or bool(candidate.diagnostics.get("wavlm_rescue"))
                                or bool(candidate.diagnostics.get("wavlm_local_rescue"))
                            ),
                        )
                        if exclusion is not None:
                            candidate.diagnostics.update(exclusion)
                        if (
                            exclusion
                            and not exclusion.get("excluded_role_rejected")
                            and self._needs_final_multimodel_exclusion(candidate, exclusion)
                        ):
                            if fourth_verifier is None or fourth_profile is None:
                                fourth_verifier, fourth_profile = verifier.quaternary_pair(profile)
                            candidate_waveform = self._waveform_span(
                                target_waveform,
                                TimeSpan(candidate.start, candidate.end),
                            )
                            fourth_embedding = fourth_verifier.embeddings_from_waveforms(
                                [candidate_waveform]
                            )[0]
                            multimodel_exclusion = verifier.multimodel_exclusion_audit(
                                audit_match,
                                profile,
                                fourth_embedding,
                                fourth_profile,
                                exclusion_profiles,
                            )
                            candidate.diagnostics["final_multimodel_exclusion_checked"] = True
                            if multimodel_exclusion is not None:
                                candidate.diagnostics.update(multimodel_exclusion)
                            if self._needs_final_multimodel_exclusion(
                                candidate,
                                multimodel_exclusion,
                                require_third_model=True,
                            ):
                                exclusion = multimodel_exclusion
                                if exclusion is not None:
                                    # The third-model decision is a real veto,
                                    # not just a diagnostic marker. The old
                                    # path set ``final_multimodel_exclusion_veto``
                                    # but left ``excluded_role_rejected`` false,
                                    # allowing the mixed candidate into STT.
                                    exclusion["excluded_role_rejected"] = True
                                candidate.diagnostics["final_multimodel_exclusion_veto"] = True
                        if exclusion and exclusion.get("excluded_role_rejected"):
                            candidate.reject_reason = (
                                f"更接近{exclusion['excluded_role']}，已按排除角色删除"
                            )
                            rejected.append(candidate)
                        else:
                            retained_turns.append(candidate)
                        progress(
                            0.80,
                            f"排除角色最终复核：{audit_index}/{len(accepted_turns)}",
                        )
                    accepted_turns = retained_turns

                progress(
                    0.80,
                    f"声纹筛选完成：保留 {len(accepted_turns)} 个目标人物回合，"
                    f"舍弃 {len(rejected)} 个回合",
                )
            finally:
                verifier.close()

            accepted: list[CandidateSentence] = []
            if accepted_turns:
                accepted_turns.sort(key=lambda item: (item.start, item.end))
                progress(0.80, f"筛选完成：仅对 {len(accepted_turns)} 个目标人物回合执行 STT")
                segmenter = WhisperSegmenter(self.device)
                transcribed = segmenter.transcribe_spans(
                    stem,
                    [TimeSpan(candidate.start, candidate.end) for candidate in accepted_turns],
                    progress=lambda value, message: progress(0.80 + 0.13 * value, message),
                )
                by_turn: dict[int, list[CandidateSentence]] = {}
                for sentence in transcribed:
                    turn_index = int(sentence.diagnostics.get("transcription_span_index", -1))
                    by_turn.setdefault(turn_index, []).append(sentence)
                for index, turn in enumerate(accepted_turns):
                    sentences = by_turn.get(index, [])
                    if not sentences:
                        turn.reject_reason = "STT 未返回文本"
                        rejected.append(turn)
                        continue
                    sentences.sort(key=lambda item: (item.start, item.end))
                    text = ""
                    languages: list[str] = []
                    for fragment in sentences:
                        text = self._join_transcript_text(text, fragment.whisper_text)
                        if fragment.language and fragment.language != "auto":
                            languages.append(fragment.language)
                    text = text.strip()
                    if not text:
                        turn.reject_reason = "STT 未返回有效文本"
                        rejected.append(turn)
                        continue
                    fragment_count = len(sentences)
                    if fragment_count >= 2:
                        ordered_fragments = sorted(
                            sentences, key=lambda item: (item.start, item.end)
                        )
                        fragment_gaps = [
                            round(max(0.0, right.start - left.end), 5)
                            for left, right in zip(
                                ordered_fragments, ordered_fragments[1:]
                            )
                        ]
                        # Whisper may split one continuous utterance at an
                        # internal punctuation mark or an omitted word
                        # timestamp.  The speaker segmentation already made
                        # this turn atomic, so fragment count alone is not
                        # evidence of a speaker switch. Only a real long gap
                        # remains a hard boundary; contiguous fragments are
                        # merged into the one verified speaker turn below.
                        hard_fragment_gap = any(
                            gap > self.options.silence_split_seconds
                            for gap in fragment_gaps
                        )
                        turn.diagnostics.update(
                            {
                                "stt_fragment_count": fragment_count,
                                "stt_fragment_gaps": fragment_gaps,
                                "stt_multi_fragment_merged": not hard_fragment_gap,
                            }
                        )
                        if (
                            hard_fragment_gap
                            and not turn.diagnostics.get("post_target_silence_merge")
                        ):
                            turn.reject_reason = "STT 显示多个独立话段，疑似连续换人"
                            rejected.append(turn)
                            continue
                    if turn.duration < self.options.min_output_seconds:
                        turn.reject_reason = "讲话片段短于训练下限"
                        rejected.append(turn)
                        continue

                    language = (
                        max(set(languages), key=languages.count)
                        if languages
                        else "auto"
                    )
                    sentence = CandidateSentence(
                        start=turn.start,
                        end=turn.end,
                        whisper_text=text,
                        language=_language_from_text(text, language),
                        text=text,
                        accepted=True,
                    )
                    sentence.speaker_score = turn.speaker_score
                    sentence.window_min_score = turn.window_min_score
                    sentence.window_p20_score = turn.window_p20_score
                    sentence.speaker_vote_ratio = turn.speaker_vote_ratio
                    sentence.window_vote_ratio = turn.window_vote_ratio
                    sentence.speaker_threshold = turn.speaker_threshold
                    sentence.overlap_score = turn.overlap_score
                    sentence.singing_score = turn.singing_score
                    sentence.speech_score = turn.speech_score
                    sentence.diagnostics.update(turn.diagnostics)
                    sentence.diagnostics.update(
                        {
                            "stt_fragment_count": len(sentences),
                            "audio_boundary_source": "silence_and_speaker",
                            "stt_changed_audio_boundary": False,
                        }
                    )
                    accepted.append(sentence)
                chinese_candidates = [candidate for candidate in accepted if candidate.language == "zh"]
                if chinese_candidates:
                    chinese_paths: list[Path] = []
                    chinese_by_path: dict[Path, CandidateSentence] = {}
                    for index, candidate in enumerate(chinese_candidates, start=1):
                        clip = paths["candidate_clips"] / f"zh_stt_{index:04d}.wav"
                        write_clip(stem, clip, candidate.start, candidate.end, sample_rate=16000)
                        chinese_paths.append(clip)
                        chinese_by_path[clip] = candidate
                    refined = vad_tools.transcribe_files(
                        chinese_paths,
                        progress=lambda value, message: progress(0.92 + 0.02 * value, message),
                    )
                    for clip, candidate in chinese_by_path.items():
                        candidate.text = refined.get(clip, candidate.whisper_text) or candidate.whisper_text
                        clip.unlink(missing_ok=True)
            else:
                progress(0.92, "筛选完成：没有目标人物回合，跳过 STT")
            manifest, transcript, archive = self._write_outputs(
                accepted,
                rejected,
                stem,
                target.name,
                paths,
                progress,
                create_archive=create_archive,
                source_media=target,
            )
            progress(1.0, f"完成：保留 {len(accepted)} 句，舍弃 {len(rejected)} 句")
            return PipelineResult(job_id, paths["output"], archive, accepted, rejected, manifest, transcript)
        finally:
            if singing_detector is not None:
                singing_detector.close()
            if self.options.cleanup_work:
                shutil.rmtree(paths["root"], ignore_errors=True)

    def run_many(
        self,
        references: Iterable[str | Path],
        targets: Iterable[str | Path],
        negative_references: Iterable[Iterable[str | Path]] | None = None,
        progress: ProgressCallback | None = None,
    ) -> BatchPipelineResult:
        """Process target files sequentially and create one batch download."""
        progress = progress or _noop_progress
        reference_paths = [Path(path) for path in references if path]
        negative_reference_groups = [
            [Path(path) for path in group if path]
            for group in (negative_references or [])
        ]
        negative_reference_groups = [
            group for group in negative_reference_groups if group
        ]
        target_paths = [Path(path) for path in targets if path]
        if not reference_paths:
            raise ValueError("请至少提供一段参考音频")
        if not target_paths:
            raise ValueError("请至少提供一段待提取音频")
        missing = [str(path) for path in target_paths if not path.exists()]
        if missing:
            raise FileNotFoundError("待提取音频不存在：" + "，".join(missing))

        batch_id = time.strftime("%Y%m%d_%H%M%S") + "_batch_" + uuid.uuid4().hex[:6]
        batch_dir = OUTPUT_ROOT / batch_id
        batch_dir.mkdir(parents=True, exist_ok=False)
        results: list[PipelineResult] = []
        target_count = len(target_paths)

        try:
            for target_index, target in enumerate(target_paths, start=1):
                prefix = f"目标文件 {target_index}/{target_count} [{target.name}]"

                def report_target(value: float, message: str, *, _prefix: str = prefix) -> None:
                    overall = ((target_index - 1) + min(1.0, max(0.0, value))) / target_count
                    progress(overall, f"{_prefix} · {message}")

                report_target(0.0, "准备处理")
                destination_name = f"{target_index:03d}_{_safe_name(target.stem, 'target')}"
                destination = batch_dir / destination_name
                child_job_id = f"{batch_id}_{target_index:03d}"
                result = self.run(
                    reference_paths,
                    target,
                    negative_references=negative_reference_groups,
                    progress=report_target,
                    job_id=child_job_id,
                    create_archive=False,
                )
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.move(str(result.output_dir), str(destination))

                old_output_dir = result.output_dir
                old_manifest_rel = result.manifest_path.relative_to(old_output_dir)
                old_transcript_rel = result.transcript_path.relative_to(old_output_dir)
                for sentence in [*result.accepted, *result.rejected]:
                    if sentence.audio_file:
                        sentence.audio_file = str(Path(destination_name) / sentence.audio_file)
                    if sentence.text_file:
                        sentence.text_file = str(Path(destination_name) / sentence.text_file)
                    if sentence.video_file:
                        sentence.video_file = str(Path(destination_name) / sentence.video_file)
                result.output_dir = destination
                result.manifest_path = destination / old_manifest_rel
                result.transcript_path = destination / old_transcript_rel
                result.archive_path.unlink(missing_ok=True)
                results.append(result)
                report_target(1.0, f"文件完成，保留 {len(result.accepted)} 句")

            records: list[dict] = []
            target_summaries: list[dict] = []
            srt_parts: list[str] = []
            srt_index = 1
            for target, result in zip(target_paths, results):
                target_summaries.append(
                    {
                        "target": target.name,
                        "output_dir": result.output_dir.name,
                        "accepted_count": len(result.accepted),
                        "rejected_count": len(result.rejected),
                        "reject_summary": {
                            reason: sum((item.reject_reason or "未知原因") == reason for item in result.rejected)
                            for reason in sorted({item.reject_reason or "未知原因" for item in result.rejected})
                        },
                    }
                )
                for sentence in [*result.accepted, *result.rejected]:
                    record = sentence.to_dict()
                    record["target"] = target.name
                    records.append(record)
                for sentence in result.accepted:
                    srt_parts.append(
                        f"{srt_index}\n{_srt_time(sentence.start)} --> {_srt_time(sentence.end)}\n"
                        f"[{target.name}] {sentence.text or sentence.whisper_text}\n"
                    )
                    srt_index += 1

            manifest_path = batch_dir / "batch_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "batch_id": batch_id,
                        "target_count": target_count,
                        "accepted_count": sum(len(result.accepted) for result in results),
                        "rejected_count": sum(len(result.rejected) for result in results),
                        "options": asdict(self.options),
                        "targets": target_summaries,
                        "sentences": records,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            csv_path = batch_dir / "batch_manifest.csv"
            fields = [
                "target",
                "audio_file",
                "text_file",
                "video_file",
                "text",
                "language",
                "start",
                "end",
                "speaker_score",
                "window_min_score",
                "window_p20_score",
                "speaker_vote_ratio",
                "window_vote_ratio",
                "speaker_threshold",
                "overlap_score",
                "singing_score",
                "accepted",
                "reject_reason",
            ]
            with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(records)

            transcript_path = batch_dir / "batch_transcript.srt"
            transcript_path.write_text("\n".join(srt_parts), encoding="utf-8")
            archive_path = OUTPUT_ROOT / f"{batch_id}.zip"
            temporary_archive = archive_path.with_suffix(".zip.tmp")
            temporary_archive.unlink(missing_ok=True)
            with zipfile.ZipFile(temporary_archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for file in batch_dir.rglob("*"):
                    # A batch download is the single archive for all targets;
                    # never nest a child archive inside it.
                    if file.is_file() and file.suffix.lower() != ".zip":
                        archive.write(file, file.relative_to(batch_dir))
            with zipfile.ZipFile(temporary_archive, "r") as check:
                bad_entry = check.testzip()
                if bad_entry is not None:
                    raise RuntimeError(f"批次压缩包校验失败：{bad_entry}")
            temporary_archive.replace(archive_path)
            progress(1.0, f"批次完成：{target_count} 个目标文件")
            return BatchPipelineResult(
                batch_id=batch_id,
                output_dir=batch_dir,
                archive_path=archive_path,
                results=results,
                manifest_path=manifest_path,
                transcript_path=transcript_path,
            )
        except Exception:
            if not results:
                shutil.rmtree(batch_dir, ignore_errors=True)
            raise
