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
    normalize_audio,
    pad_for_separator,
    probe_duration,
    trim_audio_in_place,
    write_clip,
)
from .config import OUTPUT_ROOT, WORK_ROOT, ensure_local_assets
from .filters import OverlapDetector, SingingDetector
from .speaker import (
    DualSpeakerVerifier,
    LocalSpeakerTurnSplitter,
    SpeakerBoundary,
    SpeakerMatchDecision,
    SpeakerMatchProfile,
)
from .transcription import FunASRTools, WhisperSegmenter
from .types import BatchPipelineResult, CandidateSentence, PipelineResult, TimeSpan

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineOptions:
    speaker_threshold: float = 0.70
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
    cleanup_work: bool = False


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

    def _job_paths(self, job_id: str) -> dict[str, Path]:
        root = WORK_ROOT / job_id
        return {
            "root": root,
            "normalized_refs": root / "references_normalized",
            "reference_stems": root / "reference_stems",
            "reference_clips": root / "reference_voice_clips",
            "normalized_target": root / "target_normalized.wav",
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

    @staticmethod
    def _merge_vad_spans(spans: list[TimeSpan], gap: float = 0.18) -> list[TimeSpan]:
        if not spans:
            return []
        spans = sorted(spans, key=lambda item: item.start)
        merged: list[TimeSpan] = [spans[0]]
        for span in spans[1:]:
            previous = merged[-1]
            if span.start <= previous.end + gap:
                merged[-1] = TimeSpan(previous.start, max(previous.end, span.end))
            else:
                merged.append(span)
        return merged

    def _make_reference_clips(
        self,
        reference_stems: list[Path],
        spans_by_path: dict[Path, list[TimeSpan]],
        paths: dict[str, Path],
    ) -> list[Path]:
        output: list[Path] = []
        paths["reference_clips"].mkdir(parents=True, exist_ok=True)
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
                    destination = paths["reference_clips"] / (
                        f"reference_{reference_index:03d}_{clip_index:03d}.wav"
                    )
                    write_clip(stem, destination, cursor, clip_end, sample_rate=16000)
                    output.append(destination)
                    cursor = clip_end
            if clip_index == 0:
                # Short references are still useful; the verifier pads them safely.
                destination = paths["reference_clips"] / f"reference_{reference_index:03d}_001.wav"
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
                and (left.duration < minimum_seconds or right.duration < minimum_seconds)
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

    @staticmethod
    def _needs_boundary_recovery(match: SpeakerMatchDecision, duration: float) -> bool:
        """Select near-target turns and audit every long accepted turn."""

        secondary = match.secondary
        if secondary is None or duration < 1.8:
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
        return duration >= 1.80

    def _recover_target_segments(
        self,
        span: TimeSpan,
        boundaries: list[SpeakerBoundary],
        verifier: DualSpeakerVerifier,
        waveform: torch.Tensor,
        profile: SpeakerMatchProfile,
        threshold: float,
        progress: ProgressCallback,
        recovery_index: int,
        recovery_count: int,
    ) -> tuple[list[CandidateSentence], list[CandidateSentence]] | None:
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
        accepted: list[CandidateSentence] = []
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
                }
            )
            if part.duration < minimum:
                candidate.reject_reason = "说话回合过短"
            else:
                match = self._verify_speaker_span(
                    verifier, waveform, part, profile, threshold
                )
                self._apply_speaker_match(candidate, match, profile, threshold)
                if not match.accepted:
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
                rejected.append(candidate)
            else:
                accepted.append(candidate)

        if not accepted:
            # A confirmed boundary with no independently matching piece is
            # safer to discard as a mixed turn than to export the original span.
            discarded = CandidateSentence(span.start, span.end, "")
            discarded.reject_reason = "疑似混合说话人，保守舍弃"
            discarded.diagnostics.update(
                {
                    "local_boundary_recovery": True,
                    "recovery_boundaries": candidates,
                }
            )
            return [], [discarded]
        return accepted, rejected

    @staticmethod
    def _expand_recall_candidates(
        scored_turns: list[
            tuple[TimeSpan, CandidateSentence, SpeakerMatchDecision | None]
        ],
        accepted_turns: list[CandidateSentence],
        rejected: list[CandidateSentence],
        profile: SpeakerMatchProfile,
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
            if match.tier != "weak":
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
                and candidate.window_vote_ratio >= 0.35
                and match.secondary.window_vote_ratio >= 0.35
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

    def _write_outputs(
        self,
        accepted: list[CandidateSentence],
        rejected: list[CandidateSentence],
        stem: Path,
        target_name: str,
        paths: dict[str, Path],
        progress: ProgressCallback,
    ) -> tuple[Path, Path, Path]:
        output_dir = paths["output"]
        output_dir.mkdir(parents=True, exist_ok=True)
        audio_dir = output_dir / "audio"
        text_dir = output_dir / "text"
        audio_dir.mkdir(exist_ok=True)
        text_dir.mkdir(exist_ok=True)
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
        progress: ProgressCallback | None = None,
        job_id: str | None = None,
    ) -> PipelineResult:
        progress = progress or _noop_progress
        references = [Path(path) for path in references if path]
        target = Path(target)
        if not references:
            raise ValueError("请至少提供一段参考音频")
        if not target.exists():
            raise FileNotFoundError(target)
        job_id = job_id or time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        paths = self._job_paths(job_id)
        paths["root"].mkdir(parents=True, exist_ok=True)
        try:
            progress(0.01, "检查本地模型")
            normalized_refs, reference_durations = self._prepare_references(references, paths, progress)
            target_normalized = normalize_audio(target, paths["normalized_target"], sample_rate=44100, stereo=True)
            target_duration = pad_for_separator(target_normalized)

            progress(0.14, "提取参考与目标人声：准备 UVR 分块")
            separator = UVR5Separator(self.device)
            separation_items = [
                (path, paths["reference_stems"] / f"reference_{index:03d}_vocals.wav")
                for index, path in enumerate(normalized_refs, start=1)
            ]
            separation_items.append((target_normalized, paths["stems"] / "target_vocals.wav"))
            try:
                separated = separator.separate_many(
                    separation_items,
                    progress=lambda value, message: progress(0.14 + 0.24 * value, message),
                )
            except ValueError as exc:
                if "有效信号" not in str(exc):
                    raise
                # A fully silent target has no sentence by definition. Return
                # the same empty artifact shape as a music-only target.
                output_dir = paths["output"]
                output_dir.mkdir(parents=True, exist_ok=True)
                manifest, transcript, archive = self._write_outputs(
                    [], [], target_normalized, target.name, paths, progress
                )
                progress(1.0, "完成：目标音频没有人声")
                return PipelineResult(job_id, output_dir, archive, [], [], manifest, transcript)
            reference_stems = separated[:-1]
            stem = separated[-1]
            for reference_stem, original_duration in zip(reference_stems, reference_durations):
                trim_audio_in_place(reference_stem, original_duration)
            trim_audio_in_place(stem, target_duration)

            progress(0.40, "正在检测讲话范围（STT 将在筛选完成后执行）")
            duration = probe_duration(stem)
            vad_tools = FunASRTools(self.device)
            vad_map = vad_tools.vad_many(
                [*reference_stems, stem],
                progress=lambda value, message: progress(0.40 + 0.08 * value, message),
            )
            progress(0.48, "VAD 完成：先过滤多人同时说话和歌声")
            reference_clips = self._make_reference_clips(reference_stems, vad_map, paths)
            # VAD spans are the speech-only source of truth.  A longer silence
            # starts a new dialogue block for diarization; the silence itself
            # is never emitted as an output segment.
            vad_spans = self._merge_vad_spans(
                vad_map.get(stem, []),
                gap=0.45,
            )
            if not vad_spans:
                manifest, transcript, archive = self._write_outputs(
                    [],
                    [],
                    stem,
                    target.name,
                    paths,
                    progress,
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
            singing_detector = None
            if self.options.use_singing_detector:
                singing_detector = SingingDetector("cpu")

            clean_spans = vad_spans
            if overlap_detector is not None:
                clean_spans, overlap_spans = overlap_detector.clean_spans(
                    stem,
                    clean_spans,
                    self.options.overlap_threshold,
                    progress=lambda value, message: progress(0.48 + 0.035 * value, message),
                )
                rejected.extend(
                    CandidateSentence(
                        span.start,
                        span.end,
                        "",
                        reject_reason="检测到多人同时发声",
                        overlap_score=self.options.overlap_threshold,
                    )
                    for span in overlap_spans
                )
            if singing_detector is not None:
                clean_spans, singing_spans = singing_detector.clean_spans(
                    stem,
                    clean_spans,
                    self.options.singing_threshold,
                    progress=lambda value, message: progress(0.515 + 0.035 * value, message),
                )
                rejected.extend(
                    CandidateSentence(
                        span.start,
                        span.end,
                        "",
                        reject_reason="检测到唱歌或歌声",
                        singing_score=self.options.singing_threshold,
                    )
                    for span in singing_spans
                )
            progress(
                0.55,
                f"局部过滤完成：剩余 {len(clean_spans)} 个单人讲话范围，"
                f"移除多人 {sum(item.reject_reason == '检测到多人同时发声' for item in rejected)} 段、"
                f"歌声 {sum(item.reject_reason == '检测到唱歌或歌声' for item in rejected)} 段",
            )
            if not clean_spans:
                if singing_detector is not None:
                    singing_detector.close()
                manifest, transcript, archive = self._write_outputs(
                    [], rejected, stem, target.name, paths, progress
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

            verifier = DualSpeakerVerifier(
                self.device,
                status=lambda message: progress(0.56, message),
            )
            try:
                profile = verifier.build_profile(reference_clips, self.options.speaker_threshold)
                progress(0.57, f"参考声纹建立完成，开始切分 {len(clean_spans)} 个单人讲话范围")
                split_result = verifier.split_speaker_spans(
                    stem,
                    clean_spans,
                    minimum_turn_seconds=0.30,
                    context_seconds=1.20,
                    scan_hop_seconds=0.30,
                    minimum_similarity_drop=0.08,
                    minimum_separation_seconds=0.70,
                    progress=lambda value, message: progress(0.57 + 0.11 * value, message),
                )
                progress(0.68, f"换人切分完成：得到 {len(split_result)} 个独立说话回合")
                target_waveform = load_mono(stem, 16000)
                effective_threshold = max(
                    self.options.speaker_threshold,
                    profile.primary.suggested_threshold,
                )
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
                    match: SpeakerMatchDecision | None = None
                    if candidate.duration < self.options.min_sentence_seconds:
                        candidate.reject_reason = "说话回合过短"
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
                    scored_turns.append((span, candidate, match))
                    progress(
                        0.68 + 0.04 * index / max(1, len(split_result)),
                        f"初步匹配目标人物 {index}/{len(split_result)}",
                    )

                recovery_turns = [
                    (span, candidate, match)
                    for span, candidate, match in scored_turns
                    if match is not None
                    and self._needs_boundary_recovery(match, candidate.duration)
                ]
                recovery_boundaries: list[SpeakerBoundary] = []
                if recovery_turns:
                    progress(
                        0.72,
                        f"发现 {len(recovery_turns)} 个疑似混合回合，正在局部补切",
                    )
                    assert verifier.secondary is not None
                    recovery_splitter = LocalSpeakerTurnSplitter(
                        verifier.primary,
                        secondary=verifier.secondary,
                    )
                    recovery_boundaries = recovery_splitter.detect_speaker_boundaries(
                        stem,
                        [item[0] for item in recovery_turns],
                        context_seconds=0.60,
                        scan_hop_seconds=0.10,
                        primary_candidate_threshold=0.78,
                        minimum_separation_seconds=0.30,
                        progress=lambda value, message: progress(
                            0.72 + 0.025 * value, message
                        ),
                    )

                recovery_lookup = {id(candidate): index for index, (_span, candidate, _match) in enumerate(recovery_turns, start=1)}
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
                            effective_threshold,
                            progress,
                            recovery_index,
                            len(recovery_turns),
                        )
                        if recovered is not None:
                            accepted_candidates, discarded = recovered
                            accepted_turns.extend(accepted_candidates)
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

                promoted = self._expand_recall_candidates(
                    scored_turns,
                    accepted_turns,
                    rejected,
                    profile,
                )
                if promoted:
                    progress(
                        0.76,
                        f"自适应召回：根据本文件已确认声纹追加 {promoted} 个候选回合",
                    )

                progress(
                    0.78,
                    f"声纹筛选完成：保留 {len(accepted_turns)} 个目标人物回合，"
                    f"舍弃 {len(rejected)} 个回合",
                )
            finally:
                verifier.close()
                if singing_detector is not None:
                    singing_detector.close()

            accepted: list[CandidateSentence] = []
            if accepted_turns:
                # Adaptive recall appends candidates after the original turn
                # pass. Whisper sorts spans by time, so sort the owner list
                # first to keep STT diagnostics and speaker metadata paired
                # with the correct audio span.
                accepted_turns = self._merge_adjacent_target_turns(
                    accepted_turns,
                    self.options.min_output_seconds,
                )
                accepted_turns.sort(key=lambda item: (item.start, item.end))
                progress(0.78, f"筛选完成：仅对 {len(accepted_turns)} 个目标人物回合执行 STT")
                segmenter = WhisperSegmenter(self.device)
                transcribed = segmenter.transcribe_spans(
                    stem,
                    [TimeSpan(candidate.start, candidate.end) for candidate in accepted_turns],
                    progress=lambda value, message: progress(0.78 + 0.14 * value, message),
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
                    sentences = self._coalesce_short_sentences(
                        sentences,
                        self.options.min_output_seconds,
                    )
                    sentences.sort(key=lambda item: (item.start, item.end))
                    turn_sentences: list[CandidateSentence] = []
                    for sentence_index, sentence in enumerate(sentences):
                        # Whisper word timestamps often begin at the first
                        # confidently decoded token and end before the final
                        # phoneme. A single-speaker turn is a safer hard fence
                        # for training audio than those token-level edges.
                        if len(sentences) == 1:
                            sentence.start = turn.start
                            sentence.end = turn.end
                        elif sentence_index == 0:
                            sentence.start = turn.start
                        elif sentence_index == len(sentences) - 1:
                            sentence.end = turn.end
                        if sentence.duration < self.options.min_output_seconds:
                            turn.reject_reason = "STT 鍒嗗潡杩囩煭"
                            break
                        sentence.text = sentence.whisper_text
                        sentence.accepted = True
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
                        turn_sentences.append(sentence)
                    if turn.reject_reason == "STT 鍒嗗潡杩囩煭":
                        rejected.append(turn)
                    else:
                        accepted.extend(turn_sentences)
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
            )
            progress(1.0, f"完成：保留 {len(accepted)} 句，舍弃 {len(rejected)} 句")
            return PipelineResult(job_id, paths["output"], archive, accepted, rejected, manifest, transcript)
        finally:
            if self.options.cleanup_work:
                shutil.rmtree(paths["root"], ignore_errors=True)

    def run_many(
        self,
        references: Iterable[str | Path],
        targets: Iterable[str | Path],
        progress: ProgressCallback | None = None,
    ) -> BatchPipelineResult:
        """Process target files sequentially and create one batch download."""
        progress = progress or _noop_progress
        reference_paths = [Path(path) for path in references if path]
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
                result = self.run(reference_paths, target, progress=report_target, job_id=child_job_id)
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
