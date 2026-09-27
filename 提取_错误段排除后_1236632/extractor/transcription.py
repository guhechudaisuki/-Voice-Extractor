from __future__ import annotations

import gc
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from .audio import load_mono, probe_duration
from .config import FFMPEG, PARAFORMER_MODEL, PUNC_MODEL, VAD_MODEL, whisper_snapshot
from .types import CandidateSentence, TimeSpan


STRONG_END_RE = re.compile(r"[。！？!?]+[\"'”’）】》]*$")
FRAGMENT_RE = re.compile(r".*?[。！？!?]+|.+$", re.S)


def is_complete_text(text: str) -> bool:
    return bool(text.strip() and STRONG_END_RE.search(text.strip()))


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "").strip()


def infer_language(text: str, reported: str = "auto") -> str:
    reported = (reported or "auto").lower()
    if re.search(r"[\u3040-\u30ff]", text):
        return "ja"
    if reported in {"chinese", "mandarin"}:
        return "zh"
    if reported in {"japanese", "ja"}:
        return "ja"
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    return reported


def _split_timed_chunk(text: str, start: float, end: float) -> list[tuple[str, float, float]]:
    fragments = [part for part in FRAGMENT_RE.findall(text) if part]
    if len(fragments) <= 1:
        return [(text, start, end)]
    weights = [max(1, len(re.sub(r"[。！？!?\s]", "", part))) for part in fragments]
    total = sum(weights)
    cursor = start
    output: list[tuple[str, float, float]] = []
    for index, (fragment, weight) in enumerate(zip(fragments, weights)):
        fragment_end = end if index == len(fragments) - 1 else cursor + (end - start) * weight / total
        output.append((fragment, cursor, fragment_end))
        cursor = fragment_end
    return output


def _deduplicate_timed_parts(
    parts: list[tuple[str, float, float]],
    tolerance: float = 0.35,
) -> list[tuple[str, float, float]]:
    """Sort global Whisper timestamps and remove overlap duplicates."""
    ordered = sorted(parts, key=lambda item: (item[1], item[2]))
    unique: list[tuple[str, float, float]] = []
    for text, start, end in ordered:
        cleaned = _clean_text(text)
        start = float(start)
        end = float(end)
        if not cleaned or end <= start:
            continue
        duplicate = False
        for index in range(len(unique) - 1, -1, -1):
            previous_text, previous_start, previous_end = unique[index]
            if start - previous_start > max(1.0, tolerance * 2.0):
                break
            if cleaned != _clean_text(previous_text):
                continue
            overlap = min(end, previous_end) - max(start, previous_start)
            same_boundary = (
                abs(start - previous_start) <= tolerance
                and abs(end - previous_end) <= max(tolerance * 2.0, 0.75)
            )
            if overlap > 0.0 or same_boundary:
                duplicate = True
                if end - start > previous_end - previous_start:
                    unique[index] = (text, start, end)
                break
        if not duplicate:
            unique.append((text, start, end))
    return unique


class WhisperSegmenter:
    def __init__(self, device: str = "cuda") -> None:
        self.device = 0 if device == "cuda" and torch.cuda.is_available() else -1

    def transcribe(
        self,
        audio_path: Path,
        max_gap: float = 1.2,
        progress: Callable[[float, str], None] | None = None,
        window_seconds: float = 60.0,
        overlap_seconds: float = 8.0,
    ) -> list[CandidateSentence]:
        from transformers import pipeline

        ffmpeg_dir = str(FFMPEG.parent)
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        progress = progress or (lambda _value, _message: None)
        duration = probe_duration(audio_path)
        if window_seconds <= 0 or overlap_seconds < 0 or overlap_seconds >= window_seconds:
            raise ValueError("Whisper 分块参数无效")
        step = window_seconds if duration <= window_seconds else window_seconds - overlap_seconds
        # Number of starts needed to cover the tail, including files just over
        # one window long.
        count = max(1, int(math.ceil(max(0.0, duration - window_seconds) / step)) + 1)
        progress(0.0, f"Whisper：正在加载模型（共 {count} 个分块）")
        dtype = torch.float16 if self.device >= 0 else torch.float32
        pipe = pipeline(
            "automatic-speech-recognition",
            model=str(whisper_snapshot()),
            device=self.device,
            torch_dtype=dtype,
            chunk_length_s=30,
        )
        # The pipeline accepts a path or an in-memory waveform.  Bounded paths
        # keep long files from becoming one opaque ASR call, while preserving
        # the existing local-FFmpeg loading behavior.
        import tempfile
        from .audio import extract_audio_range

        results: list[dict[str, Any]] = []
        try:
            for index in range(count):
                start = index * step
                end = min(duration, start + window_seconds)
                if end <= start:
                    continue
                with tempfile.TemporaryDirectory(prefix="whisper_chunk_") as temp_dir:
                    chunk_path = Path(temp_dir) / "chunk.wav"
                    progress(
                        index / max(1, count),
                        f"Whisper 分块 {index + 1}/{count}：开始（{start:.1f}-{end:.1f} 秒）",
                    )
                    extract_audio_range(audio_path, chunk_path, start, end)
                    result = pipe(
                        str(chunk_path),
                        return_timestamps="word",
                        return_language=True,
                        batch_size=1,
                        generate_kwargs={"task": "transcribe"},
                    )
                result["_offset"] = start
                results.append(result)
                progress((index + 1) / max(1, count), f"Whisper 分块 {index + 1}/{count}：完成")
        finally:
            del pipe
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        raw_timed_parts: list[tuple[str, float, float, str]] = []
        for result in results:
            language = str(result.get("language") or "auto").lower()
            offset = float(result.get("_offset", 0.0))
            chunks = result.get("chunks", [])
            for chunk in chunks:
                text = chunk.get("text", "")
                timestamp = chunk.get("timestamp") or (None, None)
                start, end = timestamp
                if start is None:
                    continue
                start = float(start) + offset
                if end is None:
                    end = float(start - offset) + max(0.25, min(1.0, len(text) * 0.18))
                end = float(end) + offset
                # Keep each model timestamp intact until all overlapping
                # windows have been merged; splitting first can turn one
                # boundary token into non-identical fragments and defeat the
                # duplicate check.
                raw_timed_parts.append((text, start, end, language))

        # Assemble sentences only after all windows have been merged.  This
        # prevents an overlap boundary from producing a duplicate or truncated
        # sentence.
        merged_timed_parts: list[tuple[str, float, float, str]] = []
        for language in sorted({item[3] for item in raw_timed_parts}):
            merged = _deduplicate_timed_parts(
                [(text, start, end) for text, start, end, item_language in raw_timed_parts if item_language == language],
                tolerance=min(1.0, overlap_seconds / 4.0),
            )
            merged_timed_parts.extend((*item, language) for item in merged)
        merged_timed_parts.sort(key=lambda item: (item[1], item[2]))
        timed_parts: list[tuple[str, float, float, str]] = []
        for text, start, end, language in merged_timed_parts:
            timed_parts.extend((*part, language) for part in _split_timed_chunk(text, start, end))

        candidates: list[CandidateSentence] = []
        buffer: list[str] = []
        buffer_languages: list[str] = []
        sentence_start: float | None = None
        previous_end: float | None = None
        for text, start, end, part_language in timed_parts:
            if previous_end is not None and start - previous_end > max_gap and buffer:
                buffer = []
                buffer_languages = []
                sentence_start = None
            if sentence_start is None:
                sentence_start = start
            buffer.append(text)
            buffer_languages.append(part_language)
            previous_end = end
            merged = _clean_text("".join(buffer))
            if is_complete_text(merged):
                candidates.append(
                    CandidateSentence(
                        start=sentence_start,
                        end=end,
                        whisper_text=merged,
                        language=infer_language(
                            merged,
                            max(set(buffer_languages), key=buffer_languages.count),
                        ),
                    )
                )
                buffer = []
                buffer_languages = []
                sentence_start = None
        if buffer and sentence_start is not None and previous_end is not None:
            merged = _clean_text("".join(buffer))
            # A tightly trimmed single utterance often lacks terminal
            # punctuation from Whisper. Keep it only when it spans nearly the
            # full input, so arbitrary long-file fragments remain excluded.
            if merged and sentence_start <= 0.35 and duration - previous_end <= 0.45:
                candidates.append(
                    CandidateSentence(
                        start=sentence_start,
                        end=previous_end,
                        whisper_text=merged,
                        language=infer_language(
                            merged,
                            max(set(buffer_languages), key=buffer_languages.count),
                        ),
                    )
                )
        return candidates

    @staticmethod
    def _normalise_spans(
        spans: Sequence[TimeSpan | Sequence[float]],
        duration: float,
    ) -> list[TimeSpan]:
        """Return valid, ordered spans without merging neighbouring turns.

        A span is a hard transcription boundary.  In particular, two spans
        which touch (or even overlap slightly) are intentionally kept as two
        records so that text from different speaker turns can never enter the
        same sentence buffer.
        """
        normalised: list[TimeSpan] = []
        for item in spans:
            try:
                start = float(item.start)
                end = float(item.end)
            except (AttributeError, TypeError, ValueError):
                # Accept a simple ``(start, end)`` pair as a convenience for
                # callers that have not materialised TimeSpan yet.
                try:
                    start = float(item[0])  # type: ignore[index]
                    end = float(item[1])  # type: ignore[index]
                except (IndexError, TypeError, ValueError):
                    continue
            if not math.isfinite(start) or not math.isfinite(end):
                continue
            start = max(0.0, start)
            end = min(max(0.0, duration), end)
            if end <= start:
                continue
            normalised.append(TimeSpan(start, end))
        # Sorting makes output deterministic while retaining every boundary;
        # do not call the VAD span merge helper here.
        return sorted(normalised, key=lambda span: (span.start, span.end))

    @staticmethod
    def _window_plan(
        spans: list[TimeSpan],
        window_seconds: float,
        overlap_seconds: float,
    ) -> list[tuple[int, int, float, float]]:
        """Expand spans into bounded Whisper windows.

        The first two tuple fields are the source-span and window indexes.  A
        separate plan lets progress be based on actual model calls and keeps
        the model lifetime independent from the number of spans.
        """
        if window_seconds <= 0 or overlap_seconds < 0 or overlap_seconds >= window_seconds:
            raise ValueError("Whisper 分块参数无效")
        step = window_seconds - overlap_seconds
        plan: list[tuple[int, int, float, float]] = []
        for span_index, span in enumerate(spans):
            span_duration = span.end - span.start
            count = max(1, int(math.ceil(max(0.0, span_duration - window_seconds) / step)) + 1)
            for window_index in range(count):
                start = span.start + window_index * step
                end = min(span.end, start + window_seconds)
                # The count formula should guarantee this, but guarding here
                # avoids an accidental zero-length FFmpeg invocation after
                # floating-point rounding.
                if end > start:
                    plan.append((span_index, window_index, start, end))
        return plan

    @staticmethod
    def _timed_parts_for_result(
        result: dict[str, Any],
        window_start: float,
        window_end: float,
        span: TimeSpan,
    ) -> list[tuple[str, float, float, str]]:
        """Convert one Whisper result to global, span-clamped timed parts."""
        language = str(result.get("language") or "auto").lower()
        chunks = result.get("chunks") or []
        timed: list[tuple[str, float, float, str]] = []
        untimed: list[str] = []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            text = str(chunk.get("text") or "")
            if not _clean_text(text):
                continue
            timestamp = chunk.get("timestamp")
            if not timestamp or not isinstance(timestamp, (tuple, list)) or len(timestamp) < 2:
                untimed.append(text)
                continue
            local_start, local_end = timestamp[0], timestamp[1]
            try:
                local_start = None if local_start is None else float(local_start)
                local_end = None if local_end is None else float(local_end)
            except (TypeError, ValueError):
                local_start = local_end = None
            if local_start is None:
                untimed.append(text)
                continue
            start = window_start + max(0.0, local_start)
            if local_end is None:
                # Whisper occasionally omits the final timestamp.  Estimate a
                # small duration from the token count, then let the span clamp
                # below provide the hard boundary.
                estimated = max(0.25, min(1.0, len(_clean_text(text)) * 0.18))
                end = start + estimated
            else:
                end = window_start + max(local_start, local_end)
            start = max(span.start, min(span.end, start))
            end = max(span.start, min(span.end, end))
            if end <= start:
                end = min(span.end, start + max(0.05, min(0.5, window_end - window_start)))
            if end > start:
                timed.append((text, start, end, language))

        # Some pipeline versions return only ``text`` (or return a mixture of
        # timed and untimed chunks).  Never lose non-empty text just because a
        # terminal timestamp was absent: cover the current window and let the
        # per-span assembly handle it independently.
        if not timed:
            fallback_text = str(result.get("text") or "")
            if not _clean_text(fallback_text) and untimed:
                fallback_text = "".join(untimed)
            if _clean_text(fallback_text):
                timed.append((fallback_text, span.start if window_start <= span.start else window_start,
                              span.end if window_end >= span.end else window_end, language))
        elif untimed:
            # Preserve untimed words which were not represented by a timed
            # chunk, appending them to the current window rather than dropping
            # speech at a model boundary.
            fallback_text = _clean_text("".join(untimed))
            if fallback_text:
                fallback_start = max(span.start, window_start)
                fallback_end = min(span.end, window_end)
                if fallback_end > fallback_start:
                    timed.append((fallback_text, fallback_start, fallback_end, language))
        return timed

    @staticmethod
    def _candidates_from_span_parts(
        parts: list[tuple[str, float, float, str]],
        span: TimeSpan,
        max_gap: float,
        span_index: int,
    ) -> list[CandidateSentence]:
        """Assemble one span only; never carry a buffer into another span."""
        if not parts:
            return []
        # Deduplicate only within this span.  Window overlap from one span is
        # safe to collapse, while identical text in a neighbouring span is a
        # legitimate separate turn and must remain separate.
        merged: list[tuple[str, float, float, str]] = []
        for language in sorted({item[3] for item in parts}):
            deduped = _deduplicate_timed_parts(
                [(text, start, end) for text, start, end, item_language in parts if item_language == language],
                tolerance=0.35,
            )
            merged.extend((*item, language) for item in deduped)
        merged.sort(key=lambda item: (item[1], item[2]))
        timed_parts: list[tuple[str, float, float, str]] = []
        for text, start, end, language in merged:
            for fragment, fragment_start, fragment_end in _split_timed_chunk(text, start, end):
                fragment_start = max(span.start, min(span.end, fragment_start))
                fragment_end = max(span.start, min(span.end, fragment_end))
                if fragment_end > fragment_start and _clean_text(fragment):
                    timed_parts.append((fragment, fragment_start, fragment_end, language))

        candidates: list[CandidateSentence] = []
        buffer: list[str] = []
        buffer_languages: list[str] = []
        sentence_start: float | None = None
        previous_end: float | None = None

        def flush(*, tail: bool) -> None:
            nonlocal buffer, buffer_languages, sentence_start, previous_end
            if not buffer or sentence_start is None or previous_end is None:
                buffer = []
                buffer_languages = []
                sentence_start = None
                return
            text = _clean_text("".join(buffer))
            if text and previous_end > sentence_start:
                language = max(set(buffer_languages), key=buffer_languages.count) if buffer_languages else "auto"
                candidate = CandidateSentence(
                    start=max(span.start, sentence_start),
                    end=min(span.end, previous_end),
                    whisper_text=text,
                    language=infer_language(text, language),
                )
                candidate.diagnostics["transcription_span_index"] = span_index
                if tail:
                    candidate.diagnostics["transcription_tail"] = True
                candidates.append(candidate)
            buffer = []
            buffer_languages = []
            sentence_start = None

        for text, start, end, language in timed_parts:
            if previous_end is not None and start - previous_end > max_gap:
                # A gap is a sentence/turn boundary even when Whisper omitted
                # punctuation.  Flush rather than discarding the preceding
                # text, then start a fresh buffer inside this same span.
                flush(tail=True)
                previous_end = None
            if sentence_start is None:
                sentence_start = start
            buffer.append(text)
            buffer_languages.append(language)
            previous_end = end
            if is_complete_text(_clean_text("".join(buffer))):
                flush(tail=False)
                previous_end = None
        # Crucially, an unterminated final turn is still a valid result because
        # its speaker boundary was supplied by the caller.
        flush(tail=True)
        return candidates

    def transcribe_spans(
        self,
        audio_path: Path,
        spans: Sequence[TimeSpan | Sequence[float]],
        max_gap: float = 1.2,
        progress: Callable[[float, str], None] | None = None,
        window_seconds: float = 60.0,
        overlap_seconds: float = 8.0,
    ) -> list[CandidateSentence]:
        """Transcribe each supplied speaker span as an independent unit.

        ``spans`` are hard fences: Whisper output from one span is assembled
        and flushed before the next span is processed.  This API is intended
        for the post-speaker-segmentation pipeline, where a change of speaker
        has already produced separate :class:`TimeSpan` values.

        The Whisper pipeline is created exactly once per call and reused for
        all windows.  A non-empty result without terminal punctuation is kept
        because the caller's span boundary, rather than punctuation, defines
        the end of the target speaker turn.
        """
        progress = progress or (lambda _value, _message: None)
        audio_path = Path(audio_path)
        if not spans:
            progress(1.0, "Whisper：没有待识别的目标段")
            return []
        duration = probe_duration(audio_path)
        normalised_spans = self._normalise_spans(list(spans), duration)
        if not normalised_spans:
            progress(1.0, "Whisper：没有有效的目标段")
            return []
        plan = self._window_plan(normalised_spans, window_seconds, overlap_seconds)
        if not plan:
            progress(1.0, "Whisper：没有可识别的目标窗口")
            return []

        from transformers import pipeline

        ffmpeg_dir = str(FFMPEG.parent)
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        progress(0.0, f"Whisper：正在加载模型（{len(normalised_spans)} 个目标段，{len(plan)} 个窗口）")
        dtype = torch.float16 if self.device >= 0 else torch.float32
        pipe = pipeline(
            "automatic-speech-recognition",
            model=str(whisper_snapshot()),
            device=self.device,
            torch_dtype=dtype,
            chunk_length_s=30,
        )

        import tempfile
        from .audio import extract_audio_range

        parts_by_span: dict[int, list[tuple[str, float, float, str]]] = {
            index: [] for index in range(len(normalised_spans))
        }
        windows_per_span = {
            span_index: sum(1 for item in plan if item[0] == span_index)
            for span_index in range(len(normalised_spans))
        }
        try:
            for plan_index, (span_index, window_index, start, end) in enumerate(plan):
                span = normalised_spans[span_index]
                span_window_count = windows_per_span[span_index]
                progress(
                    plan_index / max(1, len(plan)),
                    f"Whisper 目标段 {span_index + 1}/{len(normalised_spans)}，"
                    f"窗口 {window_index + 1}/{span_window_count}：开始（{start:.2f}-{end:.2f} 秒）",
                )
                with tempfile.TemporaryDirectory(prefix="whisper_span_") as temp_dir:
                    chunk_path = Path(temp_dir) / "span.wav"
                    extract_audio_range(audio_path, chunk_path, start, end)
                    result = pipe(
                        str(chunk_path),
                        return_timestamps="word",
                        return_language=True,
                        batch_size=1,
                        generate_kwargs={"task": "transcribe"},
                    )
                if not isinstance(result, dict):
                    result = {}
                parts_by_span[span_index].extend(
                    self._timed_parts_for_result(result, start, end, span)
                )
                progress(
                    (plan_index + 1) / max(1, len(plan)),
                    f"Whisper 目标段 {span_index + 1}/{len(normalised_spans)}，"
                    f"窗口 {window_index + 1}/{span_window_count}：完成",
                )
        finally:
            del pipe
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        candidates: list[CandidateSentence] = []
        for span_index, span in enumerate(normalised_spans):
            candidates.extend(
                self._candidates_from_span_parts(
                    parts_by_span.get(span_index, []), span, max_gap=max_gap, span_index=span_index
                )
            )
        candidates.sort(key=lambda candidate: (candidate.start, candidate.end))
        progress(1.0, f"Whisper：目标段识别完成，得到 {len(candidates)} 条文本")
        return candidates


class FunASRTools:
    def __init__(self, device: str = "cuda") -> None:
        self.device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"

    def vad(self, audio_path: Path, progress: Callable[[float, str], None] | None = None) -> list[TimeSpan]:
        return self.vad_many([audio_path], progress=progress).get(audio_path, [])

    def vad_many(
        self,
        audio_paths: list[Path],
        progress: Callable[[float, str], None] | None = None,
        window_seconds: float = 60.0,
    ) -> dict[Path, list[TimeSpan]]:
        from funasr import AutoModel

        if not audio_paths:
            return {}
        if window_seconds <= 0:
            raise ValueError("VAD 分块时长必须大于 0 秒")
        progress = progress or (lambda _value, _message: None)
        durations = [probe_duration(path) for path in audio_paths]
        chunk_counts = [max(1, int(math.ceil(duration / window_seconds))) for duration in durations]
        total_chunks = sum(chunk_counts)
        completed = 0
        progress(0.0, f"VAD：正在加载模型（共 {total_chunks} 个分块）")
        model = AutoModel(
            model=str(VAD_MODEL),
            device=self.device,
            disable_pbar=True,
            disable_update=True,
            check_latest=False,
        )
        output: dict[Path, list[TimeSpan]] = {}
        try:
            import tempfile
            from .audio import extract_audio_range

            for file_index, (audio_path, duration, chunk_count) in enumerate(
                zip(audio_paths, durations, chunk_counts),
                start=1,
            ):
                spans: list[TimeSpan] = []
                for chunk_index in range(chunk_count):
                    start_offset = chunk_index * window_seconds
                    end_offset = min(duration, start_offset + window_seconds)
                    label = (
                        f"VAD 文件 {file_index}/{len(audio_paths)}，"
                        f"块 {chunk_index + 1}/{chunk_count}"
                    )
                    progress(completed / max(1, total_chunks), f"{label}：开始")
                    with tempfile.TemporaryDirectory(prefix="vad_chunk_") as temp_dir:
                        chunk_path = Path(temp_dir) / "chunk.wav"
                        extract_audio_range(audio_path, chunk_path, start_offset, end_offset)
                        result = model.generate(input=str(chunk_path), batch_size_s=60)
                    values = result[0].get("value", []) if result else []
                    spans.extend(
                        TimeSpan(
                            start_offset + float(start) / 1000.0,
                            min(duration, start_offset + float(end) / 1000.0),
                        )
                        for start, end in values
                    )
                    completed += 1
                    progress(completed / max(1, total_chunks), f"{label}：完成")
                output[audio_path] = spans
            return output
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def transcribe_files(
        self,
        paths: list[Path],
        progress: Callable[[float, str], None] | None = None,
    ) -> dict[Path, str]:
        from funasr import AutoModel

        if not paths:
            return {}
        progress = progress or (lambda _value, _message: None)
        progress(0.0, f"中文 STT：正在加载模型（共 {len(paths)} 句）")
        model = AutoModel(
            model=str(PARAFORMER_MODEL),
            punc_model=str(PUNC_MODEL),
            device=self.device,
            disable_pbar=True,
            disable_update=True,
            check_latest=False,
        )
        output: dict[Path, str] = {}
        try:
            for index, path in enumerate(paths, start=1):
                progress((index - 1) / len(paths), f"中文 STT {index}/{len(paths)}：开始")
                result = model.generate(input=str(path), batch_size_s=60, hotword="")
                output[path] = _clean_text(result[0].get("text", "")) if result else ""
                progress(index / len(paths), f"中文 STT {index}/{len(paths)}：完成")
            return output
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def align_candidates_to_vad(
    candidates: list[CandidateSentence],
    spans: list[TimeSpan],
    audio_duration: float,
    padding: float = 0.10,
    max_extension: float = 0.25,
) -> list[CandidateSentence]:
    aligned: list[CandidateSentence] = []
    for candidate in candidates:
        overlaps = [span for span in spans if span.end > candidate.start and span.start < candidate.end]
        if not overlaps:
            candidate.reject_reason = "时间范围内未检测到讲话"
            aligned.append(candidate)
            continue
        first, last = overlaps[0], overlaps[-1]
        start_extension = min(max_extension, max(0.0, candidate.start - first.start))
        end_extension = min(max_extension, max(0.0, last.end - candidate.end))
        candidate.start = max(0.0, candidate.start - start_extension - padding)
        candidate.end = min(audio_duration, candidate.end + end_extension + padding)
        aligned.append(candidate)
    return aligned
