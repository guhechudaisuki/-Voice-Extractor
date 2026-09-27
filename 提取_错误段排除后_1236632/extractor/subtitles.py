"""Optional subtitle timing hints. Text is provenance, never a transcript."""

from __future__ import annotations

import re
import sys
from bisect import bisect_left
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Sequence

from .types import TimeSpan


SUBTITLE_SUFFIXES = {".ass", ".ssa", ".srt", ".vtt"}


def validate_subtitle_bindings(targets, bindings) -> dict[Path, Path]:
    """Validate the whole batch before any source starts expensive processing."""
    target_paths = {Path(target).resolve() for target in targets}
    result = {}
    for target, subtitle in (bindings or {}).items():
        target_path, subtitle_path = Path(target).resolve(), Path(subtitle).resolve()
        if target_path not in target_paths:
            raise ValueError(f"字幕绑定的目标不在本批次中：{target_path.name}")
        read_subtitles(subtitle_path)
        result[target_path] = subtitle_path
    return result


@dataclass(frozen=True)
class SubtitleCue:
    index: int
    start: float
    end: float
    text: str
    kind: str = "speech"


def read_subtitles(path: Path) -> list[SubtitleCue]:
    if path.suffix.lower() not in SUBTITLE_SUFFIXES:
        raise ValueError("字幕仅支持 ASS、SSA、SRT、VTT")
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("字幕文件超过 16 MB")
    # The parser ships with the application; no model/runtime download is needed.
    vendor = str(Path(__file__).resolve().parents[1] / "vendor")
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    import pysubs2

    content = path.read_bytes()
    encodings = ("utf-16",) if content[:2] in (b"\xff\xfe", b"\xfe\xff") else ("utf-8-sig", "gb18030")
    for encoding in encodings:
        try:
            text = content.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("无法识别字幕编码，请使用 UTF-8 字幕")
    parsed = pysubs2.SSAFile.from_string(text, format_=path.suffix.lower().lstrip(".").replace("ssa", "ass"))
    cues = []
    seen = set()
    for index, event in enumerate(parsed, 1):
        if event.is_comment or event.is_drawing or event.end <= event.start:
            continue
        plain = event.plaintext.strip()
        if not plain:
            continue
        kind = "speech"
        if re.search(r"(?:oped|karaoke|lyrics|song|歌词|歌詞|歌曲|片头|片尾)", event.style, re.I) or re.search(r"\\[kK](?:f|o)?\d", event.text):
            kind = "lyrics"
        elif re.search(r"\\(?:pos|move)\(", event.text) or re.search(r"(?:sign|title|staff|注释|招牌)", event.style, re.I):
            kind = "screen_text"
        key = (event.start, event.end, kind)
        if key in seen:
            continue
        seen.add(key)
        cues.append(SubtitleCue(index, event.start / 1000, event.end / 1000, plain, kind))
    if not cues:
        raise ValueError("字幕中没有有效的时间轴条目")
    return sorted(cues, key=lambda cue: (cue.start, cue.end))


def _nearest(values: Sequence[float], value: float) -> float:
    index = bisect_left(values, value)
    return min(values[max(0, index - 1):index + 1], key=lambda item: abs(item - value))


def _intersection(left: TimeSpan, right: TimeSpan) -> float:
    return max(0.0, min(left.end, right.end) - max(left.start, right.start))


@dataclass
class SubtitleGuide:
    cues: list[SubtitleCue]
    filename: str
    offset: float = 0.0
    aligned: bool = False
    anchors: list[tuple[float, float]] = field(default_factory=list)
    report: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "SubtitleGuide":
        cues = read_subtitles(path)
        return cls(cues, path.name, report={
            "file": path.name,
            "cue_count": len(cues),
            "lyrics_hints": sum(cue.kind == "lyrics" for cue in cues),
            "screen_text_hints": sum(cue.kind == "screen_text" for cue in cues),
            "text_used_for_stt": False,
        })

    def calibrate(self, speech: Sequence[TimeSpan]) -> None:
        """Estimate one offset from paired onset/offset anchors, not nearest silence alone."""
        self.aligned = False
        self.anchors = []
        starts = sorted(span.start for span in speech)
        ends = sorted(span.end for span in speech)
        cues = [cue for cue in self.cues if cue.kind == "speech" and 0.6 <= cue.end - cue.start <= 12]
        if not starts or len(cues) < 6:
            self.report.update(status="insufficient_anchors", aligned=False)
            return
        def anchors_for(shift):
            pairs = []
            for cue in cues:
                ds = _nearest(starts, cue.start + shift) - cue.start
                de = _nearest(ends, cue.end + shift) - cue.end
                if abs(ds - shift) <= 0.25 and abs(de - shift) <= 0.25 and abs(ds - de) <= 0.30:
                    pairs.append(((cue.start + cue.end) / 2, (ds + de) / 2))
            return pairs

        trials = []
        for step in range(-30, 31):
            shift = step * 0.05
            pairs = anchors_for(shift)
            trials.append((len(pairs), -abs(shift), shift, pairs))
        _, _, shift, pairs = max(trials, key=lambda row: row[:2])
        if pairs:
            shift = median(pair[1] for pair in pairs)
            pairs = [pair for pair in pairs if abs(pair[1] - shift) <= 0.20]
        # VAD often joins several subtitle lines. An absolute cue coverage
        # quota rejects valid timelines; compare against off-time coincidences.
        controls = [len(anchors_for(value)) for value in (-20, -15, -10, -5, 5, 10, 15, 20)]
        chance = median(controls)
        supported = len(pairs) >= max(6, 2 * chance + 3)
        spread = max((pair[0] for pair in pairs), default=0) - min((pair[0] for pair in pairs), default=0)
        self.aligned = supported and spread >= 10
        self.offset = round(shift, 4) if self.aligned else 0.0
        self.anchors = pairs if self.aligned else []
        self.report.update(
            status="aligned" if self.aligned else "timeline_mismatch",
            aligned=self.aligned, offset_seconds=self.offset,
            anchor_count=len(pairs), anchor_fraction=round(len(pairs) / len(cues), 4),
            off_time_anchor_median=chance,
        )

    def cue_span(self, cue: SubtitleCue) -> TimeSpan | None:
        if not self.aligned or cue.kind != "speech":
            return None
        center = (cue.start + cue.end) / 2
        nearby = [shift for time, shift in self.anchors if abs(time - center) <= 45]
        # Do not propagate a global offset across an unsupported edit or drift.
        if len(nearby) < 2:
            return None
        local = median(nearby)
        if abs(local - self.offset) > 0.25:
            return None
        return TimeSpan(max(0.0, cue.start + local), max(0.0, cue.end + local))

    def retry_windows(self, speech: Sequence[TimeSpan], duration: float) -> list[TimeSpan]:
        windows = []
        for cue in self.cues:
            span = self.cue_span(cue)
            if span is None or span.duration > 12 or span.start >= duration:
                continue
            covered = sum(_intersection(span, part) for part in speech)
            if covered >= min(0.35, span.duration * 0.5):
                continue
            window = TimeSpan(max(0.0, span.start - 0.6), min(duration, span.end + 0.6))
            if windows and window.start <= windows[-1].end and window.end - windows[-1].start <= 30:
                windows[-1] = TimeSpan(windows[-1].start, max(windows[-1].end, window.end))
            else:
                windows.append(window)
        return windows

    def groups(self, speech: Sequence[TimeSpan], max_gap: float) -> list[tuple[SubtitleCue, list[TimeSpan]]]:
        """Assign whole VAD islands; never cut a voiced sample at a subtitle timestamp."""
        by_cue: dict[int, list[tuple[int, TimeSpan]]] = {}
        usable = [(cue, self.cue_span(cue)) for cue in self.cues]
        for index, part in enumerate(sorted(speech, key=lambda span: span.start)):
            if part.duration <= 0:
                continue
            choices = []
            for cue, span in usable:
                if span is None or span.start > part.end + 0.45 or span.end < part.start - 0.45:
                    continue
                overlap = _intersection(part, span)
                if overlap >= part.duration * 0.5 and part.start >= span.start - 0.45 and part.end <= span.end + 0.45:
                    choices.append((overlap, cue))
            choices.sort(key=lambda item: item[0], reverse=True)
            if not choices or (len(choices) > 1 and choices[0][0] - choices[1][0] < 0.10):
                continue
            by_cue.setdefault(choices[0][1].index, []).append((index, part))
        groups = []
        for cue in self.cues:
            chunks: list[list[TimeSpan]] = []
            previous_index = -2
            for index, part in by_cue.get(cue.index, []):
                if chunks and index == previous_index + 1 and part.start - chunks[-1][-1].end <= max_gap + 1e-6:
                    chunks[-1].append(part)
                else:
                    chunks.append([part])
                previous_index = index
            groups.extend((cue, chunk) for chunk in chunks)
        return groups

    def annotate(self, candidate) -> None:
        span = TimeSpan(candidate.start, candidate.end)
        matches = []
        for cue in self.cues:
            aligned = self.cue_span(cue)
            if aligned is not None and _intersection(span, aligned) > 0.10:
                matches.append({**asdict(cue), "aligned_start": aligned.start, "aligned_end": aligned.end})
        candidate.diagnostics["subtitle_reference"] = matches
        candidate.diagnostics["subtitle_text_used_for_stt"] = False
