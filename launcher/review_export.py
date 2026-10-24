"""Export reviewed audio and optional portable material without a web service."""
from pathlib import Path
import tempfile
import zipfile

from launcher.training_material import (export_package, partial_keep_segments,
                                        reviewable, validated_rows)
from launcher.training_workspace import Workspace


def _clip_text(row: dict) -> str:
    text = str(row.get("text") or row.get("whisper_text") or "").strip()
    # GPT-SoVITS 训练清单按行解析，文本中的竖线与换行会破坏格式。
    return text.replace("|", "／").replace("\n", " ")


def _partial_clip_text(row: dict, review: dict, keep: tuple[float, float]) -> str:
    """partial 保留段对应的台词切片：音字同步，杜绝整句脏文本进 train.list。

    优先级：
    1. 用户在标注里显式填写的 partial_text；
    2. 字幕/分片子片段（diagnostics.subtitle_reference 等带绝对时间的
       {start, end, text} 条目）：只拼接与保留段绝对区间重叠 >= 50% 的
       子片段台词，按时间顺序连接；
    3. 无任何子片段时间戳时，按保留区间在整句中的相对位置对整句文本做
       等比例字符切片（保底）。
    """
    duration = max(1e-6, float(row.get("end") or 0) - float(row.get("start") or 0))
    explicit = str(review.get("partial_text") or "").strip()
    if explicit:
        return explicit.replace("|", "／").replace("\n", " ")
    abs_start = float(row.get("start") or 0) + float(keep[0])
    abs_end = float(row.get("start") or 0) + float(keep[1])
    fragments = []
    diagnostics = row.get("diagnostics") or {}
    for key in ("subtitle_reference", "subtitle_cues", "stt_fragments"):
        entries = diagnostics.get(key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            text = str(entry.get("text") or "").strip()
            start = entry.get("aligned_start", entry.get("start"))
            end = entry.get("aligned_end", entry.get("end"))
            if not text or not isinstance(start, (int, float)) \
                    or not isinstance(end, (int, float)):
                continue
            overlap = min(abs_end, float(end)) - max(abs_start, float(start))
            if overlap >= 0.5 * max(1e-6, float(end) - float(start)):
                fragments.append((float(start), float(end), text))
    if fragments:
        fragments.sort()
        text = "".join(piece for _s, _e, piece in fragments)
        return text.replace("|", "／").replace("\n", " ").strip()
    full = _clip_text(row)
    if not full:
        return ""
    left = max(0.0, min(1.0, keep[0] / duration))
    right = max(0.0, min(1.0, keep[1] / duration))
    begin = int(len(full) * left)
    finish = max(begin + 1, int(len(full) * right))
    return full[begin:finish].strip()


def export_review_zip(batch: Path, manifest: dict, reviews: dict, destination: Path,
                      include_material: bool = False, *, gpt_sovits_list: bool = False,
                      speaker: str = "target", language: str = "JA") -> int:
    rows = validated_rows(batch, manifest, reviews) if include_material else reviewable(batch, manifest)
    correct = [row for row in rows if reviews.get(str(row[0]), {}).get('label') == 'correct']
    partial = [row for row in rows if reviews.get(str(row[0]), {}).get('label') == 'partial']
    if not correct and not partial and not include_material:
        raise ValueError('没有标记为目标匹配的片段')
    # Finish the complete archive before replacing a user-selected destination.
    workspace = Workspace()
    exported: list[tuple[str, str]] = []
    try:
        temporary = workspace.root
        archive_path = temporary / 'reviewed.zip'
        if include_material:
            # A single ZIP contains the labels, all reviewed clips (including
            # negatives), and reference audio. No nested archive or unpack step.
            export_package(batch, manifest, reviews, archive_path)
        else:
            from launcher.audio_export import write_clip
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
                written = set()
                for ordinal, row, audio, relative, *_ in correct:
                    if relative in written:
                        continue
                    archive.write(audio, relative)
                    written.add(relative)
                    exported.append((relative, _clip_text(row)))
                    text = audio.with_suffix('.txt')
                    if text.is_file():
                        archive.write(text, str(Path(relative).with_suffix('.txt')).replace('\\', '/'))
                # partial（部分错人）自动裁出干净保留段：好肉不再整句丢弃。
                for ordinal, row, audio, relative, *_ in partial:
                    review = reviews.get(str(ordinal), {})
                    keep = max(partial_keep_segments(row, review),
                               key=lambda seg: seg[1] - seg[0], default=None)
                    if keep is None or keep[1] - keep[0] < 1.0:
                        continue
                    piece = temporary / f'partial_{ordinal:06d}.wav'
                    try:
                        write_clip(audio, piece, keep[0], keep[1], normalize_level=True)
                    except (ValueError, OSError, RuntimeError):
                        continue
                    entry = f'partial/{piece.name}'
                    archive.write(piece, entry)
                    piece_text = _partial_clip_text(row, review, keep)
                    exported.append((entry, piece_text))
                    archive.writestr(f'partial/{piece.stem}.txt', piece_text)
                if gpt_sovits_list and exported:
                    # GPT-SoVITS 标准训练索引：wav_path|speaker|language|text
                    lines = [f"{path}|{speaker}|{language}|{text}"
                             for path, text in exported]
                    archive.writestr('train.list', '\n'.join(lines) + '\n')
        save_file(archive_path, destination)
    finally:
        workspace.close()
    return len(exported) if not include_material else len(correct)


def save_file(source: Path, destination: Path):
    import os
    import shutil
    descriptor, name = tempfile.mkstemp(prefix='.voice_export_', suffix='.tmp', dir=destination.parent)
    partial = Path(name)
    try:
        with os.fdopen(descriptor, 'wb') as output, source.open('rb') as input_file:
            shutil.copyfileobj(input_file, output, length=1024 * 1024)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
