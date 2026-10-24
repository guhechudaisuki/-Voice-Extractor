"""Portable, checked review packages for local identity training."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
import zipfile

LABELS = {"correct", "wrong", "partial", "singing", "overlap", "skip"}
SCHEMA = 2


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def reviewable(batch_dir: Path, manifest: dict) -> list[tuple[int, dict, Path, str]]:
    result = []
    root = batch_dir.resolve()
    sentences = manifest.get("sentences")
    if not isinstance(sentences, list):
        return result
    for ordinal, row in enumerate(sentences, 1):
        if not isinstance(row, dict):
            continue
        diagnostics = row.get("diagnostics") or {}
        if not isinstance(diagnostics, dict):
            diagnostics = {}
        # Tier B（边缘待捡漏）片段同样可试听：音频由流水线写入 _borderline/。
        if not (row.get("accepted") or diagnostics.get("cluster_removed")
                or diagnostics.get("cluster_tier") == "B"):
            continue
        relative = str(row.get("audio_file") or "").replace("\\", "/")
        if not relative:
            continue
        clip = (root / relative).resolve()
        if root not in clip.parents:
            raise ValueError(f"片段 {ordinal} 路径越界")
        if clip.is_file():
            result.append((ordinal, row, clip, relative))
    return result


def partial_keep_segments(row: dict, review: dict) -> list[tuple[float, float]]:
    """partial 标注的干净保留区间（相对片段起点的秒数）。

    错人区间 [partial_start, partial_end] 之外的连续部分即保留段。
    """
    duration = float(row.get("end") or 0) - float(row.get("start") or 0)
    start, end = float(review.get("partial_start")), float(review.get("partial_end"))
    segments = []
    if start > 0.05:
        segments.append((0.0, start))
    if end < duration - 0.05:
        segments.append((end, duration))
    return segments


def _cut_to_archive(archive: zipfile.ZipFile, clip: Path, start: float, end: float,
                    name: str, workspace: Path) -> Path | None:
    """从片段音频切出 [start, end]（相对秒）写入压缩包；带 15ms 淡入淡出。"""
    if end - start < 0.5:
        return None
    from extractor.audio import write_clip
    temporary = workspace / (name.replace("/", "_"))
    try:
        write_clip(clip, temporary, start, end, normalize_level=True)
    except (ValueError, OSError, RuntimeError):
        return None
    archive.write(temporary, name)
    return temporary


def validated_rows(batch_dir: Path, manifest: dict, reviews: dict) -> list[tuple[int, dict, Path, str, dict]]:
    rows = reviewable(batch_dir, manifest)
    if not rows:
        raise ValueError("此批次没有可试听片段")
    if not isinstance(reviews, dict):
        raise ValueError("标注格式无效")
    result = []
    for ordinal, row, clip, relative in rows:
        item = reviews.get(str(ordinal))
        if not isinstance(item, dict) or item.get("label") not in LABELS:
            raise ValueError(f"还有未打标片段：第 {ordinal} 段")
        if item.get("ordinal") != ordinal or item.get("filename") != clip.name:
            raise ValueError(f"第 {ordinal} 段标注与音频不对应")
        if item["label"] == "partial":
            start, end = item.get("partial_start"), item.get("partial_end")
            duration = float(row.get("end") or 0) - float(row.get("start") or 0)
            if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (start, end)) or not 0 <= start < end <= duration + .02:
                raise ValueError(f"第 {ordinal} 段的错人区间无效")
        result.append((ordinal, row, clip, relative, item))
    return result


def clean_reviews(reviews: dict) -> dict:
    if not isinstance(reviews, dict):
        raise ValueError('标注格式无效')
    fields = {"ordinal", "filename", "label", "partial_start", "partial_end", "notes"}
    return {key: {name: value for name, value in item.items() if name in fields}
            for key, item in reviews.items() if isinstance(item, dict)}


def export_package(batch_dir: Path, manifest: dict, reviews: dict, destination: Path) -> dict:
    validated_rows(batch_dir, manifest, reviews)
    try:
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
            result = write_material(archive, batch_dir, manifest, reviews)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return {**result, "path": destination}


def reference_key(checksums) -> str:
    return hashlib.sha256(json.dumps(sorted(set(checksums)), separators=(",", ":")).encode()).hexdigest()


def write_material(archive: zipfile.ZipFile, batch_dir: Path, manifest: dict, reviews: dict) -> dict:
    import tempfile
    from extractor.training_context import load_training_context
    rows = validated_rows(batch_dir, manifest, reviews)
    contexts, items, written, directories = {}, [], set(), {}
    with tempfile.TemporaryDirectory(prefix='voice_material_') as scratch:
        workspace = Path(scratch)
        for ordinal, row, clip, relative, review in rows:
            # Child output folders carry their own episode and target references.
            directory = clip.parent
            while directory != batch_dir.resolve() and not (directory / '_training/context.json').is_file() and not (directory / 'manifest.json').is_file():
                directory = directory.parent
            if directory not in directories:
                context = load_training_context(directory)
                refs = []
                seen_refs = set()
                for path in context['reference_paths']:
                    checksum = digest(path)
                    if checksum in seen_refs:
                        continue
                    seen_refs.add(checksum)
                    name = f"references/{checksum}{path.suffix.lower()}"
                    if name not in written:
                        archive.write(path, name)
                        written.add(name)
                    refs.append({'audio': name, 'sha256': checksum, 'bytes': path.stat().st_size})
                ref_key = reference_key(item['sha256'] for item in refs)
                source_sha = context.get('source_sha256')
                source_name = context.get('source_name') or str(row.get('target') or batch_dir.name)
                context_id = hashlib.sha256(f"{source_sha or source_name}:{ref_key}".encode()).hexdigest()
                contexts[context_id] = {'source_name': source_name, 'source_sha256': source_sha,
                                        'reference_key': ref_key, 'references': refs}
                directories[directory] = context_id
            name = f"audio/{ordinal:06d}.wav"
            archive.write(clip, name)
            items.append({'ordinal': ordinal, 'filename': clip.name, 'audio': name,
                          'sha256': digest(clip), 'bytes': clip.stat().st_size,
                          'context': directories[directory], 'start': row.get('start'), 'end': row.get('end'),
                          'accepted': bool(row.get('accepted')), 'review': clean_reviews({'r': review})['r']})
            # partial（部分错人）自动拆分：错人切片是紧贴目标的困难负例，
            # 干净保留段是正例——一次标注同时喂给强化训练两份证据。
            if review.get('label') == 'partial':
                wrong_start = float(review.get('partial_start'))
                wrong_end = float(review.get('partial_end'))
                pieces = []
                hard_neg = _cut_to_archive(
                    archive, clip, wrong_start, wrong_end,
                    f"audio/{ordinal:06d}_hard_neg.wav", workspace)
                if hard_neg is not None:
                    pieces.append((f"audio/{ordinal:06d}_hard_neg.wav", hard_neg, 'wrong',
                                   wrong_start, wrong_end))
                keep = max(partial_keep_segments(row, review),
                           key=lambda seg: seg[1] - seg[0], default=None)
                if keep is not None:
                    clean_pos = _cut_to_archive(
                        archive, clip, keep[0], keep[1],
                        f"audio/{ordinal:06d}_clean_pos.wav", workspace)
                    if clean_pos is not None:
                        pieces.append((f"audio/{ordinal:06d}_clean_pos.wav", clean_pos, 'correct',
                                       keep[0], keep[1]))
                for entry_name, piece_path, label, seg_start, seg_end in pieces:
                    items.append({'ordinal': ordinal, 'filename': piece_path.name,
                                  'audio': entry_name, 'sha256': digest(piece_path),
                                  'bytes': piece_path.stat().st_size,
                                  'context': directories[directory],
                                  'start': float(row.get('start') or 0) + seg_start,
                                  'end': float(row.get('start') or 0) + seg_end,
                                  'accepted': label == 'correct',
                                  'derived_from': ordinal,
                                  'review': {'label': label, 'ordinal': ordinal, 'filename': clip.name}})
    archive.writestr('material.json', json.dumps({'kind': 'voice_extractor_training_material',
        'schema': SCHEMA, 'batch_id': batch_dir.name, 'contexts': contexts, 'items': items}, ensure_ascii=False, indent=2))
    return {'count': len(items), 'source_count': len(contexts)}


def import_packages(paths: list[Path], destination: Path) -> list[dict]:
    if not paths or any(path.suffix.lower() != '.zip' for path in paths):
        raise ValueError("请选择含标注、片段音频和参考音频的 ZIP，无需解压")
    try:
        return _import_packages(paths, destination)
    except (zipfile.BadZipFile, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("原料文件损坏或索引缺失，请重新导出原料") from exc


def _read_asset(archive, asset, destination, prefix):
    if not isinstance(asset, dict):
        raise ValueError('原料音频索引无效')
    name, checksum = asset.get('audio'), asset.get('sha256')
    if (not isinstance(name, str) or not re.fullmatch(prefix + r'/[a-zA-Z0-9_.-]+', name)
            or not isinstance(checksum, str) or not re.fullmatch(r'[a-f0-9]{64}', checksum)):
        raise ValueError('原料包中的音频路径或校验信息无效')
    info = archive.getinfo(name)
    if not 0 < info.file_size <= 256 * 1024**2 or info.file_size != asset.get('bytes'):
        raise ValueError('原料包中的音频大小无效')
    target = destination / (checksum + Path(name).suffix)
    sha = hashlib.sha256()
    with archive.open(info) as source, target.open('wb') as output:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            sha.update(block)
            output.write(block)
    if sha.hexdigest() != checksum:
        target.unlink(missing_ok=True)
        raise ValueError('原料包音频校验失败')
    return target


def _import_packages(paths: list[Path], destination: Path) -> list[dict]:
    unique, total_bytes = {}, 0
    destination.mkdir(parents=True, exist_ok=True)
    for path in paths:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            total_bytes += sum(item.file_size for item in members)
            if len(members) > 20001 or total_bytes > 8 * 1024**3:
                raise ValueError('原料包超出数量或大小限制')
            names = [item.filename for item in members]
            if 'material.json' not in names:
                raise ValueError('ZIP 不含训练标注，请全部打标后勾选附带训练原料重新导出 ZIP')
            if len(set(names)) != len(names) or archive.getinfo('material.json').file_size > 16 * 1024**2:
                raise ValueError('原料包索引重复或过大')
            data = json.loads(archive.read('material.json'))
            if not isinstance(data, dict) or data.get('kind') != 'voice_extractor_training_material':
                raise ValueError('原料包格式不支持')
            if data.get('schema') == 1:
                raise ValueError('旧版原料未携带目标参考音频，请从原批次重新导出 ZIP')
            if data.get('schema') != SCHEMA or not isinstance(data.get('items'), list) or not isinstance(data.get('contexts'), dict):
                raise ValueError('原料包格式不支持')
            contexts = {}
            for key, value in data['contexts'].items():
                if not isinstance(value, dict) or not isinstance(value.get('references'), list) or not value['references']:
                    raise ValueError('原料缺少目标参考音频')
                refs = value['references']
                ref_paths = sorted(set(_read_asset(archive, asset, destination, 'references') for asset in refs))
                ref_key = reference_key(asset['sha256'] for asset in refs)
                source_sha = value.get('source_sha256')
                if source_sha is not None and (not isinstance(source_sha, str) or not re.fullmatch(r'[a-f0-9]{64}', source_sha)):
                    raise ValueError('原料来源指纹无效')
                contexts[key] = {'reference_paths': list(map(str, ref_paths)), 'reference_key': ref_key,
                    'source_name': str(value.get('source_name') or ''), 'source_sha256': source_sha}
            for item in data['items']:
                if not isinstance(item, dict) or item.get('context') not in contexts:
                    raise ValueError('片段缺少对应的参考音频')
                review = item.get('review') or {}
                if not isinstance(review, dict) or review.get('label') not in LABELS:
                    raise ValueError('原料包中的标注无效')
                if review['label'] == 'partial':
                    start, end = review.get('partial_start'), review.get('partial_end')
                    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (start, end)) or not 0 <= start < end:
                        raise ValueError('原料包中的错人区间无效')
                target = _read_asset(archive, item, destination, 'audio')
                context = contexts[item['context']]
                key = (item['sha256'], context['reference_key'])
                source_group = context['source_sha256'] or f"{data.get('batch_id')}:{context['source_name']}"
                previous = unique.get(key)
                signature = lambda value: tuple(value.get(name) for name in ('label', 'partial_start', 'partial_end'))
                if previous and signature(previous['review']) != signature(review):
                    raise ValueError('相同音频针对相同参考出现冲突标签')
                if previous:
                    previous['source_groups'] = sorted(set(previous['source_groups']) | {source_group})
                else:
                    unique[key] = {**item, **context, 'path': str(target), 'batch_id': data.get('batch_id'),
                                   'source_groups': [source_group], 'review': clean_reviews({'r': review})['r']}
    return list(unique.values())
