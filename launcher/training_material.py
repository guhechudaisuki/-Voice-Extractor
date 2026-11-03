"""Portable, checked review packages for local identity training."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from extractor.model_store import reference_key
import subprocess
import sys
import unicodedata
import zipfile

LABELS = {"correct", "wrong", "partial", "singing", "overlap", "skip"}
SCHEMA = 2


def training_text(row: dict, review: dict | None = None) -> str:
    if review and review.get('label') == 'correct' and review.get('revised_text') is not None:
        return ' '.join(str(review['revised_text']).replace('|', '／').splitlines()).strip()
    for key in ('whisper_text', 'subtitle_text', 'text'):
        value = str(row.get(key) or '').strip()
        if value:
            return ' '.join(value.replace('|', '／').splitlines()).strip()
    return ''


def prepare_training_audio(source: Path, destination: Path) -> float:
    """Apply 20 ms linear fades and preserve 1 dB of peak headroom."""
    import numpy as np
    import soundfile as sf

    samples, rate = sf.read(str(source), dtype='float64', always_2d=True)
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError('训练音频为空或包含非有限采样值')
    fade = min(round(rate * 0.020), len(samples) // 2)
    if fade:
        ramp = np.linspace(0.0, 1.0, fade)[:, None]
        samples[:fade] *= ramp
        samples[-fade:] *= ramp[::-1]
    ceiling = np.floor(10 ** (-1.0 / 20) * 32768) / 32768
    peak = float(np.max(np.abs(samples)))
    if peak > ceiling:
        samples *= ceiling / peak
    sf.write(str(destination), samples, rate, subtype='PCM_16', format='WAV')
    return len(samples) / rate


def export_language(row, fallback='JA'):
    value = str(row.get('language') or row.get('language_code') or fallback).upper()
    aliases = {'JAPANESE': 'JA', 'JP': 'JA', 'CHINESE': 'ZH', 'CN': 'ZH',
               'MANDARIN': 'ZH', 'ENGLISH': 'EN', 'KOREAN': 'KO', 'AUTO': fallback.upper()}
    value = aliases.get(value, value)
    if not value or any(char in value for char in '|\r\n'):
        raise ValueError('语言代码无效')
    return value


def write_training_manifests(archive, records: list[dict], speaker: str) -> dict:
    if not speaker.strip() or any(char in speaker for char in '|\r\n'):
        raise ValueError('说话人不可为空或包含索引分隔符')
    valid = [record for record in records if record['duration_seconds'] > 0
             and any(unicodedata.category(char)[0] in 'LN' for char in record['text'])]
    lines = [f"{record['path']}|{speaker}|{export_language(record)}|{record['text']}" for record in valid]
    archive.writestr('slicer_opt.list', ''.join(line + '\n' for line in lines))
    metadata = [{'audio_filepath': record['path'], 'text': record['text'],
                 'duration': record['duration_seconds'], 'speaker': speaker, 'language': export_language(record)} for record in valid]
    archive.writestr('metadata.jsonl', ''.join(json.dumps(item, ensure_ascii=False,
                     allow_nan=False) + '\n' for item in metadata))
    return {'valid_count': len(valid),
            'total_duration_seconds': sum(record['duration_seconds'] for record in valid),
            'slicer_opt_path': 'slicer_opt.list', 'metadata_path': 'metadata.jsonl'}


def review_rank(row: dict) -> float:
    """Rank a sentence without ever changing its manifest ordinal."""
    diagnostics = row.get('diagnostics') or {}
    for value in (diagnostics.get('tier_b_rank_score'),
                  diagnostics.get('calibrated_fusion_score'), row.get('speaker_score')):
        if isinstance(value, (int, float)) and math.isfinite(value):
            return float(value)
    return -math.inf


def information_rank(row: dict) -> float:
    diagnostics = row.get('diagnostics') or {}
    value = (diagnostics.get('identity_heatmap') or {}).get('information_score')
    if isinstance(value,(int,float)) and math.isfinite(value):
        return float(value)
    value = review_rank(row)
    return -abs(value-.5) if math.isfinite(value) else 0.


def high_confidence_borderline(row: dict) -> bool:
    diagnostics = row.get('diagnostics') or {}
    if not isinstance(diagnostics, dict) or diagnostics.get('cluster_tier') != 'B':
        return False
    if (diagnostics.get('local_classifier_multi_veto')
            or diagnostics.get('cluster_source_identity_conflict')
            or diagnostics.get('cluster_reason') == 'local_classifier_multi_veto'
            or 'local_classifier_multi_veto' in str(row.get('reject_reason') or '')):
        return False
    if diagnostics.get('clean_core_certificate'):
        return True
    # Aggregate certificates may describe other turns; require full coverage.
    certificates = diagnostics.get('clean_core_certificates')
    start, end = row.get('start'), row.get('end')
    if (isinstance(certificates, dict)
            and all(isinstance(value, (int, float)) and math.isfinite(value) for value in (start, end))
            and end > start):
        for spans in certificates.values():
            if not isinstance(spans, (list, tuple)):
                continue
            for span in spans:
                if (isinstance(span, (list, tuple)) and len(span) == 2
                        and all(isinstance(value, (int, float)) and math.isfinite(value) for value in span)
                        and span[0] <= start + 1e-6 and span[1] >= end - 1e-6):
                    return True
    return any(isinstance(value, (int, float)) and math.isfinite(value) and value >= limit
               for value, limit in ((diagnostics.get('tier_b_rank_score'), 1.38),
                                    (diagnostics.get('calibrated_fusion_score'), .70)))


def _speaker_manifest(root: Path, folder: Path):
    path = next((path for path in (folder / 'manifest.json', folder / 'batch_manifest.json',
                 root / 'batch_manifest.json', root / 'manifest.json') if path.is_file()), None)
    return (json.loads(path.read_text(encoding='utf-8')), path.parent) if path else ({}, folder)


def _speaker_operation_worker(batch: Path, operation: str, cluster_ids=None):
    """Use the same installed runtime as desktop audio export, outside the frozen UI."""
    resource = Path(sys.executable).resolve().parent / 'resource'
    python = resource / 'python/python.exe'
    script = (
        'import json,sys; from pathlib import Path; '
        'from launcher.training_material import discover_batch_speakers, export_batch_seeds; '
        'a=json.loads(sys.argv[1]); '
        'r=(discover_batch_speakers(Path(a["batch"])) if a["operation"] == "discover" '
        'else export_batch_seeds(Path(a["batch"]), a["cluster_ids"])); '
        'print(json.dumps(r, ensure_ascii=False))')
    payload = json.dumps({'batch': str(batch.resolve()), 'operation': operation,
                          'cluster_ids': cluster_ids})
    result = subprocess.run([str(python), '-B', '-X', 'utf8', '-c', script, payload],
        cwd=resource, env=os.environ.copy(), capture_output=True, text=True,
        encoding='utf-8', errors='replace', timeout=120,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result.returncode:
        raise RuntimeError('角色探索或种子导出失败：' + (result.stderr or result.stdout)[-1000:])
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise RuntimeError('角色探索或种子导出返回了无效结果') from exc


def discover_batch_speakers(batch: Path) -> list[dict]:
    """Adapt the public discovery contract for both UI clients, scoped by episode."""
    if getattr(sys, 'frozen', False):
        return _speaker_operation_worker(batch, 'discover')
    from dataclasses import asdict, is_dataclass
    from extractor.nextgen.speaker_discovery import discover_episode_speakers
    root = batch.resolve()
    speakers = []
    for path in sorted(root.rglob('cluster_report.json')):
        if root not in path.resolve().parents:
            continue
        report = json.loads(path.read_text(encoding='utf-8'))
        episode = path.parent.relative_to(root).as_posix()
        manifest, audio_base = _speaker_manifest(root, path.parent)
        sentences = manifest.get('sentences', [])
        if audio_base == root and path.parent != root:
            sentences = [row for row in sentences if str(row.get('audio_file', '')).replace('\\', '/').startswith(episode + '/')]
        for speaker in discover_episode_speakers(report, max_seeds=4):
            data = asdict(speaker) if is_dataclass(speaker) else dict(speaker)
            data.update(episode=episode, key=f"{episode}::{data['cluster_id']}")
            if not data.get('sample_texts'):
                previews = []
                for seed in data.get('recommended_seeds', ()):
                    start, end = seed.get('span') or (seed.get('start'), seed.get('end'))
                    for row in sentences:
                        text = str(row.get('text') or row.get('whisper_text') or '').strip()
                        if (text and start is not None and end is not None
                                and float(row.get('end', 0)) > start and float(row.get('start', 0)) < end
                                and text not in previews):
                            previews.append(text)
                data['sample_texts'] = previews[:4]
            speakers.append(data)
    return speakers


def export_batch_seeds(batch: Path, cluster_ids=None) -> dict:
    """Export selected episode-scoped roles; recover available seeds from clips."""
    if getattr(sys, 'frozen', False):
        return _speaker_operation_worker(batch, 'export', cluster_ids)
    from extractor.nextgen.speaker_discovery import export_speaker_seeds
    from launcher.audio_export import write_clip
    root = batch.resolve()
    speakers = discover_batch_speakers(root)
    selected = speakers
    if cluster_ids is not None:
        selected = []
        for identifier in cluster_ids:
            matches = [speaker for speaker in speakers
                       if speaker['key'] == str(identifier) or str(speaker['cluster_id']) == str(identifier)]
            if len(matches) != 1:
                raise ValueError(f'角色簇不存在或跨集重名，请使用完整角色键：{identifier}')
            if matches[0] not in selected:
                selected.append(matches[0])
    exported, warnings = [], []
    for episode in dict.fromkeys(speaker['episode'] for speaker in selected):
        folder = (root / episode).resolve()
        if folder != root and root not in folder.parents:
            raise ValueError('角色簇路径越界')
        report = json.loads((folder / 'cluster_report.json').read_text(encoding='utf-8'))
        roles = [speaker for speaker in selected if speaker['episode'] == episode and not speaker.get('is_mixed')]
        ids = {int(speaker['cluster_id']) for speaker in roles}
        if any(identifier < 0 for identifier in ids):
            raise ValueError('角色簇 ID 必须为非负整数')
        report = {**report, 'clusters': [cluster for cluster in report.get('clusters', []) if cluster.get('cluster_id') in ids]}
        manifest, audio_base = _speaker_manifest(root, folder)
        # A multi-episode batch's first input is not necessarily this episode.
        use_manifest_provenance = audio_base == folder or len({speaker['episode'] for speaker in speakers}) == 1
        provenance = (manifest if use_manifest_provenance else report).get('provenance')
        inputs = (provenance or {}).get('inputs') or []
        if not inputs:
            inputs = (report.get('provenance') or {}).get('inputs') or []
            use_manifest_provenance = False
        source = inputs[0] if inputs else None
        if isinstance(source, dict):
            source = source.get('path') or source.get('file')
        source = Path(source) if source else None
        if source is not None and not source.is_absolute():
            source = (audio_base if use_manifest_provenance else folder) / source
        destination = root / 'discovered_seeds'
        if episode != '.':
            destination /= episode
        destination = destination.resolve()
        if root not in destination.parents:
            raise ValueError('种子导出目录越界')
        destination.mkdir(parents=True, exist_ok=True)
        # A pre-existing junction must not redirect writes outside the batch.
        for identifier in ids:
            cluster_dir = (destination / f'cluster_{identifier:02d}').resolve()
            if destination not in cluster_dir.parents:
                raise ValueError('种子角色目录越界')
            if any(cluster_dir not in path.resolve().parents for path in cluster_dir.glob('*.wav')):
                raise ValueError('种子音频路径越界')
        if source is not None and source.is_file():
            try:
                result = export_speaker_seeds(source, report, destination, max_clusters=max(1, len(ids)))
                exported.extend(str(Path(path).resolve()) for paths in result.values() for path in paths[:4])
                continue
            except (OSError, ValueError, RuntimeError) as exc:
                warnings.append(f'{episode} 原轨导出失败，改用已有切片：{exc}')
        # The public exporter consumes an entire stem. When it is gone, use its
        # discovery-selected absolute intervals and translate into saved clips.
        rows = []
        for row in manifest.get('sentences', []):
            relative = str(row.get('audio_file') or '').replace('\\', '/')
            clip = (audio_base / relative).resolve()
            if relative and root in clip.parents and clip.is_file():
                # Aggregate manifests use the original episode as a prefix.
                if audio_base == root and folder != root and folder not in clip.parents:
                    continue
                rows.append((row, clip))
        for speaker in roles:
            out = destination / f"cluster_{speaker['cluster_id']:02d}"
            out.mkdir(parents=True, exist_ok=True)
            for number, seed in enumerate(speaker.get('recommended_seeds', ())[:4], 1):
                span = seed.get('span') or (seed.get('start'), seed.get('end'))
                start, end = span
                if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in span) or end <= start:
                    continue
                match = next(((row, path) for row, path in rows
                              if float(row.get('start', 0)) <= start and float(row.get('end', 0)) >= end), None)
                if match is None:
                    warnings.append(f"{speaker['key']} 的种子 {number} 缺少原轨和完整切片")
                    continue
                row, clip = match
                path = out / f'seed_{number:02d}.wav'
                write_clip(clip, path, start - float(row['start']), end - float(row['start']), normalize_level=True)
                exported.append(str(path.resolve()))
    return {'paths': list(dict.fromkeys(exported)), 'warnings': warnings,
            'output_dir': str(root / 'discovered_seeds')}


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


def partial_intervals(row: dict, review: dict) -> list[tuple[float, float]]:
    """Validate and merge relative contamination intervals; accept legacy pairs."""
    duration = float(row.get('end') or 0) - float(row.get('start') or 0)
    values = review.get('partial_intervals')
    if values is None:
        values = [(review.get('partial_start'), review.get('partial_end'))]
    if not isinstance(values, (list, tuple)) or not values or not math.isfinite(duration) or duration <= 0:
        raise ValueError('错人区间无效')
    intervals = []
    for pair in values:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in pair)):
            raise ValueError('错人区间必须是有限秒数对')
        start, end = pair
        if not 0 <= start < end <= duration + .02:
            raise ValueError('错人区间超出片段范围')
        intervals.append((float(start), min(float(end), duration)))
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def partial_keep_segments(row: dict, review: dict) -> list[tuple[float, float]]:
    duration = float(row.get('end') or 0) - float(row.get('start') or 0)
    cursor, segments = 0., []
    for start, end in partial_intervals(row, review):
        if start > cursor:
            segments.append((cursor, start))
        cursor = end
    if cursor < duration:
        segments.append((cursor, duration))
    return segments


def _cut_to_archive(archive: zipfile.ZipFile, clip: Path, start: float, end: float,
                    name: str, workspace: Path) -> Path | None:
    """从片段音频切出保留区间，添加 20ms 淡入淡出及峰值保护。"""
    if end - start < 0.5:
        return None
    from launcher.audio_export import write_clip
    temporary = workspace / (name.replace("/", "_"))
    try:
        write_clip(clip, temporary, start, end, normalize_level=True)
        prepare_training_audio(temporary, temporary)
    except (ValueError, OSError, RuntimeError):
        return None
    archive.write(temporary, name)
    return temporary


def validated_rows(batch_dir: Path, manifest: dict, reviews: dict, *, require_all: bool = False) -> list[tuple[int, dict, Path, str, dict]]:
    rows = reviewable(batch_dir, manifest)
    if not rows:
        raise ValueError("此批次没有可试听片段")
    if not isinstance(reviews, dict):
        raise ValueError("标注格式无效")
    result = []
    for ordinal, row, clip, relative in rows:
        item = reviews.get(str(ordinal))
        if item is None and not require_all:
            continue
        if not isinstance(item, dict) or item.get("label") not in LABELS:
            raise ValueError(f"第 {ordinal} 段标注无效或未打标")
        if item.get("ordinal") != ordinal or item.get("filename") != clip.name:
            raise ValueError(f"第 {ordinal} 段标注与音频不对应")
        if item["label"] == "partial":
            partial_intervals(row, item)
        result.append((ordinal, row, clip, relative, item))
    if not result:
        raise ValueError("没有已标注的可导出片段")
    return result


def clean_reviews(reviews: dict) -> dict:
    if not isinstance(reviews, dict):
        raise ValueError('标注格式无效')
    fields = {"ordinal", "filename", "label", "partial_start", "partial_end",
              "partial_text", "partial_intervals", "partial_texts", "notes", "revised_text"}
    return {key: {name: value for name, value in item.items() if name in fields}
            for key, item in reviews.items() if isinstance(item, dict)}


def export_package(batch_dir: Path, manifest: dict, reviews: dict, destination: Path,
                   *, speaker: str = 'target', language: str = 'JA') -> dict:
    validated_rows(batch_dir, manifest, reviews)
    try:
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
            result = write_material(archive, batch_dir, manifest, reviews, speaker=speaker, language=language)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return {**result, "path": destination}


def write_material(archive: zipfile.ZipFile, batch_dir: Path, manifest: dict, reviews: dict,
                   *, speaker: str = 'target', language: str = 'JA') -> dict:
    import tempfile
    from extractor.training_context import load_training_context
    rows = validated_rows(batch_dir, manifest, reviews)
    contexts, items, written, directories = {}, [], set(), {}
    records = []
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
                source_name = str(context.get('source_name') or row.get('target') or batch_dir.name)
                source_name = source_name.replace('\\', '/').rsplit('/', 1)[-1]
                context_id = hashlib.sha256(f"{source_sha or source_name}:{ref_key}".encode()).hexdigest()
                contexts[context_id] = {'source_name': source_name, 'source_sha256': source_sha,
                                        'reference_key': ref_key, 'references': refs, 'speaker_name':speaker,
                                        'reference_names':context.get('reference_names', [p.name for p in context['reference_paths']]),
                                        'original_reference_paths':context.get('original_reference_paths', [])}
                if context.get('role_reference_sha256'):
                    contexts[context_id]['role_reference_sha256'] = context['role_reference_sha256']
                directories[directory] = context_id
            name = f"audio/{ordinal:06d}.wav"
            prepared = workspace / f'{ordinal:06d}.wav'
            duration = prepare_training_audio(clip, prepared)
            archive.write(prepared, name)
            text = training_text(row, review)
            archive.writestr(str(Path(name).with_suffix('.txt')).replace('\\', '/'), text)
            if review.get('label') == 'correct':
                records.append({'path': name, 'text': text, 'duration_seconds': duration, 'language': export_language(row, language)})
            items.append({'ordinal': ordinal, 'filename': clip.name, 'audio': name,
                          'sha256': digest(prepared), 'bytes': prepared.stat().st_size,
                          'context': directories[directory], 'start': row.get('start'), 'end': row.get('end'),
                          'accepted': bool(row.get('accepted')), 'review': clean_reviews({'r': review})['r']})
            # Keep partial-derived samples identifiable after package import.
            if review.get('label') == 'partial':
                pieces = []
                for label, kind, intervals in (
                        ('wrong', 'hard_neg', partial_intervals(row, review)),
                        ('correct', 'clean_pos', partial_keep_segments(row, review))):
                    for index, (start, end) in enumerate(intervals, 1):
                        word_audit = None
                        if label == 'correct':
                            from extractor.word_boundaries import partial_word_bounds
                            (start, end), word_audit = partial_word_bounds(
                                row, (start, end), partial_intervals(row, review))
                        suffix = '' if index == 1 else f'_{index:02d}'
                        name = f"audio/{ordinal:06d}_{kind}{suffix}.wav"
                        piece = _cut_to_archive(archive, clip, start, end, name, workspace)
                        if piece is not None:
                            pieces.append((name, piece, label, start, end, word_audit))
                for entry_name, piece_path, label, seg_start, seg_end, word_audit in pieces:
                    from launcher.review_export import _partial_clip_text
                    import soundfile as sf
                    piece_text = _partial_clip_text(row, review if label == 'correct' else {},
                                                    (seg_start, seg_end))
                    archive.writestr(str(Path(entry_name).with_suffix('.txt')).replace('\\', '/'), piece_text)
                    if label == 'correct':
                        info = sf.info(str(piece_path))
                        records.append({'path': entry_name, 'text': piece_text,
                                        'duration_seconds': info.frames / info.samplerate, 'language': export_language(row, language)})
                    items.append({'ordinal': ordinal, 'filename': piece_path.name,
                                  'audio': entry_name, 'sha256': digest(piece_path),
                                  'bytes': piece_path.stat().st_size,
                                  'context': directories[directory],
                                  'start': float(row.get('start') or 0) + seg_start,
                                  'end': float(row.get('start') or 0) + seg_end,
                                  'accepted': label == 'correct',
                                  'derived_from': ordinal,
                                  'word_boundary_refinement': word_audit,
                                  'sample_kind': 'partial_negative' if label == 'wrong' else 'partial_positive',
                                  'review': {'label': label, 'ordinal': ordinal, 'filename': clip.name}})
    archive.writestr('material.json', json.dumps({'kind': 'voice_extractor_training_material',
        'schema': SCHEMA, 'batch_id': batch_dir.name, 'contexts': contexts, 'items': items}, ensure_ascii=False, indent=2))
    stats = write_training_manifests(archive, records, speaker)
    return {'count': len(items), 'source_count': len(contexts), **stats}


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
        package_hash = digest(path)
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            total_bytes += sum(item.file_size for item in members)
            if len(members) > 20001 or total_bytes > 8 * 1024**3:
                raise ValueError('原料包超出数量或大小限制')
            names = [item.filename for item in members]
            if 'material.json' not in names:
                raise ValueError('ZIP 不含训练标注，请标注片段后勾选附带训练原料重新导出 ZIP')
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
                contexts[key]['speaker_name'] = str(value.get('speaker_name') or '未命名角色')[:128]
                for field in ('reference_names', 'original_reference_paths'):
                    names = value.get(field, [])
                    if not isinstance(names, list) or len(names) > 100 or any(not isinstance(n,str) or len(n)>4096 for n in names):
                        raise ValueError('角色参考元数据无效')
                    contexts[key][field] = names
                role_hashes = value.get('role_reference_sha256')
                if role_hashes is not None:
                    if (not isinstance(role_hashes, list) or not role_hashes
                            or any(not isinstance(v, str) or not re.fullmatch(r'[a-f0-9]{64}', v) for v in role_hashes)):
                        raise ValueError('角色参考指纹无效')
                    contexts[key]['role_key'] = reference_key(role_hashes)
                    contexts[key]['role_reference_sha256'] = sorted(set(role_hashes))
            for item in data['items']:
                if not isinstance(item, dict) or item.get('context') not in contexts:
                    raise ValueError('片段缺少对应的参考音频')
                review = item.get('review') or {}
                if not isinstance(review, dict) or review.get('label') not in LABELS:
                    raise ValueError('原料包中的标注无效')
                if review['label'] == 'partial':
                    partial_intervals(item, review)
                target = _read_asset(archive, item, destination, 'audio')
                context = contexts[item['context']]
                key = (item['sha256'], context['reference_key'], context.get('role_key'))
                source_group = context['source_sha256'] or f"{data.get('batch_id')}:{context['source_name']}"
                previous = unique.get(key)
                signature = lambda value: json.dumps({name: value.get(name) for name in ('label', 'partial_start', 'partial_end', 'partial_intervals')}, sort_keys=True)
                if previous and signature(previous['review']) != signature(review):
                    raise ValueError('相同音频针对相同参考出现冲突标签')
                if previous:
                    previous['_package_sha256'] = sorted(set(previous['_package_sha256']) | {package_hash})
                    previous['source_groups'] = sorted(set(previous['source_groups']) | {source_group})
                    if item.get('derived_from') is not None:
                        previous['derived_from'] = item['derived_from']
                    if item.get('sample_kind') in {'partial_negative', 'partial_positive'}:
                        previous['sample_kind'] = item['sample_kind']
                else:
                    unique[key] = {**item, **context, 'path': str(target), 'batch_id': data.get('batch_id'),
                                   '_package_sha256': [package_hash],
                                   'source_groups': [source_group], 'review': clean_reviews({'r': review})['r']}
    return list(unique.values())
