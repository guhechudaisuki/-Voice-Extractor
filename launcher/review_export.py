"""Export reviewed audio and optional portable material without a web service."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
import tempfile
import unicodedata
import wave
import zipfile

from launcher.training_material import (export_package, partial_keep_segments,
                                        reviewable, validated_rows, training_text,
                                        prepare_training_audio, write_training_manifests, export_language)
from launcher.training_workspace import Workspace
from extractor.recall import MIN_TARGET_SECONDS


def _list_text(text: str) -> str:
    # GPT-SoVITS 训练清单按行解析，文本中的竖线与换行会破坏格式。
    return ' '.join(text.replace('|', '／').splitlines()).strip()


def _partial_clip_text(row: dict, review: dict, keep: tuple[float, float]) -> str:
    """Use explicit crop text or complete, genuinely timed ASR units only.

    Missing text keeps the WAV available for transcription; it must never
    invent a training label from duration ratios or translated subtitles.
    """
    for item in review.get('partial_texts') or ():
        if (isinstance(item, dict) and item.get('start') == keep[0]
                and item.get('end') == keep[1] and item.get('text')):
            return _list_text(str(item['text']))
    explicit = str(review.get("partial_text") or "").strip()
    if explicit:
        keeps = partial_keep_segments(row,review)
        if len(keeps)==1:
            return _list_text(explicit)
        parts = explicit.replace('；；','\n').replace(';;','\n').splitlines()
        if len(parts)==len(keeps) and keep in keeps:
            return _list_text(parts[keeps.index(keep)])
    abs_start = float(row.get("start") or 0) + float(keep[0])
    abs_end = float(row.get("start") or 0) + float(keep[1])
    full = (_list_text(str(review['revised_text'])) if review.get('revised_text') is not None
            else training_text(row, review))
    from extractor.timed_text import complete_timing, slice_timed_text
    diagnostics = row.get("diagnostics") or {}
    for key in ("word_timestamps", "stt_fragments"):
        fragments = complete_timing(full, diagnostics.get(key),
                                    (float(row.get('start') or 0), float(row.get('end') or 0)))
        if fragments:
            return _list_text(slice_timed_text(full, fragments, (abs_start, abs_end)))
    return ''


def _finite(value):
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def _audio_identity(path: Path, fallback_duration: float) -> tuple[str, float]:
    """Hash PCM frames, ignoring WAV tags and filenames; stream to bound memory."""
    checksum = hashlib.sha256()
    try:
        with wave.open(str(path), 'rb') as audio:
            checksum.update(str((audio.getnchannels(), audio.getsampwidth(), audio.getframerate())).encode())
            duration = audio.getnframes() / audio.getframerate()
            while block := audio.readframes(65536):
                checksum.update(block)
    except (wave.Error, EOFError):
        # Fall back to a file digest for audio without readable PCM frames.
        checksum = hashlib.sha256()
        with path.open('rb') as audio:
            for block in iter(lambda: audio.read(1024 * 1024), b''):
                checksum.update(block)
        duration = fallback_duration
    return checksum.hexdigest(), duration


def _quality(row: dict, audio: Path, *, cropped=False) -> dict:
    diagnostics = row.get('diagnostics') or {}
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    qc = diagnostics.get('acoustic_qc') or {}
    if not cropped and isinstance(qc, dict) and qc.get('quality_grade') in {'Gold', 'Silver', 'Review'}:
        return qc
    try:
        import soundfile as sf
        from extractor.acoustic_qc import evaluate_clip_acoustics
        samples, rate = sf.read(str(audio), dtype='float32', always_2d=True)
        measured = evaluate_clip_acoustics(samples.mean(axis=1), rate,
            speaker_score=_finite(row.get('speaker_score')) or 0.0,
            window_min=_finite(row.get('window_min_score')) or 0.0)
        if isinstance(measured, dict):
            return measured
    except (ImportError, OSError, ValueError, RuntimeError):
        pass
    return {'quality_grade': 'Unknown'}


def _language(row: dict, fallback: str) -> str:
    try:
        return export_language(row, fallback)
    except ValueError:
        return export_language({}, fallback)


def _dataset_report(records: list[dict], skipped: dict, indexed: bool, minimum) -> dict:
    count = len(records)
    durations = [record['duration_seconds'] for record in records]
    total_seconds = sum(durations)
    distribution = {}
    for grade in ('Gold', 'Silver', 'Review', 'Unknown'):
        amount = sum(record['quality_grade'] == grade for record in records)
        distribution[grade] = {'count': amount, 'ratio': amount / count if count else 0.0}
    snr = [value for record in records if (value := _finite(record['acoustic_qc'].get('snr_db'))) is not None]
    clipping = [value for record in records if (value := _finite(record['acoustic_qc'].get('clipping_ratio'))) is not None]
    languages = {code: 0 for code in ('JA', 'ZH', 'EN')}
    buckets = dict.fromkeys(('<1.2s', '1.2~3s', '3~6s', '6~10s', '>10s'), 0)
    for record in records:
        code = record['language']
        languages[code] = languages.get(code, 0) + 1
        duration = record['duration_seconds']
        bucket = '<1.2s' if duration < 1.2 else '1.2~3s' if duration < 3 else '3~6s' if duration < 6 else '6~10s' if duration <= 10 else '>10s'
        buckets[bucket] += 1
    return {
        'schema': 1, 'scope': 'exported_dataset_audio', 'min_quality_grade': minimum,
        'total_sentences': count,
        'correct_count': sum(record['label'] == 'correct' for record in records),
        'partial_count': sum(record['label'] == 'partial' for record in records),
        'total_duration_seconds': total_seconds, 'total_duration_minutes': total_seconds / 60,
        'average_duration_seconds': total_seconds / count if count else 0.0,
        'min_duration_seconds': min(durations, default=0.0),
        'max_duration_seconds': max(durations, default=0.0),
        'quality_distribution': distribution,
        'mean_snr_db': sum(snr) / len(snr) if snr else None,
        'snr_measured_count': len(snr), 'qc_measured_count': len(clipping),
        'zero_clipping_rate': sum(value == 0 for value in clipping) / len(clipping) if clipping else None,
        'language_distribution': languages,
        'language_basis': 'sentence language metadata, falling back to export language',
        'train_list_valid_lines': sum(record['valid_text'] for record in records) if indexed else 0,
        'train_list_included': indexed,
        'average_characters': sum(sum(not char.isspace() for char in record['text']) for record in records) / count if count else 0.0,
        'duration_buckets': {key: {'count': amount, 'ratio': amount / count if count else 0.0} for key, amount in buckets.items()},
        **skipped, 'clips': records,
    }


def _report_markdown(report: dict) -> str:
    def measured(value, unit=''):
        return '—（未测量）' if value is None else f'{value:.2f}{unit}'
    count = report['total_sentences']
    lines = ['# 数据集质量体检报告', '', '## 📊 总体规模', '',
        '| 指标 | 数值 |', '|---|---:|',
        f"| 导出音频 | {count} 句 |",
        f"| 完整正确 / 精裁救回 | {report['correct_count']} / {report['partial_count']} 句 |",
        f"| 总时长 | {report['total_duration_seconds']:.2f} 秒 / {report['total_duration_minutes']:.2f} 分钟 |",
        f"| 平均 / 最短 / 最长 | {report['average_duration_seconds']:.2f} / {report['min_duration_seconds']:.2f} / {report['max_duration_seconds']:.2f} 秒 |",
        '', '## 🏅 声学质量', '', '| 等级 | 句数 | 占比 |', '|---|---:|---:|']
    for grade, item in report['quality_distribution'].items():
        lines.append(f"| {grade} | {item['count']} | {item['ratio']:.1%} |")
    lines += ['', f"平均 SNR：{measured(report['mean_snr_db'], ' dB')}（{report['snr_measured_count']} 句有测量值）。",
        f"零削波通过率：{measured(None if report['zero_clipping_rate'] is None else report['zero_clipping_rate'] * 100, '%')}（分母为 {report['qc_measured_count']} 句已测音频）。",
        '', '## 🗣️ 语言与文本', '', '| 语言 | 句数 |', '|---|---:|']
    for language, amount in report['language_distribution'].items():
        lines.append(f'| {language} | {amount} |')
    lines += ['', f"train.list 有效索引：{report['train_list_valid_lines']} 行；平均每句 {report['average_characters']:.2f} 字符。",
        '语言按片段元数据统计，缺失时使用导出语言。',
        f"空白或纯标点台词未写入索引：{len(report['skipped_empty_text'])} 句；音频仍保留。",
        '', '## ⏱️ 时长分桶', '', '| 时长 | 句数 | 占比 |', '|---|---:|---:|']
    for bucket, item in report['duration_buckets'].items():
        lines.append(f"| {bucket} | {item['count']} | {item['ratio']:.1%} |")
    lines += ['', '分桶边界：1.2 ≤ t < 3，3 ≤ t < 6，6 ≤ t ≤ 10；未知质量不计为通过。',
        f"质量过滤：{len(report['skipped_quality'])} 句；同内容去重：{len(report['duplicate_ordinals'])} 句；裁切失败或不足 {MIN_TARGET_SECONDS:g} 秒：{len(report['skipped_partial'])} 句。", '']
    return '\n'.join(lines)


def export_review_zip(batch: Path, manifest: dict, reviews: dict, destination: Path,
                      include_material: bool = False, *, gpt_sovits_list: bool = False,
                      speaker: str = "target", language: str = "JA",
                      min_quality_grade: str | None = None,
                      restore_high_frequencies: bool = False,
                      stats: dict | None = None) -> int:
    """Export a dataset; material mode intentionally retains every reviewed label.

    Gold admits Gold; Silver admits Gold and Silver; Review admits all measured
    grades. None also retains unknown grades. Empty text only excludes train.list
    lines. Quality statistics refer to audio actually contained in the archive.
    """
    grades = {'Review': 0, 'Silver': 1, 'Gold': 2}
    if min_quality_grade is not None and min_quality_grade not in grades:
        raise ValueError('min_quality_grade 必须为 Gold、Silver、Review 或 None')
    if not str(speaker).strip() or not str(language).strip() or any(char in str(speaker) + str(language) for char in '|\r\n'):
        raise ValueError('说话人和语言不可为空或包含索引分隔符')
    rows = validated_rows(batch, manifest, reviews) if include_material else reviewable(batch, manifest)
    selected = [row for row in rows if reviews.get(str(row[0]), {}).get('label') in {'correct', 'partial'}]
    if not selected and not include_material:
        raise ValueError('没有标记为目标匹配的片段')
    workspace = Workspace()
    records = []
    skipped = {'skipped_empty_text': [], 'skipped_quality': [], 'duplicate_ordinals': [], 'skipped_partial': []}
    try:
        temporary = workspace.root
        archive_path = temporary / 'reviewed.zip'
        if include_material:
            # Training material is a portable, complete annotation history.
            export_stats = export_package(batch, manifest, reviews, archive_path, speaker=speaker, language=language)
        else:
            from launcher.audio_export import write_clip
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
                identities, written = set(), set()
                for ordinal, row, source_audio, source_relative, *_ in selected:
                    review = reviews.get(str(ordinal), {})
                    label = review['label']
                    cropped = label == 'partial'
                    source_duration = max(0.0, float(row.get('end') or 0) - float(row.get('start') or 0))
                    try:
                        keeps = ([span for span in partial_keep_segments(row, review)
                                  if span[1] - span[0] >= MIN_TARGET_SECONDS - 1e-9]
                                 if cropped else [(0., source_duration)])
                    except (TypeError, ValueError):
                        keeps = []
                    if not keeps:
                        skipped['skipped_partial'].append(ordinal)
                    for piece_index, keep in enumerate(keeps, 1):
                        audio, relative, text = source_audio, source_relative, training_text(row, review)
                        duration = keep[1] - keep[0]
                        if cropped:
                            from extractor.word_boundaries import partial_word_bounds
                            from launcher.training_material import partial_intervals
                            keep,word_audit = partial_word_bounds(row,keep,partial_intervals(row,review))
                            suffix = '' if piece_index == 1 else f'_{piece_index:02d}'
                            piece = temporary / f'partial_{ordinal:06d}{suffix}.wav'
                            try:
                                write_clip(source_audio, piece, *keep, normalize_level=True)
                            except (TypeError, ValueError, OSError, RuntimeError):
                                skipped['skipped_partial'].append(ordinal)
                                continue
                            audio, relative = piece, f'partial/{piece.name}'
                            text = _partial_clip_text(row, review, keep)
                        qc = _quality(row, audio, cropped=cropped)
                        grade = qc.get('quality_grade')
                        grade = grade if grade in grades else 'Unknown'
                        if min_quality_grade is not None and grades.get(grade, -1) < grades[min_quality_grade]:
                            skipped['skipped_quality'].append(ordinal)
                            continue
                        identity, duration = _audio_identity(audio, duration)
                        if identity in identities or relative in written:
                            skipped['duplicate_ordinals'].append(ordinal)
                            continue
                        identities.add(identity)
                        written.add(relative)
                        prepared = temporary / f'prepared_{ordinal:06d}_{piece_index:02d}.wav'
                        # Old clients may still send the retired flag. It can
                        # never authorize adding any sample from the raw mix.
                        hf_audit = ({'applied': False, 'reason': 'stem_only_policy'}
                                    if restore_high_frequencies else None)
                        duration = prepare_training_audio(audio, prepared)
                        valid_text = any(unicodedata.category(char)[0] in 'LN' for char in text)
                        if not valid_text:
                            skipped['skipped_empty_text'].append(ordinal)
                        language_code = _language(row, language)
                        # Manifest metadata may not inject extra train.list columns.
                        if not language_code or any(char in language_code for char in '|\r\n'):
                            language_code = language
                        records.append({'ordinal': ordinal, 'path': relative, 'label': label,
                            'duration_seconds': duration, 'text': text, 'valid_text': valid_text,
                            'language': language_code, 'quality_grade': grade,
                            'word_boundary_refinement':word_audit if cropped else None,
                            'high_frequency_restoration':hf_audit,
                            'acoustic_qc': {key: _finite(qc.get(key)) for key in ('snr_db', 'clipping_ratio', 'effective_speech_seconds')}})
                        from launcher.corpus_curation import acoustic_dimensions
                        records[-1]['curation_features'] = acoustic_dimensions(prepared,text,language_code)
                        from launcher.golden_prompts import provenance
                        scope = str(batch.resolve()) + '/' + source_relative.split('/')[0]
                        records[-1].update(provenance(row,keep,scope,source_relative))
                        archive.write(prepared, relative)
                        text_entry = str(Path(relative).with_suffix('.txt')).replace('\\', '/')
                        archive.writestr(text_entry, text)
                export_stats = write_training_manifests(archive, records, speaker)
                if gpt_sovits_list:
                    lines = [f"{record['path']}|{speaker}|{record['language']}|{record['text']}" for record in records if record['valid_text']]
                    archive.writestr('train.list', '\n'.join(lines) + ('\n' if lines else ''))
                report = _dataset_report(records, skipped, gpt_sovits_list, min_quality_grade)
                from launcher.corpus_curation import write_curation
                report['curation'] = write_curation(archive,records,speaker)
                archive.writestr('dataset_report.json', json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
                archive.writestr('dataset_report.md', _report_markdown(report))
        save_file(archive_path, destination)
        if stats is not None:
            stats.update({key: export_stats[key] for key in
                          ('valid_count', 'total_duration_seconds', 'slicer_opt_path', 'metadata_path')})
            for key in ('slicer_opt_path', 'metadata_path'):
                stats[key] = f'{destination}!/{stats[key]}'
    finally:
        workspace.close()
    return len(records) if not include_material else sum(reviews.get(str(row[0]), {}).get('label') == 'correct' for row in rows)


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
