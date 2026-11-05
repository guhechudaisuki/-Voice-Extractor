"""Compile reviewed batches with global decoded-PCM deduplication and LUFS mastering."""
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
import zipfile
import numpy as np
import soundfile as sf
from launcher.review_export import export_review_zip, _audio_identity, _dataset_report, _report_markdown, _quality, save_file
from launcher.training_material import write_training_manifests


def _short_clip_rms_master(source, destination, target_lufs, true_peak):
    from scipy.signal import resample_poly
    samples, rate = sf.read(source, dtype='float64', always_2d=True)
    if not samples.size or not np.isfinite(samples).all():
        raise ValueError('短片段音频为空或包含非有限采样值')

    def rms(audio):
        return float(np.sqrt(np.mean(audio ** 2)))

    def peak(audio):
        # Include inter-sample peaks, preserving all channels and the source rate.
        return float(np.max(np.abs(resample_poly(audio, 8, 1, axis=0))))

    def db(value):
        return 20 * math.log10(max(value, 1e-12))

    input_rms, input_peak = rms(samples), peak(samples)
    ceiling = 10 ** (true_peak / 20)
    # Reserve PCM16 quantization headroom; limit linear gain rather than distort peaks.
    safe_peak = ceiling - 4 / 32768
    gain = min(10 ** ((target_lufs + .691) / 20) / input_rms,
               safe_peak / input_peak) if input_rms and input_peak else 1.
    sf.write(destination, np.clip(samples * gain, -safe_peak, safe_peak), rate, subtype='PCM_16')
    output, _ = sf.read(destination, dtype='float64', always_2d=True)
    output_peak = peak(output)
    if output_peak > ceiling:
        raise ValueError('短片段标准化输出未通过真峰值检查')
    lufs = db(rms(output)) - .691
    return {'input_lufs': db(input_rms) - .691, 'output_lufs': lufs,
            'true_peak_dbtp': db(output_peak), 'target_lufs': target_lufs,
            'target_true_peak_dbtp': true_peak, 'target_reached': abs(lufs - target_lufs) <= .5,
            'method': 'short_clip_rms_fallback', 'loudness_basis': 'rms_equivalent',
            'gain_db': db(gain), 'sample_rate': rate}


def master_loudness(source, destination, *, target_lufs=-23., true_peak=-1.):
    from extractor.config import FFMPEG
    if not (math.isfinite(target_lufs) and -36 <= target_lufs <= -12
            and math.isfinite(true_peak) and -6 <= true_peak <= -1):
        raise ValueError('响度须为 -36 至 -12 LUFS，真峰值须为 -6 至 -1 dBTP')
    info = sf.info(source)
    rate = info.samplerate
    base = f'loudnorm=I={target_lufs}:TP={true_peak}:LRA=11'
    def run(filters, output):
        command = [str(FFMPEG), '-hide_banner', '-nostdin', '-y', '-i', str(source),
                   '-af', filters + ':print_format=json', '-ar', str(rate), *output]
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace',
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), timeout=180)
        if result.returncode:
            raise ValueError('响度标准化失败：' + result.stderr[-1000:])
        blocks = re.findall(r'\{\s*"input_i".*?\}', result.stderr, re.S)
        if not blocks:
            raise ValueError('无法测量片段 LUFS')
        return json.loads(blocks[-1])
    measured = run(base, ['-f', 'null', '-'])
    names = {'measured_I': 'input_i', 'measured_TP': 'input_tp', 'measured_LRA': 'input_lra',
             'measured_thresh': 'input_thresh', 'offset': 'target_offset'}
    if not all(math.isfinite(float(measured[key])) for key in names.values()):
        if info.duration < 1.5:
            return _short_clip_rms_master(source, destination, target_lufs, true_peak)
        raise ValueError('静音或过短片段无法进行 LUFS 标准化')
    filters = base + ':linear=true' + ''.join(f':{name}={measured[key]}' for name, key in names.items())
    result = run(filters, ['-c:a', 'pcm_s16le', str(destination)])
    lufs, peak = float(result['output_i']), float(result['output_tp'])
    if info.duration < 1.5 and (not math.isfinite(lufs) or not math.isfinite(peak)):
        return _short_clip_rms_master(source, destination, target_lufs, true_peak)
    if not math.isfinite(lufs) or not math.isfinite(peak) or peak > true_peak + .1:
        raise ValueError('标准化输出未通过真峰值检查')
    return {'input_lufs': float(measured['input_i']), 'output_lufs': lufs, 'true_peak_dbtp': peak,
            'target_lufs': target_lufs, 'target_true_peak_dbtp': true_peak,
            'target_reached': abs(lufs-target_lufs) <= .5, 'method': result['normalization_type'], 'sample_rate': rate}


def merge_datasets(batches, destination, *, speaker='target', language='JA', target_lufs=-23.,
                   true_peak=-1., min_quality_grade=None, restore_high_frequencies=False, progress=None):
    """batches contains (directory, manifest, reviews); never modifies source audio."""
    if not batches:
        raise ValueError('请选择至少一个已复核批次')
    progress = progress or (lambda _: None)
    records, seen, duplicates, sources, skipped_quality = [], set(), [], [], []
    with tempfile.TemporaryDirectory(prefix='voice_corpus_') as tmp:
        root = Path(tmp)
        package = root / 'corpus.zip'
        with zipfile.ZipFile(package, 'w', zipfile.ZIP_DEFLATED) as archive:
            for index, (batch, manifest, reviews) in enumerate(batches):
                batch = Path(batch)
                progress(f'总装 {index+1}/{len(batches)}：{batch.name}')
                reviewed = root / f'batch_{index}.zip'
                export_review_zip(batch, manifest, reviews, reviewed, speaker=speaker, language=language,
                    min_quality_grade=min_quality_grade, restore_high_frequencies=restore_high_frequencies)
                with zipfile.ZipFile(reviewed) as incoming:
                    report = json.loads(incoming.read('dataset_report.json'))
                    sources.append({'batch_id': batch.name, 'selected': report['total_sentences'],
                                    'skipped_quality': list(report['skipped_quality']),
                                    'skipped_partial': report['skipped_partial'],
                                    'skipped_empty_text': report['skipped_empty_text']})
                    skipped_quality.extend({'batch_id': batch.name, 'ordinal': ordinal}
                                           for ordinal in report['skipped_quality'])
                    for clip in report['clips']:
                        source = root / 'source.wav'
                        source.write_bytes(incoming.read(clip['path']))
                        identity, _ = _audio_identity(source, clip['duration_seconds'])
                        if identity in seen:
                            duplicates.append({'batch_id': batch.name, 'ordinal': clip['ordinal']})
                            continue
                        mastered = root / 'mastered.wav'
                        audit = master_loudness(source, mastered, target_lufs=target_lufs, true_peak=true_peak)
                        sentences = manifest.get('sentences') or []
                        ordinal = clip['ordinal']
                        row = (sentences[ordinal-1] if isinstance(sentences, list) and isinstance(ordinal, int)
                               and 0 < ordinal <= len(sentences) else {})
                        qc = _quality(row if isinstance(row, dict) else {}, mastered, cropped=True)
                        ranks = {'Review': 0, 'Silver': 1, 'Gold': 2}
                        if min_quality_grade and ranks.get(qc.get('quality_grade'), -1) < ranks[min_quality_grade]:
                            skipped_quality.append({'batch_id': batch.name, 'ordinal': ordinal,
                                                    'quality_grade': qc.get('quality_grade', 'Unknown'),
                                                    'reason': 'post_mastering_quality'})
                            sources[-1]['skipped_quality'].append(ordinal)
                            continue
                        seen.add(identity)
                        name = f'audio/{len(records)+1:06d}.wav'
                        records.append({**clip, 'path': name, 'batch_id': batch.name,
                                        'source_pcm_sha256': identity, 'loudness': audit,
                                        'quality_grade': qc.get('quality_grade', 'Unknown'),
                                        'acoustic_qc': {k: qc.get(k) for k in ('snr_db', 'clipping_ratio', 'effective_speech_seconds')}})
                        from launcher.corpus_curation import acoustic_dimensions
                        records[-1]['curation_features'] = acoustic_dimensions(mastered,clip['text'],clip['language'])
                        archive.write(mastered, name)
                        archive.writestr(str(Path(name).with_suffix('.txt')).replace('\\', '/'), clip['text'])
            if not records:
                raise ValueError('没有可导出的已复核音频片段')
            stats = write_training_manifests(archive, records, speaker)
            archive.writestr('train.list', ''.join(
                f"{r['path']}|{speaker}|{r['language']}|{r['text']}\n"
                for r in records if r['valid_text']))
            skipped_empty_text = [{'batch_id': r['batch_id'], 'ordinal': r['ordinal']}
                                  for r in records if not r['valid_text']]
            report = _dataset_report(records, {'duplicate_clips': duplicates, 'duplicate_ordinals': duplicates,
                'skipped_empty_text': skipped_empty_text, 'skipped_quality': skipped_quality,
                'skipped_partial': []}, True, min_quality_grade)
            report.update(batches=sources, target_lufs=target_lufs, true_peak_dbtp=true_peak,
                          lufs_min=min(r['loudness']['output_lufs'] for r in records),
                          lufs_max=max(r['loudness']['output_lufs'] for r in records),
                          loudness_target_misses=[r['path'] for r in records if not r['loudness']['target_reached']])
            from launcher.corpus_curation import write_curation
            report['curation'] = write_curation(archive,records,speaker)
            archive.writestr('dataset_report.json', json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
            archive.writestr('dataset_report.md', _report_markdown(report))
        save_file(package, Path(destination))
    return {**stats, 'batch_count': len(batches), 'duplicate_count': len(duplicates),
            'audio_count': len(records), 'total_audio_duration_seconds': report['total_duration_seconds'],
            'skipped_empty_text': skipped_empty_text, 'skipped_quality': skipped_quality,
            'loudness_target_misses': report['loudness_target_misses']}
