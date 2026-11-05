"""Continue the installed WavLM speaker projection using reference/clip pairs."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from launcher.training_material import import_packages
from launcher.reinforcement_validation import classification_metrics, non_regressing, paired_decision_audit
from launcher.reinforcement_history import completion_message, read_history, stage_checkpoint
from launcher.role_samples import role_hashes, sample_fingerprint
from extractor.model_store import (active_speaker_path, file_sha256, reinforcement_lock,
                                   replace_model, snapshot_global_models, atomic_save_projection)

EPOCHS = 60
EARLY_STOPPING_PATIENCE = 8
BATCH_SIZE = 64
PARTIAL_NEGATIVE_WEIGHT = 2.0
PARTIAL_POSITIVE_WEIGHT = 1.5


def _sample_weight(item) -> float:
    """Cropped dialogue carries stronger evidence than an unrelated clip."""
    label = item['review']['label']
    if label not in {'correct', 'wrong'}:
        return 1.0
    if (item.get('derived_from') is None
            and item.get('sample_kind') not in {'partial_negative', 'partial_positive'}):
        return 1.0
    return (PARTIAL_NEGATIVE_WEIGHT if label == 'wrong'
            else PARTIAL_POSITIVE_WEIGHT)


def rebind_classifier_references(items, progress, *, classifier_path: Path | None = None,
                                 scorer_fn=None, replay_entries=(), model_path=None) -> dict | None:
    """Calibrate one reference-conditioned metric across every supplied character."""
    import numpy as np
    import torch
    from extractor.config import CLASSIFIER_V2_MODEL, WAVLM_SV_MODEL
    from extractor.audio import load_mono
    from extractor.local_identity_classifier import (window_starts, score_window,
                                                     reference_conditioned_classifier)
    from extractor.set_relational import SetRelationalScorer
    from launcher.rank_calibration import fit_temperature
    path = Path(classifier_path) if classifier_path else Path(CLASSIFIER_V2_MODEL)
    replay = {sample_fingerprint(entry['item']): entry for entry in replay_entries
              if entry.get('calibration_queries') is not None}
    # Current PCM is optional once the frozen 0.5 s features enter global replay.
    items = [{**item, '_calibration_entry': replay.get(sample_fingerprint(item)) if item.get('sha256') else None}
             for item in items]
    current_keys = {sample_fingerprint(item) for item in items if item.get('sha256')}
    items += [{**entry['item'], '_calibration_entry': entry} for key, entry in replay.items()
              if key not in current_keys]
    positives = [item for item in items if item.get('review', {}).get('label') == 'correct']
    negatives = [item for item in items if item.get('review', {}).get('label') == 'wrong']
    if not positives or not negatives:
        progress('全局分类器校准需要正例和负例，保留现有校准')
        return None
    models_by_reference = {}
    if scorer_fn is None:
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        directory = Path(model_path).parent if model_path else Path(WAVLM_SV_MODEL)
        encoder = WavLMForXVector.from_pretrained(str(directory),
            local_files_only=True, use_safetensors=active_speaker_path(directory).suffix == '.safetensors').to(device).eval()
        encoder.requires_grad_(False)
        processor = Wav2Vec2FeatureExtractor.from_pretrained(str(directory), local_files_only=True)

    def scores(item):
        entry = item.get('_calibration_entry')
        if entry is not None and scorer_fn is None:
            with torch.no_grad():
                projection = encoder.feature_extractor
                queries = torch.nn.functional.normalize(projection(entry['calibration_queries'].to(device)), dim=-1)
                references = torch.nn.functional.normalize(projection(entry['calibration_references'].to(device)), dim=-1)
                values = SetRelationalScorer()(queries, references)
            if not torch.isfinite(values).all():
                raise ValueError('回放分类器得分无效，原模型未修改')
            return values.cpu().tolist()
        if scorer_fn is None:
            key = tuple(role_hashes(item))
            if key not in models_by_reference:
                classifier = reference_conditioned_classifier(encoder, processor, device,
                    [Path(p) for p in item['reference_paths']],
                    calibration={'scale':1., 'bias':0.}, relational=True)
                models_by_reference[key] = (encoder, processor, classifier, 0., device)
            score = lambda wave: score_window(models_by_reference[key], wave)
        else:
            score = scorer_fn
        wave = load_mono(Path(item['path']), 16000)
        offsets = window_starts(len(wave)) or ([0] if len(wave) >= 5600 else [])
        values = []
        for offset in offsets:
            local = wave[offset:offset + 8000]
            if float(local.square().mean()) < 1e-8:
                continue
            value = float(score(local))
            if not math.isfinite(value):
                raise ValueError('分类器窗口得分无效，原模型未修改')
            values.append(value)
        return values

    positive_rows = [(item,scores(item)) for item in positives]
    negative_rows = [(item,scores(item)) for item in negatives]
    raw = [v for _,values in positive_rows+negative_rows for v in values]
    labels = [1 for _,values in positive_rows for v in values] + [0 for _,values in negative_rows for v in values]
    if not raw or set(labels) != {0,1}:
        progress('全局分类器校准跳过：正/负例窗口证据不足')
        return None
    calibration = fit_temperature(raw, labels)
    transform = lambda value: value*calibration['scale']+calibration['bias']
    positive_limits = [transform(v) - .1*_sample_weight(item) for item,values in positive_rows for v in values]
    negative_limits = [float(np.mean([transform(v) for v in values])) + .1*_sample_weight(item)
                       for item,values in negative_rows if values]
    if not positive_limits or not negative_limits:
        progress('全局分类器校准跳过：正/负例窗口证据不足')
        return None
    reject = float(np.quantile(positive_limits, .05))
    rescue = float(np.quantile(negative_limits, .95))
    separated = rescue < reject - .2
    payload = dict(schema=2, scope='universal', scoring='set_relational_v2',
        calibration=calibration,
        threshold=rescue, calibration_status='reinforced',
        veto_status='active' if separated else 'disabled',
        window_seconds=.5, stride_seconds=.25,
        calibration_reference_sets=len({tuple(role_hashes(i)) for i in items}),
        sample_weighting={'partial_negative': PARTIAL_NEGATIVE_WEIGHT,
                          'partial_positive': PARTIAL_POSITIVE_WEIGHT,
                          'positive_clips': len(positives), 'negative_clips': len(negatives)})
    projection_path = Path(model_path) if model_path else active_speaker_path(WAVLM_SV_MODEL)
    if projection_path.is_file():
        payload['projection_sha256'] = file_sha256(projection_path)
    if separated:
        payload['reject_threshold'] = reject
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and not (Path(WAVLM_SV_MODEL) / 'global_rollback.json').exists():
        import shutil
        shutil.copyfile(path, path.with_suffix(path.suffix + '.previous'))
    atomic_save_projection(path, payload)
    progress(f'全局参考条件分类器已校准：{payload["calibration_reference_sets"]} 组参考；'
             + ('持续负窗否决启用' if separated else '分布重叠，仅启用正类救援'))
    return {'reject_threshold': payload.get('reject_threshold'), 'threshold': rescue,
            'reference_sets': payload['calibration_reference_sets'], 'scope': 'universal'}


def train(paths: list[Path], destination: Path, progress, *, model_path: Path | None = None) -> dict:
    from extractor.config import WAVLM_SV_MODEL
    current = Path(model_path) if model_path else active_speaker_path(WAVLM_SV_MODEL)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with reinforcement_lock(current):
        current = _canonical_global_checkpoint(current)
        items = import_packages(paths, destination / 'imported_audio')
        if not items:
            raise ValueError('ZIP 没有已标注样本')
        report = _train_current(paths, destination, progress, current, items=items)
        (destination / 'training_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        return report


def _canonical_global_checkpoint(current: Path) -> Path:
    """Migrate legacy global weights under the training lock; no gradient update."""
    if current.suffix != '.safetensors':
        return current
    binary = current.parent / 'pytorch_model.bin'
    if binary.is_file():
        return binary
    from launcher.reinforcement_history import stage_binary_checkpoint
    from launcher.role_replay import load_bank, save_bank
    import tempfile
    descriptor, name = tempfile.mkstemp(prefix='.global_binary_', suffix='.bin', dir=current.parent)
    os.close(descriptor)
    staged = Path(name)
    try:
        old_hash = file_sha256(current)
        stage_binary_checkpoint(current, staged)
        if (current.parent / 'replay_bank.pt').is_file():
            bank = load_bank(current.parent, old_hash)
            new_hash = file_sha256(staged)
            bank['base_model_sha256'] = new_hash
            bank['compatible_model_sha256'] = sorted(set(bank.get('compatible_model_sha256', [old_hash])) | {new_hash})
            save_bank(current.parent, bank)
        staged.replace(binary)
    finally:
        staged.unlink(missing_ok=True)
    return binary


def source_groups(items):
    # The same query appearing in several packages/contexts must stay on one
    # side of a source holdout, even when its target references differ.
    parents, owners = {}, {}
    def find(key):
        parents.setdefault(key, key)
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key
    groups = []
    for item in items:
        group = find(item['source_groups'][0])
        for other in item['source_groups'][1:]:
            parents[find(other)] = group
        previous = owners.setdefault(item.get('pcm_sha256') or item['sha256'], group)
        parents[find(previous)] = group
        groups.append(group)
    return [find(group) for group in groups]


def _calibrate_global_classifier(directory, bank, current_items, group_by_key, held_out,
                        report, progress, *, snapshot=True):
    """Calibrate the classifier independently of projection validation.

    Held-out sources never enter classifier calibration.
    Pending samples remain pending for future multi-source gradient training.
    """
    reserved = {source for source, split in bank.get('source_split', {}).items() if split == 'holdout'}
    reserved.update(held_out)
    chosen = [k for k in bank['entries'] if group_by_key[k] not in reserved
              and not reserved.intersection(bank['entries'][k]['item'].get('source_groups', ()))]
    if not chosen:
        return
    from extractor.config import CLASSIFIER_V2_MODEL
    if snapshot:
        snapshot_global_models(active_speaker_path(directory), CLASSIFIER_V2_MODEL)
    training_items = [current_items[k] for k in chosen if k in current_items]
    try:
        from extractor.config import CLASSIFIER_V2_MODEL
        classifier = Path(CLASSIFIER_V2_MODEL)
        rebound = rebind_classifier_references(training_items, progress, classifier_path=classifier,
            replay_entries=[bank['entries'][k] for k in chosen], model_path=active_speaker_path(directory))
        report['channel_updates']['local_classifier'] = 'recalibrated' if rebound else 'unchanged'
    except Exception as exc:  # 重绑定失败不影响已完成的模型强化
        report['channel_updates']['local_classifier'] = 'failed'
        progress(f"classifier_v2 重绑定失败（其他已完成通道不受影响）: {type(exc).__name__}: {exc}")


def _train_current(paths, destination, progress, current, *, items=None):
    import copy
    import torch
    import torch.nn.functional as F
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
    from extractor.audio import load_mono
    from extractor.local_identity_classifier import pooled_features, window_starts
    from launcher.role_replay import (load_bank, save_bank, add_entry, source_partition,
                              replay_keys, write_meta)

    base_sha256 = file_sha256(current)
    directory = current.parent
    history = read_history(current)
    bank = load_bank(directory, base_sha256)
    material_hashes = [file_sha256(Path(path)) for path in paths]
    seen = set(history['material_sha256'])
    new_paths = []
    for path, checksum in zip(paths, material_hashes):
        if checksum not in seen:
            new_paths.append(path)
            seen.add(checksum)
    report = {'schema': 3, 'packages': len(paths), 'unique_clips': 0,
              'new_packages': len(new_paths), 'duplicate_packages': len(paths) - len(new_paths),
              'training_clips': 0, 'held_out_clips': 0, 'metrics': {},
              'source_count': 0, 'reference_sets': 0,
              'model_path': str(current.resolve()), 'base_model_sha256': base_sha256,
              'scope': 'universal',
              'initialization': 'existing_model_weights', 'component': 'wavlm_speaker_projection',
              'channel_updates': {}}
    report_path = destination / 'training_report.json'
    if not new_paths:
        report['status'] = 'already_applied'
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        progress(completion_message(report))
        return report
    items = items if items is not None else import_packages(new_paths, destination / 'audio')
    new_hashes = {file_sha256(Path(path)) for path in new_paths}
    items = [item for item in items if not item.get('_package_sha256')
             or new_hashes.intersection(item['_package_sha256'])]
    selected = [item for item in items if item['review']['label'] in {'correct', 'wrong'}]
    trained = set(history.get('sample_fingerprints', ()))
    unique = {}
    for item in selected:
        fingerprint = sample_fingerprint(item)
        if fingerprint not in trained:
            unique.setdefault(fingerprint, item)
    selected = list(unique.values())
    if not selected:
        report['status'] = 'already_applied'
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        return report
    report.update(unique_clips=len(items), excluded_ambiguous_clips=len(items) - len(selected),
                  source_count=len({group for item in items for group in item['source_groups']}),
                  reference_sets=len({item['reference_key'] for item in items}),
                  sample_weighting={
                      'partial_negative': PARTIAL_NEGATIVE_WEIGHT,
                      'partial_positive': PARTIAL_POSITIVE_WEIGHT,
                      'hardness_multiplier_max': 2.0,
                      'partial_negative_clips': sum(item['review']['label'] == 'wrong'
                          and _sample_weight(item) > 1 for item in selected),
                      'partial_positive_clips': sum(item['review']['label'] == 'correct'
                          and _sample_weight(item) > 1 for item in selected),
                  })
    progress(f"ZIP 哈希比对：新增 {len(new_paths)} 份，跳过重复 {report['duplicate_packages']} 份")
    progress('正在加载现用声纹模型，按每段对应的参考音频构建训练对')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = WavLMForXVector.from_pretrained(str(current.parent), local_files_only=True,
        use_safetensors=current.suffix == '.safetensors').to(device).eval()
    encoder.requires_grad_(False)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(str(current.parent), local_files_only=True)
    projection = encoder.feature_extractor
    original = copy.deepcopy(projection).eval()
    projection.requires_grad_(True)
    feature_cache, local_features, reference_features = {}, {}, {}
    window_features = {}
    pcm_hashes = {}

    def encode(path, *, local=False):
        path = Path(path)
        key = file_sha256(path)
        if key in feature_cache and (not local or key in local_features):
            return feature_cache[key]
        wave = load_mono(path, 16000)
        pcm_hashes[key] = hashlib.sha256(wave.detach().cpu().float().contiguous().numpy().tobytes()).hexdigest()
        if len(wave) < 5600:
            raise ValueError('音频或参考不足 0.35 秒，请重新导出有效原料')
        if len(wave) < 8000:
            wave = F.pad(wave, (0, 8000 - len(wave)))
        window = min(len(wave), 28800)
        starts = window_starts(len(wave), window=window, hop=14400)
        if len(starts) > 40:
            starts = [starts[round(i * (len(starts) - 1) / 39)] for i in range(40)]
        def feature_at(offset, size):
            window_key = (key, offset, size)
            if window_key not in window_features:
                pooled = pooled_features(encoder, processor, device, wave[offset:offset + size]).cpu()
                width = pooled.numel() // 3
                feature = torch.cat((pooled[:width], pooled[-width:]))
                if feature.numel() != projection.in_features or not torch.isfinite(feature).all():
                    raise ValueError('音频的声纹统计特征无效')
                window_features[window_key] = feature
            return window_features[window_key]
        features = []
        for offset in starts:
            features.append(feature_at(offset, window))
        reference_features[key] = torch.stack(features)
        feature_cache[key] = reference_features[key].mean(0)
        if local:
            offsets = window_starts(len(wave))
            if len(offsets) > 40:
                offsets = [offsets[round(i * (len(offsets) - 1) / 39)] for i in range(40)]
            voiced = [feature_at(offset, 8000) for offset in offsets
                      if float(wave[offset:offset+8000].square().mean()) >= 1e-8]
            local_features[key] = torch.stack(voiced) if voiced else torch.empty((0, projection.in_features))
        return feature_cache[key]

    current_items = {sample_fingerprint(item):item for item in selected}
    added = []
    for index, item in enumerate(selected):
        key = sample_fingerprint(item)
        if key in bank['entries']:
            continue
        query = encode(item['path'], local=True)
        references = [encode(p) for p in item['reference_paths']]
        if add_entry(bank,item,query, references,
                     pcm_sha256=pcm_hashes[file_sha256(Path(item['path']))]):
            bank['entries'][key]['calibration_queries'] = local_features[file_sha256(Path(item['path']))]
            bank['entries'][key]['calibration_references'] = torch.cat([
                reference_features[file_sha256(Path(p))] for p in item['reference_paths']])
            added.append(key)
        if (index + 1) % 10 == 0 or index + 1 == len(selected):
            progress(f'声纹特征已提取 {index + 1}/{len(selected)}')
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    keys = list(bank['entries'])
    all_items = [bank['entries'][k]['item'] for k in keys]
    groups = source_groups(all_items)
    held_out = source_partition(bank,groups,keys,history)
    save_bank(directory,bank)
    report.update(replay_bank_clips=len(keys), new_samples=len(added),
                  source_count=len(set(groups)), source_fingerprints_complete=all(i.get('source_sha256') for i in all_items))
    # Pending samples survive a failed cold-start validation without changing weights.
    if not held_out:
        _calibrate_global_classifier(directory, bank, current_items, dict(zip(keys, groups)),
                            held_out, report, progress)
        report.update(status='validation_unavailable', validation_gate={'passed':False,
            'reason':'已保存经验特征；需要至少三个独立来源，且训练与留出集均有正负样本'})
        write_meta(directory,bank,report)
        report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        progress(completion_message(report))
        return report
    pending = {k for k in keys if k not in trained}
    chosen = replay_keys(bank,pending,groups,keys,held_out)
    group_by_key = dict(zip(keys,groups))
    selected = [bank['entries'][k]['item'] for k in chosen]
    groups = [group_by_key[k] for k in chosen]
    verified = all(item.get('source_sha256') for item in selected)
    new_train = sum(k in pending and group_by_key[k] not in held_out for k in chosen)
    old_train = sum(k not in pending and group_by_key[k] not in held_out for k in chosen)
    if not new_train:
        report['status'] = 'already_applied'
        write_meta(directory,bank,report)
        report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        return report
    report['replayed_training_clips'] = old_train
    replay_weight = min(1., new_train/max(1,old_train))
    from launcher.relational_training import train_projection
    samples = []
    for key, item, group in zip(chosen, selected, groups):
        entry = bank['entries'][key]
        references = entry.get('calibration_references')
        if references is None or not len(references):
            references = torch.stack(list(entry['references']))
        weight = _sample_weight(item) * (1. if key in pending or group in held_out else replay_weight)
        samples.append((entry['query'], references, int(item['review']['label']=='correct'), group, weight))
    best_state, initial_loss, validation, metrics = train_projection(
        projection, original, samples, held_out, epochs=EPOCHS, patience=EARLY_STOPPING_PATIENCE,
        batch_size=BATCH_SIZE, progress=progress, device=device)
    report['validation_gate'] = validation
    if best_state is None:
        _calibrate_global_classifier(directory, bank, current_items, group_by_key,
                            held_out, report, progress)
        report.update(status='rejected_holdout', initial_loss=initial_loss,
                      training_clips=sum(g not in held_out for g in groups),
                      held_out_clips=sum(g in held_out for g in groups), metrics=metrics)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        write_meta(directory,bank,report)
        progress(completion_message(report))
        return report
    if not all(torch.isfinite(value).all() for value in projection.state_dict().values()):
        raise ValueError('模型权重无效，原模型未修改')
    held_out_clips = sum(group in held_out for group in groups)
    report.update(training_clips=len(selected) - held_out_clips, held_out_clips=held_out_clips,
                  metrics=metrics, source_fingerprints_complete=verified,
                  status='updated_in_place', initial_loss=initial_loss,
                  limitations='Reference-conditioned speaker verification; frozen backbone. '
                              'Training accuracy does not establish improvement on independent speech.')
    history['material_sha256'] = sorted(set(history['material_sha256']) | set(material_hashes))
    history['sample_fingerprints'] = sorted(trained | {sample_fingerprint(item) for item in selected})
    if file_sha256(current) != base_sha256:
        raise ValueError('基础模型在训练期间发生变化，本次未发布')
    history.update(validation_gate=report['validation_gate'],
                   training_sources=sorted(set(groups) - held_out), held_out_sources=sorted(held_out))
    staged = destination / current.name
    stage_checkpoint(current, staged, projection, history)
    from extractor.config import CLASSIFIER_V2_MODEL
    snapshot_global_models(current, CLASSIFIER_V2_MODEL)
    updated_sha256 = file_sha256(staged)
    bank['base_model_sha256'] = updated_sha256
    bank['compatible_model_sha256'] = sorted(set(bank.get('compatible_model_sha256', [base_sha256])) | {updated_sha256})
    save_bank(directory, bank)
    # Register the staged hash before publication; a crash cannot orphan replay.
    replace_model(staged, current, base_sha256)
    report['channel_updates']['wavlm'] = 'weights_updated'
    _calibrate_global_classifier(directory, bank, current_items, group_by_key, held_out,
                        report, progress, snapshot=False)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    write_meta(directory,bank,report)
    progress(completion_message(report))
    return report


if __name__ == "__main__":
    directory = Path(sys.argv[1])
    owner_path = directory.parent / "owner.json"
    if owner_path.is_file():
        import threading
        import time
        from launcher.training_workspace import _alive
        owner = json.loads(owner_path.read_text(encoding="utf-8"))
        def watch_owner():
            while True:
                time.sleep(1)
                if not _alive(owner):
                    os._exit(2)
        threading.Thread(target=watch_owner, daemon=True).start()
    paths = sorted(directory.glob("upload_*.zip"))
    try:
        train(paths, directory, lambda message: print(message, flush=True))
    except Exception as exc:
        print(str(exc), flush=True)
        raise SystemExit(1)
