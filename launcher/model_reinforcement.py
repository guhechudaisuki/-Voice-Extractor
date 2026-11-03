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
from launcher.reinforcement_validation import classification_metrics, non_regressing
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
                values = ((queries @ references.T).topk(min(2, len(references)), dim=-1).values.mean(-1) - .8) * 20.
            if not torch.isfinite(values).all():
                raise ValueError('回放分类器得分无效，原模型未修改')
            return values.cpu().tolist()
        if scorer_fn is None:
            key = tuple(role_hashes(item))
            if key not in models_by_reference:
                classifier = reference_conditioned_classifier(encoder, processor, device,
                    [Path(p) for p in item['reference_paths']])
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

    positive_limits = [value - .1 * _sample_weight(item) for item in positives for value in scores(item)]
    negative_limits = [float(np.mean(values)) + .1 * _sample_weight(item)
                       for item in negatives if (values := scores(item))]
    if not positive_limits or not negative_limits:
        progress('全局分类器校准跳过：正/负例窗口证据不足')
        return None
    reject = float(np.quantile(positive_limits, .05))
    rescue = float(np.quantile(negative_limits, .95))
    separated = rescue < reject - .2
    payload = dict(schema=2, scope='universal', scoring='reference_cosine_v1',
        threshold=rescue, calibration_status='reinforced',
        veto_status='active' if separated else 'disabled',
        window_seconds=.5, stride_seconds=.25,
        calibration_reference_sets=len({tuple(role_hashes(i)) for i in items}),
        sample_weighting={'partial_negative': PARTIAL_NEGATIVE_WEIGHT,
                          'partial_positive': PARTIAL_POSITIVE_WEIGHT,
                          'positive_clips': len(positives), 'negative_clips': len(negatives)})
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


def _negative_matrix(values, name):
    import numpy as np
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError(f"负例原型 {name} 维度或数值无效，历史原型未修改")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError(f"负例原型 {name} 含零向量，历史原型未修改")
    return matrix / np.maximum(norms, 1e-12)


def _read_negative_prototypes(path):
    """Read legacy libraries as well as the cumulative-support schema."""
    import numpy as np
    if not path.is_file():
        return None, None, [], set()
    try:
        with np.load(path, allow_pickle=False) as data:
            primary = _negative_matrix(data['primary'], 'primary')
            secondary = _negative_matrix(data['secondary'], 'secondary')
            if len(primary) != len(secondary):
                raise ValueError('负例原型双通道数量不一致')
            counts = np.asarray(data.get('support_counts', data.get('new_counts',
                                np.ones(len(primary)))), dtype=np.float64)
            if (counts.shape != (len(primary),) or not np.isfinite(counts).all()
                    or np.any(counts < 0) or np.any(counts != np.floor(counts))
                    or np.any(counts >= np.iinfo(np.int64).max)):
                raise ValueError('负例原型支持度无效')
            # Legacy new_counts recorded only that run, including zeros for
            # untouched historical speakers. Every existing prototype has
            # at least one supporting sample.
            counts = np.maximum(counts, 1).astype(np.int64).tolist()
            hashes = np.asarray(data.get('sample_sha256', []))
            if hashes.ndim != 1:
                raise ValueError('负例原型样本索引无效')
            seen = {str(value) for value in hashes}
            if any(len(value) != 64 or any(c not in '0123456789abcdef' for c in value)
                   for value in seen):
                raise ValueError('负例原型样本哈希无效')
        return primary, secondary, counts, seen
    except (OSError, ValueError, KeyError, EOFError) as exc:
        raise ValueError(f'无法读取历史负例原型，已保留原文件：{exc}') from exc


def _extract_negative_embeddings(paths, progress):
    """Keep model loading separate from the persistent-library transaction."""
    import numpy as np
    import torch
    from extractor.speaker import CAMPlusVerifier, SpeakerVerifier
    from extractor.audio import load_mono
    from extractor.config import ASSET_ROOT
    from extractor.nextgen.anime_embedding import AnimeSpeakerOnnx, _MODEL_FILES
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    primary = secondary = None
    anime = {}
    try:
        primary = SpeakerVerifier(device=device)
        secondary = CAMPlusVerifier(device=device)
        models = ASSET_ROOT / 'models'
        if all((models / path).is_file() for path in _MODEL_FILES.values()):
            for variant in ('char', 'va'):
                anime['anime_' + variant] = AnimeSpeakerOnnx(models, variant, device=device)
        else:
            progress('动漫声纹模型未安装，保留已有动漫原型，本次仅更新常规原型')
        vectors = {name: [] for name in ('primary', 'secondary', *anime)}
        for index, path in enumerate(paths, 1):
            wave = load_mono(Path(path), 16000)
            vectors['primary'].append(primary._embedding_from_waveform(wave).detach().cpu().numpy().reshape(-1))
            vectors['secondary'].append(secondary._embedding_from_waveform(wave).detach().cpu().numpy().reshape(-1))
            for name, encoder in anime.items():
                vectors[name].append(encoder.encode(wave, sample_rate=16000))
            if index % 20 == 0:
                progress(f'负例原型提取 {index}/{len(paths)}')
        return {name: np.asarray(rows) for name, rows in vectors.items()}
    finally:
        if primary is not None:
            primary.close()
        if secondary is not None:
            secondary.close()
        for encoder in anime.values():
            encoder.close()


def _read_anime_prototypes(path, count):
    """Missing views have zero support; their storage rows are never embeddings."""
    import numpy as np
    views = {}
    if not path.is_file():
        return views
    with np.load(path, allow_pickle=False) as data:
        for channel in ('anime_char', 'anime_va'):
            if channel not in data:
                continue
            matrix = np.asarray(data[channel], dtype=np.float64)
            support = np.asarray(data.get(channel + '_support_counts', np.ones(count)), dtype=np.float64)
            if (matrix.shape != (count, 192) or support.shape != (count,)
                    or not np.isfinite(matrix).all() or not np.isfinite(support).all()
                    or np.any(support < 0) or np.any(support != np.floor(support))
                    or np.any(support >= np.iinfo(np.int64).max)):
                raise ValueError(f'历史 {channel} 原型或支持度无效，原文件未修改')
            valid = support > 0
            matrix[valid] = _negative_matrix(matrix[valid], channel)
            matrix[~valid] = 0.
            views[channel] = (matrix, support.astype(np.int64))
    return views


def _merge_negative_prototypes(old_primary, old_secondary, support_counts,
                               new_primary, new_secondary, *, old_views=None, new_views=None):
    """Primary cosine drives identity; both channels get a 0.2 moving update."""
    import numpy as np
    fresh_p = _negative_matrix(new_primary, 'new primary')
    fresh_s = _negative_matrix(new_secondary, 'new secondary')
    if len(fresh_p) != len(fresh_s):
        raise ValueError('新负例双通道数量不一致')
    if old_primary is not None and (old_primary.shape[1] != fresh_p.shape[1]
                                   or old_secondary.shape[1] != fresh_s.shape[1]):
        raise ValueError('负例原型维度与当前模型不一致，历史原型未修改')
    primary = [] if old_primary is None else list(old_primary.copy())
    secondary = [] if old_secondary is None else list(old_secondary.copy())
    counts = list(support_counts)
    new_counts = [0] * len(counts)
    extra_views = old_views is not None or new_views is not None
    old_views, new_views = old_views or {}, new_views or {}
    fresh_views, view_vectors, view_supports = {}, {}, {}
    for channel in sorted(set(old_views) | set(new_views)):
        if channel in new_views:
            fresh_views[channel] = _negative_matrix(new_views[channel], channel)
            if fresh_views[channel].shape != (len(fresh_p), 192):
                raise ValueError(f'{channel} 嵌入维度或数量与音频不一致')
        old, supports = old_views.get(channel, (np.zeros((len(counts), 192)), np.zeros(len(counts))))
        if old.shape != (len(counts), 192) or len(supports) != len(counts):
            raise ValueError(f'{channel} 历史原型未与常规通道对齐')
        view_vectors[channel], view_supports[channel] = list(old.copy()), list(supports)
    added = 0
    for index, (vector_p, vector_s) in enumerate(zip(fresh_p, fresh_s)):
        similarities = np.asarray(primary) @ vector_p if primary else np.empty(0)
        best = int(np.argmax(similarities)) if len(similarities) else -1
        if best >= 0 and similarities[best] >= .82:
            for vectors, fresh in ((primary, vector_p), (secondary, vector_s)):
                merged = .8 * vectors[best] + .2 * fresh
                vectors[best] = merged / np.linalg.norm(merged)
            counts[best] += 1
            new_counts[best] += 1
        else:
            best = len(primary)
            primary.append(vector_p.copy())
            secondary.append(vector_s.copy())
            counts.append(1)
            new_counts.append(1)
            added += 1
            for channel in view_vectors:
                view_vectors[channel].append(np.zeros(192))
                view_supports[channel].append(0)
        for channel, fresh in fresh_views.items():
            previous = view_vectors[channel][best]
            merged = (.8 * previous + .2 * fresh[index]
                      if view_supports[channel][best] else fresh[index].copy())
            norm = np.linalg.norm(merged)
            if norm <= 1e-12:
                raise ValueError(f'{channel} 原型更新发生方向抵消，历史原型未修改')
            view_vectors[channel][best] = merged / norm
            view_supports[channel][best] += 1
    # Stable order resolves equal-support ties in favour of historical memory.
    keep = sorted(range(len(counts)), key=lambda i: (-counts[i], i))[:64]
    result = (np.asarray(primary, dtype=np.float32)[keep],
            np.asarray(secondary, dtype=np.float32)[keep],
            np.asarray(counts, dtype=np.int64)[keep],
            np.asarray(new_counts, dtype=np.int64)[keep], added)
    if extra_views:
        views = {channel: (np.asarray(rows, dtype=np.float32)[keep],
                          np.asarray(view_supports[channel], dtype=np.int64)[keep])
                 for channel, rows in view_vectors.items()}
        return (*result, views)
    return result


def update_auto_negative_prototypes(items, model_dir: Path, progress, *, status=None, replay_vectors=None) -> int:
    """Merge unique wrong-speaker clips into aligned multiview memory.

    A primary cosine of at least 0.82 updates an existing speaker by a moving
    average. Otherwise a new prototype is added. Cumulative sample support
    selects the strongest 64 prototypes. The enclosing training operation
    already holds reinforcement_lock for the installed model.
    """
    import numpy as np
    import tempfile
    wrong = [item for item in items if item.get('review', {}).get('label') == 'wrong']
    if not wrong:
        if status is not None:
            status.update(state='no_wrong_samples', channels={})
        return 0
    model_dir = Path(model_dir)
    existing = model_dir / 'auto_negatives.npz'
    old_p, old_s, counts, seen = _read_negative_prototypes(existing)
    old_views = _read_anime_prototypes(existing, len(counts))
    from extractor.config import ASSET_ROOT
    from extractor.nextgen.anime_embedding import _MODEL_FILES
    anime_available = all((ASSET_ROOT / 'models' / path).is_file()
                          for path in _MODEL_FILES.values())
    missing_anime = any(channel not in old_views or np.any(old_views[channel][1] == 0)
                        for channel in ('anime_char', 'anime_va'))
    pending, backfill = {}, {}
    for item in sorted(wrong, key=lambda value: str(value['path'])):
        checksum = str(item.get('sha256') or '').lower()
        if len(checksum) != 64 or any(c not in '0123456789abcdef' for c in checksum):
            checksum = file_sha256(Path(item['path']))
        if checksum not in seen:
            pending.setdefault(checksum, Path(item['path']))
        elif anime_available and missing_anime:
            backfill.setdefault(checksum, Path(item['path']))
    if not pending and not backfill:
        progress('负例原型无需更新：本次错人音频均已学习')
        if status is not None:
            status.update(state='already_learned', channels={
                channel: 'preserved' if channel in ('primary', 'secondary') or channel in old_views else 'unavailable'
                for channel in ('primary', 'secondary', 'anime_char', 'anime_va')})
        return 0
    requested = [*pending.values(), *backfill.values()]
    by_path = {str(item['path']):item['sha256'] for item in wrong}
    stored = [dict((replay_vectors or {}).get(by_path[str(p)], {})) for p in requested]
    if replay_vectors:
        missing = [i for i, views in enumerate(stored) if not {'primary', 'secondary'} <= views.keys()]
        if missing:
            fresh = _extract_negative_embeddings([requested[i] for i in missing], progress)
            if not isinstance(fresh, dict):
                fresh = dict(zip(('primary', 'secondary'), fresh))
            for offset, index in enumerate(missing):
                stored[index] = {channel: matrix[offset] for channel, matrix in fresh.items()}
        channels = set.intersection(*(set(v) for v in stored))
        extracted = {c:np.stack([np.asarray(v[c]) for v in stored]) for c in channels}
    else:
        extracted = _extract_negative_embeddings(requested, progress)
    if isinstance(extracted, dict):
        fresh_p, fresh_s = extracted['primary'], extracted['secondary']
        fresh_views = {name: extracted[name] for name in ('anime_char', 'anime_va') if name in extracted}
    else:
        fresh_p, fresh_s = extracted
        fresh_views = {}
    if len(fresh_p) != len(requested) or len(fresh_s) != len(requested):
        raise ValueError('负例嵌入数量与音频不一致，历史原型未修改')
    new_size = len(pending)
    if pending:
        primary, secondary, supports, new_counts, added, views = _merge_negative_prototypes(
            old_p, old_s, counts, fresh_p[:new_size], fresh_s[:new_size], old_views=old_views,
            new_views={channel: matrix[:new_size] for channel, matrix in fresh_views.items()})
    else:
        primary, secondary = old_p.copy(), old_s.copy()
        supports, new_counts = np.asarray(counts, dtype=np.int64), np.zeros(len(counts), dtype=np.int64)
        added = 0
        views = {channel: (matrix.copy(), support.copy())
                 for channel, (matrix, support) in old_views.items()}
    backfilled = set()
    if backfill and fresh_views:
        probes_p = _negative_matrix(fresh_p[new_size:], 'backfill primary')
        probes_s = _negative_matrix(fresh_s[new_size:], 'backfill secondary')
        for channel, matrix in fresh_views.items():
            probes = _negative_matrix(matrix[new_size:], channel)
            if probes.shape != (len(backfill), 192):
                raise ValueError(f'{channel} 补齐嵌入维度或数量与音频不一致')
            vectors, support = views.setdefault(channel, (
                np.zeros((len(primary), 192), dtype=np.float32),
                np.zeros(len(primary), dtype=np.int64)))
            missing = support == 0
            for index, (probe_p, probe_s) in enumerate(zip(probes_p, probes_s)):
                matches_p, matches_s = primary @ probe_p, secondary @ probe_s
                best = int(np.argmax(matches_p))
                # Legacy libraries have no clip-to-prototype map. Both views must
                # select the same existing speaker before filling an absent view.
                if (matches_p[best] < .82 or best != int(np.argmax(matches_s))
                        or not missing[best]):
                    continue
                merged = (.8 * vectors[best] + .2 * probes[index]
                          if support[best] else probes[index])
                norm = float(np.linalg.norm(merged))
                if norm <= 1e-12:
                    raise ValueError(f'{channel} 补齐原型发生方向抵消，历史原型未修改')
                vectors[best] = merged / norm
                support[best] += 1
                backfilled.add(channel)
    if not pending and not backfilled:
        progress('动漫记忆尚未补齐：本次旧样本没有与缺失原型形成双通道一致匹配')
        if status is not None:
            status.update(state='backfill_pending', samples=0, channels={
                channel: 'preserved' if channel in ('primary', 'secondary') or channel in old_views else 'unavailable'
                for channel in ('primary', 'secondary', 'anime_char', 'anime_va')})
        return 0
    extra = {key: value for channel, (matrix, channel_counts) in views.items()
             for key, value in ((channel, matrix), (channel + '_support_counts', channel_counts))}
    model_dir.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.auto_negatives_', suffix='.tmp', dir=model_dir)
    temporary = Path(name)
    try:
        # Passing a handle prevents numpy silently appending '.npz' to '.tmp'.
        with os.fdopen(descriptor, 'wb') as stream:
            # Only numeric arrays and validated content hashes cross this boundary.
            np.savez(stream, primary=primary, secondary=secondary,
                     support_counts=supports, new_counts=new_counts,
                     sample_sha256=np.asarray(sorted(seen | set(pending)), dtype='U64'),
                     schema=np.asarray(3, dtype=np.int32), **extra)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, existing)
    finally:
        temporary.unlink(missing_ok=True)
    progress(f'负例原型库已更新：共 {len(primary)} 个原型；本次新增样本 {len(pending)} 条')
    if backfilled:
        progress('已补齐历史原型的动漫证据，常规声纹与累计样本支持数未重复更新')
    if status is not None:
        status.update(state='updated', samples=len(pending), prototypes=len(primary),
                      channels={channel: 'updated' if (pending and channel in ('primary', 'secondary'))
                                or (pending and channel in fresh_views) or channel in backfilled
                                else 'preserved' if channel in ('primary', 'secondary') or channel in views else 'unavailable'
                                for channel in ('primary', 'secondary', 'anime_char', 'anime_va')})
    return added



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


def _update_global_memory(directory, bank, chosen, current_items, group_by_key, held_out,
                        report, progress, *, snapshot=True):
    """Publish non-gradient updates independently of projection validation.

    Held-out sources never enter either the prototypes or classifier calibration.
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
        memory = report['negative_memory'] = {}
        # Historical audio may already be deleted; cached vectors remain usable.
        prototype_items = [(current_items[k] if k in current_items else
                            {**bank['entries'][k]['item'], 'path': 'replay:'+k})
                           for k in chosen if k in current_items or
                           {'primary', 'secondary'} <= bank['entries'][k].get('vectors', {}).keys()]
        replay_vectors = {bank['entries'][k]['item']['sha256']:bank['entries'][k]['vectors']
                          for k in chosen if bank['entries'][k].get('vectors')}
        update_auto_negative_prototypes(prototype_items or training_items, directory, progress,
                                        status=memory, replay_vectors=replay_vectors)
        report['channel_updates'].update({
            channel: 'memory_' + state for channel, state in memory.get('channels', {}).items()})
    except Exception as exc:  # 原型库失败不影响已完成的模型强化
        report['negative_memory'] = {'state': 'failed', 'error': str(exc)}
        report['channel_updates'].update({name: 'failed' for name in (
            'primary', 'secondary', 'anime_char', 'anime_va')})
        progress(f"负例原型库更新失败（其他已完成通道不受影响）: {type(exc).__name__}: {exc}")
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
    for item in selected:
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
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if added:
        try:
            vectors = _extract_negative_embeddings([current_items[k]['path'] for k in added],progress)
            for index,key in enumerate(added):
                bank['entries'][key]['vectors'] = {
                    name:torch.as_tensor(matrix[index]).float().cpu() for name,matrix in vectors.items()}
        except Exception as exc:
            report['replay_auxiliary_error'] = str(exc)
            progress(f'辅助声纹特征暂不可用，已保留 WavLM 经验特征：{exc}')
    keys = list(bank['entries'])
    all_items = [bank['entries'][k]['item'] for k in keys]
    groups = source_groups(all_items)
    held_out = source_partition(bank,groups,keys,history)
    save_bank(directory,bank)
    report.update(replay_bank_clips=len(keys), new_samples=len(added),
                  source_count=len(set(groups)), source_fingerprints_complete=all(i.get('source_sha256') for i in all_items))
    # Pending samples survive a failed cold-start validation without changing weights.
    if not held_out:
        _update_global_memory(directory, bank, keys, current_items, dict(zip(keys, groups)),
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
    pairs = []
    for key,item,group in zip(chosen,selected,groups):
        entry = bank['entries'][key]
        # A clip with more reference files must not receive more total gradient
        # mass than an equally weighted clip from another extraction batch.
        weight = (_sample_weight(item) * (1. if key in pending or group in held_out else replay_weight)
                  / len(entry['references']))
        for reference in entry['references']:
            pairs.append((entry['query'],reference,int(item['review']['label']=='correct'),group,weight))
    # Replay owns frozen features; only the existing projection receives gradients.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    train_idx = [i for i, pair in enumerate(pairs) if pair[3] not in held_out]
    test_idx = [i for i, pair in enumerate(pairs) if pair[3] in held_out]
    def batch(indexes):
        queries = torch.stack([pairs[i][0] for i in indexes]).to(device)
        references = torch.stack([pairs[i][1] for i in indexes]).to(device)
        labels = torch.tensor([pairs[i][2] for i in indexes], dtype=torch.float32, device=device)
        weights = torch.tensor([pairs[i][4] for i in indexes], dtype=torch.float32, device=device)
        return queries, references, labels, weights
    def embeddings(head, query, reference):
        return F.normalize(head(query), dim=-1), F.normalize(head(reference), dim=-1)
    def evaluate(indexes):
        total_loss, total_weight, correct = 0., 0., 0
        all_scores, all_labels = [], []
        with torch.no_grad():
            for offset in range(0, len(indexes), BATCH_SIZE):
                portion = indexes[offset:offset+BATCH_SIZE]
                query, reference, labels, weights = batch(portion)
                q, r = embeddings(projection, query, reference)
                scores = (q * r).sum(-1)
                all_scores.extend(scores.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                losses = F.binary_cross_entropy_with_logits((scores - .8) * 20, labels, reduction='none')
                total_loss += float((losses * weights).sum())
                total_weight += float(weights.sum())
                correct += int(((scores >= .8) == labels.bool()).sum())
        return {**classification_metrics(all_scores, all_labels), 'pairs': len(indexes), 'sources': len({pairs[i][3] for i in indexes}),
                'positive': sum(pairs[i][2] for i in indexes), 'negative': sum(1-pairs[i][2] for i in indexes),
                'loss': total_loss / total_weight if total_weight else None,
                'accuracy_at_cosine_080': correct / len(indexes) if indexes else None}
    initial_loss = evaluate(train_idx)['loss']
    baseline = evaluate(test_idx)
    best_state, best_metrics, best_epoch, stale = None, baseline, 0, 0
    optimizer = torch.optim.Adam(projection.parameters(), lr=1e-5)
    generator = torch.Generator().manual_seed(20260930)
    for epoch in range(EPOCHS):
        order = [train_idx[i] for i in torch.randperm(len(train_idx), generator=generator).tolist()]
        for offset in range(0, len(order), BATCH_SIZE):
            query, reference, labels, weights = batch(order[offset:offset+BATCH_SIZE])
            optimizer.zero_grad()
            q, r = embeddings(projection, query, reference)
            with torch.no_grad():
                old_q, old_r = embeddings(original, query, reference)
            # Positive pairs must clear an additional cosine margin.
            cosine = (q * r).sum(-1)
            margin = 0.08 * labels
            logits = ((cosine - margin) - .8) * 20
            # Detach hardness so lowering a sample's weight cannot lower loss.
            error_probability = (logits.detach().sigmoid() - labels).abs()
            weights = weights * (1.0 + error_probability)
            losses = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
            loss = (losses * weights).sum() / weights.sum()
            loss = loss + .1 * ((q-old_q).square().mean() + (r-old_r).square().mean())
            # Anchor the projection to its initial weights on small datasets.
            l2_sp = sum(
                (param - original_param).square().sum()
                for param, original_param in zip(projection.parameters(), original.parameters()))
            loss = loss + .05 * l2_sp
            if not torch.isfinite(loss):
                raise ValueError('训练损失无效，原模型未修改')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(projection.parameters(), 1.)
            optimizer.step()
        measured = evaluate(test_idx)
        if non_regressing(measured, baseline) and measured['loss'] < best_metrics['loss'] - 1e-7:
            best_state = {k: v.detach().cpu().clone() for k, v in projection.state_dict().items()}
            best_metrics, best_epoch, stale = measured, epoch + 1, 0
        else:
            stale += 1
        progress(f"训练 {epoch + 1}/{EPOCHS}，留出损失 {measured['loss']:.5f}，最佳轮次 {best_epoch}")
        if stale >= EARLY_STOPPING_PATIENCE:
            break
    report['validation_gate'] = {'passed': best_state is not None, 'baseline': baseline,
                                  'best': best_metrics, 'best_epoch': best_epoch, 'epochs_run': epoch + 1}
    if best_state is None:
        _update_global_memory(directory, bank, chosen, current_items, group_by_key,
                            held_out, report, progress)
        report.update(status='rejected_holdout', initial_loss=initial_loss,
                      training_clips=sum(g not in held_out for g in groups),
                      held_out_clips=sum(g in held_out for g in groups),
                      metrics={'training': evaluate(train_idx), 'held_out_sources': evaluate(test_idx)})
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        write_meta(directory,bank,report)
        progress(completion_message(report))
        return report
    projection.load_state_dict(best_state)
    metrics = {'training': evaluate(train_idx), 'held_out_sources': evaluate(test_idx)}
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
    _update_global_memory(directory, bank, chosen, current_items, group_by_key, held_out,
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
