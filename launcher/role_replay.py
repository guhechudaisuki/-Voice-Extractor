"""Universal query/reference replay with source-disjoint validation."""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

from extractor.model_store import atomic_save_projection
from .role_samples import sample_fingerprint, role_hashes
from .reinforcement_validation import holdout_sources

SCHEMA = 'universal-wavlm-mean-std-preprojection-v2'


def load_bank(directory, base_sha256):
    import torch
    path = Path(directory) / 'replay_bank.pt'
    if not path.exists():
        return dict(schema=SCHEMA, base_model_sha256=base_sha256,
                    compatible_model_sha256=[base_sha256],
                    entries={}, source_split={})
    bank = torch.load(path, map_location='cpu', weights_only=True)
    if (bank.get('schema') != SCHEMA
            or base_sha256 not in bank.get('compatible_model_sha256', [bank.get('base_model_sha256')])):
        raise ValueError('全局经验池与基础模型不匹配，请使用对应模型或恢复备份')
    if not isinstance(bank.get('entries'), dict) or not isinstance(bank.get('source_split'), dict):
        raise ValueError('经验池索引无效')
    for entry in bank['entries'].values():
        features = [entry['query'], *entry['references']]
        if (not features or not entry['references'] or any(not isinstance(v, torch.Tensor)
                or v.ndim != 1 or v.shape != features[0].shape or not torch.isfinite(v).all()
                for v in features) or entry['item']['review']['label'] not in {'correct', 'wrong'}):
            raise ValueError('经验池包含无效特征或标签')
        for name in ('calibration_queries', 'calibration_references'):
            matrix = entry.get(name)
            if matrix is not None and (not isinstance(matrix, torch.Tensor)
                    or matrix.ndim != 2 or matrix.shape[1] != features[0].numel()
                    or not torch.isfinite(matrix).all()
                    or (name == 'calibration_references' and len(matrix) == 0)):
                raise ValueError('经验池包含无效的局部校准特征')
        if (entry.get('calibration_queries') is None) != (entry.get('calibration_references') is None):
            raise ValueError('经验池的局部校准特征不完整')
    return bank


def save_bank(directory, bank):
    Path(directory).mkdir(parents=True, exist_ok=True)
    atomic_save_projection(Path(directory) / 'replay_bank.pt', bank)


def add_entry(bank, item, query, references, *, pcm_sha256=None, vectors=None):
    """Deduplicate PCM aliases; contradictory labels require explicit correction."""
    key = sample_fingerprint(item)
    content = (item.get('source_sha256'), pcm_sha256 or item['sha256'])
    for previous in bank['entries'].values():
        if tuple(previous['content']) == content and role_hashes(previous['item']) == role_hashes(item):
            if previous['item']['review']['label'] != item['review']['label']:
                raise ValueError('同一来源音频存在冲突标签，请统一标注后重新导出')
            return False
    keep = ('sha256', 'source_sha256', 'source_groups', 'review', 'sample_kind',
            'reference_key', 'role_reference_sha256', 'speaker_name',
            'reference_names', 'original_reference_paths')
    metadata = {k: item[k] for k in keep if k in item}
    metadata['pcm_sha256'] = pcm_sha256 or item['sha256']
    metadata['role_reference_sha256'] = role_hashes(item)
    bank['entries'][key] = dict(item=metadata, query=query.detach().cpu().clone(),
        references=[v.detach().cpu().clone() for v in references], content=list(content),
        vectors=vectors or {})
    return True


def source_partition(bank, groups, keys, previous=None):
    """Never promote a previously trained source into a future holdout."""
    entries = bank['entries']
    split = dict(bank['source_split'])
    previous = previous or {}
    for name, value in (('training_sources', 'train'), ('held_out_sources', 'holdout')):
        for source in previous.get(name, ()):
            split.setdefault(source, value)
    labels = [int(entries[k]['item']['review']['label'] == 'correct') for k in keys]
    verified = all(entries[k]['item'].get('source_sha256') for k in keys)
    if not verified or len(set(groups)) < 3:
        return set()
    resolved = {}
    for key, group in zip(keys, groups):
        source_names = entries[key]['item']['source_groups']
        states = {split[s] for s in [group, *source_names] if s in split}
        if len(states) > 1:
            return set()  # Duplicate audio connects a trained and held-out source.
        if states:
            state = states.pop()
            if group in resolved and resolved[group] != state:
                return set()
            resolved[group] = state
    held = {g for g,v in resolved.items() if v == 'holdout'}
    if not held:
        choices = [g for g in groups if resolved.get(g) != 'train']
        candidates = holdout_sources(groups, labels, True)
        if candidates.intersection(g for g,v in resolved.items() if v == 'train'):
            candidates = set()
            for group in sorted(set(choices)):
                if ({y for g,y in zip(groups,labels) if g == group} == {0,1}
                        and {y for g,y in zip(groups,labels) if g != group} == {0,1}):
                    candidates = {group}
                    break
        held = candidates
    if (not held or {y for g,y in zip(groups,labels) if g in held} != {0,1}
            or {y for g,y in zip(groups,labels) if g not in held} != {0,1}):
        return set()
    for key,group in zip(keys,groups):
        state = 'holdout' if group in held else 'train'
        for source in [group, *entries[key]['item']['source_groups']]:
            split[source] = state
    bank['source_split'] = split
    return held


def replay_keys(bank, new_keys, groups, keys, held, limit=256):
    """Bound historical gradient work; every held-out observation stays validation-only."""
    selected = set(new_keys)
    buckets = defaultdict(list)
    for key,group in zip(keys,groups):
        if group in held:
            selected.add(key)
        elif key not in selected:
            buckets[(group,bank['entries'][key]['item']['review']['label'])].append(key)
    for values in buckets.values():
        values.sort()
    # Round-robin balances both sources and labels, not ZIP sizes.
    while buckets and limit > 0:
        for bucket in list(sorted(buckets)):
            selected.add(buckets[bucket].pop(0)); limit -= 1
            if not buckets[bucket]:
                del buckets[bucket]
            if limit <= 0:
                break
    return [key for key in keys if key in selected]


def write_meta(directory, bank, report=None):
    directory = Path(directory)
    path = directory / 'reinforcement_meta.json'
    old = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
    items = [e['item'] for e in bank['entries'].values()]
    counts = {label: sum(i['review']['label'] == label for i in items) for label in ('correct','wrong')}
    metrics = (report or {}).get('metrics', {}).get('held_out_sources')
    meta = {**old, 'schema':2, 'scope':'universal',
        'positive_count':counts['correct'], 'negative_count':counts['wrong'],
        'reference_names':sorted({n for i in items for n in i.get('reference_names',())}),
        'reference_paths':sorted({n for i in items for n in i.get('original_reference_paths',())}),
        'reference_sha256':sorted({h for i in items for h in i.get('role_reference_sha256',())}),
        'updated_at':datetime.now(timezone.utc).isoformat(),
        'status':(report or {}).get('status','features_saved')}
    if metrics:
        meta['latest_validation_attempt'] = metrics
        if (report or {}).get('status')=='updated_in_place':
            meta['latest_holdout'] = metrics
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)
    return meta
