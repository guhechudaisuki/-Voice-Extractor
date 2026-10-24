"""Continue the installed WavLM speaker projection using reference/clip pairs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from launcher.training_material import import_packages
from launcher.reinforcement_history import read_history, stage_checkpoint, completion_message
from extractor.model_store import active_speaker_path, file_sha256, reinforcement_lock, replace_model

EPOCHS = 60
BATCH_SIZE = 64


def train(paths: list[Path], destination: Path, progress, *, model_path: Path | None = None) -> dict:
    from extractor.config import WAVLM_SV_MODEL
    current = model_path or active_speaker_path(WAVLM_SV_MODEL)
    with reinforcement_lock(current):
        return _train_current(paths, destination, progress, current)


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
        previous = owners.setdefault(item['sha256'], group)
        parents[find(previous)] = group
        groups.append(group)
    return [find(group) for group in groups]


def _train_current(paths, destination, progress, current):
    import copy
    import torch
    import torch.nn.functional as F
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
    from extractor.audio import load_mono
    from extractor.local_identity_classifier import pooled_features, window_starts

    base_sha256 = file_sha256(current)
    history = read_history(current)
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
              'initialization': 'existing_model_weights', 'component': 'wavlm_speaker_projection'}
    report_path = destination / 'training_report.json'
    if not new_paths:
        report['status'] = 'already_applied'
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        progress(completion_message(report))
        return report
    items = import_packages(new_paths, destination / 'audio')
    selected = [item for item in items if item['review']['label'] in {'correct', 'wrong'}]
    if not selected:
        raise ValueError('ZIP 没有可训练的目标匹配或整句非目标标注')
    report.update(unique_clips=len(items), excluded_ambiguous_clips=len(items) - len(selected),
                  source_count=len({group for item in items for group in item['source_groups']}),
                  reference_sets=len({item['reference_key'] for item in items}))
    labels = [int(item['review']['label'] == 'correct') for item in selected]
    if not history['material_sha256'] and (labels.count(0) < 2 or labels.count(1) < 2):
        raise ValueError('首次强化至少需要 2 段目标匹配和 2 段整句非目标音频')
    progress(f"ZIP 哈希比对：新增 {len(new_paths)} 份，跳过重复 {report['duplicate_packages']} 份")
    progress('正在加载现用声纹模型，按每段对应的参考音频构建训练对')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = WavLMForXVector.from_pretrained(str(current.parent), local_files_only=True).to(device).eval()
    encoder.requires_grad_(False)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(str(current.parent), local_files_only=True)
    projection = encoder.feature_extractor
    original = copy.deepcopy(projection).eval()
    projection.requires_grad_(True)
    feature_dir = destination / 'features'
    feature_dir.mkdir()
    feature_paths = {}

    def encode(path):
        path = Path(path)
        key = file_sha256(path)
        if key in feature_paths:
            return feature_paths[key]
        wave = load_mono(path, 16000)
        if len(wave) < 8000:
            raise ValueError('音频或参考不足 0.5 秒，请重新导出有效原料')
        window = min(len(wave), 28800)
        starts = window_starts(len(wave), window=window, hop=14400)
        if len(starts) > 40:
            starts = [starts[round(i * (len(starts) - 1) / 39)] for i in range(40)]
        mean = None
        for offset in starts:
            pooled = pooled_features(encoder, processor, device, wave[offset:offset + window]).cpu()
            width = pooled.numel() // 3
            feature = torch.cat((pooled[:width], pooled[-width:]))
            if feature.numel() != projection.in_features or not torch.isfinite(feature).all():
                raise ValueError('音频的声纹统计特征无效')
            mean = feature.clone() if mean is None else mean + feature
        feature_path = feature_dir / (key + '.pt')
        torch.save(mean / len(starts), feature_path)
        feature_paths[key] = feature_path
        return feature_path

    groups = source_groups(selected)
    pairs = []
    for number, (item, group) in enumerate(zip(selected, groups), 1):
        query = encode(item['path'])
        for reference in item['reference_paths']:
            pairs.append((query, encode(reference), int(item['review']['label'] == 'correct'), group))
        progress(f'整理各集声纹训练对 {number}/{len(selected)}')
    # Backbone features are on disk. Release the frozen network while training
    # only its existing projection, and bound the feature cache and batch size.
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sources = sorted(set(groups), key=lambda value: hashlib.sha256(value.encode()).hexdigest())
    verified = all(item.get('source_sha256') for item in selected)
    held_out = set(sources[:max(1, len(sources) // 5)]) if len(sources) >= 3 and verified else set()
    train_idx = [i for i, pair in enumerate(pairs) if pair[3] not in held_out]
    test_idx = [i for i, pair in enumerate(pairs) if pair[3] in held_out]
    if not train_idx or {pairs[i][2] for i in train_idx} != set(labels):
        # Few sources are allowed; avoid requiring more episodes solely to split.
        train_idx, test_idx = list(range(len(pairs))), []
        held_out = set()
    @lru_cache(maxsize=256)
    def feature(path):
        return torch.load(path, map_location='cpu', weights_only=True)
    def batch(indexes):
        queries = torch.stack([feature(pairs[i][0]) for i in indexes]).to(device)
        references = torch.stack([feature(pairs[i][1]) for i in indexes]).to(device)
        labels = torch.tensor([pairs[i][2] for i in indexes], dtype=torch.float32, device=device)
        return queries, references, labels
    def embeddings(head, query, reference):
        return F.normalize(head(query), dim=-1), F.normalize(head(reference), dim=-1)
    def evaluate(indexes):
        total_loss, correct = 0., 0
        with torch.no_grad():
            for offset in range(0, len(indexes), BATCH_SIZE):
                portion = indexes[offset:offset+BATCH_SIZE]
                query, reference, labels = batch(portion)
                q, r = embeddings(projection, query, reference)
                scores = (q * r).sum(-1)
                total_loss += float(F.binary_cross_entropy_with_logits((scores - .8) * 20, labels, reduction='sum'))
                correct += int(((scores >= .8) == labels.bool()).sum())
        return {'pairs': len(indexes), 'sources': len({pairs[i][3] for i in indexes}),
                'positive': sum(pairs[i][2] for i in indexes), 'negative': sum(1-pairs[i][2] for i in indexes),
                'loss': total_loss / len(indexes) if indexes else None,
                'accuracy_at_cosine_080': correct / len(indexes) if indexes else None}
    initial_loss = evaluate(train_idx)['loss']
    optimizer = torch.optim.Adam(projection.parameters(), lr=1e-5)
    generator = torch.Generator().manual_seed(20260930)
    for epoch in range(EPOCHS):
        order = [train_idx[i] for i in torch.randperm(len(train_idx), generator=generator).tolist()]
        for offset in range(0, len(order), BATCH_SIZE):
            query, reference, labels = batch(order[offset:offset+BATCH_SIZE])
            optimizer.zero_grad()
            q, r = embeddings(projection, query, reference)
            with torch.no_grad():
                old_q, old_r = embeddings(original, query, reference)
            # 加性角度边际（CosFace 式 AAM）：正样本对需要在边际内"更像"，
            # 把相似女声负例推出决策带；负样本对保持原判定带不变。
            cosine = (q * r).sum(-1)
            margin = 0.08 * labels
            loss = F.binary_cross_entropy_with_logits(((cosine - margin) - .8) * 20, labels)
            loss = loss + .1 * ((q-old_q).square().mean() + (r-old_r).square().mean())
            # L2-SP：与初始权重的锚定正则，防止小样本增量强化破坏预训练
            # 特征空间（灾难性遗忘）。
            l2_sp = sum(
                (param - original_param).square().sum()
                for param, original_param in zip(projection.parameters(), original.parameters()))
            loss = loss + .05 * l2_sp
            if not torch.isfinite(loss):
                raise ValueError('训练损失无效，原模型未修改')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(projection.parameters(), 1.)
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            progress(f'训练 {epoch + 1}/{EPOCHS}')
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
    staged = destination / current.name
    stage_checkpoint(current, staged, projection, history)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    replace_model(staged, current, base_sha256)
    try:
        staged.unlink(missing_ok=True)
    except OSError:
        progress('模型已保存，暂存副本将在清理临时文件时移除')
    feature.cache_clear()
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
