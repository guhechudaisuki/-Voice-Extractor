"""Keep successful ZIP hashes inside the atomically published checkpoint."""
from collections import OrderedDict
import copy
import json
from pathlib import Path
import re

HISTORY_KEY = 'voice_extract_reinforcement'


def _validated(value):
    if value is None:
        return {'schema': 1, 'material_sha256': []}
    if (not isinstance(value, dict) or value.get('schema') != 1
            or not isinstance(value.get('material_sha256'), list)):
        raise ValueError('模型内的训练哈希记录无效，未修改模型')
    if any(not isinstance(key, str) or not re.fullmatch(r'[a-f0-9]{64}', key) for key in value['material_sha256']):
        raise ValueError('模型内的训练哈希记录无效，未修改模型')
    return copy.deepcopy(value)


def read_history(path: Path) -> dict:
    if path.suffix == '.safetensors':
        from safetensors import safe_open
        with safe_open(str(path), framework='pt', device='cpu') as checkpoint:
            value = (checkpoint.metadata() or {}).get(HISTORY_KEY)
        return _validated(json.loads(value) if value is not None else None)
    import torch
    state = torch.load(path, map_location='cpu', weights_only=True)
    return _validated(getattr(state, '_metadata', {}).get('', {}).get(HISTORY_KEY))


def stage_binary_checkpoint(source: Path, destination: Path):
    """Convert a legacy global safetensors checkpoint without changing weights."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    from extractor.model_store import atomic_save_projection

    state = OrderedDict(load_file(str(source), device='cpu'))
    with safe_open(str(source), framework='pt', device='cpu') as checkpoint:
        metadata = checkpoint.metadata() or {}
    state._metadata = {'': {HISTORY_KEY: read_history(source),
                           'voice_extract_safetensors_metadata': metadata}}
    atomic_save_projection(destination, state)
    checked = torch.load(destination, map_location='cpu', weights_only=True)
    if (set(checked) != set(state) or any(not torch.equal(value, checked[key]) for key, value in state.items())
            or read_history(destination) != read_history(source)):
        raise ValueError('全局模型格式迁移校验失败，原模型未修改')


def stage_checkpoint(current: Path, staged: Path, projection, history: dict):
    """Preserve unrelated tensors/metadata and verify both weights and history."""
    import torch
    history = _validated(history)
    if current.suffix == '.safetensors':
        from safetensors import safe_open
        from safetensors.torch import load_file, save_file
        state = load_file(str(current), device='cpu')
        with safe_open(str(current), framework='pt', device='cpu') as checkpoint:
            metadata = checkpoint.metadata() or {}
        metadata.update(format='pt', **{HISTORY_KEY: json.dumps(history, separators=(',', ':'))})
    else:
        state = torch.load(current, map_location='cpu', weights_only=True)
        metadata = copy.deepcopy(getattr(state, '_metadata', {}))
        state = OrderedDict(state)
        metadata.setdefault('', {})[HISTORY_KEY] = history
        state._metadata = metadata
    for key, value in projection.state_dict().items():
        state['feature_extractor.' + key] = value.detach().cpu().contiguous()
    if current.suffix == '.safetensors':
        save_file(state, str(staged), metadata=metadata)
        checked = load_file(str(staged), device='cpu')
    else:
        from extractor.model_store import atomic_save_projection
        atomic_save_projection(staged, state)
        checked = torch.load(staged, map_location='cpu', weights_only=True)
    for key, value in projection.state_dict().items():
        if not torch.equal(checked['feature_extractor.' + key], value.detach().cpu()):
            raise ValueError('模型保存校验失败，原模型未修改')
    del state, checked
    if read_history(staged) != history:
        raise ValueError('训练哈希记录保存校验失败，原模型未修改')


def completion_message(report: dict) -> str:
    auxiliary = report.get('channel_updates', {})
    state = auxiliary.get('local_classifier')
    detail = ('；参考条件分类器已校准' if state == 'recalibrated' else
              '；参考条件分类器校准失败，保留原配置' if state == 'failed' else '')
    if report['status'] == 'validation_unavailable':
        return '全局 WavLM 权重保留：来源不足，经验已进入全局回放池' + detail
    if report['status'] == 'rejected_holdout':
        return '全局 WavLM 权重保留：未通过留出集非退化验证' + detail
    if report['status'] == 'already_applied':
        return f"已跳过 {report['duplicate_packages']} 份重复 ZIP；没有新增训练样本"
    return (f"全局强化完成：处理 {report['new_packages']} 份新 ZIP，跳过 {report['duplicate_packages']} 份重复 ZIP；"
            '全局通用 WavLM 已原子更新，.previous 可回滚' + detail)
