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
        torch.save(state, staged)
        checked = torch.load(staged, map_location='cpu', weights_only=True)
    for key, value in projection.state_dict().items():
        if not torch.equal(checked['feature_extractor.' + key], value.detach().cpu()):
            raise ValueError('模型保存校验失败，原模型未修改')
    del state, checked
    if read_history(staged) != history:
        raise ValueError('训练哈希记录保存校验失败，原模型未修改')


def completion_message(report: dict) -> str:
    if report['status'] == 'already_applied':
        return f"已跳过 {report['duplicate_packages']} 份重复 ZIP，没有新增 ZIP，模型未改动"
    return (f"强化完成：处理 {report['new_packages']} 份新 ZIP，跳过 {report['duplicate_packages']} 份重复 ZIP；"
            '已自动保存回现用模型')
