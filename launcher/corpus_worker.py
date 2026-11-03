"""Run model/audio corpus operations in the packaged Python, outside frozen Tk."""
import json
import os
from pathlib import Path
import subprocess
import sys


def global_model_paths():
    from extractor.model_store import active_speaker_path
    if getattr(sys, 'frozen', False):
        resource = Path(sys.executable).resolve().parent / 'resource'
        model_dir = resource / 'model/speaker/wavlm-base-plus-sv'
        classifier = resource / 'models/anime_t3/classifier_v2.pt'
    else:
        from extractor.config import WAVLM_SV_MODEL, CLASSIFIER_V2_MODEL
        model_dir, classifier = Path(WAVLM_SV_MODEL), Path(CLASSIFIER_V2_MODEL)
    return {'speaker': active_speaker_path(model_dir),
            'auto_negatives': model_dir / 'auto_negatives.npz', 'classifier': classifier}


def _rollback_targets(paths):
    marker = paths['speaker'].parent / 'global_rollback.json'
    if not marker.is_file():
        return {name: True for name, path in paths.items() if path.with_suffix(path.suffix + '.previous').is_file()}
    metadata = json.loads(marker.read_text(encoding='utf-8'))
    if not isinstance(metadata, dict):
        raise ValueError('全局回滚备份清单无效')
    expected = [str(path.resolve()) for path in paths.values()]
    present = metadata.get('present')
    if (metadata.get('paths') != expected or not isinstance(present, list)
            or any(name not in expected for name in present)):
        raise ValueError('全局回滚备份与当前安装路径不一致')
    targets = {name: str(path.resolve()) in present for name, path in paths.items()}
    for name, existed in targets.items():
        path = paths[name]
        if existed and not path.with_suffix(path.suffix + '.previous').is_file():
            raise ValueError(f'全局回滚备份不完整：{path.name}')
    return targets


def rollback_status():
    try:
        targets = _rollback_targets(global_model_paths())
        return {'available': bool(targets), 'backups': list(targets), 'model_scope': 'global'}
    except (ValueError, OSError) as exc:
        return {'available': False, 'backups': [], 'model_scope': 'global', 'message': str(exc)}


def rollback_global_models():
    """Restore available global backups under the training lock, undoing partial failure."""
    from contextlib import ExitStack
    import shutil
    import tempfile
    from extractor.model_store import reinforcement_lock
    paths = global_model_paths()
    with reinforcement_lock(paths['speaker']), ExitStack() as stack:
        targets = _rollback_targets(paths)
        staged, restored = {}, []
        for name, existed in targets.items():
            path = paths[name]
            backup = path.with_suffix(path.suffix + '.previous')
            if not existed and not path.exists():
                continue
            directory = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='.rollback_', dir=path.parent)))
            if existed:
                shutil.copyfile(backup, directory / 'restore')
            if path.is_file():
                shutil.copyfile(path, directory / 'current')
            staged[name] = directory
        if not targets:
            raise ValueError('没有可回滚的全局模型备份')
        try:
            for name, directory in staged.items():
                if targets[name]:
                    os.replace(directory / 'restore', paths[name])
                else:
                    paths[name].unlink()
                restored.append(name)
        except OSError:
            for name in reversed(restored):
                current = staged[name] / 'current'
                if current.is_file():
                    os.replace(current, paths[name])
                else:
                    paths[name].unlink(missing_ok=True)
            raise
    return {'status': 'rolled_back', 'model_scope': 'global', 'restored': restored}


def invoke(operation, payload):
    from launcher.portable_resources import portable_environment
    resource = Path(sys.executable).resolve().parent / 'resource'
    # portable_environment takes the installation root and appends resource itself.
    environment = {**os.environ, **portable_environment(resource.parent)}
    environment['PYTHONPATH'] = os.pathsep.join(str(path) for path in
        (resource, resource / 'vendor', resource / 'python_packages', resource / 'dependencies'))
    environment['VOICE_EXTRACT_WAVLM_MODEL'] = str(resource / 'model/speaker/wavlm-base-plus-sv')
    environment['VOICE_EXTRACT_CLASSIFIER_V2_MODEL'] = str(resource / 'models/anime_t3/classifier_v2.pt')
    environment['VOICE_EXTRACT_SV_CODE'] = str(resource / 'dependencies/eres2net')
    environment['VOICE_EXTRACT_UVR_ROOT'] = str(resource / 'dependencies/uvr5')
    script = 'import json,sys; from launcher.corpus_worker import execute; print(json.dumps(execute(sys.argv[1],json.loads(sys.argv[2])),ensure_ascii=False))'
    result = subprocess.run([str(resource/'python/python.exe'), '-B', '-X', 'utf8', '-c', script,
                             operation, json.dumps(payload, ensure_ascii=False)],
        cwd=resource, env=environment, capture_output=True, text=True, encoding='utf-8', errors='replace',
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), timeout=7200)
    if result.returncode:
        raise RuntimeError('批次处理失败：' + (result.stderr or result.stdout)[-1500:])
    return json.loads(result.stdout.strip().splitlines()[-1])


def execute(operation, payload):
    if operation == 'rollback_status':
        return rollback_status()
    if operation == 'rollback':
        return rollback_global_models()
    if operation == 'rescore':
        from launcher.role_rescore import rescore_batch
        return rescore_batch(Path(payload['batch']))
    if operation == 'merge':
        from launcher.role_rescore import manifest_path, apply_rescore
        from launcher.review_store import read_review
        from launcher.dataset_merger import merge_datasets
        paths = [Path(path) for path in payload['batches']]
        batches = [(path, apply_rescore(path, json.loads(manifest_path(path).read_text(encoding='utf-8'))),
                    read_review(path).get('reviews', {})) for path in paths]
        return merge_datasets(batches, Path(payload['destination']), **payload['options'])
    raise ValueError('未知批次操作')
