"""Select and atomically update installed models used by extraction."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def reference_key(checksums):
    """Identify a reference set without selecting or creating a private model."""
    return hashlib.sha256(json.dumps(sorted(set(checksums)), separators=(",", ":")).encode()).hexdigest()


def atomic_save_projection(path, payload):
    """Publish a complete Torch artifact with an atomic file replacement."""
    import torch
    path = Path(path)
    descriptor, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def active_classifier_path(asset_root: Path) -> Path:
    directory = asset_root / 'models' / 'anime_t3'
    reviewed = directory / 'classifier_v2.pt'
    return reviewed if reviewed.is_file() else directory / 'classifier.pt'


def active_speaker_path(model_dir: Path) -> Path:
    binary = model_dir / 'pytorch_model.bin'
    if binary.is_file():
        return binary
    safe = model_dir / 'model.safetensors'
    return safe if safe.is_file() else model_dir / 'pytorch_model.bin'


def snapshot_global_models(current: Path, classifier: Path):
    """Back up all global inference artifacts, including previously absent files."""
    paths = [Path(current), Path(classifier)]
    marker = Path(current).parent / 'global_rollback.json'
    # Finish every potentially large copy before replacing the previous backup
    # generation. A failed auxiliary copy must leave the old rollback intact.
    with tempfile.TemporaryDirectory(prefix='.global_snapshot_', dir=marker.parent) as temporary:
        staging = Path(temporary)
        present = []
        for index, path in enumerate(paths):
            if path.is_file():
                shutil.copyfile(path, staging / str(index))
                present.append(str(path.resolve()))
        staged_marker = staging / 'marker.json'
        staged_marker.write_text(json.dumps({'paths': [str(p.resolve()) for p in paths],
                                             'present': present}), encoding='utf-8')
        old, published = [], []
        try:
            for index, path in enumerate(paths):
                backup = path.with_suffix(path.suffix + '.previous')
                if backup.exists():
                    previous = staging / f'old_{index}'
                    backup.replace(previous)
                    old.append((previous, backup))
                if (staging / str(index)).exists():
                    (staging / str(index)).replace(backup)
                    published.append(backup)
            staged_marker.replace(marker)
        except OSError:
            for backup in published:
                backup.unlink(missing_ok=True)
            for previous, backup in old:
                previous.replace(backup)
            raise


def rollback_global_models(model_dir: Path, classifier: Path):
    """Restore the last global update under the same training writer lock."""
    import json
    current = active_speaker_path(Path(model_dir))
    with reinforcement_lock(current):
        marker = Path(model_dir) / 'global_rollback.json'
        metadata = json.loads(marker.read_text(encoding='utf-8'))
        paths = [current, Path(classifier)]
        present = set(metadata['present'])
        if not {str(p.resolve()) for p in paths}.issubset(metadata['paths']):
            raise ValueError('全局回滚备份与当前安装路径不一致')
        for path in paths:
            if str(path.resolve()) in present and not path.with_suffix(path.suffix + '.previous').is_file():
                raise ValueError(f'全局回滚备份不完整：{path.name}')
        for path in paths:
            if str(path.resolve()) in present:
                temporary = path.with_suffix(path.suffix + '.rollback')
                shutil.copyfile(path.with_suffix(path.suffix + '.previous'), temporary)
                temporary.replace(path)
            else:
                path.unlink(missing_ok=True)
        marker.unlink()
    return {'status': 'rolled_back', 'scope': 'universal', 'model_path': str(current)}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def reinforcement_lock(path: Path):
    if not path.is_file():
        raise ValueError(f'未找到现用人物识别模型：{path}')
    # Stable across a safetensors -> binary publication in this model directory.
    lock = (path.parent / '.reinforce.lock').open('a+b')
    acquired = False
    try:
        if lock.tell() == 0:
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError as exc:
            raise ValueError('此模型正在被另一个窗口强化，请等待完成') from exc
        yield
    finally:
        if acquired:
            lock.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def replace_model(staged: Path, current: Path, expected_sha256: str):
    """Caller holds reinforcement_lock; readers see one complete checkpoint."""
    if file_sha256(current) != expected_sha256:
        raise ValueError('现用模型在训练期间发生变化，本次未覆盖，请重新强化')
    descriptor, name = tempfile.mkstemp(prefix='.reinforced_', suffix='.tmp', dir=current.parent)
    temporary = Path(name)
    backup = current.with_suffix(current.suffix + '.previous')
    backup_temp = temporary.with_suffix('.backup')
    try:
        with os.fdopen(descriptor, 'wb') as output, staged.open('rb') as source:
            shutil.copyfileobj(source, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        shutil.copyfile(current, backup_temp)
        backup_temp.replace(backup)
        temporary.replace(current)
    finally:
        temporary.unlink(missing_ok=True)
        backup_temp.unlink(missing_ok=True)
