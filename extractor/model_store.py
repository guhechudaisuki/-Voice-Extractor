"""Select and atomically update installed models used by extraction."""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import shutil
import tempfile


def active_classifier_path(asset_root: Path) -> Path:
    directory = asset_root / 'models' / 'anime_t3'
    reviewed = directory / 'classifier_v2.pt'
    return reviewed if reviewed.is_file() else directory / 'classifier.pt'


def active_speaker_path(model_dir: Path) -> Path:
    safe = model_dir / 'model.safetensors'
    return safe if safe.is_file() else model_dir / 'pytorch_model.bin'


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
    lock = path.with_suffix(path.suffix + '.reinforce.lock').open('a+b')
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
