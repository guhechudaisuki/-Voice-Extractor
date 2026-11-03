"""Versioned, content-addressed target preprocessing (never reference identity)."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import soundfile as sf

from .config import CACHE_ROOT

SCHEMA_VERSION = 5
SPAN_FIELDS = ("pre_singing_spans", "initial_vad", "raw_initial_vad",
               "residual_singing", "singing_blocks", "overlap_evidence", "overlap_blocks")


def is_wave_file(path: Path) -> bool:
    """Check the RIFF container before handing a cache file to a decoder."""
    try:
        with path.open("rb") as stream:
            header = stream.read(12)
            size = os.fstat(stream.fileno()).st_size
        if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
            return False
        declared_size = int.from_bytes(header[4:8], "little") + 8
        return 44 <= declared_size <= size
    except OSError:
        return False


def _audio_info(path: Path, duration: float, *, raw: bool = False):
    if not is_wave_file(path):
        raise ValueError("Cached audio must be a complete RIFF/WAVE file")
    info = sf.info(str(path))
    if info.frames <= 0 or abs(info.duration - duration) > .05:
        raise ValueError("Cached audio must preserve the original timeline")
    if raw and (info.samplerate != 16000 or info.channels != 1):
        raise ValueError("Original cache audio must be 16 kHz mono")
    return info


def media_cache_key(source_sha256: str, singing_threshold: float,
                    use_singing: bool, use_overlap: bool, *,
                    uvr_model_sha256: str = "", overlap_threshold: float = .35) -> str:
    payload = [SCHEMA_VERSION, source_sha256, uvr_model_sha256,
               round(singing_threshold, 4), bool(use_singing),
               round(overlap_threshold, 4), bool(use_overlap)]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def media_cache_dir(source_sha256: str, singing_threshold: float,
                    use_singing: bool, use_overlap: bool, **settings) -> Path:
    key = media_cache_key(source_sha256, singing_threshold, use_singing, use_overlap, **settings)
    return CACHE_ROOT / "prepared_media" / key[:2] / key


def _valid_front(data: dict) -> bool:
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        return False
    duration = data.get("target_duration")
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        return False
    for field in (*SPAN_FIELDS, "detector_spans"):
        spans = data.get(field)
        if not isinstance(spans, list):
            return False
        for span in spans:
            if not isinstance(span, list) or len(span) != 2:
                return False
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in span):
                return False
            if not 0 <= span[0] < span[1] <= duration + .05:
                return False
    return True


def load_media_cache(cache_dir: Path) -> dict:
    """Read only fully published, valid audio pairs; legacy/partial entries miss."""
    try:
        data = json.loads((cache_dir / "front_stages.json").read_text(encoding="utf-8"))
        if not _valid_front(data) or data.get("cache_key") != cache_dir.name:
            return {}
        files = data.get("audio_files", {})
        for name in ("target_vocals.wav", "target_raw_16k.wav"):
            path = cache_dir / name
            _audio_info(path, data["target_duration"], raw=name == "target_raw_16k.wav")
            if path.stat().st_size != files.get(name, {}).get("bytes"):
                return {}
        return data
    except (OSError, RuntimeError, ValueError, TypeError, AttributeError):
        return {}


def media_cache_hit(source_sha256: str, singing_threshold: float,
                    use_singing: bool, use_overlap: bool, **settings) -> Path | None:
    if os.environ.get('VOICE_EXTRACT_PERSIST_CACHE') != '1':
        return None
    directory = media_cache_dir(source_sha256, singing_threshold, use_singing, use_overlap, **settings)
    data = load_media_cache(directory)
    if data.get("source_sha256") == source_sha256 and data.get("cache_key") == directory.name:
        return directory
    return None


def _atomic_copy(source: Path, destination: Path) -> None:
    fd, temporary = tempfile.mkstemp(dir=str(destination.parent), suffix=".wav.tmp")
    try:
        with os.fdopen(fd, "wb") as output, source.open("rb") as input_file:
            shutil.copyfileobj(input_file, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def _writer_lock(cache_dir: Path):
    """One publisher per key; OS locks are released even after process death.

    A contending pipeline can finish using its own work files. It need not wait
    for another writer to publish the identical cache key. Keep the lock inode
    in place so a third writer cannot acquire a different file with that name.
    """
    with (cache_dir / ".write.lock").open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def save_media_cache(cache_dir: Path, source_sha256: str,
                     vocals_src: Path, raw_16k_src: Path | None,
                     front_stages: dict) -> None:
    """Publish audio first, JSON last; interrupted writes cannot be cache hits.

    Writers for a key are serialized by an OS lock. Each file is replaced
    atomically; JSON length/header checks reject partial entries. A valid
    existing entry is never modified or copied onto itself.
    """
    if os.environ.get('VOICE_EXTRACT_PERSIST_CACHE') != '1':
        return
    data = dict(front_stages, schema_version=SCHEMA_VERSION,
                source_sha256=source_sha256, cache_key=cache_dir.name)
    if not _valid_front(data):
        raise ValueError("Incomplete or invalid media preprocessing metadata")
    if raw_16k_src is None:
        raise ValueError("The unmasked original 16 kHz waveform is required")
    cache_dir.mkdir(parents=True, exist_ok=True)
    with _writer_lock(cache_dir) as acquired:
        if acquired:
            _publish_media_cache(cache_dir, vocals_src, raw_16k_src, data)


def _publish_media_cache(cache_dir: Path, vocals_src: Path, raw_16k_src: Path, data: dict) -> None:
    existing = load_media_cache(cache_dir)
    if existing.get("source_sha256") == data["source_sha256"]:
        return
    sources = ((vocals_src, "target_vocals.wav"), (raw_16k_src, "target_raw_16k.wav"))
    for source, name in sources:
        _audio_info(source, data["target_duration"], raw=name == "target_raw_16k.wav")
    meta = cache_dir / "front_stages.json"
    meta.unlink(missing_ok=True)
    data["audio_files"] = {}
    for source, name in sources:
        destination = cache_dir / name
        _atomic_copy(source, destination)
        _audio_info(destination, data["target_duration"], raw=name == "target_raw_16k.wav")
        data["audio_files"][name] = {"bytes": destination.stat().st_size}
    fd, temporary = tempfile.mkstemp(dir=str(cache_dir), suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(data, output, ensure_ascii=False, allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, meta)
    finally:
        Path(temporary).unlink(missing_ok=True)
