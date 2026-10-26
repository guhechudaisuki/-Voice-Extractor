"""Content-addressed persistent cache for UVR/VAD/Singing stage outputs.

同一集视频（如 24 分钟 MKV）在多角色提取时会被重复跑 UVR + SingingDetector
+ VAD + OverlapDetector 多次。此模块按源文件 SHA-256 缓存全部中间产物，
同一集换角色重跑时 0.5 秒直达声纹核验阶段（提速 10 倍+）。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from .config import CACHE_ROOT


def media_cache_key(source_sha256: str, singing_threshold: float,
                    use_singing: bool, use_overlap: bool) -> str:
    return hashlib.sha256(json.dumps(
        [source_sha256, singing_threshold, use_singing, use_overlap],
        separators=(",", ":")).encode()).hexdigest()


def media_cache_dir(source_sha256: str, singing_threshold: float,
                    use_singing: bool, use_overlap: bool) -> Path:
    key = media_cache_key(source_sha256, singing_threshold, use_singing, use_overlap)
    return CACHE_ROOT / "prepared_media" / key[:2] / key


def media_cache_hit(source_sha256: str, singing_threshold: float,
                    use_singing: bool, use_overlap: bool) -> Path | None:
    cache_dir = media_cache_dir(source_sha256, singing_threshold, use_singing, use_overlap)
    meta = cache_dir / "front_stages.json"
    if not meta.is_file():
        return None
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    vocals = cache_dir / "target_vocals.wav"
    raw_16k = cache_dir / "target_raw_16k.wav"
    if not vocals.is_file():
        return None
    if data.get("source_sha256") != source_sha256:
        return None
    return cache_dir


def save_media_cache(cache_dir: Path, source_sha256: str,
                     vocals_src: Path, raw_16k_src: Path | None,
                     front_stages: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    vocals_dst = cache_dir / "target_vocals.wav"
    if not vocals_dst.is_file():
        shutil.copyfile(vocals_src, vocals_dst)
    if raw_16k_src is not None and raw_16k_src.is_file():
        raw_dst = cache_dir / "target_raw_16k.wav"
        if not raw_dst.is_file():
            shutil.copyfile(raw_16k_src, raw_dst)
    front_stages["source_sha256"] = source_sha256
    meta = cache_dir / "front_stages.json"
    fd, tmp = tempfile.mkstemp(dir=str(cache_dir), suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(front_stages, f, ensure_ascii=False)
        os.replace(tmp, str(meta))
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load_media_cache(cache_dir: Path) -> dict:
    meta = cache_dir / "front_stages.json"
    if not meta.is_file():
        return {}
    try:
        return json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
