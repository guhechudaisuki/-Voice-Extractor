"""Content-addressed, atomic dual-channel embedding cache."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_key(source_sha256: str, start: float, end: float,
              model_sha256: str, sample_rate: int = 16000) -> str:
    # Exact sample indexes avoid collisions between distinct cuts near a 10ms boundary.
    return hashlib.sha256(json.dumps([
        source_sha256, round(start * sample_rate), round(end * sample_rate),
        model_sha256, sample_rate, "mono-trim20ms-v1",
    ], separators=(",", ":")).encode()).hexdigest()


class EmbeddingCache:
    def __init__(self, root: Path, enabled: bool = True):
        self.root, self.enabled = Path(root), enabled
        self.hits = self.misses = 0

    def get_or_compute(self, key: str, compute):
        path = self.root / key[:2] / (key + ".npz")
        if self.enabled and path.is_file():
            try:
                with np.load(path, allow_pickle=False) as data:
                    vectors = (data["primary"].copy(), data["secondary"].copy())
                    if all(v.ndim == 1 and np.isfinite(v).all() and np.linalg.norm(v) > 0 for v in vectors):
                        self.hits += 1
                        return vectors
            except (OSError, ValueError, KeyError):
                pass
        self.misses += 1
        vectors = tuple(np.asarray(v, dtype=np.float32) for v in compute())
        if not all(v.ndim == 1 and np.isfinite(v).all() and np.linalg.norm(v) > 0 for v in vectors):
            raise ValueError("Invalid embedding; refusing to cache")
        if self.enabled:
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    np.savez_compressed(stream, primary=vectors[0], secondary=vectors[1])
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return vectors
