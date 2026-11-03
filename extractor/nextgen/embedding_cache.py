"""Content-addressed, atomic dual-channel embedding cache."""
from __future__ import annotations

import hashlib
import json
import io
import os
from collections import OrderedDict
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
from .feature_store import FeatureStore


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


def canonical_interval(start, end, sample_rate=16000):
    """Half-open sample bounds, with no implicit edge trimming."""
    if not np.isfinite(start) or not np.isfinite(end) or end <= max(0., start):
        raise ValueError("Invalid audio interval")
    return round(max(0., float(start)) * sample_rate) / sample_rate, round(float(end) * sample_rate) / sample_rate


class EmbeddingCache:
    def __init__(self, root: Path, enabled: bool = True, *, writable=True):
        self.root, self.enabled = Path(root), enabled
        self.hits = self.misses = 0
        self.memory = OrderedDict()
        self.store = None
        self.writable = writable
        self.persist = os.environ.get("VOICE_EXTRACT_PERSIST_CACHE") == "1"

    def _store(self):
        if self.store is None:
            self.store = FeatureStore(self.root / "features.sqlite3")
        return self.store

    @staticmethod
    def _valid(vectors):
        return len(vectors) == 2 and all(
            v.ndim == 1 and np.isfinite(v).all() and np.linalg.norm(v) > 0 for v in vectors)

    def _remember(self, key, vectors):
        self.memory[key] = vectors
        self.memory.move_to_end(key)

    def get_many(self, keys):
        result = {key: self.memory[key] for key in keys if key in self.memory}
        missing = list(dict.fromkeys(key for key in keys if key not in result))
        blobs = self._store().get_many(missing) if self.enabled and self.persist and missing else {}
        for key in missing:
            legacy = self.root / key[:2] / (key + ".npz")
            source = io.BytesIO(blobs[key]) if key in blobs else legacy
            if not self.enabled or not self.persist or (key not in blobs and not legacy.is_file()):
                continue
            try:
                with np.load(source, allow_pickle=False) as data:
                    vectors = (data["primary"].copy(), data["secondary"].copy())
                if self._valid(vectors):
                    result[key] = vectors
                    self._remember(key, vectors)
            except (OSError, ValueError, KeyError, EOFError, BadZipFile):
                continue
        self.hits += len(result)
        self.misses += len(set(keys) - result.keys())
        return result

    def put_many(self, values):
        blobs = []
        for key, vectors in values:
            vectors = tuple(np.asarray(v, dtype=np.float32) for v in vectors)
            if not self._valid(vectors):
                raise ValueError("Invalid embedding; refusing to cache")
            self._remember(key, vectors)
            if self.enabled and self.persist and self.writable:
                stream = io.BytesIO()
                np.savez(stream, primary=vectors[0], secondary=vectors[1])
                blobs.append((key, stream.getvalue()))
        if blobs:
            self._store().put_many(blobs)

    def get_or_compute(self, key: str, compute):
        cached = self.get_many([key])
        if key in cached:
            return cached[key]
        vectors = tuple(np.asarray(v, dtype=np.float32) for v in compute())
        self.put_many([(key, vectors)])
        return vectors
