"""Read paired audio intervals and cache their acoustic measurements separately."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .embedding_cache import cache_key, file_digest, canonical_interval
from .physiology import analyze_pair
from .feature_store import FeatureStore


def _read_interval(path, start, end, sample_rate):
    with sf.SoundFile(str(path)) as stream:
        first, last = round(start * stream.samplerate), round(end * stream.samplerate)
        if first < 0 or last > len(stream) or last <= first:
            raise ValueError("Audio interval exceeds the paired recording")
        stream.seek(first)
        samples = stream.read(last - first, dtype="float64", always_2d=True).mean(axis=1)
        source_rate = stream.samplerate
    if source_rate != sample_rate:
        divisor = math.gcd(source_rate, sample_rate)
        samples = resample_poly(samples, sample_rate // divisor, source_rate // divisor)
    expected = round(end * sample_rate) - round(start * sample_rate)
    if abs(len(samples) - expected) > 1:
        raise ValueError("Resampled interval has inconsistent duration")
    return samples[:expected]


class PhysiologyProbe:
    def __init__(self, stem, raw, source_hash, cache_root=None, *, enabled=True):
        self.stem, self.raw = Path(stem), Path(raw)
        persist = os.environ.get('VOICE_EXTRACT_PERSIST_CACHE') == '1'
        self.cache_root = Path(cache_root) if cache_root is not None and enabled and persist else None
        self.store = None
        self.sample_rate = 16000
        self.inputs = {"stem_sha256": source_hash, "raw_sha256": file_digest(self.raw)}
        code = [Path(__file__), Path(__file__).with_name("physiology.py")]
        self.version = hashlib.sha256("".join(file_digest(p) for p in code).encode()).hexdigest()
        self.source_key = hashlib.sha256(json.dumps(self.inputs, sort_keys=True).encode()).hexdigest()
        self.memory = {}
        self.decoded = {}
        self.hits = self.computed = self.unavailable = self.cache_write_errors = 0
        if self.cache_root:
            try:
                self.store = FeatureStore(self.cache_root / "features.sqlite3")
            except (OSError, sqlite3.Error):
                self.cache_write_errors += 1

    @staticmethod
    def _valid(value):
        return (isinstance(value, dict)
                and isinstance(value.get("available"), bool)
                and all(isinstance(value.get(name), (float, int))
                        and math.isfinite(value[name]) and 0. <= value[name] <= 1.
                        for name in ("artifact_confidence", "reliability"))
                and value["artifact_confidence"] <= value["reliability"]
                and (value["available"] or value["artifact_confidence"] == 0.))

    def measure(self, start, end):
        a, b = canonical_interval(start, end)
        key = cache_key(self.source_key, a, b, self.version, self.sample_rate)
        if key in self.memory:
            return self.memory[key]
        path = self.cache_root / key[:2] / (key + ".json") if self.cache_root else None
        try:
            blob = self.store.get_many([key]).get(key) if self.store else None
        except (OSError, sqlite3.Error):
            self.cache_write_errors += 1
            blob = None
        if blob is not None or (path is not None and path.is_file()):
            try:
                result = json.loads(blob if blob is not None else path.read_text(encoding="utf-8"))
                if self._valid(result):
                    self.hits += 1
                    self.memory[key] = result
                    return result
            except (OSError, ValueError):
                pass
        try:
            stem = self._read(self.stem, a, b)
            raw = self._read(self.raw, a, b)
            if abs(len(stem) - len(raw)) <= 1:
                length = min(len(stem), len(raw))
                stem, raw = stem[:length], raw[:length]
            result = analyze_pair(stem, raw, self.sample_rate)
            if not self._valid(result):
                raise ValueError("Invalid physiology measurement")
        except (OSError, RuntimeError, ValueError) as exc:
            result = {"available": False, "reason": type(exc).__name__,
                      "reliability": 0., "artifact_confidence": 0.}
            self.unavailable += 1
            self.memory[key] = result
            return result
        self.computed += 1
        result = {**result, "span": [float(start), float(end)], "version": self.version}
        self.memory[key] = result
        if self.store is not None:
            try:
                self.store.put_many([(key, json.dumps(result, allow_nan=False,
                    separators=(",", ":")).encode("utf-8"))])
            except (OSError, sqlite3.Error):
                self.cache_write_errors += 1
        return result

    def _read(self, path, start, end):
        if path not in self.decoded:
            try:
                return _read_interval(path, start, end, self.sample_rate)
            except sf.LibsndfileError:
                from ..audio import load_mono
                self.decoded[path] = load_mono(path, self.sample_rate).detach().cpu().numpy()
        waveform = self.decoded[path]
        first, last = round(start * self.sample_rate), round(end * self.sample_rate)
        if first < 0 or last > len(waveform) or last <= first:
            raise ValueError("Audio interval exceeds the paired recording")
        return waveform[first:last]

    def diagnostics(self):
        return {"enabled": True, "inputs": self.inputs, "version": self.version,
                "cache_hits": self.hits, "computed": self.computed,
                "unavailable": self.unavailable, "cache_write_errors": self.cache_write_errors,
                "measured_spans": len(self.memory),
                "reliable_spans": sum(bool(row.get("available")) and row.get("reliability", 0.) > 0.
                                      for row in self.memory.values())}
