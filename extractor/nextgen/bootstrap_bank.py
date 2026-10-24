"""Target embedding banks with validated, content-bound training provenance."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class BootstrapVector:
    source_sha256: str
    start: float
    end: float
    duration: float
    primary: np.ndarray
    secondary: np.ndarray
    confidence: float = 0.0
    # Transitive evidence used to certify this pseudo-label, excluding itself.
    supports: tuple[tuple[str, float, float], ...] = ()


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    if vector.ndim != 1 or not vector.size:
        raise ValueError("bootstrap embedding must be a non-empty vector")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-8:
        raise ValueError("bootstrap vectors must be finite and non-zero")
    return vector / norm


def deduplicate(vectors: Iterable[BootstrapVector], *, max_items: int = 500,
                cosine_floor: float = 0.95) -> list[BootstrapVector]:
    """Keep high-confidence, duration-weighted evidence while removing repeats."""
    if max_items <= 0 or not 0.0 <= cosine_floor <= 1.0:
        raise ValueError("invalid bootstrap bank limits")
    candidates = sorted(vectors, key=lambda item: (-item.confidence, -item.duration,
                                                   item.source_sha256, item.start, item.end))
    kept: list[BootstrapVector] = []
    primary: list[np.ndarray] = []
    secondary: list[np.ndarray] = []
    for item in candidates:
        if (not np.isfinite([item.start, item.end, item.duration, item.confidence]).all()
                or not item.source_sha256
                or item.start < 0 or item.end <= item.start or item.duration <= 0
                or item.duration > item.end - item.start + 1e-5):
            raise ValueError("invalid bootstrap interval or duration")
        if any(not source or not np.isfinite([a, b]).all() or a < 0 or b <= a
               for source, a, b in item.supports):
            raise ValueError("invalid bootstrap evidence interval")
        p, s = _unit(item.primary), _unit(item.secondary)
        if primary and (p.shape != primary[0].shape or s.shape != secondary[0].shape):
            raise ValueError("bootstrap embedding dimensions are inconsistent")
        if any(float(p @ other) >= cosine_floor and float(s @ other_s) >= cosine_floor
               for other, other_s in zip(primary, secondary)):
            continue
        kept.append(BootstrapVector(item.source_sha256, float(item.start), float(item.end),
                                    max(0.0, float(item.duration)), p, s,
                                    float(item.confidence), item.supports))
        primary.append(p)
        secondary.append(s)
        if len(kept) >= max_items:
            break
    return kept


def save(path: Path, vectors: Iterable[BootstrapVector], *, source_ids: Iterable[str],
         schema: int = 1, context: dict | None = None) -> dict:
    selected = deduplicate(vectors)
    if not selected:
        raise ValueError("cannot save an empty bootstrap bank")
    meta = {
        "schema": schema,
        "count": len(selected),
        "source_ids": sorted(set(source_ids)),
        "dedup_cosine_floor": 0.95,
        "purpose": "development_target_evidence_only",
    }
    if context is not None:
        meta.update(schema=2, purpose="automatic_cluster_bootstrap", context=context)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(
                stream,
                source_sha256=np.array([item.source_sha256 for item in selected]),
                start=np.array([item.start for item in selected]),
                end=np.array([item.end for item in selected]),
                duration=np.array([item.duration for item in selected]),
                confidence=np.array([item.confidence for item in selected]),
                primary=np.stack([item.primary for item in selected]),
                secondary=np.stack([item.secondary for item in selected]),
                supports=np.array([json.dumps(item.supports) for item in selected]),
                meta=np.array(json.dumps(meta, ensure_ascii=False)),
            )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    meta["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return meta


def load(path: Path) -> tuple[dict, list[BootstrapVector]]:
    with np.load(path, allow_pickle=False) as payload:
        meta = json.loads(str(payload["meta"]))
        if type(meta.get("count")) is not int or meta["count"] <= 0:
            raise ValueError("bootstrap bank must have a positive vector count")
        names = ("source_sha256", "start", "end", "duration", "primary", "secondary", "confidence")
        if any(len(payload[name]) != meta.get("count") for name in names):
            raise ValueError("bootstrap bank array lengths mismatch")
        supports = ([json.loads(str(row)) for row in payload["supports"]]
                    if "supports" in payload else [[] for _ in range(meta["count"])])
        if not supports or len(supports) != meta["count"]:
            raise ValueError("bootstrap bank evidence lengths mismatch")
        vectors = [BootstrapVector(str(source), float(start), float(end), float(duration),
                                    _unit(primary), _unit(secondary), float(confidence),
                                    tuple((str(h), float(a), float(b)) for h, a, b in evidence))
                   for source, start, end, duration, primary, secondary, confidence, evidence in zip(
                       payload["source_sha256"], payload["start"], payload["end"],
                       payload["duration"], payload["primary"], payload["secondary"],
                       payload["confidence"], supports)]
    if meta.get("count") != len(vectors):
        raise ValueError("bootstrap bank metadata count mismatch")
    # Validate every row; do not silently truncate malformed banks.
    deduplicate(vectors, max_items=max(1, len(vectors)))
    return meta, vectors


def independent_vectors(vectors, source_sha256, start, end):
    """Exclude the current turn, overlapping turns and their descendants."""
    return [row for row in vectors if not any(
        source == source_sha256 and min(end, b) > max(start, a)
        for source, a, b in ((row.source_sha256, row.start, row.end), *row.supports))]


def evidence_spans(vectors):
    return tuple(sorted({span for row in vectors
                         for span in ((row.source_sha256, row.start, row.end), *row.supports)}))


def merge_profile(profile, vectors):
    """Duration-weighted evidence with half the weight anchored to user input."""
    from .identity_cluster import TargetProfile
    rows = list(vectors)
    if not rows:
        return profile
    weights = np.array([row.duration for row in rows], dtype=np.float64)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("bootstrap duration weights must be positive and finite")
    weights /= weights.sum()
    centers = []
    for name, original in (("primary", profile.primary_centroid),
                           ("secondary", profile.secondary_centroid)):
        matrix = np.stack([getattr(row, name) for row in rows])
        if matrix.shape[1:] != original.shape:
            raise ValueError("bootstrap embedding dimensions differ from reference")
        # Keep the original identity anchor at half the weight irrespective of
        # bank size. Reference floors remain fixed across all rounds.
        centers.append(_unit(_unit(weights @ matrix) + _unit(original)))
    return TargetProfile(*centers, profile.primary_reference_floor,
                         profile.secondary_reference_floor, source="merged")
