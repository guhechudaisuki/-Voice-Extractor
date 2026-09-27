"""Immutable grouped references; never absorb episode predictions as labels."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Iterable

from .timeline import SampleSpan, require_digest


@dataclass(frozen=True)
class Reference:
    clip_digest: str
    source_digest: str
    source_span: SampleSpan
    role: str  # "target" or an opaque user-provided exclusion-group ID
    raw_digest: str
    stem_digest: str
    # Near-duplicates are supplied by a verified audio-fingerprint stage,
    # not inferred from speaker cosine (same voice is not duplicate audio).
    duplicate_group: str | None = None

    def __post_init__(self) -> None:
        for value in (self.clip_digest, self.source_digest, self.raw_digest, self.stem_digest):
            require_digest(value)
        if not self.role:
            raise ValueError("Reference requires a role group")


class ReferenceBank:
    def __init__(self, entries: Iterable[Reference]):
        unique: dict[str, Reference] = {}
        duplicates: dict[str, str] = {}
        seen_clips: dict[str, Reference] = {}
        for entry in sorted(entries, key=lambda row: row.clip_digest):
            key = entry.duplicate_group or entry.clip_digest
            previous_role = duplicates.get(key)
            if previous_role is not None and previous_role != entry.role:
                raise ValueError("Same reference audio occurs in conflicting role groups")
            previous = seen_clips.get(entry.clip_digest)
            if previous is not None and previous != entry:
                raise ValueError("Same clip fingerprint has conflicting metadata")
            seen_clips[entry.clip_digest] = entry
            if previous_role is not None:
                continue
            unique[entry.clip_digest] = entry
            duplicates[key] = entry.role
        self._entries = tuple(unique.values())
        if not self.target:
            raise ValueError("At least one target reference is required")

    @property
    def entries(self) -> tuple[Reference, ...]:
        return self._entries

    @property
    def target(self) -> tuple[Reference, ...]:
        return tuple(row for row in self._entries if row.role == "target")

    @property
    def excluded(self) -> dict[str, tuple[Reference, ...]]:
        roles = sorted({row.role for row in self._entries if row.role != "target"})
        return {role: tuple(row for row in self._entries if row.role == role) for role in roles}

    @property
    def digest(self) -> str:
        payload = json.dumps([asdict(row) for row in self._entries], sort_keys=True,
                             separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def independent_of(self, source_digest: str, span: SampleSpan) -> tuple[Reference, ...]:
        """Return only references with no waveform overlap with the query."""
        require_digest(source_digest)
        return tuple(row for row in self._entries if row.source_digest != source_digest
                     or row.source_span.intersection(span) is None)

    def cache_key(self, model_digest: str, preprocessing_version: str) -> str:
        require_digest(model_digest)
        if not preprocessing_version:
            raise ValueError("Cache key requires the preprocessing version")
        return hashlib.sha256(json.dumps([self.digest, model_digest, preprocessing_version])
                              .encode("utf-8")).hexdigest()
