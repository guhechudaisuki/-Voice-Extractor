"""Frozen local anime-speaker witness used by the production merge stage.

This is not a trained per-person classifier. It compares a candidate part to
the user-supplied target references and optional anonymous exclusion groups
with two existing, locally installed anime-domain encoders. The witness can
rescue a same-speaker join only when both sides independently support the
target; unresolved evidence never opens the production gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .nextgen.anime_embedding import AnimeSpeakerOnnx


@dataclass(frozen=True)
class AnimeDomainScore:
    state: str
    target_margins: tuple[float, ...]
    other_roles: tuple[str, ...]


def _paired_files(stem: Sequence[Path], raw: Sequence[Path]) -> list[tuple[Path, Path]]:
    stem_by_name = {Path(path).name: Path(path) for path in stem if path}
    raw_by_name = {Path(path).name: Path(path) for path in raw if path}
    if not stem_by_name or stem_by_name.keys() != raw_by_name.keys():
        raise ValueError("Anime witness requires matching raw/stem reference clips")
    return [(stem_by_name[name], raw_by_name[name]) for name in sorted(stem_by_name)]


class AnimeDomainWitness:
    """A conservative four-view witness: char/va x stem/raw."""

    def __init__(self, models_root: str | Path,
                 target_stem: Sequence[Path], target_raw: Sequence[Path],
                 negative_stem: Sequence[Sequence[Path]] = (),
                 negative_raw: Sequence[Sequence[Path]] = ()):
        self.encoders = {
            variant: AnimeSpeakerOnnx(models_root, variant)
            for variant in ("char", "va")
        }
        target_pairs = _paired_files(target_stem, target_raw)
        if not target_pairs:
            raise ValueError("Anime witness needs at least one target reference")
        raw_groups = list(negative_raw)
        stem_groups = list(negative_stem)
        if len(raw_groups) != len(stem_groups):
            raise ValueError("Anime witness negative raw/stem groups differ")
        self._vectors: dict[tuple[str, str, str], np.ndarray] = {}
        self._roles: list[str] = []
        groups = [("target", target_pairs)]
        for index, (stem_group, raw_group) in enumerate(zip(stem_groups, raw_groups), 1):
            pairs = _paired_files(stem_group, raw_group)
            if pairs:
                groups.append((f"role_{index:03d}", pairs))
        for variant, encoder in self.encoders.items():
            for channel in ("stem", "raw"):
                for role, pairs in groups:
                    vectors = [
                        encoder.encode(self._read(path, channel), sample_rate=16000)
                        for stem_path, raw_path in pairs
                        for path in ([stem_path] if channel == "stem" else [raw_path])
                    ]
                    self._vectors[(variant, channel, role)] = np.stack(vectors)
        self._roles = [role for role, _pairs in groups if role != "target"]

    @staticmethod
    def _read(path: Path, _channel: str) -> torch.Tensor:
        from .audio import load_mono

        return load_mono(path, 16000)

    def score(self, stem: torch.Tensor, raw: torch.Tensor) -> AnimeDomainScore:
        if abs(stem.numel() - raw.numel()) > 800:
            return AnimeDomainScore("unresolved", (), ())
        margins: list[float] = []
        winners: list[str] = []
        for variant, encoder in self.encoders.items():
            for channel, waveform in (("stem", stem), ("raw", raw)):
                try:
                    vector = encoder.encode(waveform, sample_rate=16000)
                except ValueError:
                    return AnimeDomainScore("unresolved", (), ())
                target = float(np.median(self._vectors[(variant, channel, "target")] @ vector))
                role_scores = {
                    role: float((self._vectors[(variant, channel, role)] @ vector).max())
                    for role in self._roles
                }
                if not role_scores:
                    return AnimeDomainScore("unresolved", (), ())
                role = max(role_scores, key=role_scores.get)
                margins.append(target - role_scores[role])
                winners.append(role)
        positive = sum(value > 0.0 for value in margins)
        target_supported = positive >= 2 and any(
            margins[index] > 0.0 for index in (0, 2)
        )
        other_supported = (
            winners[0] == winners[2]
            and all(margins[index] < 0.0 for index in (0, 2))
            and any(winners[index] == winners[0] and margins[index] < 0.0
                    for index in (1, 3))
        )
        state = "target_supported" if target_supported else (
            "other_supported" if other_supported else "unresolved"
        )
        return AnimeDomainScore(state, tuple(margins), tuple(winners))
