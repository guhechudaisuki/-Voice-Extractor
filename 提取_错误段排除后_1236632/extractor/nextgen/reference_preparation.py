"""Build an immutable, multi-reference bank from prepared single-speaker clips.

Only clean VAD-confirmed intervals contribute reference tokens. This stage
does not identify a person from a filename or character name. Distinct user
roles stay distinct; target and optional exclusion groups are never averaged.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import torch

from .features import WavLMSpeakerFeatures
from .inference import EncodedReference
from .reference_bank import Reference, ReferenceBank
from .scene_adapter import PreparedScene
from .timeline import SampleSpan


@dataclass(frozen=True)
class ReferenceMaterial:
    role: str
    scene: PreparedScene

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("Reference role cannot be empty")


def _fingerprint(samples: np.ndarray) -> str:
    """Conservative same-waveform fingerprint, invariant to a global gain."""
    if samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError("Invalid reference waveform")
    scale = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    if scale < 1e-5:
        raise ValueError("Silent reference cannot create a speaker token")
    quantized = np.round(np.clip(samples / scale, -8, 8) * 8).astype(np.int8)
    return hashlib.sha256(len(samples).to_bytes(8, "little") + quantized.tobytes()).hexdigest()


def _clean_islands(scene: PreparedScene) -> tuple[SampleSpan, ...]:
    blocked = (*scene.singing, *scene.overlap)
    return tuple(span for span in scene.stem_voice
                 if not any(span.intersection(mask) for mask in blocked))


def prepare_reference_bank(materials: tuple[ReferenceMaterial, ...],
                           encoder: WavLMSpeakerFeatures, *,
                           context_samples: int = 6000,
                           minimum_valid_frames: int = 12) -> tuple[ReferenceBank, tuple[EncodedReference, ...]]:
    if not materials or not any(item.role == "target" for item in materials):
        raise ValueError("At least one prepared target reference is required")
    if (type(context_samples) is not int or context_samples < 0
            or type(minimum_valid_frames) is not int or minimum_valid_frames < 1):
        raise ValueError("Invalid reference-context or evidence limits")
    entries: list[Reference] = []
    encoded: dict[str, EncodedReference] = {}
    roles_by_clip: dict[str, str] = {}
    for material in materials:
        scene = material.scene
        audio, source = scene.audio, scene.audio.source
        clean = set(_clean_islands(scene))
        islands = scene.stem_voice
        for index, speech in enumerate(islands):
            if speech not in clean:
                continue
            raw_signal, _ = audio.read_pair(speech)
            clip_digest = _fingerprint(raw_signal)
            previous_role = roles_by_clip.get(clip_digest)
            if previous_role is not None:
                if previous_role != material.role:
                    raise ValueError("Identical reference audio occurs in target and exclusion roles")
                continue
            previous_end = islands[index - 1].end if index else 0
            next_start = islands[index + 1].start if index + 1 < len(islands) else source.total_samples
            for mask in (*scene.singing, *scene.overlap):
                if mask.end <= speech.start:
                    previous_end = max(previous_end, mask.end)
                elif mask.start >= speech.end:
                    next_start = min(next_start, mask.start)
            # A reference token may attend to its entire local transformer
            # context. Never let a neighboring VAD island enter that context.
            context = SampleSpan(max(0, previous_end, speech.start - context_samples),
                                 min(source.total_samples, next_start, speech.end + context_samples))
            if context.length < getattr(getattr(encoder, "geometry", None),
                                        "convolution_support", 4000):
                continue
            raw, stem = audio.encode(context, encoder)
            _, signal = audio.read_pair(context)
            levels = []
            geometrically_valid = []
            for cell in stem.cells:
                geometrically_valid.append(speech.contains(cell))
                local = signal[cell.start - context.start:cell.end - context.start]
                levels.append(float(np.sqrt(np.mean(local.astype(np.float64) ** 2))))
            levels_tensor = torch.tensor(levels, dtype=torch.float32)
            valid = torch.tensor(geometrically_valid, dtype=torch.bool)
            if not valid.any():
                continue
            typical = float(levels_tensor[valid].median())
            if typical < 1e-5:
                continue
            # Quiet VAD-internal frames do not become identity evidence merely
            # because the whole reference file passed VAD.
            valid &= levels_tensor >= max(1e-5, typical * .15)
            if int(valid.sum()) < minimum_valid_frames:
                continue
            quality = (levels_tensor / max(typical, 1e-5)).clamp(0.0, 1.0)
            quality[~valid] = 0.0
            alignment = audio.alignment_report
            reference = Reference(clip_digest, source.source_sha256, context,
                                  material.role, alignment.raw_digest,
                                  alignment.stem_digest, duplicate_group=clip_digest)
            entries.append(reference)
            encoded[clip_digest] = EncodedReference(clip_digest, raw, stem, valid, quality)
            roles_by_clip[clip_digest] = material.role
    bank = ReferenceBank(entries)
    retained = tuple(encoded[item.clip_digest] for item in bank.entries)
    return bank, retained
