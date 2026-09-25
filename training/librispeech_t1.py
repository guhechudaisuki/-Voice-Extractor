"""Build speaker-disjoint, full-utterance T1 feature scenes from LibriSpeech.

This is an offline research data builder, never part of the desktop runtime.
Synthetic joins are not a substitute for naturally alternating anime dialogue.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extractor.speaker import WavLMSpeakerVerifier  # noqa: E402


PATTERNS = (
    ("target", "other"),
    ("other", "target"),
    ("target", "other", "target"),
    ("target", "target"),
    ("other",),
    ("other", "other"),
)
PATTERN_NAMES = (
    "target_other", "other_target", "target_other_target",
    "target_target", "other_only", "other_other",
)
GAPS_SECONDS = (0.0, 0.12, 0.35, 0.75)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_utterances(
    corpus: Path, *, minimum_seconds: float = 1.2, maximum_seconds: float = 4.5,
) -> dict[str, list[Path]]:
    """Only intact mono 16 kHz files are eligible; no sentence is cropped."""
    speakers: dict[str, list[Path]] = {}
    for path in sorted(corpus.glob("*/*/*.flac")):
        info = sf.info(path)
        if (
            info.samplerate != 16000 or info.channels != 1
            or not minimum_seconds <= info.duration <= maximum_seconds
        ):
            continue
        speakers.setdefault(path.parent.parent.name, []).append(path)
    return {speaker: files for speaker, files in speakers.items() if len(files) >= 6}


def split_speakers(
    speakers: dict[str, list[Path]], *, seed: int, validation_fraction: float = 0.2,
) -> tuple[list[str], list[str]]:
    if not 0 < validation_fraction < 1 or len(speakers) < 10:
        raise ValueError("Need at least 10 speakers and a nonempty validation split")
    names = sorted(speakers)
    random.Random(seed).shuffle(names)
    validation_count = max(2, round(len(names) * validation_fraction))
    return sorted(names[validation_count:]), sorted(names[:validation_count])


def _different_reference(
    files: list[Path], excluded: set[Path], rng: random.Random,
) -> Path:
    options = [path for path in files if path not in excluded]
    if not options:
        raise ValueError("Reference must be a different utterance from every query")
    chapters = {path.parent for path in excluded}
    cross_chapter = [path for path in options if path.parent not in chapters]
    return rng.choice(cross_chapter or options)


def plan_scene(
    files_by_speaker: dict[str, list[Path]], speaker_pool: list[str],
    index: int, rng: random.Random,
) -> dict:
    if len(speaker_pool) < 2:
        raise ValueError("A scene needs distinct target and other speakers")
    target, other = rng.sample(speaker_pool, 2)
    pattern = PATTERNS[index % len(PATTERNS)]
    target_count = pattern.count("target")
    other_count = pattern.count("other")
    target_query = rng.sample(files_by_speaker[target], target_count)
    other_query = rng.sample(files_by_speaker[other], other_count)
    target_refs: list[Path] = []
    excluded_target = set(target_query)
    requested_target_references = 1 + index % 7
    reference_count = min(requested_target_references, len(files_by_speaker[target]) - target_count)
    for _ in range(reference_count):
        reference = _different_reference(files_by_speaker[target], excluded_target, rng)
        target_refs.append(reference)
        excluded_target.add(reference)
    negative_group_count = min(index % 6, len(speaker_pool) - 1)
    negative_speakers: list[str] = []
    if negative_group_count and (index // 6) % 2 == 0:
        negative_speakers.append(other)
    distractors = [speaker for speaker in speaker_pool if speaker not in (target, other)]
    negative_speakers.extend(
        rng.sample(distractors, negative_group_count - len(negative_speakers))
    )
    negative_groups = []
    for group_index, speaker in enumerate(negative_speakers):
        excluded = set(other_query) if speaker == other else set()
        group_refs = []
        for _ in range(1 + (index + group_index) % 2):
            reference = _different_reference(files_by_speaker[speaker], excluded, rng)
            group_refs.append(reference)
            excluded.add(reference)
        negative_groups.append({
            "speaker": speaker,
            "paths": [str(path) for path in group_refs],
        })
    roles = {"target": iter(target_query), "other": iter(other_query)}
    return {
        "target_speaker": target,
        "other_speaker": other,
        "pattern_name": PATTERN_NAMES[index % len(PATTERNS)],
        "pattern": list(pattern),
        "query_paths": [str(next(roles[role])) for role in pattern],
        "target_reference_paths": [str(path) for path in target_refs],
        "negative_reference_groups": negative_groups,
        "gap_seconds": [rng.choice(GAPS_SECONDS) for _ in range(len(pattern) - 1)],
    }


def load_audio(path: Path) -> torch.Tensor:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if sample_rate != 16000 or audio.ndim != 1:
        raise ValueError(f"Expected 16 kHz mono audio: {path}")
    return torch.from_numpy(np.ascontiguousarray(audio))


def assemble_scene(plan: dict) -> tuple[torch.Tensor, list[tuple[int, int, str]]]:
    pieces: list[torch.Tensor] = []
    intervals: list[tuple[int, int, str]] = []
    cursor = 0
    for index, (path, role) in enumerate(zip(plan["query_paths"], plan["pattern"])):
        if index:
            gap = torch.zeros(round(plan["gap_seconds"][index - 1] * 16000))
            pieces.append(gap)
            cursor += gap.numel()
        waveform = load_audio(Path(path))
        pieces.append(waveform)
        intervals.append((cursor, cursor + waveform.numel(), role))
        cursor += waveform.numel()
    return torch.cat(pieces), intervals


def frame_labels(
    frame_count: int, intervals: list[tuple[int, int, str]],
    *, frame_hop_samples: int = 320, frame_center_samples: int = 200,
    uncertain_edge_samples: int = 1600,
    waveform: torch.Tensor | None = None,
) -> torch.Tensor:
    """Conservative labels: edges and quiet in-utterance frames are unknown."""
    labels = torch.zeros(frame_count, 4)
    if not intervals or frame_count < 1:
        raise ValueError("Expected a nonempty scene and feature timeline")
    centers = torch.arange(frame_count) * frame_hop_samples + frame_center_samples
    energy = None
    if waveform is not None:
        if waveform.ndim != 1:
            raise ValueError("Expected mono waveform for speech-activity labels")
        samples = torch.arange(-200, 200)[None, :] + centers[:, None]
        valid = (samples >= 0) & (samples < waveform.numel())
        samples = samples.clamp(0, waveform.numel() - 1)
        energy = ((waveform[samples] * valid).square().mean(dim=1) + 1e-12).sqrt()
    for start, end, role in intervals:
        if role not in ("target", "other"):
            raise ValueError(f"Unknown role: {role}")
        inside = (centers >= start + uncertain_edge_samples) & (
            centers < end - uncertain_edge_samples
        )
        near_edge = ((centers >= start - uncertain_edge_samples) & (
            centers < start + uncertain_edge_samples
        )) | ((centers >= end - uncertain_edge_samples) & (
            centers < end + uncertain_edge_samples))
        labels[inside, 0 if role == "target" else 1] = 1.0
        labels[inside, 2] = 1.0
        if energy is not None and inside.any():
            # A clean audiobook file can still contain internal pauses. Such
            # frames have no speaker identity evidence and must not become
            # positive training examples for either speaker or speech.
            utterance_energy = energy[(centers >= start) & (centers < end)]
            reference_level = torch.quantile(utterance_energy, 0.90)
            quiet = inside & (energy < max(0.0005, float(reference_level) * 0.02))
            labels[quiet, :3] = -1.0
        labels[near_edge, :3] = -1.0
    for (_, _, left_role), (right_start, _, right_role) in zip(intervals, intervals[1:]):
        if left_role != right_role:
            near_change = (centers >= right_start - uncertain_edge_samples) & (
                centers < right_start + uncertain_edge_samples
            )
            labels[near_change, 3] = 1.0
    return labels


def encode_frames(
    verifier: WavLMSpeakerVerifier, waveform: torch.Tensor, layer: int,
) -> torch.Tensor:
    inputs = verifier.feature_extractor(
        waveform.numpy(), sampling_rate=16000, return_tensors="pt",
    )
    with torch.inference_mode():
        states = verifier.model.wavlm(
            input_values=inputs["input_values"].to(verifier.device),
            output_hidden_states=True,
        ).hidden_states
    if layer < 0 or layer >= len(states):
        raise ValueError(f"WavLM layer {layer} is unavailable")
    return states[layer][0].float().cpu()


def build_features(
    corpus: Path, output_dir: Path, *, train_scenes: int, validation_scenes: int,
    seed: int, layer: int,
) -> dict:
    if min(train_scenes, validation_scenes) < 1:
        raise ValueError("Both training and validation require scenes")
    files = index_utterances(corpus)
    train_speakers, validation_speakers = split_speakers(files, seed=seed)
    verifier = WavLMSpeakerVerifier()
    model_hash = sha256(ROOT / "model/speaker/wavlm-base-plus-sv/pytorch_model.bin")
    manifest = {
        "schema_version": 1,
        "purpose": "Frozen-feature T1 prototype; synthetic joins are not a real-dialogue test",
        "dataset": "LibriSpeech dev-clean, CC BY 4.0",
        "corpus": str(corpus.resolve()),
        "model_sha256": model_hash,
        "wavlm_hidden_layer": layer,
        "seed": seed,
        "train_speakers": train_speakers,
        "validation_speakers": validation_speakers,
        "scenes": [],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for split, pool, count in (
            ("train", train_speakers, train_scenes),
            ("validation", validation_speakers, validation_scenes),
        ):
            (output_dir / split).mkdir(exist_ok=True)
            rng = random.Random(seed + (0 if split == "train" else 1_000_000))
            for index in range(count):
                plan = plan_scene(files, pool, index, rng)
                scene, intervals = assemble_scene(plan)
                query = encode_frames(verifier, scene, layer)
                targets = torch.stack([
                    F.normalize(encode_frames(verifier, load_audio(Path(path)), layer).mean(0), dim=0)
                    for path in plan["target_reference_paths"]
                ])
                negative_groups = [torch.stack([
                    F.normalize(encode_frames(verifier, load_audio(Path(path)), layer).mean(0), dim=0)
                    for path in group["paths"]
                ]) for group in plan["negative_reference_groups"]]
                negatives = None
                negative_mask = None
                if negative_groups:
                    maximum_group_size = max(len(group) for group in negative_groups)
                    negatives = torch.zeros(
                        len(negative_groups), maximum_group_size, query.shape[1],
                    )
                    negative_mask = torch.zeros(
                        len(negative_groups), maximum_group_size, dtype=torch.bool,
                    )
                    for group_index, group in enumerate(negative_groups):
                        negatives[group_index, :len(group)] = group
                        negative_mask[group_index, :len(group)] = True
                labels = frame_labels(query.shape[0], intervals, waveform=scene)
                relative_file = f"{split}/{index:04d}.pt"
                torch.save({
                    "query": query.half(),
                    "target_references": targets.half(),
                    "negative_references": negatives.half() if negatives is not None else None,
                    "negative_mask": negative_mask,
                    "labels": labels.to(torch.int8),
                }, output_dir / relative_file)
                manifest["scenes"].append({
                    "file": relative_file,
                    "split": split,
                    "frames": query.shape[0],
                    "duration_seconds": round(scene.numel() / 16000, 5),
                    "intervals_samples": [list(item) for item in intervals],
                    **plan,
                })
                print(f"{split} {index + 1}/{count}: {query.shape[0]} frames", flush=True)
    finally:
        verifier.close()
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--train-scenes", type=int, default=120)
    parser.add_argument("--validation-scenes", type=int, default=36)
    parser.add_argument("--seed", type=int, default=240926)
    parser.add_argument("--layer", type=int, default=9)
    args = parser.parse_args()
    build_features(
        args.corpus, args.output_dir, train_scenes=args.train_scenes,
        validation_scenes=args.validation_scenes, seed=args.seed, layer=args.layer,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
