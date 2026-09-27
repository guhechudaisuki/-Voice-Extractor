"""Bounded CC BY 4.0 LibriSpeech T1 data for a research-only temporal head.

The source utterances have speaker IDs, but synthetic joins are not anime
dialogue. Raw and stem are deliberately identical here; success cannot prove
UVR-domain performance, singing/overlap rejection or complete boundaries.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extractor.nextgen.features import WavLMSpeakerFeatures, file_digest  # noqa: E402
from extractor.nextgen.identity_model import HEADS, ReferenceFeatures  # noqa: E402
from extractor.nextgen.timeline import SampleSpan, SourceTimeline, ViewAlignment  # noqa: E402
from training.librispeech_t1 import (assemble_scene, index_utterances, load_audio,  # noqa: E402
                                     plan_scene, split_speakers)
from training.nextgen_data import DataRecord  # noqa: E402
from training.nextgen_examples import (save_feature_example, write_manifest)  # noqa: E402


def _encoded(encoder: WavLMSpeakerFeatures, waveform: torch.Tensor):
    waveform = waveform.contiguous().float()
    digest = hashlib.sha256(waveform.numpy().tobytes()).hexdigest()
    source = SourceTimeline(digest, 16000, waveform.numel())
    aligned = ViewAlignment(source, 16000, waveform.numel(), 0, True)
    return encoder.encode(waveform, aligned, SampleSpan(0, waveform.numel()))


def _reference(encoder: WavLMSpeakerFeatures, paths: tuple[str, ...],
               cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    tokens, masks, quality = [], [], []
    for name in paths:
        if name not in cache:
            waveform = load_audio(Path(name))
            encoded = _encoded(encoder, waveform)
            values = encoded.values.float()
            rms = []
            for cell in encoded.cells:
                local = waveform[cell.start:cell.end]
                rms.append(float(local.square().mean().sqrt()))
            levels = torch.tensor(rms)
            typical = float(torch.quantile(levels, .9))
            valid = levels >= max(.0005, typical * .02)
            if not valid.any():
                raise ValueError(f"Reference has no measurable speech: {name}")
            cache[name] = values, valid, (levels / max(typical, 1e-5)).clamp(0, 1)
        values, valid, q = cache[name]
        tokens.append(values)
        masks.append(valid)
        quality.append(q)
    joined = torch.cat(tokens)
    mask = torch.cat(masks)
    confidence = torch.cat(quality)
    return ReferenceFeatures(joined, joined.clone(), mask, confidence)


def _labels(cells: tuple[SampleSpan, ...], waveform: torch.Tensor,
            intervals: list[tuple[int, int, str]]) -> torch.Tensor:
    columns = {name: index for index, name in enumerate(HEADS)}
    labels = torch.full((len(cells), len(HEADS)), -1.0)
    labels[:, [columns[name] for name in ("target", "other", "speech", "singing", "overlap", "change")]] = 0
    centers = torch.tensor([(cell.start + cell.end) // 2 for cell in cells])
    for start, end, role in intervals:
        inside = (centers >= start + 1600) & (centers < end - 1600)
        near = ((centers >= start - 1600) & (centers < start + 1600)) | (
            (centers >= end - 1600) & (centers < end + 1600))
        labels[near, :3] = -1
        if not inside.any():
            continue
        samples = torch.arange(-200, 200)[None, :] + centers[:, None]
        samples = samples.clamp(0, waveform.numel() - 1)
        energy = waveform[samples].square().mean(dim=1).sqrt()
        utterance = energy[(centers >= start) & (centers < end)]
        reference = float(torch.quantile(utterance, .9))
        quiet = inside & (energy < max(.0005, reference * .02))
        active = inside & ~quiet
        labels[active, columns[role]] = 1
        labels[active, columns["speech"]] = 1
        labels[quiet, :3] = -1
    for (_, _, left_role), (right_start, _, right_role) in zip(intervals, intervals[1:]):
        if left_role != right_role:
            labels[(centers >= right_start - 1600) & (centers < right_start + 1600),
                   columns["change"]] = 1
    return labels


def _different_chapters(plan: dict) -> bool:
    query = {Path(path).parent for path in plan["query_paths"]}
    references = [Path(path) for path in plan["target_reference_paths"]]
    references += [Path(path) for group in plan["negative_reference_groups"]
                   for path in group["paths"]]
    return all(path.parent not in query for path in references)


def build(corpus: Path, destination: Path, *, train_scenes: int,
          development_scenes: int, seed: int, device: str) -> dict:
    if train_scenes < 1 or development_scenes < 1 or destination.exists():
        raise ValueError("Need new output and nonempty speaker-disjoint train/development sets")
    files = index_utterances(corpus)
    train_speakers, dev_speakers = split_speakers(files, seed=seed)
    encoder = WavLMSpeakerFeatures(ROOT / "model/speaker/wavlm-base-plus-sv", device=device)
    destination.mkdir(parents=True)
    records = []
    cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    @lru_cache(maxsize=None)
    def content_digest(path: str) -> str:
        return file_digest(Path(path))

    for split, pool, count in (("train", train_speakers, train_scenes),
                               ("development", dev_speakers, development_scenes)):
        rng = random.Random(seed + (0 if split == "train" else 1_000_000))
        for index in range(count):
            plan = None
            for attempt in range(20):
                candidate = plan_scene(files, pool, index + attempt, rng)
                if _different_chapters(candidate):
                    plan = candidate
                    break
            if plan is None:
                raise ValueError("Could not make cross-chapter reference/query pairs")
            speech, intervals = assemble_scene(plan)
            pad = torch.zeros(6400)
            waveform = torch.cat((pad, speech, pad))
            intervals = [(start + pad.numel(), end + pad.numel(), role)
                         for start, end, role in intervals]
            query = _encoded(encoder, waveform)
            target = _reference(encoder, tuple(plan["target_reference_paths"]), cache)
            groups = [_reference(encoder, tuple(group["paths"]), cache)
                      for group in plan["negative_reference_groups"]]
            output = SampleSpan(pad.numel(), pad.numel() + speech.numel())
            allowed = torch.tensor([cell.intersection(output) is not None for cell in query.cells])
            labels = _labels(query.cells, waveform, intervals)
            labels[~allowed] = -1
            tensors = {"raw": query.values.float(), "stem": query.values.float().clone(),
                       "allowed": allowed, "labels.frames": labels,
                       "labels.purity": torch.tensor(float(all(role == "target" for role in plan["pattern"]))),
                       "labels.boundaries": torch.tensor([-1.0, -1.0])}
            for prefix, reference in (("target", target),
                                      *((f"excluded.{i:03d}", group)
                                        for i, group in enumerate(groups))):
                for field in ("raw", "stem", "valid", "quality"):
                    tensors[f"{prefix}.{field}"] = getattr(reference, field)
            relative = f"{split}/{index:04d}.safetensors"
            digest = save_feature_example(tensors, destination / relative)
            paths = [*plan["query_paths"], *plan["target_reference_paths"],
                     *(path for group in plan["negative_reference_groups"]
                       for path in group["paths"])]
            speakers = {plan["target_speaker"], plan["other_speaker"],
                        *(group["speaker"] for group in plan["negative_reference_groups"])}
            sources = {f"LibriSpeech/{Path(path).parent.parent.name}/{Path(path).parent.name}"
                       for path in paths}
            records.append(DataRecord(
                f"{split}-{index:04d}", split, tuple(sorted(sources)),
                tuple(sorted(speakers)), tuple(sorted({content_digest(path) for path in paths})),
                relative, digest, encoder.digest,
                "OpenSLR 12 LibriSpeech dev-clean; CC BY 4.0", True, False, True,
            ))
            print(f"{split} {index + 1}/{count}: {len(query.cells)} frames", flush=True)
    audit = write_manifest(tuple(records), destination / "manifest.json")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--train-scenes", type=int, default=24)
    parser.add_argument("--development-scenes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    print(build(args.corpus, args.destination, train_scenes=args.train_scenes,
                development_scenes=args.development_scenes, seed=args.seed,
                device=args.device))


if __name__ == "__main__":
    main()
