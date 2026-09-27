"""Probe an experimental T1 head on corrected anime same/change pairs.

This reports frame scores only. It never exports audio or changes production
decisions, and the six first-episode pairs are development cases, not a blind test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from librispeech_t1 import encode_frames  # noqa: E402
from train_temporal_t1 import TemporalIdentityHead  # noqa: E402
from extractor.audio import load_mono  # noqa: E402
from extractor.speaker import WavLMSpeakerVerifier  # noqa: E402


def read_span(waveform: torch.Tensor, start: float, end: float) -> torch.Tensor:
    return waveform[round(start * 16000):round(end * 16000)]


def reference_vectors(
    verifier: WavLMSpeakerVerifier, paths: list[Path], layer: int,
) -> torch.Tensor:
    return torch.stack([
        F.normalize(encode_frames(verifier, load_mono(path, 16000), layer).mean(dim=0), dim=0)
        for path in paths
    ])


def score_half(
    logits: torch.Tensor, frames: torch.Tensor,
    references: torch.Tensor, begin: float, end: float, start: float,
) -> dict:
    times = start + (torch.arange(frames.shape[0]) * 320 + 200) / 16000
    core = (times >= begin + 0.10) & (times < end - 0.10)
    if not core.any():
        core = (times >= begin) & (times < end)
    probabilities = torch.sigmoid(logits[core].float())
    query = F.normalize(frames[core].float(), dim=-1)
    reference_cosine = (query @ F.normalize(references.float(), dim=-1).T).max(dim=-1).values
    return {
        "frames": int(core.sum()),
        "target_mean": round(float(probabilities[:, 0].mean()), 5),
        "target_p10": round(float(torch.quantile(probabilities[:, 0], 0.10)), 5),
        "other_mean": round(float(probabilities[:, 1].mean()), 5),
        "speech_mean": round(float(probabilities[:, 2].mean()), 5),
        "target_minus_other_mean": round(float((probabilities[:, 0] - probabilities[:, 1]).mean()), 5),
        "reference_cosine_mean": round(float(reference_cosine.mean()), 5),
    }


def probe(
    work_dir: Path, checkpoint_path: Path, cases_path: Path,
    *, target_reference_count: int | None = None, use_negatives: bool = True,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    layer = int(checkpoint["wavlm_hidden_layer"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = TemporalIdentityHead(feature_dim=int(checkpoint["feature_dim"]))
    head.load_state_dict(checkpoint["state_dict"])
    head.eval().to(device)
    source = work_dir / "stems/target_vocals.wav"
    targets = sorted((work_dir / "reference_voice_clips").glob("*.wav"))
    if target_reference_count is not None:
        if target_reference_count < 1:
            raise ValueError("At least one target reference is required")
        targets = targets[:target_reference_count]
    negative_groups = [
        sorted(group.glob("*.wav"))
        for group in sorted((work_dir / "negative_reference_voice_clips").glob("role_*"))
    ]
    negative_groups = [group for group in negative_groups if group] if use_negatives else []
    if not targets:
        raise ValueError("Missing target reference clips")
    verifier = WavLMSpeakerVerifier()
    try:
        source_waveform = load_mono(source, 16000)
        target_vectors = reference_vectors(verifier, targets, layer)
        negative_vectors = [reference_vectors(verifier, group, layer) for group in negative_groups]
        negative_tensor = None
        negative_mask = None
        if negative_vectors:
            max_references = max(len(group) for group in negative_vectors)
            negative_tensor = torch.zeros(1, len(negative_vectors), max_references, target_vectors.shape[1])
            negative_mask = torch.zeros(1, len(negative_vectors), max_references, dtype=torch.bool)
            for group_index, group in enumerate(negative_vectors):
                negative_tensor[0, group_index, :len(group)] = group
                negative_mask[0, group_index, :len(group)] = True
        cases = json.loads(cases_path.read_text(encoding="utf-8"))["cases"]
        rows = []
        for case in cases:
            start, end = float(case["left"][0]), float(case["right"][1])
            query = read_span(source_waveform, start, end)
            frames = encode_frames(verifier, query, layer)
            with torch.inference_mode():
                logits = head(
                    frames.unsqueeze(0).to(device),
                    target_vectors.unsqueeze(0).to(device),
                    negative_references=(negative_tensor.to(device) if negative_tensor is not None else None),
                    negative_mask=(negative_mask.to(device) if negative_mask is not None else None),
                )[0].cpu()
            left = score_half(logits, frames, target_vectors, *case["left"], start)
            right = score_half(logits, frames, target_vectors, *case["right"], start)
            rows.append({
                "id": case["id"],
                "historical_same_target_pair": bool(case["same_target"]),
                "left": left,
                "right": right,
                "target_mean_difference": round(abs(left["target_mean"] - right["target_mean"]), 5),
                "target_minus_other_difference": round(
                    abs(left["target_minus_other_mean"] - right["target_minus_other_mean"]), 5,
                ),
            })
        return {
            "warning": "English synthetic-scene T1 head on six first-episode development pairs; uncalibrated anime scores, not accepted/rejected audio.",
            "checkpoint": str(checkpoint_path.resolve()),
            "target_reference_count": len(targets),
            "negative_group_count": len(negative_groups),
            "cases": rows,
        }
    finally:
        verifier.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("join_cases.json"))
    parser.add_argument("--target-references", type=int,
                        help="Use the first N sorted target clips for a fixed ablation; default all")
    parser.add_argument("--no-negatives", action="store_true",
                        help="Disable optional exclusion references for a fixed ablation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = probe(
        args.work_dir, args.checkpoint, args.cases,
        target_reference_count=args.target_references,
        use_negatives=not args.no_negatives,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PROBE={args.output} CASES={len(report['cases'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
