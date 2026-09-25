"""Inspect unpooled local WavLM states on frozen same/change-speaker pairs.

This is a representation diagnostic. It does not select a deployment layer,
threshold, or output an accepted audio segment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extractor.audio import load_mono  # noqa: E402
from extractor.speaker import WavLMSpeakerVerifier  # noqa: E402


LAYERS = (3, 6, 9, 12)
CASES = Path(__file__).with_name("join_cases.json")


def encode(verifier: WavLMSpeakerVerifier, waveform: torch.Tensor) -> dict[int, torch.Tensor]:
    inputs = verifier.feature_extractor(
        waveform.detach().float().cpu().numpy(),
        sampling_rate=16000,
        return_tensors="pt",
    )
    with torch.inference_mode():
        output = verifier.model.wavlm(
            input_values=inputs["input_values"].to(verifier.device),
            output_hidden_states=True,
        )
    # Each sequence remains intact here; only the diagnostic score below pools.
    return {
        layer: F.normalize(output.hidden_states[layer][0].float().cpu(), dim=-1)
        for layer in LAYERS
    }


def pool(frames: torch.Tensor) -> torch.Tensor:
    return F.normalize(frames.mean(dim=0), dim=0)


def stats(
    left: dict[int, torch.Tensor],
    right: dict[int, torch.Tensor],
    references: dict[int, torch.Tensor],
    reference_vectors: dict[int, torch.Tensor],
    exclusion_vectors: dict[int, torch.Tensor],
) -> dict[str, dict]:
    rows = {}
    for layer in LAYERS:
        left_frames, right_frames = left[layer], right[layer]
        left_vector, right_vector = pool(left_frames), pool(right_frames)
        target_vector = references[layer]
        target_scores = [
            reference_vectors[layer] @ vector
            for vector in (left_vector, right_vector)
        ]
        negative_scores = [
            exclusion_vectors[layer] @ vector
            for vector in (left_vector, right_vector)
        ] if exclusion_vectors[layer].numel() else []
        rows[str(layer)] = {
            "mean_pair_cosine": round(float(left_vector @ right_vector), 5),
            "left_target_cosine": round(float(left_vector @ target_vector), 5),
            "right_target_cosine": round(float(right_vector @ target_vector), 5),
            "left_frame_target_p20": round(float(torch.quantile(left_frames @ target_vector, 0.20)), 5),
            "right_frame_target_p20": round(float(torch.quantile(right_frames @ target_vector, 0.20)), 5),
            "left_target_reference_max": round(float(target_scores[0].max()), 5),
            "right_target_reference_max": round(float(target_scores[1].max()), 5),
            "left_target_reference_median": round(float(target_scores[0].median()), 5),
            "right_target_reference_median": round(float(target_scores[1].median()), 5),
            "left_exclusion_reference_max": round(float(negative_scores[0].max()), 5) if negative_scores else None,
            "right_exclusion_reference_max": round(float(negative_scores[1].max()), 5) if negative_scores else None,
        }
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("--channel", choices=("stem", "raw"), default="stem")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    work = args.work_dir.resolve()
    source = work / ("stems/target_vocals.wav" if args.channel == "stem" else "target_normalized.wav")
    reference_dir = work / (
        "reference_voice_clips" if args.channel == "stem" else "reference_original_voice_clips"
    )
    reference_paths = sorted(reference_dir.glob("*.wav"))
    if not reference_paths:
        parser.error(f"No references in {reference_dir}")
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    waveform = load_mono(source, 16000)
    verifier = WavLMSpeakerVerifier()
    try:
        encoded_references = [encode(verifier, load_mono(path, 16000)) for path in reference_paths]
        negative_dir = work / (
            "negative_reference_voice_clips" if args.channel == "stem"
            else "negative_reference_original_voice_clips"
        )
        negative_groups = sorted(negative_dir.glob("role_*"))
        encoded_exclusions = [
            [encode(verifier, load_mono(path, 16000)) for path in sorted(group.glob("*.wav"))]
            for group in negative_groups
        ]
        reference_centroids = {
            layer: pool(torch.cat([entry[layer] for entry in encoded_references], dim=0))
            for layer in LAYERS
        }
        reference_vectors = {
            layer: torch.stack([pool(entry[layer]) for entry in encoded_references])
            for layer in LAYERS
        }
        exclusion_vectors = {
            layer: torch.stack([
                pool(torch.cat([entry[layer] for entry in group], dim=0))
                for group in encoded_exclusions if group
            ]) if any(encoded_exclusions) else torch.empty((0, 768))
            for layer in LAYERS
        }
        pairs = []
        for case in cases:
            spans = (case["left"], case["right"])
            encoded = [
                encode(verifier, waveform[round(start * 16000):round(end * 16000)])
                for start, end in spans
            ]
            pairs.append({
                "id": case["id"],
                "expected_same_target": case["same_target"],
                "left": case["left"],
                "right": case["right"],
                "layer_stats": stats(
                    encoded[0], encoded[1], reference_centroids,
                    reference_vectors, exclusion_vectors,
                ),
            })
        report = {
            "warning": "Development-only representation probe; no layer or threshold chosen for inference.",
            "channel": args.channel,
            "reference_count": len(reference_paths),
            "exclusion_group_count": len(negative_groups),
            "pairs": pairs,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"TEMPORAL_PROBE={args.output} PAIRS={len(pairs)}")
        return 0
    finally:
        verifier.close()


if __name__ == "__main__":
    raise SystemExit(main())
