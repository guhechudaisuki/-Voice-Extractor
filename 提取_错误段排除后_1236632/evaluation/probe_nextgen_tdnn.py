"""Bounded development check of the new speaker-TDNN frame representation.

Scores are diagnostic only: no threshold is fitted and no output is accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extractor.audio import load_mono  # noqa: E402
from extractor.nextgen.features import WavLMSpeakerFeatures  # noqa: E402
from extractor.nextgen.timeline import SampleSpan, SourceTimeline, ViewAlignment  # noqa: E402


def encode(encoder: WavLMSpeakerFeatures, waveform: torch.Tensor) -> torch.Tensor:
    waveform = waveform.contiguous()
    digest = hashlib.sha256(waveform.numpy().tobytes()).hexdigest()
    source = SourceTimeline(digest, 16000, waveform.numel())
    alignment = ViewAlignment(source, 16000, waveform.numel(), 0, True)
    frames = encoder.encode(waveform, alignment, SampleSpan(0, waveform.numel())).values
    return F.normalize(frames.float(), dim=-1)


def pooled(frames: torch.Tensor) -> torch.Tensor:
    # TDNN layers are ReLU-sparse. Coordinate-wise median can become the all-
    # zero vector even for clearly audible speech, making every cosine zero.
    return F.normalize(frames.mean(dim=0), dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("--channel", choices=("raw", "stem"), default="stem")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    work = args.work_dir.resolve(strict=True)
    stem = args.channel == "stem"
    source_path = work / ("stems/target_vocals.wav" if stem else "target_normalized.wav")
    reference_dir = work / ("reference_voice_clips" if stem else "reference_original_voice_clips")
    negative_dir = work / ("negative_reference_voice_clips" if stem
                           else "negative_reference_original_voice_clips")
    cases = json.loads((ROOT / "evaluation/join_cases.json").read_text(encoding="utf-8"))["cases"]
    model = WavLMSpeakerFeatures(ROOT / "model/speaker/wavlm-base-plus-sv",
                                  device="cuda" if torch.cuda.is_available() else "cpu")
    references = [pooled(encode(model, load_mono(path, 16000)))
                  for path in sorted(reference_dir.glob("*.wav"))]
    exclusions = [pooled(encode(model, load_mono(path, 16000)))
                  for path in sorted(negative_dir.glob("role_*/*.wav"))]
    if not references:
        raise ValueError("No local target reference clips")
    waveform = load_mono(source_path, 16000)
    rows = []
    for case in cases:
        sides = []
        for start, end in (case["left"], case["right"]):
            frames = encode(model, waveform[round(start * 16000):round(end * 16000)])
            vector = pooled(frames)
            target_scores = torch.stack(references) @ vector
            negative_scores = torch.stack(exclusions) @ vector if exclusions else None
            sides.append({
                "target_reference_max": round(float(target_scores.max()), 5),
                "target_reference_median": round(float(target_scores.median()), 5),
                "negative_reference_max": (round(float(negative_scores.max()), 5)
                                           if negative_scores is not None else None),
                "query_frame_target_max_p20": round(float(torch.quantile(
                    frames @ torch.stack(references).T, .2, dim=0).max()), 5),
                "vector": vector,
            })
        pair_cosine = float(sides[0].pop("vector") @ sides[1].pop("vector"))
        rows.append({"id": case["id"], "expected_same_target": case["same_target"],
                     "pair_cosine": round(pair_cosine, 5), "sides": sides})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "schema": 1, "purpose": "development_representation_probe_only",
        "channel": args.channel, "encoder_digest": model.digest,
        "reference_count": len(references), "exclusion_count": len(exclusions),
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} development cases to {args.output}")


if __name__ == "__main__":
    main()
