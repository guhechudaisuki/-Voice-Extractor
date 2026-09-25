"""Offline local anime-speaker embedding probe on corrected speech pairs.

Uses the already-installed OmnVoice Python environment and locally cached
char/va weights. Scores are diagnostics, not a production acceptance rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from anime_speaker_embedding.model import AnimeSpeakerEmbedding


ROOT = Path(__file__).resolve().parents[1]
CASES = Path(__file__).with_name("join_cases.json")
MODELS = {
    "char": ROOT / "models/anime-speaker-char/embedding_model.pth",
    "va": ROOT / "models/anime-speaker-va/embedding_model.pth",
}


def read_audio(path: Path, start: float | None = None, end: float | None = None) -> torch.Tensor:
    with sf.SoundFile(path) as stream:
        sample_rate = stream.samplerate
        if start is not None:
            stream.seek(round(start * sample_rate))
        frames = -1 if end is None else round((end - (start or 0.0)) * sample_rate)
        data = stream.read(frames, dtype="float32", always_2d=True)
    mono = np.ascontiguousarray(data.mean(axis=1))
    if sample_rate != 16000:
        mono = np.ascontiguousarray(librosa.resample(mono, orig_sr=sample_rate, target_sr=16000))
    return torch.from_numpy(mono)


def encode(model: AnimeSpeakerEmbedding, waveform: torch.Tensor) -> torch.Tensor:
    if waveform.numel() < 3200:
        raise ValueError("Audio is too short for this diagnostic")
    with torch.inference_mode():
        vector = model(waveform.unsqueeze(0).to(model.device)).reshape(-1).float().cpu()
    return F.normalize(vector, dim=0)


def scores(
    vector: torch.Tensor, target_vectors: torch.Tensor,
    negative_groups: list[torch.Tensor],
) -> dict:
    target_scores = target_vectors @ vector
    negative_scores = [float((group @ vector).max()) for group in negative_groups]
    return {
        "target_max": round(float(target_scores.max()), 5),
        "target_median": round(float(target_scores.median()), 5),
        "negative_group_max": [round(value, 5) for value in negative_scores],
        "target_minus_nearest_negative": (
            round(float(target_scores.max()) - max(negative_scores), 5)
            if negative_scores else None
        ),
    }


def run(
    work_dir: Path, *, variant: str, channel: str, cases_path: Path = CASES,
) -> dict:
    if variant not in MODELS or channel not in ("stem", "raw"):
        raise ValueError("Unknown model variant or channel")
    weight = MODELS[variant]
    if not weight.is_file():
        raise FileNotFoundError(weight)
    model = AnimeSpeakerEmbedding(variant=variant, ckpt_path=weight).eval()
    source = work_dir / ("stems/target_vocals.wav" if channel == "stem" else "target_normalized.wav")
    reference_dir = work_dir / (
        "reference_voice_clips" if channel == "stem" else "reference_original_voice_clips"
    )
    negative_dir = work_dir / (
        "negative_reference_voice_clips" if channel == "stem"
        else "negative_reference_original_voice_clips"
    )
    references = sorted(reference_dir.glob("*.wav"))
    if not references:
        raise ValueError("No target reference clips")
    target_vectors = torch.stack([encode(model, read_audio(path)) for path in references])
    negative_groups = [
        torch.stack([encode(model, read_audio(path)) for path in sorted(group.glob("*.wav"))])
        for group in sorted(negative_dir.glob("role_*")) if any(group.glob("*.wav"))
    ]
    cases = json.loads(cases_path.read_text(encoding="utf-8"))["cases"]
    rows = []
    for case in cases:
        left, right = (encode(model, read_audio(source, *case[side])) for side in ("left", "right"))
        whole = encode(model, read_audio(source, case["left"][0], case["right"][1]))
        rows.append({
            "id": case["id"],
            "historical_same_target_pair": (
                bool(case["same_target"])
                if case.get("same_target") is not None else None
            ),
            "left": scores(left, target_vectors, negative_groups),
            "right": scores(right, target_vectors, negative_groups),
            "whole": scores(whole, target_vectors, negative_groups),
            "pair_cosine": round(float(left @ right), 5),
        })
    return {
        "warning": "Development pairs only; cosine scores are not calibrated identity probabilities or a deployable rule.",
        "cases_file": str(cases_path),
        "variant": variant,
        "channel": channel,
        "target_reference_count": len(references),
        "negative_group_count": len(negative_groups),
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("--variant", choices=tuple(MODELS), required=True)
    parser.add_argument("--channel", choices=("stem", "raw"), required=True)
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.work_dir, variant=args.variant, channel=args.channel,
        cases_path=args.cases,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ANIME_PROBE={args.output} CASES={len(result['cases'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
