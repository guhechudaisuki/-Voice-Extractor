"""Offline full-episode VoiceFilter test; output is unsafe development audio."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_voice_filter_local import MODEL_DIR, embedding, read_audio, rms
from src.model.modeling_enh import VoiceFilter


SAMPLE_RATE = 16000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


CASES = (
    ("reviewed_target_303", 303.38, 307.79),
    ("reviewed_target_788", 788.42, 802.42),
    ("reviewed_target_1241", 1241.17, 1244.67),
    ("reviewed_other_774", 774.82, 781.52),
    ("reviewed_other_1026", 1026.40, 1028.94),
    ("known_other_tail_938", 938.75, 939.095),
    ("known_other_tail_1245", 1245.23, 1245.46),
)


def chunks(model: VoiceFilter, waveform: torch.Tensor,
           vector: torch.Tensor, device: torch.device) -> torch.Tensor:
    outputs = []
    count = math.ceil(waveform.numel() / model.wav_chunk_size)
    for index in range(count):
        start = index * model.wav_chunk_size
        piece = waveform[start:start + model.wav_chunk_size].to(device)
        with torch.inference_mode():
            enhanced = model.do_enh(piece, vector).detach().cpu().float()
        outputs.append(enhanced)
        if (index + 1) % 20 == 0 or index + 1 == count:
            print(f"[full-episode {index + 1}/{count}] {100 * (index + 1) / count:.1f}%", flush=True)
    return torch.cat(outputs)


def measure(name: str, start: float, end: float,
            source: torch.Tensor, output: torch.Tensor) -> dict:
    left = round(start * SAMPLE_RATE)
    right = round(end * SAMPLE_RATE)
    original = source[left:right]
    enhanced = output[left:right]
    input_rms = rms(original)
    output_rms = rms(enhanced)
    return {
        "case": name, "source_span": [start, end],
        "input_rms": input_rms, "output_rms_before_global_peak_normalization": output_rms,
        "gain_db": 20 * math.log10(max(output_rms, 1e-9) / max(input_rms, 1e-9)),
        "output_input_cosine": float(F.cosine_similarity(enhanced, original, dim=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--channel", choices=("raw", "stem"), default="raw")
    parser.add_argument("--reference-index", type=int, default=0)
    args = parser.parse_args()
    work = args.work_dir.resolve(strict=True)
    destination = args.destination.resolve()
    refs = sorted((work / "reference_voice_clips").glob("*.wav"))
    if not 0 <= args.reference_index < len(refs):
        raise ValueError("Reference index is outside the local target reference set")
    source_path = (work / "target_normalized.wav" if args.channel == "raw"
                   else work / "stems/target_vocals.wav")
    destination.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    model = VoiceFilter.from_pretrained(str(MODEL_DIR), local_files_only=True).eval().to(device)
    vector = embedding(model, read_audio(refs[args.reference_index], channel_mode="first"), device)
    source = read_audio(source_path, channel_mode="first")
    original_peak = float(source.abs().max())
    if original_peak < 1e-6:
        raise ValueError("Episode source is silent")
    scaled = source / original_peak  # The upstream service normalizes the whole file once.
    print(f"Loaded {source.numel() / SAMPLE_RATE:.2f}s; {model.wav_chunk_size / SAMPLE_RATE:.1f}s chunks; "
          f"reference={refs[args.reference_index].name}; channel={args.channel}", flush=True)

    # Verify that streaming one 5-second model chunk per call is equivalent to
    # its own batched do_enh() path before applying it to the complete episode.
    prefix = scaled[:15 * SAMPLE_RATE]
    with torch.inference_mode():
        batched_prefix = model.do_enh(prefix.to(device), vector).detach().cpu().float()
    streamed_prefix = chunks(model, prefix, vector, device)
    max_difference = float((batched_prefix - streamed_prefix).abs().max())
    maximum = max(float(batched_prefix.abs().max()), 1e-6)
    relative_difference = max_difference / maximum
    print(f"Chunk equivalence relative max difference={relative_difference:.8f}", flush=True)
    if relative_difference > 1e-3:
        raise RuntimeError("Streaming differs from the model's batched do_enh path")

    output = chunks(model, scaled, vector, device)
    if output.numel() != scaled.numel() or not torch.isfinite(output).all():
        raise ValueError("Model returned an invalid or misaligned episode waveform")
    cases = [measure(*case, scaled, output) for case in CASES]
    target_gains = [row["gain_db"] for row in cases if row["case"].startswith("reviewed_target_")]
    target_reference_gain = float(np.median(target_gains))
    for row in cases:
        row["gain_relative_to_reviewed_targets_db"] = row["gain_db"] - target_reference_gain
        print(json.dumps(row, ensure_ascii=False), flush=True)
    output_peak = float(output.abs().max())
    playback = output * (0.95 / max(output_peak, 0.95))
    playback_path = destination / "FULL_EPISODE_UNVERIFIED.wav"
    sf.write(str(playback_path), playback.numpy(), SAMPLE_RATE,
             subtype="PCM_16")
    for name, start, end in CASES:
        left, right = round(start * SAMPLE_RATE), round(end * SAMPLE_RATE)
        sf.write(str(destination / f"{name}_UNVERIFIED.wav"), playback[left:right].numpy(),
                 SAMPLE_RATE, subtype="PCM_16")
    report = {
        "purpose": "development_only_not_training_audio_or_identity_acceptance",
        "source": str(source_path), "reference": str(refs[args.reference_index]),
        "source_sha256": sha256(source_path),
        "reference_sha256": sha256(refs[args.reference_index]),
        "model_weight_sha256": sha256(MODEL_DIR / "pytorch_model.bin"),
        "output_sha256": sha256(playback_path),
        "channel": args.channel, "channel_mode": "first", "sample_rate": SAMPLE_RATE,
        "duration_seconds": source.numel() / SAMPLE_RATE,
        "model_chunk_seconds": model.wav_chunk_size / SAMPLE_RATE,
        "streaming_batched_relative_difference": relative_difference,
        "source_peak": original_peak, "output_peak_before_playback_normalization": output_peak,
        "playback_global_gain": 0.95 / max(output_peak, 0.95),
        "reviewed_target_median_gain_db": target_reference_gain,
        "cases": cases,
    }
    (destination / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print(f"FULL_EPISODE_REPORT={destination / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
