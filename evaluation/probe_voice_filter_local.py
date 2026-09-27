"""Bounded, offline VoiceFilter trial; never accepts or exports product clips."""
from __future__ import annotations

import argparse
from functools import partial
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models/conv-voice-filter"
sys.path.insert(0, str(MODEL_DIR / "deps"))
sys.path.insert(0, str(MODEL_DIR / "source"))
from src.model.modeling_enh import VoiceFilter  # noqa: E402


def read_audio(path: Path, start: float | None = None, end: float | None = None, *,
               channel_mode: str = "first") -> torch.Tensor:
    with sf.SoundFile(str(path)) as handle:
        if start is not None:
            handle.seek(round(start * handle.samplerate))
        frames = -1 if end is None else round((end - (start or 0)) * handle.samplerate)
        audio = handle.read(frames, dtype="float32", always_2d=True)
        mono = audio[:, 0] if channel_mode == "first" else audio.mean(axis=1)
        waveform = torch.from_numpy(mono.copy())
        if handle.samplerate != 16000:
            waveform = torchaudio.functional.resample(waveform, handle.samplerate, 16000)
    return waveform.contiguous()


def embedding(model: VoiceFilter, waveform: torch.Tensor, device: torch.device) -> torch.Tensor:
    chunks = []
    for start in range(0, waveform.numel(), 80000):
        chunk = waveform[start:start + 80000]
        chunks.append(F.pad(chunk, (0, 80000 - chunk.numel())))
    if not chunks:
        raise ValueError("Empty reference audio")
    with torch.inference_mode():
        return model.xvector_model(torch.stack(chunks).unsqueeze(1).to(device)).mean(0)


def rms(waveform: torch.Tensor) -> float:
    return float(torch.sqrt(waveform.float().square().mean()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_dir", type=Path, help="Existing prepared episode work directory")
    parser.add_argument("destination", type=Path, help="New development-only output directory")
    parser.add_argument("--channel", choices=("raw", "stem"), default="stem")
    parser.add_argument("--channel-mix", choices=("first", "mean"), default="first")
    parser.add_argument("--review-manifest", type=Path,
                        help="Optional known-good manifest for x-vector discrimination audit")
    parser.add_argument("--cross-conditions", action="store_true",
                        help="Compare target and all supplied exclusion references on short cases")
    parser.add_argument("--target-ref-sweep", action="store_true",
                        help="Test every supplied target reference on short positive/negative cases")
    args = parser.parse_args()
    work = args.work_dir.resolve(strict=True)
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    reference = sorted((work / "reference_voice_clips").glob("*.wav"))[0]
    negative = sorted((work / "negative_reference_voice_clips").glob("role_*/*.wav"))[0]
    source = (work / "target_normalized.wav" if args.channel == "raw"
              else work / "stems/target_vocals.wav")
    read = partial(read_audio, channel_mode=args.channel_mix)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    print(f"Loading local model on {device}", flush=True)
    model = VoiceFilter.from_pretrained(str(MODEL_DIR), local_files_only=True).eval().to(device)
    target_vector = embedding(model, read(reference), device)
    wrong_vector = embedding(model, read(negative), device)
    reference_cosine = float(F.cosine_similarity(target_vector, wrong_vector, dim=0))
    cases = (
        ("target_303", 303.38, 307.79, target_vector),
        ("other_774", 774.82, 779.50, target_vector),
        ("other_1026", 1026.40, 1028.94, target_vector),
        ("change_337", 337.87, 342.2625, target_vector),
        ("mixed_tail_938", 935.55, 939.095, target_vector),
        ("mixed_tail_1245", 1241.17, 1245.46, target_vector),
        ("wrong_reference_303", 303.38, 307.79, wrong_vector),
    )
    rows = []
    for name, start, end, vector in cases:
        original = read(source, start, end)
        peak = float(original.abs().max())
        if peak < 1e-6:
            raise ValueError(f"Source scene is silent: {name}")
        scaled = original.to(device) * (1.0 / peak)
        with torch.inference_mode():
            extracted = model.do_enh(scaled, vector).detach().cpu().float()
        if extracted.numel() != original.numel():
            raise ValueError(f"Length mismatch for {name}: {extracted.numel()} vs {original.numel()}")
        input_vector = embedding(model, scaled.cpu(), device)
        output_vector = embedding(model, extracted, device)
        output_scaled = extracted * (0.95 / max(float(extracted.abs().max()), 0.95))
        sf.write(str(destination / f"{name}.wav"), output_scaled.numpy(), 16000, subtype="PCM_16")
        sf.write(str(destination / f"{name}_input.wav"), original.numpy(), 16000, subtype="PCM_16")
        row = {
            "case": name, "source_span": [start, end],
            "condition": "wrong_reference" if name.startswith("wrong_") else "target_reference",
            "source_peak": peak, "source_rms": rms(original),
            "output_rms": rms(extracted),
            "output_over_scaled_input_db": 20 * np.log10(max(rms(extracted), 1e-9) / rms(scaled.cpu())),
            "output_peak": float(extracted.abs().max()),
            "output_input_cosine": float(F.cosine_similarity(extracted, scaled.cpu(), dim=0)),
            "input_target_cosine": float(F.cosine_similarity(input_vector, target_vector, dim=0)),
            "input_wrong_cosine": float(F.cosine_similarity(input_vector, wrong_vector, dim=0)),
            "output_target_cosine": float(F.cosine_similarity(output_vector, target_vector, dim=0)),
            "output_wrong_cosine": float(F.cosine_similarity(output_vector, wrong_vector, dim=0)),
        }
        if name in ("mixed_tail_938", "mixed_tail_1245"):
            main_end, tail_start = ((938.55, 938.75) if name == "mixed_tail_938"
                                    else (1244.67, 1245.23))
            main = slice(0, round((main_end - start) * 16000))
            tail = slice(round((tail_start - start) * 16000), None)
            main_gain = rms(extracted[main]) / max(rms(scaled.cpu()[main]), 1e-9)
            tail_gain = rms(extracted[tail]) / max(rms(scaled.cpu()[tail]), 1e-9)
            row["tail_relative_gain_db"] = float(20 * np.log10(max(tail_gain, 1e-9)
                                                                  / max(main_gain, 1e-9)))
            row["tail_input_cosine"] = float(F.cosine_similarity(
                extracted[tail], scaled.cpu()[tail], dim=0))
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    review_rows = []
    if args.review_manifest is not None:
        target_vectors = torch.stack([
            F.normalize(embedding(model, read(path), device).cpu(), dim=0)
            for path in sorted((work / "reference_voice_clips").glob("*.wav"))
        ])
        negative_vectors = torch.stack([
            F.normalize(embedding(model, read(path), device).cpu(), dim=0)
            for path in sorted((work / "negative_reference_voice_clips").glob("role_*/*.wav"))
        ])
        manifest = json.loads(args.review_manifest.read_text(encoding="utf-8"))
        labelled_spans = [
            ("old_accepted", float(row["start"]), float(row["end"]))
            for row in manifest["sentences"] if row["accepted"]
        ]
        labelled_spans += [("user_reviewed_entire_other", 774.82, 781.52),
                           ("user_reviewed_entire_other", 1026.40, 1028.94)]
        for label, start, end in labelled_spans:
            wave = read(source, start, end)
            wave *= 1.0 / max(float(wave.abs().max()), 1e-6)
            vector = F.normalize(embedding(model, wave, device).cpu(), dim=0)
            targets = target_vectors @ vector
            negatives = negative_vectors @ vector
            review_rows.append({
                "label": label, "span": [start, end],
                "target_median": float(targets.median()),
                "target_max": float(targets.max()),
                "negative_max": float(negatives.max()),
                "target_median_minus_negative_max": float(targets.median() - negatives.max()),
            })
        for row in review_rows:
            print(json.dumps(row, ensure_ascii=False), flush=True)
    cross_rows = []
    if args.cross_conditions:
        ref_paths = [("target", reference)] + [
            (path.parent.name, path)
            for path in sorted((work / "negative_reference_voice_clips").glob("role_*/*.wav"))
        ]
        vectors = [(name, embedding(model, read(path), device))
                   for name, path in ref_paths]
        for case, start, end in (("target_303", 303.38, 307.79),
                                 ("other_774", 774.82, 779.50),
                                 ("other_1026", 1026.40, 1028.94)):
            input_audio = read(source, start, end)
            input_audio = input_audio.to(device) * (1.0 / max(float(input_audio.abs().max()), 1e-6))
            outputs = []
            for role, vector in vectors:
                with torch.inference_mode():
                    result = model.do_enh(input_audio, vector).detach().cpu().float()
                outputs.append({
                    "role": role, "output_rms": rms(result),
                    "output_input_cosine": float(F.cosine_similarity(result, input_audio.cpu(), dim=0)),
                })
            cross_rows.append({"case": case, "outputs": outputs})
            print(json.dumps(cross_rows[-1], ensure_ascii=False), flush=True)
    sweep_rows = []
    if args.target_ref_sweep:
        for ref_path in sorted((work / "reference_voice_clips").glob("*.wav")):
            ref_vector = embedding(model, read(ref_path), device)
            for case, start, end in (("target_303", 303.38, 307.79),
                                     ("other_774", 774.82, 779.50),
                                     ("other_1026", 1026.40, 1028.94),
                                     ("mixed_tail_938", 935.55, 939.095),
                                     ("mixed_tail_1245", 1241.17, 1245.46)):
                input_audio = read(source, start, end)
                input_audio = input_audio.to(device) / max(float(input_audio.abs().max()), 1e-6)
                with torch.inference_mode():
                    result = model.do_enh(input_audio, ref_vector).detach().cpu().float()
                row = {
                    "reference": ref_path.name, "case": case,
                    "output_rms": rms(result),
                    "output_input_cosine": float(F.cosine_similarity(result, input_audio.cpu(), dim=0)),
                }
                if case.startswith("mixed_tail_"):
                    main_end, tail_start = ((938.55, 938.75) if case == "mixed_tail_938"
                                            else (1244.67, 1245.23))
                    main = slice(0, round((main_end - start) * 16000))
                    tail = slice(round((tail_start - start) * 16000), None)
                    main_gain = rms(result[main]) / max(rms(input_audio.cpu()[main]), 1e-9)
                    tail_gain = rms(result[tail]) / max(rms(input_audio.cpu()[tail]), 1e-9)
                    row["tail_relative_gain_db"] = float(20 * np.log10(
                        max(tail_gain, 1e-9) / max(main_gain, 1e-9)))
                sweep_rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    report = {
        "purpose": "development_only_not_identity_acceptance",
        "model": str(MODEL_DIR), "source": str(source),
        "channel_mode": args.channel_mix,
        "target_reference": str(reference), "wrong_reference": str(negative),
        "reference_cosine": reference_cosine,
        "cases": rows,
        "xvector_review": review_rows,
        "cross_conditions": cross_rows,
        "target_reference_sweep": sweep_rows,
    }
    (destination / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
