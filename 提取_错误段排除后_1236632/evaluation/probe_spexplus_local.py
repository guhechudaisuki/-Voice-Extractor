"""Audio-reference SpEx+ trial on bounded clips; output is never product audio."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import soundfile as sf
import torch
import torchaudio
import yaml
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work"))
from spex_plus_upstream import SpEx_plus  # noqa: E402

MODEL_ROOT = (
    ROOT / "models" / "modelscope_cache" / "alibabasglab"
    / "log_wsj0-2mix_speech_SpEx-plus_2spk" / "checkpoints"
    / "log_wsj0-2mix_speech_SpEx-plus_2spk"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_audio(path: Path) -> torch.Tensor:
    array, rate = sf.read(path, dtype="float32", always_2d=True)
    # Upstream dataset_speech._audioread keeps channel 0 of stereo reference
    # recordings. Averaging can cancel a phase-opposed reference signal.
    waveform = torch.from_numpy(array[:, 0])
    if rate != 8000:
        waveform = torchaudio.functional.resample(waveform, rate, 8000)
    if waveform.numel() < 800:
        raise ValueError(f"Too little audio for SpEx+: {path}")
    return waveform


def load_model(device: torch.device) -> tuple[SpEx_plus, Path]:
    config = yaml.safe_load((MODEL_ROOT / "config.yaml").read_text(encoding="utf-8"))
    args = SimpleNamespace(
        causal=config["causal"],
        network_audio=SimpleNamespace(**config["network_audio"]),
    )
    model = SpEx_plus(args)
    checkpoint = MODEL_ROOT / "last_best_checkpoint.pt"
    package = torch.load(checkpoint, map_location="cpu", weights_only=True)
    prefix = "module.sep_network."
    state = {
        key[len(prefix):]: value for key, value in package["model"].items()
        if key.startswith(prefix)
    }
    model.load_state_dict(state, strict=True)
    return model.eval().to(device), checkpoint


def rms(waveform: torch.Tensor) -> float:
    return float(waveform.float().square().mean().sqrt())


def inspect_episode(
    path: Path,
    specifications: list[str],
    model: SpEx_plus,
    reference: torch.Tensor,
    reference_length: torch.Tensor,
    device: torch.device,
) -> dict:
    """Process all samples in independent training-length windows."""
    intervals = []
    with sf.SoundFile(path) as audio:
        if audio.samplerate != 8000 or audio.channels != 1:
            raise ValueError("Episode input must be an 8 kHz mono WAV")
        sample_count = len(audio)
        for specification in specifications:
            name, truth, start_text, end_text = specification.split("|", 3)
            start, end = float(start_text), float(end_text)
            if not 0 <= start < end <= sample_count / 8000:
                raise ValueError(f"Invalid review interval: {specification}")
            intervals.append({
                "name": name,
                "reviewed_truth": truth,
                "start": start,
                "end": end,
                "start_sample": round(start * 8000),
                "end_sample": round(end * 8000),
                "input_power": 0.0,
                "output_power": 0.0,
                "cross_power": 0.0,
                "samples": 0,
            })
        whole_input_power = whole_output_power = 0.0
        total_chunks = math.ceil(sample_count / 32000)
        for index, chunk in enumerate(audio.blocks(blocksize=32000, dtype="float32")):
            block_start = index * 32000
            original = torch.from_numpy(chunk)
            model_input = original
            if model_input.numel() < 800:
                model_input = F.pad(model_input, (0, 800 - model_input.numel()))
            with torch.inference_mode():
                separated = model(
                    model_input.unsqueeze(0).to(device),
                    reference.unsqueeze(0),
                    reference_length,
                )[0][0, : original.numel()].float().cpu()
            input64 = original.double()
            output64 = separated.double()
            whole_input_power += float(input64.square().sum())
            whole_output_power += float(output64.square().sum())
            for interval in intervals:
                low = max(0, interval["start_sample"] - block_start)
                high = min(original.numel(), interval["end_sample"] - block_start)
                if high <= low:
                    continue
                source_part, output_part = input64[low:high], output64[low:high]
                interval["input_power"] += float(source_part.square().sum())
                interval["output_power"] += float(output_part.square().sum())
                interval["cross_power"] += float((source_part * output_part).sum())
                interval["samples"] += high - low
            if (index + 1) % 30 == 0 or index + 1 == total_chunks:
                print(f"SpEx+ full episode: {index + 1}/{total_chunks} chunks", flush=True)
    reviews = []
    for interval in intervals:
        source_power = max(interval["input_power"], 1e-16)
        output_power = max(interval["output_power"], 1e-16)
        reviews.append({
            key: interval[key] for key in
            ("name", "reviewed_truth", "start", "end", "samples")
        } | {
            "output_over_input_db": round(
                10 * math.log10(output_power / source_power), 4
            ),
            "waveform_cosine": round(
                interval["cross_power"] / math.sqrt(source_power * output_power),
                6,
            ),
        })
    return {
        "source": str(path.resolve()),
        "source_sha256": sha256(path),
        "duration_seconds": round(sample_count / 8000, 5),
        "window_seconds": 4.0,
        "window_caveat": "Independent 4 s model windows; edges need review.",
        "global_output_over_input_db": round(
            10 * math.log10(max(whole_output_power, 1e-16)
                            / max(whole_input_power, 1e-16)),
            4,
        ),
        "reviewed_intervals": reviews,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--case", action="append",
        help="NAME|target-or-other-or-mixed|AUDIO_PATH",
    )
    parser.add_argument("--episode", type=Path, help="8 kHz mono full recording")
    parser.add_argument(
        "--review", action="append", default=[],
        help="NAME|target-or-other-or-mixed|START_SECONDS|END_SECONDS",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--write-audio", action="store_true")
    parser.add_argument("--audit-all-heads", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if not args.case and not args.episode:
        parser.error("Pass --case or --episode")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; pass --device cpu")
    device = torch.device(args.device)
    model, checkpoint = load_model(device)
    reference = load_audio(args.reference).to(device)
    reference_length = torch.tensor([reference.numel()], device=device)
    results = []
    for specification in args.case or []:
        name, truth, value = specification.split("|", 2)
        if truth not in {"target", "other", "mixed"}:
            raise ValueError(f"Unsupported truth label: {truth}")
        path = Path(value)
        source = load_audio(path).to(device)
        with torch.inference_mode():
            outputs = model(
                source.unsqueeze(0), reference.unsqueeze(0), reference_length
            )
            extracted = outputs[0][0].float().cpu()
        original = source.cpu()
        if extracted.numel() != original.numel():
            raise RuntimeError(f"Unexpected output length for {name}")
        source_rms = rms(original)
        output_rms = rms(extracted)
        record = {
            "name": name,
            "reviewed_truth": truth,
            "input": str(path.resolve()),
            "duration_seconds": round(original.numel() / 8000, 5),
            "input_rms": round(source_rms, 7),
            "output_rms": round(output_rms, 7),
            "output_over_input_db": round(
                20 * math.log10(max(output_rms, 1e-10) / max(source_rms, 1e-10)),
                4,
            ),
            "waveform_cosine": round(
                float(F.cosine_similarity(original, extracted, dim=0)), 6
            ),
        }
        if args.audit_all_heads:
            record["head_metrics"] = [
                {
                    "head": index + 1,
                    "output_over_input_db": round(
                        20 * math.log10(max(rms(output[0]), 1e-10)
                                        / max(source_rms, 1e-10)), 4
                    ),
                    "waveform_cosine": round(float(F.cosine_similarity(
                        original, output[0].float().cpu(), dim=0,
                    )), 6),
                }
                for index, output in enumerate(outputs[:3])
            ]
        if args.write_audio:
            audio_path = args.output.parent / f"{name}_UNVERIFIED.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            # The research checkpoint can emit enormous amplitudes outside
            # its training domain.  Save a listenable diagnostic, never a
            # training-ready or acoustically faithful export.
            listenable = extracted * (source_rms / max(output_rms, 1e-10))
            peak = float(listenable.abs().max())
            if peak > 0.98:
                listenable *= 0.98 / peak
            sf.write(audio_path, listenable.numpy(), 8000)
            record["output_audio"] = str(audio_path.resolve())
            record["output_audio_gain_normalized_for_review"] = True
        results.append(record)
    payload = {
        "purpose": "development_only_not_product_acceptance",
        "upstream_source": (
            "modelscope/ClearerVoice-Studio/train/target_speaker_extraction/"
            "models/SpEx_plus/SpEx_plus.py"
        ),
        "checkpoint_sha256": sha256(checkpoint),
        "reference": str(args.reference.resolve()),
        "reference_sha256": sha256(args.reference),
        "cases": results,
    }
    if args.episode:
        payload["episode"] = inspect_episode(
            args.episode, args.review, model, reference,
            reference_length, device,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output.resolve())
    for record in results:
        print(
            record["name"], record["reviewed_truth"],
            record["output_over_input_db"], record["waveform_cosine"],
        )
    if args.episode:
        print("episode_seconds=", payload["episode"]["duration_seconds"])
        for record in payload["episode"]["reviewed_intervals"]:
            print(
                record["name"], record["reviewed_truth"],
                record["output_over_input_db"], record["waveform_cosine"],
            )


if __name__ == "__main__":
    main()
