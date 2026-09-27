"""Bounded Personal VAD 2.0 trial; never accepts or exports product audio."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
PVAD_ROOT = ROOT / "work" / "personal_vad_2_src"
CAMPLUS_ROOT = (
    ROOT / "models" / "modelscope_cache" / "iic"
    / "speech_campplus_sv_zh-cn_16k-common"
)
GPT_RUNTIME_SITE_PACKAGES = (
    ROOT.parent / "GPT-SoVITS-v2pro-20250604" / "runtime"
    / "lib" / "site-packages"
)
sys.path.insert(0, str(PVAD_ROOT))
sys.path.append(str(GPT_RUNTIME_SITE_PACKAGES))

from examples.pilot4k_speaker_aware_inference import (  # noqa: E402
    CALIBRATION_SHA256,
    load_checkpoint,
    run_inference,
    sha256_file,
)
from features import PvadFeatureExtractor, load_audio  # noqa: E402
from funasr.models.campplus.model import CAMPPlus  # noqa: E402
from postprocessing import (  # noqa: E402
    TargetSpeechStateMachine,
    TargetSpeechStateMachineConfig,
)
from streaming import PvadStreamingAdapter  # noqa: E402


def reference_embedding(paths: list[Path], device: torch.device) -> torch.Tensor:
    """Use the exact 192-d CAM++ frontend expected by the pilot checkpoint."""
    model = CAMPPlus(feat_dim=80, embedding_size=192).to(device).eval()
    state = torch.load(
        CAMPLUS_ROOT / "campplus_cn_common.bin",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    embeddings = []
    for path in paths:
        waveform = load_audio(path)
        feature = torchaudio.compliance.kaldi.fbank(
            waveform.unsqueeze(0), num_mel_bins=80,
        )
        feature -= feature.mean(dim=0, keepdim=True)
        with torch.inference_mode():
            vector = model(feature.unsqueeze(0).to(device))[0].float().cpu()
        embeddings.append(F.normalize(vector, dim=0))
    return F.normalize(torch.stack(embeddings).mean(dim=0), dim=0)


def load_calibration_portable(path: Path):
    """Account for Windows Git CRLF checkout without changing the model card."""
    normalized = path.read_bytes().replace(b"\r\n", b"\n")
    digest = hashlib.sha256(normalized).hexdigest()
    if digest != CALIBRATION_SHA256:
        raise ValueError(f"Personal VAD calibration checksum mismatch: {digest}")
    payload = json.loads(normalized)
    if payload.get("status") != "dev_calibrated_control_candidate":
        raise ValueError("Unsupported Personal VAD calibration status")
    return TargetSpeechStateMachineConfig.from_dict(
        payload["target_speech_state_machine"]
    ), digest


def inspect_case(
    path: Path,
    name: str,
    truth: str,
    model: torch.nn.Module,
    embedding: torch.Tensor,
    class_names: list[str],
    calibration,
    device: torch.device,
) -> dict:
    waveform = load_audio(path)
    features = PvadFeatureExtractor().extract(waveform)
    result = run_inference(
        model, features, embedding, class_names, device,
        target_fsm_config=calibration,
    )
    counts = {
        class_names[index]: sum(
            segment["frames"] for segment in result["segments"]
            if segment["class_id"] == index
        )
        for index in range(len(class_names))
    }
    fsm = result["target_speech_fsm"]["segments"]
    return {
        "name": name,
        "reviewed_truth": truth,
        "path": str(path.resolve()),
        "audio_seconds": round(waveform.numel() / 16000, 4),
        "frame_count": result["frame_count"],
        "class_frame_fraction": {
            key: round(value / result["frame_count"], 5)
            for key, value in counts.items()
        },
        "target_fsm_seconds": round(
            sum((segment["end_ms"] - segment["start_ms"]) / 1000 for segment in fsm),
            4,
        ),
        "target_fsm_segments": fsm,
        "raw_segments": result["segments"],
    }


def inspect_episode(
    path: Path,
    reviews: list[str],
    model: torch.nn.Module,
    embedding: torch.Tensor,
    class_names: list[str],
    calibration,
    device: torch.device,
) -> dict:
    """Run an entire recording with the upstream causal streaming adapter."""
    adapter = PvadStreamingAdapter(model, embedding, device=device)
    probability_chunks = []
    frames = []
    reset_seconds = 120
    with sf.SoundFile(path) as audio:
        if audio.samplerate != 16000 or audio.channels != 1:
            raise ValueError("Episode input must be a 16 kHz mono WAV")
        duration = len(audio) / audio.samplerate
        for block_start in range(0, len(audio), reset_seconds * 16000):
            audio.seek(block_start)
            adapter.reset()
            block_frame_start = len(frames)
            remaining = min(reset_seconds * 16000, len(audio) - block_start)
            while remaining:
                chunk = audio.read(min(160000, remaining), dtype="float32")
                remaining -= len(chunk)
                output = adapter.feed_audio(torch.from_numpy(chunk))
                if output.probabilities.numel():
                    probability_chunks.append(output.probabilities.cpu())
                    frames.extend(
                        replace(
                            frame,
                            index=frame.index + block_frame_start,
                            stack_start_sample=frame.stack_start_sample + block_start,
                            decision_start_sample=frame.decision_start_sample + block_start,
                            decision_end_sample=frame.decision_end_sample + block_start,
                        )
                        for frame in output.frames
                    )
    probabilities = torch.cat(probability_chunks)
    labels = probabilities.argmax(dim=-1)
    state_machine = TargetSpeechStateMachine(calibration)
    state_machine.process(probabilities, frames)
    fsm_spans = [
        [start / 16000, end / 16000]
        for start, end in state_machine.snapshot_segments(frames[-1].decision_end_sample)
    ]
    reviewed = []
    for specification in reviews:
        name, truth, start_text, end_text = specification.split("|", 3)
        start, end = float(start_text), float(end_text)
        if not 0 <= start < end <= duration:
            raise ValueError(f"Invalid review interval: {specification}")
        indexes = [
            index for index, frame in enumerate(frames)
            if start <= (frame.decision_start_sample + frame.decision_end_sample)
            / 32000 < end
        ]
        if not indexes:
            raise ValueError(f"No decision frames in {specification}")
        values = labels[indexes]
        reviewed.append({
            "name": name,
            "reviewed_truth": truth,
            "start": start,
            "end": end,
            "class_frame_fraction": {
                class_names[class_id]: round(
                    float((values == class_id).float().mean()), 5
                )
                for class_id in range(len(class_names))
            },
            "target_fsm_overlap_seconds": round(sum(
                max(0.0, min(end, span_end) - max(start, span_start))
                for span_start, span_end in fsm_spans
            ), 4),
        })
    return {
        "path": str(path.resolve()),
        "source_sha256": sha256_file(path),
        "duration_seconds": round(duration, 5),
        "encoder_reset_seconds": reset_seconds,
        "encoder_reset_caveat": (
            "Model positional encoding is limited to 5000 frames; "
            "decisions near reset boundaries need separate review."
        ),
        "frame_count": len(frames),
        "class_frame_fraction": {
            class_names[class_id]: round(
                float((labels == class_id).float().mean()), 5
            )
            for class_id in range(len(class_names))
        },
        "target_fsm_segments": fsm_spans,
        "reviewed_intervals": reviewed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", action="append", required=True, type=Path)
    parser.add_argument(
        "--case", action="append",
        help="NAME|target-or-other|AUDIO_PATH; repeat for bounded clips",
    )
    parser.add_argument("--episode", type=Path, help="16 kHz mono full recording")
    parser.add_argument(
        "--review", action="append", default=[],
        help="NAME|target-or-other-or-mixed|START_SECONDS|END_SECONDS",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if not args.case and not args.episode:
        parser.error("Pass --case or --episode")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; pass --device cpu")
    device = torch.device(args.device)
    model, package = load_checkpoint(
        PVAD_ROOT / "checkpoints" / "pilot4k_speaker_aware_epoch38"
        / "best_inference.pt",
        device,
    )
    calibration, calibration_hash = load_calibration_portable(
        PVAD_ROOT / "checkpoints" / "pilot4k_speaker_aware_epoch38"
        / "target_fsm.json"
    )
    embedding = reference_embedding(args.reference, device)
    class_names = list(package["class_names"])
    cases = []
    for specification in args.case or []:
        name, truth, path = specification.split("|", 2)
        if truth not in {"target", "other", "mixed"}:
            raise ValueError(f"Unsupported truth label: {truth}")
        cases.append(inspect_case(
            Path(path), name, truth, model, embedding,
            class_names, calibration, device,
        ))
    output = {
        "purpose": "development_only_not_product_acceptance",
        "checkpoint_sha256": sha256_file(
            PVAD_ROOT / "checkpoints" / "pilot4k_speaker_aware_epoch38"
            / "best_inference.pt"
        ),
        "camplus_sha256": hashlib.sha256(
            (CAMPLUS_ROOT / "campplus_cn_common.bin").read_bytes()
        ).hexdigest(),
        "calibration_sha256": calibration_hash,
        "reference_paths": [str(path.resolve()) for path in args.reference],
        "reference_embedding_norm": round(float(embedding.norm()), 6),
        "cases": cases,
    }
    if args.episode:
        output["episode"] = inspect_episode(
            args.episode, args.review, model, embedding,
            class_names, calibration, device,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output.resolve())
    for case in cases:
        print(
            case["name"], case["reviewed_truth"],
            case["class_frame_fraction"],
            "target_fsm_seconds=", case["target_fsm_seconds"],
        )
    if args.episode:
        episode = output["episode"]
        print("episode_seconds=", episode["duration_seconds"])
        for review in episode["reviewed_intervals"]:
            print(
                review["name"], review["reviewed_truth"],
                review["class_frame_fraction"],
                "target_fsm_overlap_seconds=",
                review["target_fsm_overlap_seconds"],
            )


if __name__ == "__main__":
    main()
