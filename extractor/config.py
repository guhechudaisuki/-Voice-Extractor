from __future__ import annotations

import os
from pathlib import Path


TOOL_ROOT = Path(__file__).resolve().parents[1]


def _configured_path(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    return Path(value).expanduser().resolve()


def _asset_root() -> Path:
    """Find the shared tool assets when this candidate lives under versions/."""

    for candidate in (TOOL_ROOT, *TOOL_ROOT.parents):
        if (
            (candidate / "models").exists()
            and (candidate.parent / "GPT-SoVITS-v2pro-20250604").exists()
        ):
            return candidate
    return TOOL_ROOT


ASSET_ROOT = _configured_path("VOICE_EXTRACT_ASSET_ROOT") or _asset_root()
WORKSPACE_ROOT = ASSET_ROOT.parent
GPT_ROOT = (
    _configured_path("VOICE_EXTRACT_GPT_ROOT")
    or WORKSPACE_ROOT / "GPT-SoVITS-v2pro-20250604"
)
GPT_RUNTIME = GPT_ROOT / "runtime"
FFMPEG = _configured_path("VOICE_EXTRACT_FFMPEG") or GPT_RUNTIME / "ffmpeg.exe"
FFPROBE = _configured_path("VOICE_EXTRACT_FFPROBE") or GPT_RUNTIME / "ffprobe.exe"

UVR_ROOT = _configured_path("VOICE_EXTRACT_UVR_ROOT") or GPT_ROOT / "tools" / "uvr5"
UVR_MODEL = (
    _configured_path("VOICE_EXTRACT_UVR_MODEL")
    or UVR_ROOT / "uvr5_weights" / "HP2_all_vocals.pth"
)

ASR_MODELS = (
    _configured_path("VOICE_EXTRACT_ASR_MODELS")
    or GPT_ROOT / "tools" / "asr" / "models"
)
PARAFORMER_MODEL = ASR_MODELS / "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
VAD_MODEL = ASR_MODELS / "speech_fsmn_vad_zh-cn-16k-common-pytorch"
PUNC_MODEL = ASR_MODELS / "punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
PARAFORMER_MODEL = _configured_path("VOICE_EXTRACT_PARAFORMER_MODEL") or PARAFORMER_MODEL
VAD_MODEL = _configured_path("VOICE_EXTRACT_VAD_MODEL") or VAD_MODEL
PUNC_MODEL = _configured_path("VOICE_EXTRACT_PUNC_MODEL") or PUNC_MODEL

SV_CODE = (
    _configured_path("VOICE_EXTRACT_SV_CODE")
    or GPT_ROOT / "GPT_SoVITS" / "eres2net"
)
SV_MODEL = (
    _configured_path("VOICE_EXTRACT_SV_MODEL")
    or GPT_ROOT
    / "GPT_SoVITS"
    / "pretrained_models"
    / "sv"
    / "pretrained_eres2netv2w24s4ep4.ckpt"
)
CAMPLUS_MODEL = (
    _configured_path("VOICE_EXTRACT_CAMPLUS_MODEL")
    or ASSET_ROOT / "models" / "campplus_voxceleb" / "campplus_voxceleb.bin"
)
WAVLM_SV_MODEL = (
    _configured_path("VOICE_EXTRACT_WAVLM_MODEL")
    or ASSET_ROOT / "models" / "wavlm-base-plus-sv"
)
WESPEAKER_MODEL = (
    _configured_path("VOICE_EXTRACT_WESPEAKER_MODEL")
    or ASSET_ROOT / "models" / "wespeaker-resnet34-lm" / "onnx" / "model.onnx"
)

WHISPER_CACHE_ROOT = (
    _configured_path("VOICE_EXTRACT_WHISPER_CACHE")
    or WORKSPACE_ROOT / "omnvoice" / "hf_cache"
)
WHISPER_CACHE = WHISPER_CACHE_ROOT / "models--openai--whisper-large-v3-turbo"
OVERLAP_MODEL = (
    _configured_path("VOICE_EXTRACT_OVERLAP_MODEL")
    or ASSET_ROOT / "models" / "overlap" / "model.onnx"
)
PANNS_MODEL = (
    _configured_path("VOICE_EXTRACT_PANNS_MODEL")
    or ASSET_ROOT / "models" / "panns" / "Cnn10_mAP=0.380.pth"
)
PANNS_LABELS = (
    _configured_path("VOICE_EXTRACT_PANNS_LABELS")
    or ASSET_ROOT / "models" / "panns" / "class_labels_indices.csv"
)

VENDOR_ROOT = TOOL_ROOT / "vendor"
DATA_ROOT = _configured_path("VOICE_EXTRACT_DATA_ROOT") or TOOL_ROOT
WORK_ROOT = _configured_path("VOICE_EXTRACT_WORK_ROOT") or DATA_ROOT / "work"
OUTPUT_ROOT = _configured_path("VOICE_EXTRACT_OUTPUT_ROOT") or DATA_ROOT / "output"


def whisper_snapshot() -> Path:
    snapshots = WHISPER_CACHE / "snapshots"
    candidates = sorted(p for p in snapshots.glob("*") if p.is_dir())
    if not candidates:
        raise FileNotFoundError(f"未找到本地 Whisper 模型：{snapshots}")
    return candidates[-1]


def ensure_local_assets() -> None:
    required = {
        "GPT-SoVITS runtime": GPT_RUNTIME / "python.exe",
        "FFmpeg": FFMPEG,
        "UVR5 model": UVR_MODEL,
        "Paraformer": PARAFORMER_MODEL / "model.pt",
        "FSMN-VAD": VAD_MODEL / "model.pt",
        "CT-Punc": PUNC_MODEL / "model.pt",
        "ERes2NetV2": SV_MODEL,
        "CAM++ speaker verifier": CAMPLUS_MODEL,
        "WavLM speaker verifier": WAVLM_SV_MODEL / "pytorch_model.bin",
        "WavLM config": WAVLM_SV_MODEL / "config.json",
        "WavLM preprocessor": WAVLM_SV_MODEL / "preprocessor_config.json",
        "WeSpeaker verifier": WESPEAKER_MODEL,
        "overlap detector": OVERLAP_MODEL,
        "singing detector": PANNS_MODEL,
        "PANNs labels": PANNS_LABELS,
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少本地资源：\n" + "\n".join(missing))
    whisper_snapshot()
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
