from __future__ import annotations

from pathlib import Path


TOOL_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TOOL_ROOT.parent
GPT_ROOT = WORKSPACE_ROOT / "GPT-SoVITS-v2pro-20250604"
GPT_RUNTIME = GPT_ROOT / "runtime"
FFMPEG = GPT_RUNTIME / "ffmpeg.exe"
FFPROBE = GPT_RUNTIME / "ffprobe.exe"

UVR_ROOT = GPT_ROOT / "tools" / "uvr5"
UVR_MODEL = UVR_ROOT / "uvr5_weights" / "HP2_all_vocals.pth"

ASR_MODELS = GPT_ROOT / "tools" / "asr" / "models"
PARAFORMER_MODEL = ASR_MODELS / "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
VAD_MODEL = ASR_MODELS / "speech_fsmn_vad_zh-cn-16k-common-pytorch"
PUNC_MODEL = ASR_MODELS / "punc_ct-transformer_zh-cn-common-vocab272727-pytorch"

SV_CODE = GPT_ROOT / "GPT_SoVITS" / "eres2net"
SV_MODEL = GPT_ROOT / "GPT_SoVITS" / "pretrained_models" / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt"
CAMPLUS_MODEL = TOOL_ROOT / "models" / "campplus_voxceleb" / "campplus_voxceleb.bin"

WHISPER_CACHE = WORKSPACE_ROOT / "omnvoice" / "hf_cache" / "models--openai--whisper-large-v3-turbo"
OVERLAP_MODEL = TOOL_ROOT / "models" / "overlap" / "model.onnx"
PANNS_MODEL = TOOL_ROOT / "models" / "panns" / "Cnn10_mAP=0.380.pth"
PANNS_LABELS = TOOL_ROOT / "models" / "panns" / "class_labels_indices.csv"

VENDOR_ROOT = TOOL_ROOT / "vendor"
WORK_ROOT = TOOL_ROOT / "work"
OUTPUT_ROOT = TOOL_ROOT / "output"


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
