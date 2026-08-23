from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


DIRECT_ASSETS = (
    (
        "UVR5 vocal separator",
        "uvr",
        "HP2_all_vocals.pth",
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP2_all_vocals.pth?download=true",
        None,
    ),
    (
        "ERes2NetV2 speaker model",
        "eres",
        "pretrained_eres2netv2w24s4ep4.ckpt",
        "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/sv/pretrained_eres2netv2w24s4ep4.ckpt?download=true",
        None,
    ),
    (
        "CAM++ speaker model",
        "camplus",
        "campplus_voxceleb.bin",
        "https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/resolve/master/campplus_cn_common.bin",
        None,
    ),
    (
        "WavLM speaker model config",
        "wavlm",
        "config.json",
        "https://huggingface.co/microsoft/wavlm-base-plus-sv/resolve/feb593a6c23c1cc3d9510425c29b0a14d2b07b1e/config.json?download=true",
        None,
    ),
    (
        "WavLM speaker preprocessor config",
        "wavlm",
        "preprocessor_config.json",
        "https://huggingface.co/microsoft/wavlm-base-plus-sv/resolve/feb593a6c23c1cc3d9510425c29b0a14d2b07b1e/preprocessor_config.json?download=true",
        None,
    ),
    (
        "WavLM speaker model",
        "wavlm",
        "pytorch_model.bin",
        "https://huggingface.co/microsoft/wavlm-base-plus-sv/resolve/feb593a6c23c1cc3d9510425c29b0a14d2b07b1e/pytorch_model.bin?download=true",
        None,
    ),
    (
        "WeSpeaker ONNX verifier",
        "wespeaker",
        "model.onnx",
        "https://huggingface.co/onnx-community/wespeaker-voxceleb-resnet34-LM/resolve/6a61a1833ff2583aabeba044f5c8221f00b67ceb/onnx/model.onnx?download=true",
        "3955447b0499dc9e0a4541a895df08b03c69098eba4e56c02b5603e9f7f4fcbb",
    ),
    (
        "overlap detector",
        "overlap",
        "model.onnx",
        "https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/main/model.onnx?download=true",
        "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079",
    ),
    (
        "singing detector",
        "panns",
        "Cnn10_mAP=0.380.pth",
        "https://zenodo.org/records/3987831/files/Cnn10_mAP%3D0.380.pth?download=1",
        None,
    ),
    (
        "AudioSet labels",
        "panns",
        "class_labels_indices.csv",
        "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv",
        None,
    ),
)

MODELSCOPE_ASSETS = (
    (
        "Paraformer ASR",
        "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    ),
    (
        "FSMN VAD",
        "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "speech_fsmn_vad_zh-cn-16k-common-pytorch",
    ),
    (
        "CT-Punc",
        "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
        "punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    ),
)

MODULES = {
    "torch": "torch (provided by the local GPT-SoVITS runtime)",
    "torchaudio": "torchaudio (provided by the local GPT-SoVITS runtime)",
    "gradio": "gradio==4.44.1",
    "funasr": "funasr==1.0.27",
    "faster_whisper": "faster-whisper==1.1.1",
    "onnxruntime": "onnxruntime>=1.16",
    "soundfile": "soundfile>=0.12",
    "numpy": "numpy<2",
    "scipy": "scipy",
    "modelscope": "modelscope",
    "huggingface_hub": "huggingface-hub",
    "transformers": "transformers==4.43.0",
}


def say(message: str) -> None:
    print(message, flush=True)


def find_source_root(executable_root: Path) -> Path:
    candidates: list[Path] = []
    configured = os.environ.get("VOICE_EXTRACT_ROOT")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend([executable_root, executable_root.parent, Path.cwd()])
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "app.py").is_file():
            return candidate
    raise FileNotFoundError(
        "Cannot find app.py. Set VOICE_EXTRACT_ROOT to the extraction tool folder."
    )


def find_python(source_root: Path) -> Path:
    candidates: list[Path] = []
    configured = os.environ.get("VOICE_EXTRACT_PYTHON")
    if configured:
        candidates.append(Path(configured).expanduser())
    workspace = source_root.parent
    candidates.extend(
        [
            workspace / "GPT-SoVITS-v2pro-20250604" / "runtime" / "python.exe",
            source_root / "runtime" / "python.exe",
            Path(sys.executable).with_name("python.exe"),
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Cannot find the local Python runtime. Put GPT-SoVITS-v2pro-20250604 next to this folder "
        "or set VOICE_EXTRACT_PYTHON. CUDA and PyTorch are not downloaded by this launcher."
    )


def run_python(python: Path, code: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(python), "-c", code, *arguments],
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def probe_modules(python: Path) -> dict[str, bool]:
    code = r'''
import importlib
import json
import sys
result = {}
for name in sys.argv[1:]:
    try:
        importlib.import_module(name)
        result[name] = True
    except Exception:
        result[name] = False
print(json.dumps(result))
'''
    result = run_python(python, code, *MODULES)
    if result.returncode != 0:
        raise RuntimeError("Python dependency probe failed:\n" + result.stderr.strip())
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise RuntimeError("Python dependency probe returned invalid data") from exc


def install_missing_modules(python: Path, missing: list[str]) -> None:
    critical = [name for name in missing if name in {"torch", "torchaudio"}]
    if critical:
        names = ", ".join(critical)
        raise RuntimeError(
            f"Missing {names}. This launcher never downloads CUDA or PyTorch; "
            "use a GPT-SoVITS runtime that already contains them."
        )
    packages = [MODULES[name] for name in missing]
    say("Missing Python packages detected: " + ", ".join(packages))
    say("Installing missing Python packages into the local runtime...")
    result = subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            *packages,
        ],
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Python package installation failed. Check the console output and network connection."
        )


def ensure_modules(python: Path) -> None:
    states = probe_modules(python)
    missing = [name for name, available in states.items() if not available]
    if missing:
        install_missing_modules(python, missing)
        states = probe_modules(python)
        still_missing = [name for name, available in states.items() if not available]
        if still_missing:
            raise RuntimeError("Still missing Python modules: " + ", ".join(still_missing))
    say("Python runtime check: OK (CUDA/PyTorch kept local)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_file(path: Path, expected_hash: str | None = None) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    return expected_hash is None or sha256(path).lower() == expected_hash.lower()


def download_file(name: str, url: str, destination: Path, expected_hash: str | None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if partial.exists():
        partial.unlink()
    say(f"Downloading {name}...")
    request = urllib.request.Request(url, headers={"User-Agent": "VoiceExtractor/1.0"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
            total = int(response.headers.get("Content-Length") or 0)
            received = 0
            last_report = -1
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
                received += len(block)
                if total:
                    percent = int(received * 100 / total)
                    if percent >= last_report + 5 or percent == 100:
                        say(f"  {percent:3d}% ({received / 1024 / 1024:.1f} / {total / 1024 / 1024:.1f} MB)")
                        last_report = percent
                elif received // (10 * 1024 * 1024) > last_report:
                    last_report = received // (10 * 1024 * 1024)
                    say(f"  {received / 1024 / 1024:.1f} MB received")
        if expected_hash and sha256(partial).lower() != expected_hash.lower():
            raise RuntimeError(f"Checksum mismatch for {name}")
        os.replace(partial, destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    elapsed = max(0.1, time.monotonic() - started)
    say(f"  Finished {name} ({destination.stat().st_size / 1024 / 1024:.1f} MB, {elapsed:.1f}s)")


def download_modelscope_snapshot(python: Path, model_id: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    say(f"Downloading {model_id}...")
    code = r'''
import sys
from modelscope import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2])
'''
    result = subprocess.run(
        [str(python), "-c", code, model_id, str(destination)],
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ModelScope download failed for {model_id}")


def download_whisper(python: Path, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    say("Downloading Whisper large-v3-turbo...")
    code = r'''
import sys
from huggingface_hub import snapshot_download
snapshot_download("openai/whisper-large-v3-turbo", cache_dir=sys.argv[1])
'''
    result = subprocess.run(
        [str(python), "-c", code, str(cache_dir)],
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("Whisper download failed")


def whisper_ready(cache_dir: Path) -> bool:
    snapshot_root = cache_dir / "models--openai--whisper-large-v3-turbo" / "snapshots"
    return any(path.is_dir() for path in snapshot_root.glob("*"))


def ensure_assets(source_root: Path, python: Path) -> None:
    workspace = source_root.parent
    gpt_root = workspace / "GPT-SoVITS-v2pro-20250604"
    direct_paths = {
        "uvr": gpt_root / "tools" / "uvr5" / "uvr5_weights",
        "eres": gpt_root / "GPT_SoVITS" / "pretrained_models" / "sv",
        "camplus": source_root / "models" / "campplus_voxceleb",
        "wavlm": source_root / "models" / "wavlm-base-plus-sv",
        "wespeaker": source_root / "models" / "wespeaker-resnet34-lm" / "onnx",
        "overlap": source_root / "models" / "overlap",
        "panns": source_root / "models" / "panns",
    }
    for name, folder_key, filename, url, expected_hash in DIRECT_ASSETS:
        destination = direct_paths[folder_key] / filename
        if valid_file(destination, expected_hash):
            say(f"[OK] {name}")
            continue
        download_file(name, url, destination, expected_hash)
        if not valid_file(destination, expected_hash):
            raise RuntimeError(f"Downloaded file is not usable: {destination}")

    asr_root = gpt_root / "tools" / "asr" / "models"
    for name, model_id, folder_name in MODELSCOPE_ASSETS:
        destination = asr_root / folder_name
        if valid_file(destination / "model.pt"):
            say(f"[OK] {name}")
            continue
        download_modelscope_snapshot(python, model_id, destination)
        if not valid_file(destination / "model.pt"):
            raise RuntimeError(f"ModelScope model is incomplete: {destination}")

    whisper_cache = workspace / "omnvoice" / "hf_cache"
    if whisper_ready(whisper_cache):
        say("[OK] Whisper large-v3-turbo")
    else:
        download_whisper(python, whisper_cache)
        if not whisper_ready(whisper_cache):
            raise RuntimeError("Whisper download completed but no snapshot was found")

    required = {
        "Python runtime": gpt_root / "runtime" / "python.exe",
        "FFmpeg": gpt_root / "runtime" / "ffmpeg.exe",
        "FFprobe": gpt_root / "runtime" / "ffprobe.exe",
        "app.py": source_root / "app.py",
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.is_file()]
    if missing:
        raise RuntimeError("Required local files are missing:\n" + "\n".join(missing))
    say("Dependency and model check: OK")


def launch_app(source_root: Path, python: Path) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source_root) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("MODELSCOPE_CACHE", str(source_root / "models" / "modelscope_cache"))
    env.setdefault("PYTHONUNBUFFERED", "1")
    say("Starting the local service. The web UI is Chinese.")
    return subprocess.call([str(python), "-u", "app.py"], cwd=str(source_root), env=env)


def main() -> int:
    executable_root = (
        Path(sys.executable).resolve().parent
        if getattr(sys, "frozen", False)
        else Path(__file__).resolve().parent
    )
    try:
        source_root = find_source_root(executable_root)
        python = find_python(source_root)
        say("Voice Extractor dependency check")
        say("--------------------------------")
        ensure_modules(python)
        ensure_assets(source_root, python)
        if "--check-only" in sys.argv:
            say("Check complete. --check-only was supplied, so the service was not started.")
            return 0
        return launch_app(source_root, python)
    except KeyboardInterrupt:
        say("Interrupted by user.")
        return 130
    except Exception as exc:
        say(f"ERROR: {exc}")
        say("The service was not started. Press Enter to close this window.")
        try:
            input()
        except EOFError:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
