from __future__ import annotations

import hashlib
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable


APP_VERSION = "2026.08.25-desktop-2"
APP_NAME = "Voice Extractor"
GLOBAL_SETTINGS = (
    Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    / "VoiceExtractor"
    / "desktop_settings.json"
)
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

PYTHON_INSTALLER_VERSION = "3.9.13"
PYTHON_INSTALLER_URL = (
    "https://www.python.org/ftp/python/3.9.13/"
    "python-3.9.13-amd64.exe"
)
PYTHON_INSTALLER_FILENAME = f"python-{PYTHON_INSTALLER_VERSION}-amd64.exe"
PYTHON_DOWNLOAD_MB = 28
PYTHON_INSTALLED_MB = 120
PYTORCH_OFFICIAL_URL = "https://pytorch.org/get-started/locally/"
CUDA_OFFICIAL_URL = "https://developer.nvidia.com/cuda-downloads"

Reporter = Callable[[float, str], None]


MODULE_PACKAGES = {
    "torch": None,
    "torchaudio": None,
    "funasr": "funasr==1.0.27",
    "faster_whisper": "faster-whisper==1.1.1",
    "onnxruntime": "onnxruntime>=1.16",
    "soundfile": "soundfile>=0.12",
    "numpy": "numpy<2",
    "scipy": "scipy",
    "modelscope": "modelscope",
    "huggingface_hub": "huggingface-hub",
    "transformers": "transformers==4.43.0",
    "librosa": "librosa>=0.9,<0.11",
    "tqdm": "tqdm>=4.60",
}


COMPONENT_ENVIRONMENT_PATHS = {
    "uvr_model": ("VOICE_EXTRACT_UVR_MODEL",),
    "sv_model": ("VOICE_EXTRACT_SV_MODEL",),
    "camplus_model": ("VOICE_EXTRACT_CAMPLUS_MODEL",),
    "wavlm_model": ("VOICE_EXTRACT_WAVLM_MODEL",),
    "wespeaker_model": ("VOICE_EXTRACT_WESPEAKER_MODEL",),
    "overlap_model": ("VOICE_EXTRACT_OVERLAP_MODEL",),
    "panns_model": (
        "VOICE_EXTRACT_PANNS_MODEL",
        "VOICE_EXTRACT_PANNS_LABELS",
    ),
    "paraformer_model": ("VOICE_EXTRACT_PARAFORMER_MODEL",),
    "vad_model": ("VOICE_EXTRACT_VAD_MODEL",),
    "punc_model": ("VOICE_EXTRACT_PUNC_MODEL",),
    "whisper_cache": ("VOICE_EXTRACT_WHISPER_CACHE",),
}


@dataclass(frozen=True)
class DirectFile:
    filename: str
    url: str
    expected_hash: str | None = None


@dataclass(frozen=True)
class ComponentSpec:
    key: str
    label: str
    relative_path: str
    kind: str
    estimate_mb: int
    files: tuple[DirectFile, ...] = ()
    model_id: str = ""
    check_file: str = ""


COMPONENTS = (
    ComponentSpec(
        "uvr_model",
        "UVR5 人声分离模型",
        "models/uvr5",
        "direct",
        65,
        (
            DirectFile(
                "HP2_all_vocals.pth",
                "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP2_all_vocals.pth?download=true",
            ),
        ),
        check_file="HP2_all_vocals.pth",
    ),
    ComponentSpec(
        "sv_model",
        "ERes2NetV2 声纹模型",
        "models/eres2net",
        "direct",
        55,
        (
            DirectFile(
                "pretrained_eres2netv2w24s4ep4.ckpt",
                "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/sv/pretrained_eres2netv2w24s4ep4.ckpt?download=true",
            ),
        ),
        check_file="pretrained_eres2netv2w24s4ep4.ckpt",
    ),
    ComponentSpec(
        "camplus_model",
        "CAM++ 声纹模型",
        "models/campplus_voxceleb",
        "direct",
        28,
        (
            DirectFile(
                "campplus_voxceleb.bin",
                "https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/resolve/master/campplus_cn_common.bin",
            ),
        ),
        check_file="campplus_voxceleb.bin",
    ),
    ComponentSpec(
        "wavlm_model",
        "WavLM 声纹模型",
        "models/wavlm-base-plus-sv",
        "direct",
        386,
        (
            DirectFile(
                "config.json",
                "https://huggingface.co/microsoft/wavlm-base-plus-sv/resolve/feb593a6c23c1cc3d9510425c29b0a14d2b07b1e/config.json?download=true",
            ),
            DirectFile(
                "preprocessor_config.json",
                "https://huggingface.co/microsoft/wavlm-base-plus-sv/resolve/feb593a6c23c1cc3d9510425c29b0a14d2b07b1e/preprocessor_config.json?download=true",
            ),
            DirectFile(
                "pytorch_model.bin",
                "https://huggingface.co/microsoft/wavlm-base-plus-sv/resolve/feb593a6c23c1cc3d9510425c29b0a14d2b07b1e/pytorch_model.bin?download=true",
            ),
        ),
        check_file="pytorch_model.bin",
    ),
    ComponentSpec(
        "wespeaker_model",
        "WeSpeaker 声纹复核模型",
        "models/wespeaker-resnet34-lm/onnx",
        "direct",
        26,
        (
            DirectFile(
                "model.onnx",
                "https://huggingface.co/onnx-community/wespeaker-voxceleb-resnet34-LM/resolve/6a61a1833ff2583aabeba044f5c8221f00b67ceb/onnx/model.onnx?download=true",
                "3955447b0499dc9e0a4541a895df08b03c69098eba4e56c02b5603e9f7f4fcbb",
            ),
        ),
        check_file="model.onnx",
    ),
    ComponentSpec(
        "overlap_model",
        "多人重叠检测模型",
        "models/overlap",
        "direct",
        6,
        (
            DirectFile(
                "model.onnx",
                "https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/main/model.onnx?download=true",
                "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079",
            ),
        ),
        check_file="model.onnx",
    ),
    ComponentSpec(
        "panns_model",
        "PANNs 歌声检测模型",
        "models/panns",
        "direct",
        32,
        (
            DirectFile(
                "Cnn10_mAP=0.380.pth",
                "https://zenodo.org/records/3987831/files/Cnn10_mAP%3D0.380.pth?download=1",
            ),
            DirectFile(
                "class_labels_indices.csv",
                "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv",
            ),
        ),
        check_file="Cnn10_mAP=0.380.pth",
    ),
    ComponentSpec(
        "paraformer_model",
        "Paraformer 中文识别模型",
        "models/asr/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "modelscope",
        950,
        model_id="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        check_file="model.pt",
    ),
    ComponentSpec(
        "vad_model",
        "FSMN 语音活动检测模型",
        "models/asr/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "modelscope",
        6,
        model_id="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        check_file="model.pt",
    ),
    ComponentSpec(
        "punc_model",
        "CT-Punc 标点模型",
        "models/asr/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
        "modelscope",
        280,
        model_id="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
        check_file="model.pt",
    ),
    ComponentSpec(
        "whisper_cache",
        "Whisper large-v3-turbo",
        "models/whisper/hf_cache",
        "whisper",
        1550,
        model_id="openai/whisper-large-v3-turbo",
    ),
)


@dataclass
class InstallLayout:
    root: Path

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def output(self) -> Path:
        return self.root / "output"

    @property
    def work(self) -> Path:
        return self.root / "work"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def tools(self) -> Path:
        return self.root / "tools"

    @property
    def packages(self) -> Path:
        return self.root / "python_packages"

    @property
    def requests(self) -> Path:
        return self.work / "desktop_requests"

    @property
    def settings_path(self) -> Path:
        return self.root / "voice_extractor_installation.json"

    def prepare(self) -> None:
        for path in (
            self.root,
            self.models,
            self.output,
            self.work,
            self.logs,
            self.tools,
            self.packages,
            self.requests,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class BootstrapContext:
    source_root: Path
    layout: InstallLayout
    python: Path
    assets: dict[str, Path]
    uvr_code: Path
    sv_code: Path
    ffmpeg: Path
    ffprobe: Path
    extra_python_paths: list[Path] = field(default_factory=list)

    def backend_environment(self, output_root: Path | None = None) -> dict[str, str]:
        env = os.environ.copy()
        python_paths = [
            str(self.source_root),
            str(self.source_root / "vendor"),
            str(self.layout.packages),
            *(str(path) for path in self.extra_python_paths),
        ]
        existing = env.get("PYTHONPATH")
        if existing:
            python_paths.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(python_paths)
        env["PYTHONUNBUFFERED"] = "1"
        env["HF_HUB_OFFLINE"] = "1"
        env["MODELSCOPE_CACHE"] = str(self.layout.models / "modelscope_cache")
        env["VOICE_EXTRACT_DATA_ROOT"] = str(self.layout.root)
        env["VOICE_EXTRACT_ASSET_ROOT"] = str(self.layout.root)
        env["VOICE_EXTRACT_WORK_ROOT"] = str(self.layout.work)
        env["VOICE_EXTRACT_OUTPUT_ROOT"] = str(output_root or self.layout.output)
        env["VOICE_EXTRACT_FFMPEG"] = str(self.ffmpeg)
        env["VOICE_EXTRACT_FFPROBE"] = str(self.ffprobe)
        env["VOICE_EXTRACT_UVR_ROOT"] = str(self.uvr_code)
        env["VOICE_EXTRACT_UVR_MODEL"] = str(
            self.assets["uvr_model"] / "HP2_all_vocals.pth"
        )
        env["VOICE_EXTRACT_SV_CODE"] = str(self.sv_code)
        env["VOICE_EXTRACT_SV_MODEL"] = str(
            self.assets["sv_model"] / "pretrained_eres2netv2w24s4ep4.ckpt"
        )
        env["VOICE_EXTRACT_CAMPLUS_MODEL"] = str(
            self.assets["camplus_model"] / "campplus_voxceleb.bin"
        )
        env["VOICE_EXTRACT_WAVLM_MODEL"] = str(self.assets["wavlm_model"])
        env["VOICE_EXTRACT_WESPEAKER_MODEL"] = str(
            self.assets["wespeaker_model"] / "model.onnx"
        )
        env["VOICE_EXTRACT_OVERLAP_MODEL"] = str(
            self.assets["overlap_model"] / "model.onnx"
        )
        env["VOICE_EXTRACT_PANNS_MODEL"] = str(
            self.assets["panns_model"] / "Cnn10_mAP=0.380.pth"
        )
        env["VOICE_EXTRACT_PANNS_LABELS"] = str(
            self.assets["panns_model"] / "class_labels_indices.csv"
        )
        env["VOICE_EXTRACT_PARAFORMER_MODEL"] = str(
            self.assets["paraformer_model"]
        )
        env["VOICE_EXTRACT_VAD_MODEL"] = str(self.assets["vad_model"])
        env["VOICE_EXTRACT_PUNC_MODEL"] = str(self.assets["punc_model"])
        env["VOICE_EXTRACT_WHISPER_CACHE"] = str(self.assets["whisper_cache"])
        return env


def bundle_source_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS")).resolve() / "app_bundle"
    return Path(__file__).resolve().parents[1]


def current_executable() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve()
    return Path(__file__).resolve()


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def load_global_settings() -> dict[str, object]:
    return _read_json(GLOBAL_SETTINGS)


def save_settings(
    layout: InstallLayout,
    python: Path,
    assets: dict[str, Path],
) -> None:
    payload = {
        "version": APP_VERSION,
        "install_root": str(layout.root),
        "python": str(python),
        "assets": {key: str(path) for key, path in assets.items()},
    }
    layout.settings_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    GLOBAL_SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    GLOBAL_SETTINGS.write_text(
        json.dumps(
            {
                "version": APP_VERSION,
                "install_root": str(layout.root),
                "python": str(python),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def suggested_install_root() -> Path:
    configured = os.environ.get("VOICE_EXTRACT_INSTALL_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    settings = load_global_settings()
    stored = settings.get("install_root")
    if stored:
        return Path(str(stored)).expanduser().resolve()
    executable = current_executable()
    for candidate in (executable.parent, executable.parent.parent, Path.cwd()):
        if (candidate / "models").is_dir() and (candidate / "output").exists():
            return candidate.resolve()
    return (
        Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        / "Programs"
        / "VoiceExtractor"
    ).resolve()


def _python_probe_environment(extra_paths: Iterable[Path]) -> dict[str, str]:
    env = os.environ.copy()
    values = [str(path) for path in extra_paths if path]
    if env.get("PYTHONPATH"):
        values.append(env["PYTHONPATH"])
    if values:
        env["PYTHONPATH"] = os.pathsep.join(values)
    env.pop("HF_HUB_OFFLINE", None)
    return env


def _path_from_py_launcher() -> Path | None:
    """Resolve the Python selected by the Windows ``py`` launcher."""

    launcher = shutil.which("py.exe") or shutil.which("py")
    if not launcher:
        return None
    try:
        result = subprocess.run(
            [launcher, "-3", "-c", "import sys; print(sys.executable)"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=CREATE_NO_WINDOW,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in reversed(result.stdout.splitlines()):
        candidate = Path(line.strip().strip('"'))
        if candidate.is_file():
            return candidate
    return None


def _registry_python_candidates() -> list[Path]:
    candidates: list[Path] = []
    if os.name != "nt":
        return candidates
    try:
        import winreg

        roots = (
            (winreg.HKEY_CURRENT_USER, r"Software\Python\PythonCore"),
            (winreg.HKEY_LOCAL_MACHINE, r"Software\Python\PythonCore"),
            (
                winreg.HKEY_LOCAL_MACHINE,
                r"Software\WOW6432Node\Python\PythonCore",
            ),
        )
        for root, key_path in roots:
            try:
                with winreg.OpenKey(root, key_path) as versions:
                    subkey_count = winreg.QueryInfoKey(versions)[0]
                    names = [
                        winreg.EnumKey(versions, index)
                        for index in range(subkey_count)
                    ]
            except OSError:
                continue
            for version in names:
                try:
                    install_key = key_path + "\\" + version + r"\InstallPath"
                    with winreg.OpenKey(root, install_key) as install:
                        value, _ = winreg.QueryValueEx(install, "")
                except OSError:
                    continue
                candidates.append(Path(str(value)) / "python.exe")
    except ImportError:
        pass
    return candidates


def _python_candidates(preferred: Path | None = None) -> list[Path]:
    settings = load_global_settings()
    values: list[Path] = []
    if preferred is not None:
        value = preferred.expanduser()
        values.append(value / "python.exe" if value.is_dir() else value)
    if not getattr(sys, "frozen", False):
        values.append(Path(sys.executable))
    if settings.get("python"):
        values.append(Path(str(settings["python"])).expanduser())
    for name in ("VOICE_EXTRACT_PYTHON", "PYTHONHOME", "PYTHON_ROOT"):
        configured = os.environ.get(name)
        if configured:
            value = Path(configured).expanduser()
            values.append(value / "python.exe" if value.is_dir() else value)
    for name in ("CONDA_PREFIX", "VIRTUAL_ENV"):
        configured = os.environ.get(name)
        if configured:
            values.append(Path(configured).expanduser() / "python.exe")

    py_python = _path_from_py_launcher()
    if py_python:
        values.append(py_python)
    for command in ("python.exe", "python"):
        found = shutil.which(command)
        if found:
            values.append(Path(found))
    values.extend(_registry_python_candidates())

    local_app_data = Path(
        os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")
    )
    program_files = Path(os.environ.get("ProgramFiles", "C:/Program Files"))
    program_files_x86 = Path(
        os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")
    )
    common_patterns = (
        local_app_data / "Programs" / "Python" / "Python*" / "python.exe",
        program_files / "Python*" / "python.exe",
        program_files_x86 / "Python*" / "python.exe",
        Path("C:/Python*") / "python.exe",
        Path.home() / ".conda" / "envs" / "*" / "python.exe",
    )
    for pattern in common_patterns:
        values.extend(Path(path) for path in glob.glob(str(pattern)))
    values.extend(
        (
            Path.home() / "miniconda3" / "python.exe",
            Path.home() / "anaconda3" / "python.exe",
        )
    )
    conda_envs = os.environ.get("CONDA_ENVS_PATH")
    if conda_envs:
        for root in conda_envs.split(os.pathsep):
            values.extend(Path(path) for path in glob.glob(str(Path(root) / "*" / "python.exe")))

    for root in _candidate_roots():
        values.extend(
            (
                root / "GPT-SoVITS-v2pro-20250604" / "runtime" / "python.exe",
                root / "runtime" / "python.exe",
                root / "python" / "python.exe",
            )
        )
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in values:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved not in seen and resolved.is_file():
            seen.add(resolved)
            unique.append(resolved)
    return unique


def discover_any_python(preferred: Path | None = None) -> Path | None:
    """Find any runnable Python without requiring the ML packages yet."""

    for candidate in _python_candidates(preferred):
        try:
            result = subprocess.run(
                [str(candidate), "-c", "import sys; print(sys.executable)"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=CREATE_NO_WINDOW,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            return candidate
    return None


def probe_python_runtime(python: Path) -> dict[str, object]:
    """Run a real import/GPU probe inside one candidate Python runtime."""

    code = r'''
import json
import platform
import sys

result = {
    "runnable": True,
    "executable": sys.executable,
    "version": platform.python_version(),
    "architecture": platform.architecture()[0],
    "pip": False,
    "torch": False,
    "torchaudio": False,
    "cuda_available": False,
    "cuda_build": None,
    "devices": [],
}
try:
    import pip
    result["pip"] = True
except Exception as exc:
    result["pip_error"] = repr(exc)
try:
    import torch
    result["torch"] = True
    result["torch_version"] = str(torch.__version__)
    result["cuda_build"] = torch.version.cuda
    result["cuda_available"] = bool(torch.cuda.is_available())
    if result["cuda_available"]:
        result["devices"] = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
except Exception as exc:
    result["torch_error"] = repr(exc)
try:
    import torchaudio
    result["torchaudio"] = True
    result["torchaudio_version"] = str(torchaudio.__version__)
except Exception as exc:
    result["torchaudio_error"] = repr(exc)
print(json.dumps(result, ensure_ascii=False))
'''
    try:
        completed = subprocess.run(
            [str(python), "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=CREATE_NO_WINDOW,
            timeout=90,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "runnable": False,
            "executable": str(python),
            "error": str(exc),
        }
    if completed.returncode:
        return {
            "runnable": False,
            "executable": str(python),
            "error": (completed.stderr or completed.stdout)[-800:],
        }
    try:
        result = json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {
            "runnable": False,
            "executable": str(python),
            "error": "Python 探测没有返回有效结果",
        }
    result["candidate"] = str(python)
    return result


def inspect_python_runtimes(
    preferred: Path | None = None,
    reporter: Reporter | None = None,
) -> list[dict[str, object]]:
    candidates = _python_candidates(preferred)
    if not candidates:
        return []
    if reporter:
        reporter(0.08, f"找到 {len(candidates)} 个 Python 候选，正在验证 PyTorch 与 CUDA")
    output: list[dict[str, object]] = []
    workers = min(4, len(candidates))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="python-probe") as pool:
        pending = {pool.submit(probe_python_runtime, path): path for path in candidates}
        completed_count = 0
        for future in as_completed(pending):
            output.append(future.result())
            completed_count += 1
            if reporter:
                reporter(
                    0.08 + 0.20 * completed_count / len(candidates),
                    f"Python 运行环境验证 {completed_count}/{len(candidates)}",
                )
    order = {str(path): index for index, path in enumerate(candidates)}
    output.sort(key=lambda item: order.get(str(item.get("candidate", "")), 9999))
    return output


def probe_modules(python: Path, extra_paths: Iterable[Path]) -> dict[str, bool]:
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
    try:
        result = subprocess.run(
            [str(python), "-c", code, *MODULE_PACKAGES],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_python_probe_environment(extra_paths),
            creationflags=CREATE_NO_WINDOW,
            timeout=240,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {name: False for name in MODULE_PACKAGES}
    if result.returncode != 0:
        return {name: False for name in MODULE_PACKAGES}
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {name: False for name in MODULE_PACKAGES}


def _candidate_roots() -> list[Path]:
    executable = current_executable()
    values = [Path.cwd(), executable.parent, executable.parent.parent]
    values.extend(executable.parents)
    configured = os.environ.get("VOICE_EXTRACT_GPT_ROOT")
    if configured:
        values.insert(0, Path(configured).expanduser())
    output: list[Path] = []
    for value in values:
        try:
            resolved = value.resolve()
        except OSError:
            continue
        if resolved not in output:
            output.append(resolved)
    return output


def discover_python(preferred: Path | None = None) -> Path | None:
    fallback: Path | None = None
    for candidate in _python_candidates(preferred):
        state = probe_python_runtime(candidate)
        if state.get("torch") and state.get("torchaudio"):
            if state.get("cuda_available"):
                return candidate
            if fallback is None:
                fallback = candidate
    if fallback is not None:
        return fallback
    return None


def infer_gpt_root(python: Path | None) -> Path | None:
    if python and python.parent.name.lower() == "runtime":
        parent = python.parent.parent
        if (parent / "GPT_SoVITS").is_dir() and (parent / "tools").is_dir():
            return parent
    for root in _candidate_roots():
        for candidate in (root, root / "GPT-SoVITS-v2pro-20250604"):
            if (
                (candidate / "runtime" / "python.exe").is_file()
                and (candidate / "GPT_SoVITS").is_dir()
            ):
                return candidate.resolve()
    return None


def _stream_process(
    command: list[str],
    env: dict[str, str],
    reporter: Reporter,
    progress: float,
) -> None:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        creationflags=CREATE_NO_WINDOW,
    )
    assert process.stdout is not None
    for line in process.stdout:
        value = line.strip()
        if value:
            reporter(progress, value[-500:])
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"命令执行失败，退出代码 {return_code}")


def ensure_python_modules(
    python: Path,
    layout: InstallLayout,
    source_root: Path,
    reporter: Reporter,
) -> None:
    extra_paths = [layout.packages, source_root / "vendor"]
    state = probe_modules(python, extra_paths)
    critical = [name for name in ("torch", "torchaudio") if not state.get(name)]
    if critical:
        raise RuntimeError(
            "所选 Python 缺少 " + "、".join(critical) + "；本工具不会下载 CUDA/PyTorch。"
        )
    missing = [
        package
        for name, package in MODULE_PACKAGES.items()
        if package and not state.get(name)
    ]
    if not missing:
        reporter(0.12, "Python 依赖检查完成，全部可用")
        return
    reporter(0.08, "正在把缺失 Python 依赖安装到工具自己的目录")
    env = _python_probe_environment(extra_paths)
    _stream_process(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "--target",
            str(layout.packages),
            *missing,
        ],
        env,
        reporter,
        0.10,
    )
    state = probe_modules(python, extra_paths)
    still_missing = [name for name in MODULE_PACKAGES if not state.get(name)]
    if still_missing:
        raise RuntimeError("仍缺少 Python 模块：" + "、".join(still_missing))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_file(path: Path, expected_hash: str | None = None) -> bool:
    if not path.is_file():
        return False
    try:
        size = path.stat().st_size
    except OSError:
        return False
    minimum = 1
    if path.suffix.lower() in {".pth", ".pt", ".ckpt", ".bin", ".safetensors"}:
        minimum = 1024 * 1024
    elif path.suffix.lower() == ".onnx":
        minimum = 64 * 1024
    elif path.suffix.lower() == ".csv":
        minimum = 128
    if size < minimum:
        return False
    return expected_hash is None or _sha256(path).lower() == expected_hash.lower()


def _component_ready(spec: ComponentSpec, path: Path) -> bool:
    if spec.kind == "whisper":
        snapshots = path / "models--openai--whisper-large-v3-turbo" / "snapshots"
        for snapshot in snapshots.glob("*"):
            if not snapshot.is_dir():
                continue
            model = snapshot / "model.safetensors"
            if not model.is_file():
                model = snapshot / "pytorch_model.bin"
            if (
                _valid_file(model)
                and _valid_file(snapshot / "config.json")
                and (
                    _valid_file(snapshot / "tokenizer.json")
                    or _valid_file(snapshot / "vocab.json")
                )
            ):
                return True
        return False
    if spec.kind == "direct":
        return all(
            _valid_file(path / item.filename, item.expected_hash)
            for item in spec.files
        )
    return _valid_file(path / spec.check_file)


def _normalise_component_candidate(spec: ComponentSpec, value: Path) -> Path:
    value = value.expanduser()
    if spec.kind == "whisper":
        if value.name == "models--openai--whisper-large-v3-turbo":
            return value.parent
        if value.name == "snapshots" and value.parent.name == "models--openai--whisper-large-v3-turbo":
            return value.parent.parent
        return value
    if value.suffix or value.is_file():
        return value.parent
    return value


def _cache_component_paths(spec: ComponentSpec) -> list[Path]:
    candidates: list[Path] = []
    if spec.kind == "whisper":
        for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
            if os.environ.get(name):
                candidates.append(Path(os.environ[name]).expanduser())
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        candidates.extend((hf_home / "hub", hf_home))
    elif spec.kind == "modelscope":
        roots = []
        if os.environ.get("MODELSCOPE_CACHE"):
            roots.append(Path(os.environ["MODELSCOPE_CACHE"]).expanduser())
        roots.extend(
            (
                Path.home() / ".cache" / "modelscope",
                Path.home() / ".cache" / "modelscope" / "hub",
            )
        )
        model_name = spec.model_id.split("/")[-1]
        owner = spec.model_id.split("/")[0]
        for root in roots:
            candidates.extend(
                (
                    root / "hub" / "models" / owner / model_name,
                    root / "hub" / owner / model_name,
                    root / "models" / owner / model_name,
                    root / owner / model_name,
                )
            )
    return candidates


def _legacy_component_paths(
    spec: ComponentSpec,
    layout: InstallLayout,
    gpt_root: Path | None,
) -> list[Path]:
    candidates = [layout.root / spec.relative_path]
    for name in COMPONENT_ENVIRONMENT_PATHS.get(spec.key, ()):
        configured = os.environ.get(name)
        if configured:
            candidates.append(
                _normalise_component_candidate(spec, Path(configured))
            )
    candidates.extend(_cache_component_paths(spec))
    for root in _candidate_roots():
        candidates.append(root / spec.relative_path)
        candidates.append(root / "提取" / spec.relative_path)
    if gpt_root:
        legacy = {
            "uvr_model": gpt_root / "tools" / "uvr5" / "uvr5_weights",
            "sv_model": gpt_root / "GPT_SoVITS" / "pretrained_models" / "sv",
            "paraformer_model": gpt_root
            / "tools"
            / "asr"
            / "models"
            / "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            "vad_model": gpt_root
            / "tools"
            / "asr"
            / "models"
            / "speech_fsmn_vad_zh-cn-16k-common-pytorch",
            "punc_model": gpt_root
            / "tools"
            / "asr"
            / "models"
            / "punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
            "whisper_cache": gpt_root.parent / "omnvoice" / "hf_cache",
        }
        if spec.key in legacy:
            candidates.insert(0, legacy[spec.key])
        sibling_tool = gpt_root.parent / "提取" / spec.relative_path
        candidates.insert(0, sibling_tool)
    unique: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved not in unique:
            unique.append(resolved)
    return unique


def scan_assets(
    layout: InstallLayout,
    gpt_root: Path | None,
) -> tuple[dict[str, Path], list[ComponentSpec]]:
    assets: dict[str, Path] = {}
    missing: list[ComponentSpec] = []
    stored = _read_json(layout.settings_path).get("assets", {})
    stored = stored if isinstance(stored, dict) else {}
    for spec in COMPONENTS:
        candidates: list[Path] = []
        if stored.get(spec.key):
            candidates.append(Path(str(stored[spec.key])))
        candidates.extend(_legacy_component_paths(spec, layout, gpt_root))
        found = next(
            (candidate for candidate in candidates if _component_ready(spec, candidate)),
            None,
        )
        if found is None:
            missing.append(spec)
        else:
            assets[spec.key] = found
    return assets, missing


def missing_download_size_mb(missing: Iterable[ComponentSpec]) -> int:
    return sum(spec.estimate_mb for spec in missing)


def _download_file(
    source: DirectFile,
    destination: Path,
    reporter: Reporter,
    base: float,
    span: float,
    label: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    request = urllib.request.Request(
        source.url,
        headers={"User-Agent": "VoiceExtractorDesktop/1.0"},
    )
    with urllib.request.urlopen(request, timeout=90) as response, partial.open("wb") as output:
        total = int(response.headers.get("Content-Length") or 0)
        received = 0
        last_percent = -1
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            output.write(block)
            received += len(block)
            if total:
                percent = int(received * 100 / total)
                if percent >= last_percent + 2 or percent == 100:
                    reporter(
                        base + span * percent / 100.0,
                        f"{label}：{percent}%（{received / 1024 / 1024:.1f} MB）",
                    )
                    last_percent = percent
    if source.expected_hash and _sha256(partial).lower() != source.expected_hash.lower():
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"{label} 校验失败")
    os.replace(partial, destination)


def _verify_python_installer_signature(installer: Path) -> None:
    if os.name != "nt":
        return
    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if not powershell:
        raise RuntimeError("无法验证 Python 官方安装包的数字签名")
    env = os.environ.copy()
    env["VOICE_EXTRACT_PYTHON_INSTALLER"] = str(installer)
    script = (
        "$signature = Get-AuthenticodeSignature -LiteralPath "
        "$env:VOICE_EXTRACT_PYTHON_INSTALLER; "
        "Write-Output ($signature.Status.ToString() + '|' + "
        "$signature.SignerCertificate.Subject)"
    )
    result = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        creationflags=CREATE_NO_WINDOW,
        timeout=60,
        check=False,
    )
    signature = result.stdout.strip()
    if (
        result.returncode
        or not signature.startswith("Valid|")
        or "Python Software Foundation" not in signature
    ):
        raise RuntimeError("Python 安装包数字签名无效，已停止安装")


def install_python_runtime(
    install_root: Path,
    reporter: Reporter,
) -> Path:
    """Install an official per-user Python without asking for python.exe."""

    layout = InstallLayout(install_root.expanduser().resolve())
    layout.prepare()
    destination = layout.root / "python"
    python = destination / "python.exe"
    if python.is_file() and probe_python_runtime(python).get("runnable"):
        reporter(1.0, f"Python 已存在：{python}")
        return python

    installer = layout.work / PYTHON_INSTALLER_FILENAME
    reporter(0.02, f"正在下载 Python {PYTHON_INSTALLER_VERSION} 官方安装包")
    _download_file(
        DirectFile(PYTHON_INSTALLER_FILENAME, PYTHON_INSTALLER_URL),
        installer,
        reporter,
        0.02,
        0.58,
        f"Python {PYTHON_INSTALLER_VERSION}",
    )
    reporter(0.62, "正在验证 Python 官方数字签名")
    _verify_python_installer_signature(installer)
    reporter(0.68, "正在安装本工具使用的 Python（无需填写路径）")
    command = [
        str(installer),
        "/quiet",
        "InstallAllUsers=0",
        "PrependPath=1",
        "Include_pip=1",
        "Include_launcher=1",
        "InstallLauncherAllUsers=0",
        "Include_doc=0",
        "Include_test=0",
        "Include_debug=0",
        "Include_symbols=0",
        "Shortcuts=0",
        "AssociateFiles=0",
        f"TargetDir={destination}",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=CREATE_NO_WINDOW,
            timeout=900,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Python 安装超时") from exc
    finally:
        installer.unlink(missing_ok=True)
    if completed.returncode or not python.is_file():
        detail = (completed.stderr or completed.stdout)[-1000:]
        raise RuntimeError(
            f"Python 安装失败，退出代码 {completed.returncode}\n{detail}"
        )
    state = probe_python_runtime(python)
    if not state.get("runnable"):
        raise RuntimeError("Python 安装完成，但运行验证失败")
    reporter(1.0, f"Python {PYTHON_INSTALLER_VERSION} 安装完成")
    return python


def _download_modelscope(
    python: Path,
    spec: ComponentSpec,
    destination: Path,
    layout: InstallLayout,
    source_root: Path,
    reporter: Reporter,
    progress: float,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    code = (
        "from modelscope import snapshot_download; "
        "import sys; snapshot_download(sys.argv[1], local_dir=sys.argv[2])"
    )
    env = _python_probe_environment([layout.packages, source_root / "vendor"])
    env["MODELSCOPE_CACHE"] = str(layout.models / "modelscope_cache")
    _stream_process(
        [str(python), "-c", code, spec.model_id, str(destination)],
        env,
        reporter,
        progress,
    )


def _download_whisper(
    python: Path,
    spec: ComponentSpec,
    destination: Path,
    layout: InstallLayout,
    source_root: Path,
    reporter: Reporter,
    progress: float,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    code = (
        "from huggingface_hub import snapshot_download; "
        "import sys; snapshot_download(sys.argv[1], cache_dir=sys.argv[2])"
    )
    env = _python_probe_environment([layout.packages, source_root / "vendor"])
    env.pop("HF_HUB_OFFLINE", None)
    _stream_process(
        [str(python), "-c", code, spec.model_id, str(destination)],
        env,
        reporter,
        progress,
    )


def ensure_assets(
    python: Path,
    layout: InstallLayout,
    source_root: Path,
    gpt_root: Path | None,
    reporter: Reporter,
) -> dict[str, Path]:
    assets, missing = scan_assets(layout, gpt_root)
    if not missing:
        reporter(0.92, "本地模型检查完成，全部可用")
        return assets
    total = len(missing)
    for index, spec in enumerate(missing):
        destination = layout.root / spec.relative_path
        base = 0.18 + 0.70 * index / total
        span = 0.70 / total
        reporter(base, f"正在安装 {spec.label}（{index + 1}/{total}）")
        if spec.kind == "direct":
            for file_index, item in enumerate(spec.files):
                _download_file(
                    item,
                    destination / item.filename,
                    reporter,
                    base + span * file_index / len(spec.files),
                    span / len(spec.files),
                    spec.label,
                )
        elif spec.kind == "modelscope":
            _download_modelscope(
                python,
                spec,
                destination,
                layout,
                source_root,
                reporter,
                base,
            )
        else:
            _download_whisper(
                python,
                spec,
                destination,
                layout,
                source_root,
                reporter,
                base,
            )
        if not _component_ready(spec, destination):
            raise RuntimeError(f"{spec.label} 安装后仍不完整")
    assets, missing = scan_assets(layout, gpt_root)
    if missing:
        raise RuntimeError("仍缺少模型：" + "、".join(spec.label for spec in missing))
    return assets


def _bundled_dependency(source_root: Path, name: str) -> Path | None:
    candidate = source_root / "dependencies" / name
    return candidate if candidate.is_dir() else None


def ensure_local_tools(
    layout: InstallLayout,
    source_root: Path,
    gpt_root: Path | None,
    reporter: Reporter,
) -> tuple[Path, Path, Path, Path]:
    bundled_tools = source_root / "tools"
    installed_ffmpeg = layout.tools / "ffmpeg.exe"
    installed_ffprobe = layout.tools / "ffprobe.exe"
    if installed_ffmpeg.is_file() and installed_ffprobe.is_file():
        ffmpeg, ffprobe = installed_ffmpeg, installed_ffprobe
    elif gpt_root and (
        (gpt_root / "runtime" / "ffmpeg.exe").is_file()
        and (gpt_root / "runtime" / "ffprobe.exe").is_file()
    ):
        # Reuse the existing binaries read-only. Nothing is moved or modified.
        ffmpeg = gpt_root / "runtime" / "ffmpeg.exe"
        ffprobe = gpt_root / "runtime" / "ffprobe.exe"
        reporter(0.04, "已复用本地 FFmpeg，不复制也不修改原文件")
    else:
        bundled_ffmpeg = bundled_tools / "ffmpeg.exe"
        bundled_ffprobe = bundled_tools / "ffprobe.exe"
        if not bundled_ffmpeg.is_file() or not bundled_ffprobe.is_file():
            raise RuntimeError("安装包和本地环境中均未找到 FFmpeg/FFprobe")
        reporter(0.03, "正在安装本工具专用 FFmpeg")
        shutil.copy2(bundled_ffmpeg, installed_ffmpeg)
        reporter(0.04, "正在安装本工具专用 FFprobe")
        shutil.copy2(bundled_ffprobe, installed_ffprobe)
        ffmpeg, ffprobe = installed_ffmpeg, installed_ffprobe

    uvr_code = _bundled_dependency(source_root, "uvr5")
    sv_code = _bundled_dependency(source_root, "eres2net")
    if uvr_code is None and gpt_root:
        candidate = gpt_root / "tools" / "uvr5"
        if candidate.is_dir():
            uvr_code = candidate
    if sv_code is None and gpt_root:
        candidate = gpt_root / "GPT_SoVITS" / "eres2net"
        if candidate.is_dir():
            sv_code = candidate
    if uvr_code is None or sv_code is None:
        raise RuntimeError("安装包缺少 UVR/ERes2Net 后端代码")
    return ffmpeg, ffprobe, uvr_code, sv_code


def install_executable(layout: InstallLayout) -> Path:
    executable = current_executable()
    destination = layout.root / "VoiceExtractor.exe"
    if getattr(sys, "frozen", False) and executable != destination:
        try:
            if not destination.is_file() or _sha256(destination) != _sha256(executable):
                shutil.copy2(executable, destination)
        except OSError:
            pass
    return destination if destination.is_file() else executable


def inspect_local_tools(
    layout: InstallLayout,
    source_root: Path,
    gpt_root: Path | None,
) -> dict[str, object]:
    bundled_tools = source_root / "tools"
    ffmpeg_candidates = [
        layout.tools / "ffmpeg.exe",
        bundled_tools / "ffmpeg.exe",
    ]
    ffprobe_candidates = [
        layout.tools / "ffprobe.exe",
        bundled_tools / "ffprobe.exe",
    ]
    if gpt_root:
        ffmpeg_candidates.insert(0, gpt_root / "runtime" / "ffmpeg.exe")
        ffprobe_candidates.insert(0, gpt_root / "runtime" / "ffprobe.exe")
    ffmpeg = next((path for path in ffmpeg_candidates if path.is_file()), None)
    ffprobe = next((path for path in ffprobe_candidates if path.is_file()), None)

    uvr_candidates = [source_root / "dependencies" / "uvr5"]
    sv_candidates = [source_root / "dependencies" / "eres2net"]
    if gpt_root:
        uvr_candidates.append(gpt_root / "tools" / "uvr5")
        sv_candidates.append(gpt_root / "GPT_SoVITS" / "eres2net")
    uvr = next((path for path in uvr_candidates if path.is_dir()), None)
    sv = next((path for path in sv_candidates if path.is_dir()), None)
    rows = {
        "FFmpeg": ffmpeg,
        "FFprobe": ffprobe,
        "UVR 后端代码": uvr,
        "ERes2Net 后端代码": sv,
    }
    return {
        "rows": rows,
        "missing": [label for label, path in rows.items() if path is None],
    }


def inspect_nvidia_environment() -> dict[str, object]:
    result: dict[str, object] = {
        "driver_available": False,
        "gpus": [],
        "driver_version": "",
        "cuda_toolkit": "",
    }
    nvidia_smi = shutil.which("nvidia-smi.exe") or shutil.which("nvidia-smi")
    if not nvidia_smi:
        conventional = (
            Path(os.environ.get("ProgramFiles", "C:/Program Files"))
            / "NVIDIA Corporation"
            / "NVSMI"
            / "nvidia-smi.exe"
        )
        if conventional.is_file():
            nvidia_smi = str(conventional)
    if nvidia_smi:
        try:
            completed = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=name,driver_version",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=CREATE_NO_WINDOW,
                timeout=20,
                check=False,
            )
            rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
            if completed.returncode == 0 and rows:
                result["driver_available"] = True
                gpus: list[str] = []
                drivers: list[str] = []
                for row in rows:
                    name, _, driver = row.rpartition(",")
                    gpus.append(name.strip() or row)
                    if driver.strip():
                        drivers.append(driver.strip())
                result["gpus"] = gpus
                result["driver_version"] = ", ".join(dict.fromkeys(drivers))
        except (OSError, subprocess.TimeoutExpired):
            pass

    nvcc = shutil.which("nvcc.exe") or shutil.which("nvcc")
    if nvcc:
        try:
            completed = subprocess.run(
                [nvcc, "--version"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=CREATE_NO_WINDOW,
                timeout=20,
                check=False,
            )
            match = re.search(r"release\s+([0-9.]+)", completed.stdout)
            if match:
                result["cuda_toolkit"] = match.group(1)
        except (OSError, subprocess.TimeoutExpired):
            pass
    if not result["cuda_toolkit"] and os.environ.get("CUDA_PATH"):
        match = re.search(r"v([0-9.]+)", os.environ["CUDA_PATH"], re.IGNORECASE)
        result["cuda_toolkit"] = match.group(1) if match else "已设置 CUDA_PATH"
    return result


def prepare_context(
    install_root: Path,
    preferred_python: Path | None,
    reporter: Reporter,
) -> BootstrapContext:
    source_root = bundle_source_root()
    if not (source_root / "desktop_worker.py").is_file():
        raise RuntimeError("安装包内缺少 desktop_worker.py")
    layout = InstallLayout(install_root.expanduser().resolve())
    layout.prepare()
    install_executable(layout)

    reporter(0.01, "正在复核本地 CUDA/PyTorch 运行环境")
    installed_python = layout.root / "python" / "python.exe"
    python = discover_python(preferred_python or installed_python)
    if python is None:
        raise RuntimeError(
            "未找到同时包含 PyTorch 与 torchaudio 的 Python；"
            "请使用安装页提供的 PyTorch/CUDA 官方链接。"
        )
    runtime = probe_python_runtime(python)
    if not runtime.get("cuda_available"):
        raise RuntimeError(
            "已找到 PyTorch，但 CUDA 当前不可用；"
            "请使用安装页提供的 PyTorch/CUDA 官方链接。"
        )
    gpt_root = infer_gpt_root(python)
    ffmpeg, ffprobe, uvr_code, sv_code = ensure_local_tools(
        layout,
        source_root,
        gpt_root,
        reporter,
    )
    ensure_python_modules(python, layout, source_root, reporter)
    assets = ensure_assets(
        python,
        layout,
        source_root,
        gpt_root,
        reporter,
    )
    save_settings(layout, python, assets)
    reporter(1.0, "安装与资源检查完成")
    return BootstrapContext(
        source_root=source_root,
        layout=layout,
        python=python,
        assets=assets,
        uvr_code=uvr_code,
        sv_code=sv_code,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        extra_python_paths=[layout.packages],
    )


def inspect_installation(
    install_root: Path,
    preferred_python: Path | None = None,
    reporter: Reporter | None = None,
) -> dict[str, object]:
    layout = InstallLayout(install_root.expanduser().resolve())
    source_root = bundle_source_root()
    preferred = preferred_python or (layout.root / "python" / "python.exe")
    if reporter:
        reporter(0.02, "正在枚举环境变量、PATH、py.exe、注册表和常见 Python 目录")
    python_states = inspect_python_runtimes(preferred, reporter)
    runnable_states = [item for item in python_states if item.get("runnable")]
    ml_states = [
        item
        for item in runnable_states
        if item.get("torch") and item.get("torchaudio")
    ]
    gpu_states = [item for item in ml_states if item.get("cuda_available")]
    selected_state = next(iter(gpu_states or ml_states), None)
    any_state = next(iter(runnable_states), None)
    python = (
        Path(str(selected_state.get("candidate")))
        if selected_state is not None
        else None
    )
    any_python = (
        Path(str(any_state.get("candidate"))) if any_state is not None else None
    )
    gpt_root = infer_gpt_root(python or any_python)

    if reporter:
        reporter(0.32, "正在验证 Python 依赖模块")
    module_state = (
        probe_modules(python, [layout.packages, source_root / "vendor"])
        if python is not None
        else {name: False for name in MODULE_PACKAGES}
    )
    missing_modules = [name for name, available in module_state.items() if not available]

    if reporter:
        reporter(0.58, "正在检查全部模型权重和本地模型缓存")
    assets, missing = scan_assets(layout, gpt_root)
    component_rows = [
        {
            "key": spec.key,
            "label": spec.label,
            "ready": spec.key in assets,
            "path": assets.get(spec.key),
            "download_mb": 0 if spec.key in assets else spec.estimate_mb,
        }
        for spec in COMPONENTS
    ]

    if reporter:
        reporter(0.82, "正在检查 FFmpeg、UVR/ERes2Net 后端和 NVIDIA 驱动")
    tools = inspect_local_tools(layout, source_root, gpt_root)
    nvidia = inspect_nvidia_environment()
    model_mb = missing_download_size_mb(missing)
    python_install_required = any_python is None
    runtime_ready = selected_state is not None and bool(selected_state.get("cuda_available"))
    supplementary_missing = [
        name for name in missing_modules if name not in {"torch", "torchaudio"}
    ]
    total_download_mb = model_mb + (PYTHON_DOWNLOAD_MB if python_install_required else 0)
    ready = (
        runtime_ready
        and not missing
        and not supplementary_missing
        and not tools["missing"]
    )
    if reporter:
        reporter(1.0, "本机环境、全部依赖与模型扫描完成")
    return {
        "python": python,
        "any_python": any_python,
        "python_states": python_states,
        "runtime": selected_state,
        "module_state": module_state,
        "missing_modules": missing_modules,
        "supplementary_missing": supplementary_missing,
        "python_install_required": python_install_required,
        "runtime_ready": runtime_ready,
        "nvidia": nvidia,
        "gpt_root": gpt_root,
        "assets": assets,
        "missing": missing,
        "component_rows": component_rows,
        "tools": tools,
        "missing_mb": total_download_mb,
        "model_missing_mb": model_mb,
        "python_download_mb": PYTHON_DOWNLOAD_MB if python_install_required else 0,
        "python_installed_mb": PYTHON_INSTALLED_MB if python_install_required else 0,
        "ready": ready,
        "scan_complete": True,
    }
