# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path


ROOT = Path(SPECPATH).resolve()
GPT_ROOT = Path(
    os.environ.get(
        "VOICE_EXTRACT_BUILD_GPT_ROOT",
        ROOT.parent / "GPT-SoVITS-v2pro-20250604",
    )
).resolve()


def add_tree(datas, source, destination, excluded_parts=(), excluded_suffixes=()):
    source = Path(source)
    if not source.is_dir():
        raise SystemExit(f"Missing build dependency: {source}")
    for file_path in source.rglob("*"):
        if not file_path.is_file():
            continue
        relative = file_path.relative_to(source)
        if any(part in excluded_parts for part in relative.parts):
            continue
        if file_path.suffix.lower() in excluded_suffixes:
            continue
        target = Path("app_bundle") / destination / relative.parent
        datas.append((str(file_path), str(target)))


datas = [(str(ROOT / "desktop_worker.py"), "app_bundle")]
add_tree(datas, ROOT / "extractor", "extractor", {"__pycache__"}, {".pyc"})
add_tree(datas, ROOT / "vendor", "vendor", {"__pycache__"}, {".pyc"})
add_tree(
    datas,
    GPT_ROOT / "tools" / "uvr5",
    Path("dependencies") / "uvr5",
    {"__pycache__", "uvr5_weights"},
    {".pyc", ".pth", ".onnx"},
)
add_tree(
    datas,
    GPT_ROOT / "GPT_SoVITS" / "eres2net",
    Path("dependencies") / "eres2net",
    {"__pycache__"},
    {".pyc", ".pth", ".ckpt", ".onnx"},
)

for tool_name in ("ffmpeg.exe", "ffprobe.exe"):
    tool_path = GPT_ROOT / "runtime" / tool_name
    if not tool_path.is_file():
        raise SystemExit(f"Missing build dependency: {tool_path}")
    datas.append((str(tool_path), str(Path("app_bundle") / "tools")))


a = Analysis(
    [str(ROOT / "launcher" / "voice_extractor_desktop.py")],
    pathex=[str(ROOT / "launcher"), str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=["tkinter.filedialog", "tkinter.messagebox", "tkinter.ttk"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="VoiceExtractor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
