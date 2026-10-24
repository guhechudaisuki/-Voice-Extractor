# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

ROOT = Path(SPECPATH).resolve()

a = Analysis(
    [str(ROOT / "launcher" / "voice_extractor_desktop.py")],
    pathex=[str(ROOT / "launcher"), str(ROOT)],
    binaries=[],
    datas=[],
    hiddenimports=["tkinter.filedialog", "tkinter.messagebox", "tkinter.ttk"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["extractor.audio", "torch", "torchaudio", "transformers", "funasr", "onnxruntime",
              "librosa", "numpy", "scipy", "matplotlib", "tensorflow", "pandas"],
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
