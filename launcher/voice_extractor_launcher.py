from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def find_python(root: Path) -> Path:
    candidates = []
    if os.environ.get("VOICE_EXTRACT_PYTHON"):
        candidates.append(Path(os.environ["VOICE_EXTRACT_PYTHON"]))
    candidates.extend(
        [
            root.parent / "GPT-SoVITS-v2pro-20250604" / "runtime" / "python.exe",
            root / "runtime" / "python.exe",
            Path(sys.executable).with_name("python.exe"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "找不到 Python runtime。请设置 VOICE_EXTRACT_PYTHON，或将 GPT-SoVITS-v2pro-20250604 放在仓库同级目录。"
    )


def main() -> int:
    executable_root = (
        Path(sys.executable).resolve().parent
        if getattr(sys, "frozen", False)
        else Path(__file__).resolve().parent
    )
    # The EXE is intentionally separate from the source checkout. Users can
    # keep it in ``dist`` and point to the checkout explicitly, or place it in
    # the repository directory next to app.py.
    roots = []
    if os.environ.get("VOICE_EXTRACT_ROOT"):
        roots.append(Path(os.environ["VOICE_EXTRACT_ROOT"]))
    roots.extend([executable_root, executable_root.parent])
    source_root = next((candidate for candidate in roots if (candidate / "app.py").exists()), None)
    if source_root is None:
        raise FileNotFoundError(
            "找不到 app.py。请设置 VOICE_EXTRACT_ROOT 指向提取程序目录。"
        )
    app = source_root / "app.py"
    python = find_python(source_root)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    return subprocess.call([str(python), "-u", str(app)], cwd=str(source_root), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
