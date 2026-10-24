"""Wait for a mapped, responsive native desktop before leaving the WebUI."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

READY_ENV = "VOICE_EXTRACT_DESKTOP_READY_FILE"


def announce_ready(root):
    """Called by a Tk timer: acknowledges both window mapping and event progress."""
    path = os.environ.get(READY_ENV)
    if not path:
        return
    if not root.winfo_viewable():
        root.after(100, lambda: announce_ready(root))
        return
    root.deiconify()
    root.lift()
    root.attributes("-topmost", True)
    root.focus_force()
    root.after(250, lambda: root.attributes("-topmost", False))
    destination = Path(path)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps({"pid": os.getpid(), "visible": True, "mode": "tkinter"}), encoding="utf-8")
    temporary.replace(destination)


def launch_desktop(script: Path, base_dir: Path, *, timeout=15.0, port: int | None = None):
    directory = base_dir / "work" / "desktop_handoff"
    directory.mkdir(parents=True, exist_ok=True)
    log_path = base_dir / "logs" / "desktop_startup.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="launch_", dir=directory) as temporary:
        ready_path = Path(temporary) / "ready.json"
        # Native desktop does not depend on the web service being handed off.
        environment = {**os.environ, READY_ENV: str(ready_path), "VOICE_EXTRACT_DESKTOP_SHELL": "tk"}
        with log_path.open("ab") as log:
            process = subprocess.Popen([sys.executable, "-B", str(script)], cwd=base_dir,
                env=environment, stdout=log, stderr=subprocess.STDOUT, close_fds=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
                              | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        acknowledged = False
        try:
            deadline = time.monotonic() + timeout
            while process.poll() is None and time.monotonic() < deadline:
                if ready_path.is_file():
                    ready = json.loads(ready_path.read_text(encoding="utf-8"))
                    if isinstance(ready, dict) and ready.get("pid") == process.pid and ready.get("visible") is True:
                        acknowledged = True
                        return ready
                time.sleep(.1)
            raise RuntimeError(f"桌面窗口未能显示，网页仍可使用。启动日志：{log_path}")
        finally:
            if not acknowledged:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
