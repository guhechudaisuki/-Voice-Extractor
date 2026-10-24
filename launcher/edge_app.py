"""Run the existing WebUI in an Edge application window on Windows."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import subprocess
import time

import psutil


def edge_executable() -> Path | None:
    candidates = [
        Path(os.environ.get("ProgramFiles(x86)", "")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("ProgramFiles", "")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/Edge/Application/msedge.exe",
    ]
    return next((path for path in candidates if path.is_file()), None)


def _edge_processes(profile: Path) -> set[int]:
    marker = f"--user-data-dir={profile}"
    matches = set()
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if process.info["name"] and process.info["name"].lower() == "msedge.exe" and marker in (process.info["cmdline"] or []):
                matches.add(process.info["pid"])
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return matches


def profile_path(base_dir: Path, port: int) -> Path:
    return (base_dir / "work" / f"desktop_edge_profile_{port}").resolve()


def visible_window(pids: set[int]) -> tuple[int, int] | None:
    if os.name != "nt" or not pids:
        return None
    user = ctypes.windll.user32
    user.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user.IsWindowVisible.argtypes = [wintypes.HWND]
    windows = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def collect(handle, _):
        owner = wintypes.DWORD()
        user.GetWindowThreadProcessId(handle, ctypes.byref(owner))
        if owner.value in pids and user.IsWindowVisible(handle):
            title = ctypes.create_unicode_buffer(512)
            user.GetWindowTextW(handle, title, 512)
            if "Voice Extractor" in title.value:
                windows.append((int(handle), owner.value))
        return True

    user.EnumWindows(collect, 0)
    return windows[0] if windows else None


def _focus(handle: int) -> None:
    user = ctypes.windll.user32
    user.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user.SetForegroundWindow.argtypes = [wintypes.HWND]
    user.ShowWindow(handle, 9)
    user.SetForegroundWindow(handle)


def launch_edge(base_dir: Path, port: int, *, timeout: float = 15.0) -> dict:
    edge = edge_executable()
    if edge is None:
        raise RuntimeError("未找到 Microsoft Edge")
    profile = profile_path(base_dir, port)
    profile.mkdir(parents=True, exist_ok=True)
    existing = _edge_processes(profile)
    window = visible_window(existing)
    if window is not None:
        _focus(window[0])
        return {"pid": window[1], "visible": True, "mode": "edge_app", "retain_server": True}

    url = f"http://127.0.0.1:{port}/?desktop=1"
    process = subprocess.Popen(
        [str(edge), f"--app={url}", f"--user-data-dir={profile}",
         "--no-first-run", "--disable-background-mode"],
        cwd=str(base_dir), close_fds=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
                      | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pids = _edge_processes(profile)
        window = visible_window(pids)
        if window is not None:
            _focus(window[0])
            return {"pid": window[1],
                    "visible": True, "mode": "edge_app", "retain_server": True}
        if process.poll() is not None and not pids:
            break
        time.sleep(.1)
    if process.poll() is None:
        process.terminate()
    raise RuntimeError("Edge 桌面窗口未能显示")
