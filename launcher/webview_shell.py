"""桌面端 WebView2 套壳：无边框窗口里加载本机的 WebUI。

为什么套壳：tkinter 能做的视觉上限很低（没有 CSS、没有阴影、渐变要逐像素画），
而两个界面各维护一套代码会导致功能不同步。套壳之后桌面端与网页端是同一份
界面代码，功能和外观天然一致。

依赖 Microsoft Edge WebView2 运行时（Windows 11 与打过补丁的 Windows 10 自带）。
没有它时本模块返回失败，由调用方回退到原来的 tkinter 界面——不能让桌面端
因为缺少一个可选依赖就打不开。

用法：

    python -m launcher.webview_shell --port 7860
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

LAUNCHER_DIR = Path(__file__).resolve().parent
BASE_DIR = LAUNCHER_DIR.parent
SETTINGS_PATH = BASE_DIR / "voice_extractor_settings.json"
SERVER_PY = LAUNCHER_DIR / "webui_server.py"

# 与网页版 CSS 的 --bg 保持一致，窗口底色相同才不会在加载时闪白。
WINDOW_BG = "#0c0d10"


def load_port() -> int:
    if SETTINGS_PATH.is_file():
        try:
            return int(json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
                       .get("webui_port", 7860))
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return 7860


def port_is_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.4)
        return probe.connect_ex((host, port)) == 0


def wait_for_server(port: int, timeout: float = 40.0, process=None) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            return False
        if server_is_healthy(port, timeout=.5):
            return True
        time.sleep(0.35)
    return False


def server_is_healthy(port: int, timeout: float = 2.0) -> bool:
    """确认端口后面真的是本项目的服务，而不是恰好占用了同一端口的别人。"""
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/settings", timeout=timeout
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return isinstance(payload, dict) and "webui_port" in payload
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return False


def start_server(port: int) -> subprocess.Popen | None:
    """在后台拉起 WebUI 服务。已经在跑就返回 None。"""
    if not 1 <= port <= 65535:
        raise ValueError("WebUI 端口必须在 1–65535 之间")
    if port_is_open(port):
        if server_is_healthy(port):
            return None
        raise RuntimeError(
            f"端口 {port} 已被别的程序占用，且它不是本项目的服务。\n"
            f"请在设置里换一个端口，或关掉占用该端口的程序。"
        )
    if not SERVER_PY.is_file():
        raise RuntimeError(f"找不到 {SERVER_PY.name}")

    creationflags = 0
    if os.name == "nt":
        # 不弹控制台窗口，并让服务独立于套壳进程存活
        creationflags = 0x08000000 | 0x00000200
    log_path = BASE_DIR / "logs" / "webui.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            [sys.executable, "-B", str(SERVER_PY), "--port", str(port)],
            cwd=str(BASE_DIR), stdout=log, stderr=subprocess.STDOUT,
            creationflags=creationflags, close_fds=True,
        )
    if not wait_for_server(port, process=process):
        if process.poll() is None:
            process.terminate()
        raise RuntimeError(f"WebUI 服务启动失败或超时，请查看日志：{log_path}")
    return process


def run_shell(port: int) -> int:
    """打开 WebView2 窗口。返回非零表示调用方应该回退到 tkinter。"""
    try:
        import webview
    except ImportError:
        return run_edge_shell(port)

    try:
        import clr  # noqa: F401  (pythonnet 是 pywebview 在 Windows 上的依赖)
    except ImportError:
        return run_edge_shell(port)

    url = f"http://127.0.0.1:{port}/?desktop=1"
    if not server_is_healthy(port):
        print("本机服务未就绪，回退到 tkinter 界面。", file=sys.stderr)
        return 2

    window = webview.create_window(
        "Voice Extractor",
        url,
        width=1440,
        height=900,
        min_size=(1120, 720),
        background_color=WINDOW_BG,
        text_select=False,
    )

    # 无边框 + 自绘标题栏只有在 pywebview 的 edgechromium 后端才稳，
    # 这里保持系统边框，避免不同 WebView2 版本上出现拖不动或关不掉的情况。
    try:
        webview.start(
            gui="edgechromium",
            debug=False,
            private_mode=False,   # 保留音频/设置的本地存储
        )
    except Exception as exc:  # noqa: BLE001 - 任何后端异常都回退
        print(f"WebView2 启动失败（{exc}），回退到 tkinter 界面。", file=sys.stderr)
        return 2

    del window
    return 0


def run_edge_shell(port: int) -> int:
    """Use installed Edge when optional pywebview bindings are absent."""
    from launcher.edge_app import launch_edge, visible_window, _edge_processes, profile_path
    try:
        ready = launch_edge(BASE_DIR, port)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    profile = profile_path(BASE_DIR, port)
    while visible_window(_edge_processes(profile)) is not None:
        time.sleep(.3)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--no-start-server", action="store_true",
        help="不自己拉起服务，只开窗口（服务已在别处运行时用）",
    )
    args = parser.parse_args(argv)

    port = args.port if args.port is not None else load_port()
    owned: subprocess.Popen | None = None
    try:
        if not args.no_start_server:
            owned = start_server(port)
        code = run_shell(port)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    finally:
        # 服务由外部启动时不要关它；自己起的在窗口关闭后收掉。
        if owned is not None and owned.poll() is None:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/lifecycle", timeout=2) as response:
                    retain = json.loads(response.read()).get("retain_server")
                if not retain:
                    request = urllib.request.Request(f"http://127.0.0.1:{port}/api/close", data=b"", method="POST")
                    urllib.request.urlopen(request, timeout=15).close()
                    owned.wait(timeout=20)
            except (OSError, subprocess.TimeoutExpired):
                if owned.poll() is None:
                    owned.terminate()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
