from __future__ import annotations

import argparse
import json
import subprocess
import sys
import webbrowser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LAUNCHER_DIR = Path(__file__).resolve().parent
SETTINGS_PATH = BASE_DIR / "voice_extractor_settings.json"

for d in (str(LAUNCHER_DIR), str(BASE_DIR)):
    if d not in sys.path:
        sys.path.insert(0, d)


def get_ui_mode() -> str:
    if SETTINGS_PATH.exists():
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("ui_mode", "desktop")
        except Exception:
            pass
    return "desktop"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Voice Extractor 界面启动入口")
    parser.add_argument("--ui", choices=("desktop", "webui"))
    args = parser.parse_args(argv)
    try:
        if (args.ui or get_ui_mode()) == "webui":
            from launcher.webview_shell import load_port, start_server
            port = load_port()
            # Reuse a healthy existing server; only open once HTTP is ready.
            start_server(port)
            url = f"http://127.0.0.1:{port}/"
            print(f"WebUI 已就绪：{url}", flush=True)
            if not webbrowser.open(url):
                print(f"请在浏览器中打开：{url}", flush=True)
            return 0
        desktop_py = LAUNCHER_DIR / "voice_extractor_desktop.py"
        return subprocess.call([sys.executable, "-B", str(desktop_py)], cwd=str(BASE_DIR))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"界面启动失败：{exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
