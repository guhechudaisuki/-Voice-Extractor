import json
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "launcher"))
from launcher import app_entry, webview_shell
import voice_extractor_desktop as desktop


class LauncherTests(unittest.TestCase):
    def test_web_entry_waits_for_ready_and_uses_configured_port(self):
        calls = []
        with patch.object(app_entry, "get_ui_mode", return_value="desktop"), \
                patch.object(webview_shell, "load_port", return_value=18785), \
                patch.object(webview_shell, "start_server", side_effect=lambda port: calls.append(("ready", port))), \
                patch.object(app_entry.webbrowser, "open", side_effect=lambda url: calls.append(("open", url)) or True):
            self.assertEqual(app_entry.main(["--ui", "webui"]), 0)
        self.assertEqual(calls, [("ready", 18785), ("open", "http://127.0.0.1:18785/")])

    def test_web_entry_does_not_open_browser_on_startup_failure(self):
        with patch.object(webview_shell, "start_server", side_effect=RuntimeError("server failed")), \
                patch.object(app_entry.webbrowser, "open") as browser:
            self.assertEqual(app_entry.main(["--ui", "webui"]), 1)
            browser.assert_not_called()

    def test_healthy_server_is_reused_and_foreign_port_is_rejected(self):
        with patch.object(webview_shell, "port_is_open", return_value=True), \
                patch.object(webview_shell, "server_is_healthy", return_value=True) as health, \
                patch.object(webview_shell.subprocess, "Popen") as spawn:
            self.assertIsNone(webview_shell.start_server(18785))
            health.return_value = False
            with self.assertRaises(RuntimeError):
                webview_shell.start_server(18785)
            spawn.assert_not_called()

    def test_dead_server_process_ends_readiness_wait_immediately(self):
        process = MagicMock()
        process.poll.return_value = 1
        with patch.object(webview_shell, "server_is_healthy") as health:
            self.assertFalse(webview_shell.wait_for_server(18785, process=process))
            health.assert_not_called()

    def test_desktop_entry_opens_native_window_without_web_shell(self):
        with patch.dict(sys.modules, {"webview_shell": webview_shell}), \
                patch.object(webview_shell, "main", return_value=0) as shell, \
                patch.object(desktop.tk, "Tk") as root, \
                patch.object(desktop, "VoiceExtractorDesktop") as window:
            self.assertEqual(desktop.main(), 0)
            shell.assert_not_called()
            window.assert_called_once_with(root.return_value)
            root.return_value.mainloop.assert_called_once()

    def test_explicit_shell_port_is_passed_to_server_and_window(self):
        with patch.object(webview_shell, "start_server", return_value=None) as start, \
                patch.object(webview_shell, "run_shell", return_value=0) as run:
            self.assertEqual(webview_shell.main(["--port", "18785"]), 0)
            start.assert_called_once_with(18785)
            run.assert_called_once_with(18785)


class DirectServerLaunchTests(unittest.TestCase):
    def test_script_launch_on_requested_port_can_validate_subtitles(self):
        # Same interpreter and standalone script as the BAT/WebView launchers,
        # with a different cwd so a parent process cannot mask missing imports.
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "server.log"
            with log_path.open("wb") as stream:
                process = subprocess.Popen([sys.executable, "-B", str(ROOT / "launcher/webui_server.py"),
                    "--port", str(port)], cwd=directory, stdout=stream, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                deadline = time.monotonic() + 15
                ready = False
                while process.poll() is None and time.monotonic() < deadline:
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/settings", timeout=.3) as response:
                            ready = response.status == 200
                            break
                    except OSError:
                        time.sleep(.1)
                self.assertTrue(ready, log_path.read_text(encoding="utf-8", errors="replace"))
                source = str(ROOT / "desktop_worker.py")
                payload = {"targets": [source], "references": [source],
                           "subtitles": {source: str(Path(directory) / "missing.srt")}}
                request = urllib.request.Request(f"http://127.0.0.1:{port}/api/extract",
                    data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(request, timeout=10)
                self.assertEqual(caught.exception.code, 400,
                    log_path.read_text(encoding="utf-8", errors="replace"))
                self.assertIn("字幕绑定无效", caught.exception.read().decode())
            finally:
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
