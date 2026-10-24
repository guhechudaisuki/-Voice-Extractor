"""Check a real isolated Edge app window and cleanup after its native close."""
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--port', type=int, default=7862)
    parser.add_argument('--settings')
    args = parser.parse_args()
    if args.serve:
        from launcher import webui_server as server
        server.SETTINGS_PATH = Path(args.settings)
        directory = server.TRAINING_WORKSPACE.job('closing_probe')
        (directory / 'temporary_probe.bin').write_bytes(b'temporary training data')
        worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'],
                                  creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        server.TRAINING_WORKSPACE.add_worker(worker.pid)
        server.TRAINING_PROCESSES['closing_probe'] = worker
        server.TRAINING_JOBS['closing_probe'] = {'state': 'running', 'temporary_path': str(directory),
                                                'message': 'cleanup probe', 'worker_pid': worker.pid}
        server.run_webui(port=args.port)
        return

    from launcher.webview_shell import port_is_open, wait_for_server
    from launcher.edge_app import visible_window
    import psutil
    if port_is_open(args.port):
        raise RuntimeError('Isolated test port is already in use')
    output = ROOT / 'work/cluster_p0p2_logs'
    output.mkdir(parents=True, exist_ok=True)
    base = f'http://127.0.0.1:{args.port}'

    def request(path, method='GET'):
        call = urllib.request.Request(base + path, method=method, data=b'' if method == 'POST' else None)
        with urllib.request.urlopen(call, timeout=25) as response:
            return json.load(response)

    with tempfile.TemporaryDirectory(prefix='voice_desktop_smoke_') as temporary, \
            (output / 'training_desktop_smoke.log').open('wb') as log:
        process = subprocess.Popen([sys.executable, '-B', __file__, '--serve', '--port', str(args.port),
                                    '--settings', str(Path(temporary) / 'settings.json')], cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            assert wait_for_server(args.port, process=process), 'Service did not start'
            state = request('/api/training/status/closing_probe')
            directory = Path(state['temporary_path'])
            ready = request('/api/switch_to_desktop', 'POST')
            assert ready['mode'] == 'edge_app' and ready['visible']
            window = visible_window({ready['pid']})
            assert window is not None
            # Only close the newly opened test window identified by its returned PID.
            from ctypes import wintypes
            ctypes.windll.user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
            assert ctypes.windll.user32.PostMessageW(window[0], 0x0010, 0, 0)
            process.wait(timeout=25)
            assert process.returncode == 0
            assert not directory.parent.exists(), 'Training workspace remained after native close'
            assert not psutil.pid_exists(state['worker_pid']), 'Training worker remained after native close'
            result = {'passed': True, 'port': args.port, 'checks': ['edge_app_visible', 'native_window_close',
                      'training_worker_stopped', 'temporary_workspace_removed'], 'real_training': False}
            (output / 'training_desktop_smoke.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            print(json.dumps(result))
        finally:
            if process.poll() is None:
                try:
                    request('/api/close', 'POST')
                    process.wait(timeout=25)
                except (OSError, subprocess.TimeoutExpired):
                    process.terminate()
                    process.wait(timeout=5)


if __name__ == '__main__':
    main()
