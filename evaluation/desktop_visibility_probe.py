"""Disposable real desktop startup probe; never starts extraction or installation."""
import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def visible_windows(pid):
    user = ctypes.windll.user32
    user.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user.IsWindowVisible.argtypes = [wintypes.HWND]
    user.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    rows = []
    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def callback(handle, _):
        owner = wintypes.DWORD()
        user.GetWindowThreadProcessId(handle, ctypes.byref(owner))
        if owner.value == pid and user.IsWindowVisible(handle):
            title = ctypes.create_unicode_buffer(512)
            rect = wintypes.RECT()
            user.GetWindowTextW(handle, title, 512)
            user.GetWindowRect(handle, ctypes.byref(rect))
            rows.append({'handle': handle, 'title': title.value,
                         'rect': [rect.left, rect.top, rect.right, rect.bottom]})
        return True
    user.EnumWindows(callback, 0)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--disable-sash', action='store_true')
    parser.add_argument('--disable-selection', action='store_true')
    args = parser.parse_args()
    child = """
import faulthandler
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'launcher'))
import voice_extractor_desktop as desktop
original = desktop.VoiceExtractorDesktop.__init__
def init(self, root):
    original(self, root)
    if sys.argv[1] == 'disable':
        body = self.main_notebook.winfo_children()[0].winfo_children()[0]
        body.unbind('<Configure>')
    if sys.argv[2] == 'disable':
        self.review_tab.tree.unbind('<<TreeviewSelect>>')
    print('constructed', flush=True)
    root.after(800, lambda: print('event-loop-responsive', root.winfo_viewable(), flush=True))
desktop.VoiceExtractorDesktop.__init__ = init
faulthandler.dump_traceback_later(12, repeat=False)
desktop.main()
"""
    directory = ROOT / 'work/cluster_p0p2_logs'
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / ('desktop_probe_no_sash.log' if args.disable_sash else 'desktop_probe.log')
    with log_path.open('wb') as log:
        process = subprocess.Popen([sys.executable, '-B', '-c', child,
            'disable' if args.disable_sash else 'normal', 'disable' if args.disable_selection else 'normal'],
            cwd=ROOT, env={**os.environ, 'VOICE_EXTRACT_DESKTOP_SHELL': 'tk'},
            stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    windows = []
    try:
        deadline = time.monotonic() + 15
        while process.poll() is None and time.monotonic() < deadline:
            windows = [row for row in visible_windows(process.pid) if row['title'] == 'Voice Extractor Desktop']
            if windows and 'event-loop-responsive' in log_path.read_text(encoding='utf-8', errors='replace'):
                break
            time.sleep(.2)
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=5)
    log = log_path.read_text(encoding='utf-8', errors='replace')
    passed = bool(windows) and 'event-loop-responsive 1' in log
    print(json.dumps({'passed':passed, 'windows':windows, 'log':log}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
