"""Native desktop training lifecycle, independent of HTTP and browsers."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import uuid
from launcher.training_workspace import Workspace
from launcher.reinforcement_history import completion_message

ROOT = Path(__file__).resolve().parents[1]


class TrainingSession:
    def __init__(self, base: Path | None = None):
        self.workspace = Workspace(base)
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.thread = None
        self.process = None
        self.directory = None
        self.state = 'idle'
        self.message = '选择标注与音频 ZIP 开始强化，无需解压'
        self.report = None

    def start(self, paths: list[Path], python: str = sys.executable, environment: dict | None = None):
        if not 1 <= len(paths) <= 30 or any(path.suffix.lower() != '.zip' for path in paths):
            raise ValueError('请选择 1 至 30 个含标注和音频的 ZIP 文件')
        sizes = [path.stat().st_size for path in paths]
        if any(size > 4 * 1024**3 for size in sizes) or sum(sizes) > 8 * 1024**3:
            raise ValueError('单个原料最多 4 GB，本次合计最多 8 GB')
        with self.lock:
            if self.state != 'idle':
                raise ValueError('请先清理本次训练的临时文件，再开始下一次强化')
            self.directory = self.workspace.job(uuid.uuid4().hex)
            self.cancel.clear()
            self.state, self.message = 'running', '正在复制原料到临时磁盘'
            self.thread = threading.Thread(target=self._run, args=(list(paths), python, environment), daemon=True)
            self.thread.start()

    def _run(self, paths, python, environment):
        process = None
        try:
            source_root = Path((environment or {}).get('VOICE_EXTRACT_SOURCE_ROOT', ROOT))
            total = 0
            for index, path in enumerate(paths):
                size = 0
                with path.open('rb') as source, (self.directory / f'upload_{index}.zip').open('wb') as target:
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        if self.cancel.is_set():
                            return
                        size += len(block)
                        total += len(block)
                        if size > 4 * 1024**3 or total > 8 * 1024**3:
                            raise ValueError('原料文件超过大小限制')
                        target.write(block)
            with self.lock:
                if self.cancel.is_set():
                    return
                process = subprocess.Popen([python, '-B', str(source_root / 'launcher/model_reinforcement.py'), str(self.directory)],
                    cwd=source_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                    env={**os.environ, **(environment or {}), 'PYTHONIOENCODING': 'utf-8'}, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                self.process = process
                self.workspace.add_worker(process.pid)
            for line in process.stdout:
                with self.lock:
                    self.message = line.strip()
            process.wait()
            if self.cancel.is_set():
                return
            if process.returncode:
                raise RuntimeError(self.message or '训练失败')
            report = json.loads((self.directory / 'training_report.json').read_text(encoding='utf-8'))
            with self.lock:
                self.report = report
                self.state, self.message = 'complete', completion_message(report)
        except Exception as exc:
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
            with self.lock:
                self.state, self.message = 'failed', str(exc)
            try:
                if self.directory and self.directory.exists():
                    shutil.rmtree(self.directory)
            except OSError as cleanup_error:
                with self.lock:
                    self.message += f'；清理失败，请重试：{cleanup_error}'
        finally:
            if process is not None and process.stdout:
                process.stdout.close()
            with self.lock:
                self.process = None

    def snapshot(self):
        with self.lock:
            return dict(state=self.state, message=self.message, report=self.report,
                        temporary_path=str(self.directory or self.workspace.root))

    def discard(self):
        self.cancel.set()
        with self.lock:
            process, thread = self.process, self.thread
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if thread is not None:
            thread.join(timeout=20)
            if thread.is_alive():
                raise RuntimeError('原料仍在读取，请稍后重试清理')
        with self.lock:
            if self.directory and self.directory.is_dir():
                if self.directory.resolve().parent != self.workspace.root.resolve():
                    raise ValueError('训练目录位置无效')
                shutil.rmtree(self.directory)
            self.directory, self.thread, self.report = None, None, None
            self.state, self.message = 'idle', '本次训练临时文件已清理'

    def close(self):
        self.discard()
        self.workspace.close()
