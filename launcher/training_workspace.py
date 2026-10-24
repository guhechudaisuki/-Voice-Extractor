"""Disk-backed temporary training sessions with conservative stale cleanup."""
from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time

import psutil

PREFIX = "voice_extractor_training_"


def _alive(marker: dict) -> bool:
    try:
        process = psutil.Process(int(marker["pid"]))
        return abs(process.create_time() - float(marker["created"])) < 2 and process.is_running()
    except psutil.AccessDenied:
        return True
    except (KeyError, TypeError, ValueError, psutil.NoSuchProcess):
        return False


def clean_stale(base: Path) -> int:
    count = 0
    for path in base.glob(PREFIX + "*"):
        if not path.is_dir() or path.resolve().parent != base.resolve():
            continue
        marker_path = path / "owner.json"
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (isinstance(marker, dict) and marker.get("application") == "voice-extractor"
                and not _alive(marker) and not any(_alive(worker) for worker in marker.get("workers", []))):
            try:
                shutil.rmtree(path)
                count += 1
            except OSError:
                # A locked stale folder must not prevent opening the application.
                continue
    return count


class Workspace:
    def __init__(self, base: Path | None = None):
        base = base or Path(tempfile.gettempdir())
        self.base = base.resolve()
        self.lock = threading.RLock()
        clean_stale(base)
        self.root = Path(tempfile.mkdtemp(prefix=PREFIX, dir=base))
        self.created = psutil.Process(os.getpid()).create_time()
        (self.root / "owner.json").write_text(json.dumps({"application": "voice-extractor",
            "pid": os.getpid(), "created": self.created, "timestamp": time.time()}), encoding="utf-8")
        atexit.register(self.close)

    def close(self):
        with self.lock:
            if self.root.resolve().parent != self.base or not self.root.name.startswith(PREFIX):
                raise ValueError("训练临时目录位置无效，拒绝清理")
            if self.root.is_dir():
                shutil.rmtree(self.root)

    def add_worker(self, pid: int):
        path = self.root / "owner.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("workers", []).append({"pid": pid, "created": psutil.Process(pid).create_time()})
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data), encoding="utf-8")
        temporary.replace(path)

    def job(self, job_id: str) -> Path:
        if not job_id or Path(job_id).name != job_id or job_id in {'.', '..'}:
            raise ValueError("训练任务标识无效")
        path = self.root / job_id
        path.mkdir()
        return path
