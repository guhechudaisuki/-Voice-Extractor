"""Keep desktop audio operations in the installed Python worker."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def write_clip(source, destination, start, end, **options):
    if not getattr(sys, 'frozen', False):
        from extractor.audio import write_clip as render
        return render(source, destination, start, end, **options)
    root = Path(sys.executable).resolve().parent
    resource = root / 'resource'
    python = resource / 'python/python.exe'
    script = ('import json,sys; from extractor.audio import write_clip; '
              'a=json.loads(sys.argv[1]); write_clip(*a["args"], **a["options"])')
    payload = json.dumps({'args': [str(source), str(destination), start, end], 'options': options})
    result = subprocess.run([str(python), '-c', script, payload], cwd=resource,
        env=os.environ.copy(), capture_output=True, text=True, encoding='utf-8', errors='replace',
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), timeout=120)
    if result.returncode:
        raise RuntimeError('音频导出失败：' + (result.stderr or result.stdout)[-1000:])
