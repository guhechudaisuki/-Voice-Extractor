"""Keep desktop audio operations in the installed Python worker."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def write_sequence(source, destination, intervals):
    """Audition all clean pieces, using the installed runtime in frozen UI builds."""
    if getattr(sys,'frozen',False):
        root = Path(sys.executable).resolve().parent/'resource'
        script = ('import json,sys; from launcher.audio_export import write_sequence; '
                  'print(write_sequence(*json.loads(sys.argv[1])))')
        result = subprocess.run([str(root/'python/python.exe'),'-c',script,
            json.dumps([str(source),str(destination),intervals])],cwd=root,
            env=os.environ.copy(),capture_output=True,text=True,encoding='utf-8',errors='replace',
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),timeout=120)
        if result.returncode:
            raise RuntimeError('保留段试听准备失败：'+(result.stderr or result.stdout)[-1000:])
        return float(result.stdout.strip().splitlines()[-1])
    import soundfile as sf
    import numpy as np
    samples,rate = sf.read(source,always_2d=True)
    pieces = [samples[round(a*rate):round(b*rate)] for a,b in intervals]
    silence = np.zeros((round(.15*rate),samples.shape[1]))
    audio = np.concatenate([part for i,piece in enumerate(pieces)
                            for part in ((silence,piece) if i else (piece,))])
    sf.write(destination,audio,rate,subtype='PCM_16')
    return len(audio)/rate


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
