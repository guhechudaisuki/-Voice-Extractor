"""Visual cuts propose coordinates only; they never confer voice identity."""
import json
import re
from pathlib import Path
import numpy as np
from .types import TimeSpan


def shot_cut_proposals(path, source_sha256):
    from .audio import _run, has_video_stream
    from .config import FFMPEG
    from .media_cache import CACHE_ROOT
    from .stages.base import ArtifactCache, ArtifactKey
    from .model_store import file_sha256
    if Path(path).suffix.lower() not in {'.mp4','.mkv','.mov','.avi','.webm','.m4v','.ts'}:
        return []
    key = ArtifactKey('shot_cuts',source_sha256,file_sha256(Path(__file__)),
                      'scale160-fps6-scene035')
    cache = ArtifactCache(Path(CACHE_ROOT)/'shot_cuts')
    stored = cache.read(key)
    if stored is not None:
        return json.loads(stored)
    if not has_video_stream(Path(path)):
        return []
    result = _run([str(FFMPEG),'-hide_banner','-nostdin','-i',str(path),'-an',
        '-vf',"scale=160:-2,fps=6,select='gt(scene,0.35)',showinfo",'-f','null','-'])
    points = sorted({float(v) for v in re.findall(r'pts_time:([0-9.]+)',result.stderr)})
    cache.write(key,json.dumps(points).encode())
    return points


def split_at_shots(spans, proposals, waveform):
    """Snap visual proposals to actual quiet audio valleys without dropping PCM."""
    result = []
    for span in spans:
        cuts = []
        for point in proposals:
            if not span.start+.35 < point < span.end-.35:
                continue
            a,b = max(span.start,point-.18), min(span.end,point+.18)
            wave = waveform[round(a*16000):round(b*16000)].detach().float().cpu().numpy()
            count = len(wave)//160
            if count < 3:
                continue
            energy = np.mean(wave[:count*160].reshape(count,160)**2,axis=1)
            index = int(np.argmin(energy))
            if energy[index] > max(1e-8,float(np.median(energy))*.2):
                continue
            cut = a+(index+.5)*.01
            if cut-(cuts[-1] if cuts else span.start) >= .35 and span.end-cut >= .35:
                cuts.append(cut)
        boundaries = [span.start,*cuts,span.end]
        result.extend(TimeSpan(a,b) for a,b in zip(boundaries,boundaries[1:]))
    return result
