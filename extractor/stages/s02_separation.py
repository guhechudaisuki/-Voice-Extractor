"""Universal separation stage, cached independently of identity and STT options."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Callable

from .base import ArtifactCache, ArtifactKey, Stage
from .. import pipeline_common as common
from ..model_store import file_sha256


@dataclass
class SeparationInput:
    items: list[tuple[Path, Path]]
    device: str
    progress: Callable[[float, str], None]


class SeparationStage(Stage[SeparationInput, list[Path]]):
    name = 's02_separation'

    def run(self, value):
        if os.environ.get('VOICE_EXTRACT_PERSIST_CACHE') != '1':
            return common.UVR5Separator(value.device).separate_many(value.items,progress=value.progress)
        from ..config import UVR_MODEL
        from ..media_cache import CACHE_ROOT
        cache = ArtifactCache(CACHE_ROOT / 'stages' / self.name)
        files = [Path(__file__), Path(__file__).parents[1] / 'uvr_memory.py',
                 Path(__file__).parents[1] / 'audio.py', UVR_MODEL]
        version = hashlib.sha256(''.join(file_sha256(p) for p in files).encode()).hexdigest()
        keys, missing = {}, []
        for source,destination in value.items:
            key = ArtifactKey(self.name,file_sha256(source),version,'hp2-60s-2s')
            keys[destination] = key
            marker = cache.read(key)
            cached = cache.root / key.digest / 'vocals.wav'
            if marker and cached.is_file() and file_sha256(cached) == json.loads(marker)['audio_sha256']:
                destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(cached,destination)
            else:
                missing.append((source,destination))
        if missing:
            common.UVR5Separator(value.device).separate_many(missing,progress=value.progress)
        for _,destination in missing:
            key = keys[destination]
            root = cache.root / key.digest
            root.mkdir(parents=True,exist_ok=True)
            from ..media_cache import _atomic_copy
            _atomic_copy(destination,root / 'vocals.wav')
            cache.write(key,json.dumps({'audio_sha256':file_sha256(destination)}).encode())
        return [destination for _,destination in value.items]
