"""Typed stage boundary and atomic, content-addressed universal artifacts."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Generic, TypeVar

I = TypeVar('I')
O = TypeVar('O')


class Stage(ABC, Generic[I, O]):
    name: str
    version = 1

    @abstractmethod
    def run(self, value: I) -> O:
        raise NotImplementedError


@dataclass(frozen=True)
class ArtifactKey:
    stage: str
    source_sha256: str
    implementation_sha256: str
    settings: str

    @property
    def digest(self):
        return hashlib.sha256(json.dumps([self.stage, self.source_sha256,
            self.implementation_sha256, self.settings], ensure_ascii=True).encode()).hexdigest()


class ArtifactCache:
    """For reference-independent media/features only; identity remains in RAM."""
    def __init__(self, root: Path):
        self.root = Path(root)

    def read(self, key: ArtifactKey):
        path = self.root / key.digest
        try:
            payload = (path / 'payload').read_bytes()
            metadata = json.loads((path / 'artifact.json').read_text(encoding='utf-8'))
            if metadata['key'] == key.digest and metadata['sha256'] == hashlib.sha256(payload).hexdigest():
                return payload
        except (OSError, ValueError, KeyError):
            pass
        return None

    def write(self, key: ArtifactKey, payload: bytes):
        path = self.root / key.digest
        path.mkdir(parents=True, exist_ok=True)
        for name, content in (('payload', payload), ('artifact.json', json.dumps({
                'schema': 1, 'key': key.digest, 'sha256': hashlib.sha256(payload).hexdigest()
                }).encode())):
            descriptor, temporary = tempfile.mkstemp(prefix=name + '-', dir=path)
            try:
                with os.fdopen(descriptor, 'wb') as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path / name)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
