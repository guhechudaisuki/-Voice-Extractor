"""Persist role-independent acoustic probes, without persisting identity decisions."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from .embedding_cache import file_digest
from .feature_store import FeatureStore


class AcousticProbeCache:
    def __init__(self, analyzer, root):
        code = [Path(__file__), Path(__file__).with_name('physiology.py'),
                Path(__file__).with_name('cluster_runtime.py'), Path(__file__).parents[1]/'extraction_guards.py']
        self.key = hashlib.sha256(json.dumps([analyzer.source_hash,
            [file_digest(p) for p in analyzer.references], [file_digest(p) for p in code]]).encode()).hexdigest()
        self.store = (FeatureStore(Path(root)/'rescore_acoustics/features.sqlite3')
                      if os.environ.get('VOICE_EXTRACT_PERSIST_CACHE') == '1' else None)
        found = self.store.get_many([self.key]) if self.store else {}
        self.values = json.loads(found[self.key]) if self.key in found else {}
        self.changed = False
        for name in ('vocal_geometry_evidence', 'prosody', 'reference_prosody', 'relative_window_energy'):
            original = getattr(analyzer, name)
            def probe(*args, _name=name, _original=original):
                key = json.dumps([_name, args], separators=(',', ':'))
                if key not in self.values:
                    value = _original(*args)
                    self.values[key] = self.encode(value)
                    self.changed = True
                return self.decode(self.values[key])
            setattr(analyzer, name, probe)

    @classmethod
    def encode(cls, value):
        if isinstance(value, np.ndarray):
            return {'__array__': value.tolist()}
        if isinstance(value, dict):
            return {k: cls.encode(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls.encode(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    @classmethod
    def decode(cls, value):
        if isinstance(value, dict):
            if '__array__' in value:
                return np.asarray(value['__array__'])
            return {k: cls.decode(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls.decode(v) for v in value]
        return value

    def save(self):
        if self.changed and self.store:
            self.store.put_many([(self.key, json.dumps(self.values, allow_nan=False, separators=(',', ':')).encode())])
