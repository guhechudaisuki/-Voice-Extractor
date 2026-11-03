"""Content-addressed batching shared by all callers of an encoder instance."""
from functools import wraps
import hashlib

from .nextgen.embedding_cache import EmbeddingCache, file_digest


def configure_cache(encoder, weight, *, implementation=None):
    from pathlib import Path
    from .config import CACHE_ROOT
    import torch
    code = Path(implementation) if implementation else Path(__file__).with_name("speaker.py")
    version = "|".join((file_digest(Path(weight)), file_digest(code),
                        file_digest(Path(__file__)), torch.__version__,
                        str(getattr(encoder, "device", "cpu"))))
    encoder.feature_version = hashlib.sha256(version.encode()).hexdigest()
    encoder.feature_cache = EmbeddingCache(CACHE_ROOT / "speaker_features")


def waveform_key(version, waveform):
    samples = waveform.detach().cpu().float().contiguous().numpy()
    return hashlib.sha256(version.encode() + samples.tobytes()).hexdigest()


def cached_batch(method):
    @wraps(method)
    def wrapper(self, waveforms, batch_size=48, progress=None):
        import torch
        cache = getattr(self, "feature_cache", None)
        if cache is None or not waveforms:
            return method(self, waveforms, batch_size=batch_size, progress=progress)
        keys = [waveform_key(self.feature_version, w) for w in waveforms]
        found = cache.get_many(keys)
        missing = {}
        for key, wave in zip(keys, waveforms):
            if key not in found:
                missing.setdefault(key, wave)
        # Equal lengths prevent WavLM's padded batch from changing a cached vector.
        groups = {}
        for key, wave in missing.items():
            groups.setdefault(max(8000, wave.numel()), []).append((key, wave))
        fresh = []
        for group in groups.values():
            vectors = method(self, [wave for _, wave in group], batch_size=batch_size)
            for (key, _), vector in zip(group, vectors):
                value = vector.detach().cpu().numpy()
                found[key] = (value, value)
                fresh.append((key, found[key]))
        cache.put_many(fresh)
        if progress is not None:
            progress(len(waveforms), len(waveforms))
        return torch.stack([torch.from_numpy(found[key][0].copy()) for key in keys])
    return wrapper
