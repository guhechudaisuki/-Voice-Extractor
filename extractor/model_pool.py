"""Process-local reusable inference engines; never cache reference profiles here."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
import logging
import os
import threading

LOGGER = logging.getLogger(__name__)


def resource_key(name, *paths):
    """Invalidate universal weights when an atomic model update replaces them."""
    records = []
    for value in paths:
        path = Path(value)
        if path.is_dir():
            records.extend((str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns)
                           for p in sorted(path.rglob('*')) if p.is_file())
        elif path.exists():
            stat = path.stat()
            records.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
        else:
            records.append((str(path.resolve()), None, None))
    return name, tuple(records)


def is_memory_error(error):
    message = str(error).lower()
    return any(term in message for term in
               ('out of memory', 'cuda_error_out_of_memory', 'cublas_status_alloc_failed',
                'failed to allocate memory', 'cuda failure 2'))


class ModelPool:
    """Serialized leases protect inference from concurrent LRU device migration.

    Only role-independent engines belong in this pool. Stateful wrappers such
    as DualSpeakerVerifier and ClusterAnalyzer are always created per task.
    Torch engines can move to CPU RAM without deserializing their weights.
    Other runtimes may implement ``to_device`` (e.g. CTranslate2 unload/load).
    """
    def __init__(self, max_gpu_models=3):
        self.max_gpu_models = max(1, int(max_gpu_models))
        self._models = OrderedDict()
        self._lock = threading.RLock()
        self._active = set()
        self.loads = {}

    def get(self, key, factory, *, device='cpu'):
        with self._lock:
            if key not in self._models:
                self._make_room()
                value = factory()
                self._models[key] = value
                self.loads[key] = self.loads.get(key, 0) + 1
            return _PooledEngine(self, key, device)

    @staticmethod
    def _on_gpu(value):
        return str(getattr(value, 'device', 'cpu')).lower().startswith('cuda')

    @staticmethod
    def _move(value, device):
        import torch
        if hasattr(value, 'to_device'):
            return value.to_device(device) is not False
        model = getattr(value, 'model', None)
        if isinstance(model, torch.nn.Module):
            # CPU kernels may not support half; restore the preferred GPU dtype
            # when bringing the same weights back into VRAM.
            if not hasattr(value, '_pool_gpu_dtype'):
                value._pool_gpu_dtype = next(model.parameters()).dtype
            model.to(device=device, dtype=(value._pool_gpu_dtype if device == 'cuda'
                                          else torch.float32))
            old = getattr(value, 'device', 'cpu')
            value.device = torch.device(device) if isinstance(old, torch.device) else device
            return True
        session = getattr(value, 'session', None)
        if session is not None and hasattr(session, 'set_providers'):
            providers = ['CPUExecutionProvider']
            if device == 'cuda':
                providers.insert(0, 'CUDAExecutionProvider')
            # ORT has no unload-to-RAM API. Reconfigure only under pressure;
            # ordinary cross-episode reuse keeps the same resident session.
            session.set_providers(providers)
            actual = session.get_providers()[0]
            value.device = (actual if 'ExecutionProvider' in str(getattr(value, 'device', ''))
                            else 'cuda' if actual == 'CUDAExecutionProvider' else 'cpu')
            return True
        return False

    def _make_room(self, exclude=None):
        resident = sum(self._on_gpu(v) for v in self._models.values())
        for key, value in self._models.items():
            if resident < self.max_gpu_models:
                break
            if key != exclude and key not in self._active and self._on_gpu(value):
                if self._move(value, 'cpu'):
                    resident -= 1

    @contextmanager
    def lease(self, key, device):
        with self._lock:
            value = self._models[key]
            if device == 'cuda' and not self._on_gpu(value):
                self._make_room(exclude=key)
                try:
                    self._move(value, device)
                except Exception as error:
                    if not is_memory_error(error):
                        raise
                    self._move(value, 'cpu')
                    LOGGER.warning('GPU memory exhausted; using CPU for %s', key[0])
            self._models.move_to_end(key)
            self._active.add(key)
            try:
                yield value
            finally:
                self._active.discard(key)

    def clear_task_state(self):
        """Release content embeddings and caller caches after every episode."""
        with self._lock:
            for value in self._models.values():
                cache = getattr(value, 'feature_cache', None)
                if cache is not None:
                    cache.memory.clear()

    def clear(self):
        with self._lock:
            for value in self._models.values():
                close = getattr(value, 'close', None)
                if close is not None:
                    close()
            self._models.clear()


class _PooledEngine:
    def __init__(self, pool, key, device):
        object.__setattr__(self, '_pool', pool)
        object.__setattr__(self, '_key', key)
        object.__setattr__(self, '_device', device)

    def __getattr__(self, name):
        value = getattr(self._pool._models[self._key], name)
        if not callable(value):
            return value
        def invoke(*args, **kwargs):
            with self._pool.lease(self._key, self._device) as engine:
                try:
                    return getattr(engine, name)(*args, **kwargs)
                except Exception as error:
                    if not is_memory_error(error) or not self._pool._on_gpu(engine):
                        raise
                if hasattr(engine, 'cpu_fallback'):
                    engine.cpu_fallback()
                elif not self._pool._move(engine, 'cpu'):
                    raise RuntimeError('Engine does not support CPU memory fallback')
                LOGGER.warning('Retrying %s on CPU after GPU memory exhaustion', name)
                return getattr(engine, name)(*args, **kwargs)
        return invoke

    def __setattr__(self, name, value):
        setattr(self._pool._models[self._key], name, value)

    def close(self):
        # The owning pool, rather than one episode, controls model lifetime.
        pass

    def __call__(self, *args, **kwargs):
        return self.__getattr__('__call__')(*args, **kwargs)


MODEL_POOL = ModelPool(int(os.environ.get('VOICE_EXTRACT_GPU_MODEL_LIMIT', '3')))


def pooled_encoder(factory, device, *weights, **kwargs):
    key = resource_key(getattr(factory, '__name__', type(factory).__qualname__), *weights)
    def create():
        try:
            return factory(device=device, **kwargs)
        except Exception as error:
            if device != 'cuda' or not is_memory_error(error):
                raise
        return factory(device='cpu', **kwargs)
    return MODEL_POOL.get((key, device, factory), create, device=device)
