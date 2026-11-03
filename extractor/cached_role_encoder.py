"""Lazy encoders for batch re-evaluation; frozen WavLM statistics survive training."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from .embedding_reuse import configure_cache, waveform_key
from .nextgen.embedding_cache import EmbeddingCache
from .model_store import file_sha256, active_speaker_path


class CachedEncoder:
    def __init__(self, factory, weight, *, device, implementation=None, anime=False, public_method=False):
        self.factory, self.model = factory, None
        self.device, self.anime = device, anime
        self.public_method = public_method
        configure_cache(self, weight, implementation=implementation)
        if public_method:
            self.feature_version += '|whole-window-mean-v1'

    def _load(self):
        if self.model is None:
            self.model = self.factory()
            if not self.public_method and getattr(self.model, 'feature_cache', None) is not None:
                self.model.feature_cache.writable = False
        return self.model

    @property
    def session(self):
        # Pyramid batching resolves this only after finding uncached voiced
        # features, so a warm pass never instantiates the ONNX session.
        return self._load().session

    def _embeddings_from_waveforms(self, waves, batch_size=16, progress=None):
        keys = [waveform_key(self.feature_version, wave) for wave in waves]
        found = self.feature_cache.get_many(keys)
        missing = {key: wave for key, wave in zip(keys, waves) if key not in found}
        if missing:
            self._load()
            if self.anime:
                values = [self.model.encode(w, sample_rate=16000) for w in missing.values()]
            else:
                method = self.model.embeddings_from_waveforms if self.public_method else self.model._embeddings_from_waveforms
                values = method(list(missing.values()), batch_size=batch_size)
            for key, value in zip(missing, values):
                value = np.asarray(value.cpu() if hasattr(value, 'cpu') else value, dtype=np.float32)
                found[key] = (value, value)
            self.feature_cache.put_many([(key, found[key]) for key in missing])
        return torch.stack([torch.from_numpy(found[key][0].copy()) for key in keys])

    def _embedding_from_waveform(self, wave):
        return self._embeddings_from_waveforms([wave])[0]

    def embeddings_from_waveforms(self, waves, **kwargs):
        return self._embeddings_from_waveforms(waves, **kwargs)

    def embedding_from_file(self, path):
        from .audio import load_mono
        return self._embedding_from_waveform(load_mono(path, 16000))

    def encode(self, wave, **_):
        from .nextgen.anime_embedding import InsufficientSpeech
        # Match the encoder's deterministic no-evidence cases before loading
        # weights. Silence has no vector to persist in an embedding cache.
        if wave.numel() < 3200 or float(wave.square().mean().sqrt()) < 1e-5:
            raise InsufficientSpeech('Insufficient speech for anime identity')
        return self._embedding_from_waveform(wave).numpy()

    def close(self):
        if self.model is not None:
            close = getattr(self.model, 'close', None)
            if close:
                close()
            self.model = None


class CachedWavLM:
    """Cache TDNN statistics in RAM and apply the current global projection."""
    def __init__(self, model_dir, cache_root, role_key=None, device='cpu'):
        self.model_dir, self.device = Path(model_dir), device
        self.encoder = self.processor = None
        self.version = hashlib.sha256((file_sha256(active_speaker_path(self.model_dir))
            + file_sha256(Path(__file__).with_name('local_identity_classifier.py'))
            + '|frozen-tdnn-mean-max-std-v1|' + torch.__version__).encode()).hexdigest()
        self.feature_version = self.version
        self.cache = EmbeddingCache(Path(cache_root) / 'wavlm_frozen')
        self.head_path = self.cache.root / (self.version + '_head.pt')
        self.head = None

    def _load(self):
        if self.encoder is None:
            from transformers import WavLMForXVector, Wav2Vec2FeatureExtractor
            self.encoder = WavLMForXVector.from_pretrained(str(self.model_dir), local_files_only=True,
                use_safetensors=active_speaker_path(self.model_dir).suffix == '.safetensors').to(self.device).eval()
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(str(self.model_dir), local_files_only=True)
            state = {k: v.detach().cpu() for k, v in self.encoder.feature_extractor.state_dict().items()}
            self.head = state
            if self.cache.persist:
                self.head_path.parent.mkdir(parents=True, exist_ok=True)
                from .model_store import atomic_save_projection
                atomic_save_projection(self.head_path, state)

    def pooled(self, wave):
        from .local_identity_classifier import pooled_features
        wave = wave.detach().cpu().float().flatten()
        if wave.numel() < 8000:
            wave = torch.nn.functional.pad(wave, (0, 8000-wave.numel()))
        key = waveform_key(self.version, wave)
        found = self.cache.get_many([key])
        if key not in found:
            self._load()
            value = pooled_features(self.encoder, self.processor, self.device, wave).cpu().numpy()
            width = len(value) // 3
            pair = (np.r_[value[:width], value[-width:]], value)
            self.cache.put_many([(key, pair)])
        else:
            pair = found[key]
        return pair

    def _embedding_from_waveform(self, wave):
        stats, _ = self.pooled(wave)
        if self.head is None:
            if not self.cache.persist or not self.head_path.is_file():
                self._load()
            else:
                self.head = torch.load(self.head_path, map_location='cpu', weights_only=True)
        value = torch.nn.functional.linear(torch.from_numpy(stats), self.head['weight'], self.head.get('bias'))
        return torch.nn.functional.normalize(value, dim=0)

    def _embeddings_from_waveforms(self, waves, **_):
        return torch.stack([self._embedding_from_waveform(wave) for wave in waves])

    def close(self):
        if self.encoder is not None:
            self.encoder = self.processor = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
