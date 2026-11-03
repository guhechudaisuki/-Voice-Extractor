"""Equal-length ONNX batches for the pyramid, preserving encoder/cache contracts."""
import numpy as np


def anime_batch(encoders, waveforms, batch_size=8):
    from .anime_embedding import speechbrain_compatible_fbank, InsufficientSpeech
    from ..embedding_reuse import waveform_key
    features, output = {}, []
    for encoder in encoders:
        cache = encoder.feature_cache
        keys = [waveform_key(encoder.feature_version,wave) for wave in waveforms]
        found = cache.get_many(keys)
        groups = {}
        for index,key in enumerate(keys):
            if key in found:
                continue
            if index not in features:
                try:
                    features[index] = speechbrain_compatible_fbank(waveforms[index]).numpy()
                except InsufficientSpeech:
                    features[index] = None
            feature = features[index]
            if feature is not None:
                groups.setdefault(feature.shape, []).append(index)
        fresh = []
        for indexes in groups.values():
            for start in range(0,len(indexes),batch_size):
                batch = indexes[start:start+batch_size]
                vectors = np.asarray(encoder.session.run(None,{
                    'features':np.stack([features[i] for i in batch])})[0],dtype=np.float32)
                if vectors.shape != (len(batch),192) or not np.isfinite(vectors).all():
                    raise ValueError('Invalid batched anime embeddings')
                norms = np.linalg.norm(vectors,axis=1,keepdims=True)
                if np.any(norms <= 1e-8):
                    raise ValueError('Zero batched anime embedding')
                for index,vector in zip(batch,vectors/norms):
                    found[keys[index]] = (vector,vector)
                    fresh.append((keys[index],(vector,vector)))
        cache.put_many(fresh)
        output.append([found[key][0] if key in found else None for key in keys])
    return [tuple(channel[i] for channel in output)
            if all(channel[i] is not None for channel in output) else None
            for i in range(len(waveforms))]
