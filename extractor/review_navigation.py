"""Exact-range multiview navigation for lossless all-sentence exports."""
import logging
import numpy as np
import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


def annotate_navigation(candidates, waveform, verifier, profile):
    """Annotate in place; scores and clusters never change inclusion or boundaries."""
    if not candidates:
        return
    waves = [waveform[round(c.start*16000):round(c.end*16000)] for c in candidates]
    channels = [('primary',verifier.primary,profile.primary)]
    try:
        secondary = verifier._ensure_secondary(profile)
        channels.append(('secondary',verifier.secondary,secondary))
    except (OSError,RuntimeError,ValueError) as exc:
        LOGGER.warning('复核导航次通道不可用：%s',exc)
    embeddings = {}
    for name,encoder,reference in channels:
        try:
            values = encoder._embeddings_from_waveforms(waves,batch_size=24)
            if not isinstance(values,torch.Tensor) or values.ndim != 2 or len(values) != len(candidates):
                raise ValueError('导航嵌入未返回每句独立向量')
            values = F.normalize(values.detach().float().cpu(),dim=-1)
            if not torch.isfinite(values).all():
                raise ValueError('导航嵌入非有限')
            refs = F.normalize(reference.embeddings.detach().float().cpu(),dim=-1)
            scores = (values @ refs.T).topk(min(2,len(refs)),dim=-1).values.mean(-1)
            embeddings[name] = values.numpy()
            for c,vector,score in zip(candidates,values,scores):
                navigation = c.diagnostics.setdefault('review_navigation',{})
                navigation.setdefault('embeddings',{})[name] = vector.tolist()
                navigation.setdefault('reference_scores',{})[name] = float(score)
                if name == 'primary':
                    c.speaker_score = float(score)
        except (OSError,RuntimeError,ValueError,AttributeError) as exc:
            for c in candidates:
                c.diagnostics.setdefault('review_navigation',{}).setdefault('unavailable',{})[name] = str(exc)
            LOGGER.warning('保留全部单句，导航 %s 不可用：%s',name,exc)
    if embeddings:
        joined = np.concatenate(list(embeddings.values()),axis=1)/np.sqrt(len(embeddings))
        # Conservative online spherical clustering, stable in timeline order.
        centers,counts,ids = [],[],[]
        for vector in joined:
            scores = np.asarray(centers) @ vector if centers else np.array([])
            best = int(np.argmax(scores)) if len(scores) else -1
            if best < 0 or scores[best] < .78:
                best = len(centers)
                centers.append(vector.copy()); counts.append(1)
            else:
                centers[best] = centers[best]*counts[best]+vector
                counts[best] += 1
                centers[best] /= max(np.linalg.norm(centers[best]),1e-12)
            ids.append(best)
    else:
        ids = [None]*len(candidates)
    for candidate,cluster_id in zip(candidates,ids):
        nav = candidate.diagnostics.setdefault('review_navigation',{})
        nav.update(scope='review_navigation_only',span=[candidate.start,candidate.end],
                   cluster_id=cluster_id,confirmed_target=False,method='exact_multiview_v1')
        candidate.diagnostics['cluster_ids'] = [] if cluster_id is None else [cluster_id]
        candidate.diagnostics['all_sentence_export'] = True
    # Retain no fitted centroids or task reference state outside this call.
