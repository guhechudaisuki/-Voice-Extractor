"""Shared offline/live cluster analysis; no human labels enter this module."""
from __future__ import annotations

import hashlib
import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np

from .embedding_cache import EmbeddingCache, cache_key, file_digest, canonical_interval
from .exclusion_reliability import audit_exclusion_views
from .identity_cluster import (TargetProfile, TurnEvidence, cluster_turns,
                               assign_target, profile_score, eligible_competitor)
from ..recall import MIN_TARGET_SECONDS


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def window_spans(start, end, width=1.8, hop=0.9):
    if end - start <= width:
        return [(start, end)]
    starts = list(np.arange(start, end - width, hop)) + [end - width]
    return [(float(a), float(a + width)) for a in starts]


# Minimum independently exportable clean interval.
MIN_CLEAN_SUBSPAN = MIN_TARGET_SECONDS
WHOLE_TURN_PASS_RATIO = 0.80
# Bridge a small score dip only between exclusive target windows.
BREATH_DIP_TOLERANCE = 0.03
BREATH_NEIGHBOR_FLOOR = 0.06
# Margins against the strongest competing speaker cluster.
WINDOW_COHORT_PRIMARY = 0.055
WINDOW_COHORT_SECONDARY = 0.02
WINDOW_COHORT_STRONG = 0.10
SHORT_TURN_SECONDS = 3.0
WEAK_SECONDARY_SLACK = 0.10


def _short_dialogue_path(duration, p_slack, s_slack, rows, t_strong, cohort_margin,
                         *, scene_distance=None):
    """Independent complementary certificates; every window must pass."""
    if not rows or not all(r.get("accepted") and not r.get("smoothed") for r in rows):
        return None
    def margins(primary, secondary):
        return all(isinstance(r.get("cm_p"), (int, float)) and r["cm_p"] >= primary
                   and isinstance(r.get("cm_s"), (int, float)) and r["cm_s"] >= secondary
                   for r in rows)
    if p_slack >= .08 - 1e-9 and s_slack >= -.02 - 1e-9 and margins(.09, .14):
        return "A_dual_cohort"
    if t_strong and p_slack >= .05 - 1e-9 and s_slack >= -.02 - 1e-9 and margins(.08, .10):
        return "B_three_channel"
    if (duration >= 1.85 and p_slack >= .04 - 1e-9
            and s_slack >= .02 - 1e-9 and cohort_margin >= .08):
        return "C_stable_dialogue"
    if (p_slack >= .08 - 1e-9 and s_slack >= -.045 - 1e-9
            and ((t_strong and margins(.10, .25)) or margins(.20, .25))):
        return "D_cohort_dominant"
    if (1.05 <= duration <= SHORT_TURN_SECONDS and scene_distance is not None
            and scene_distance <= 20.0 and p_slack >= .03 - 1e-9
            and s_slack >= -.04 - 1e-9 and margins(.08, .15)):
        return "E_scene_dialogue"
    return None


def _clean_core_certified(rows, profile, t_strong):
    """Certify only accepted core windows; complementary evidence keeps a secondary floor."""
    p_floor, s_floor = profile.primary_reference_floor, profile.secondary_reference_floor
    min_p = min(r["primary"] for r in rows)
    min_s = min(r["secondary"] for r in rows)

    def margins(primary, secondary):
        return all(isinstance(r.get("cm_p"), (int, float)) and r["cm_p"] >= primary
                   and isinstance(r.get("cm_s"), (int, float)) and r["cm_s"] >= secondary
                   for r in rows)

    dual = min_p >= p_floor + .05 and min_s >= s_floor + .02 and margins(.08, .08)
    mean_support = (min_p >= p_floor and min_s >= s_floor
                    and sum(r["primary"] for r in rows) / len(rows) >= p_floor + .04
                    and sum(r["secondary"] for r in rows) / len(rows) >= s_floor + .04
                    and margins(.15, .25))
    complementary = (min_p >= p_floor + .14 and min_s >= s_floor - .02
                     and margins(.12, .20) and t_strong)
    return dual or complementary or mean_support


class ClusterAnalyzer:
    # 类级默认值：测试与离线回放会用 __new__ 构造不加载模型的桩实例。
    tertiary = None
    tertiary_hash = ""
    tertiary_cache = None
    tertiary_negatives = []
    quaternary = None
    raw_stem = None
    raw_hash = ""
    raw_cache = None
    raw_waveform = None
    _physiology_probe = None
    anime_encoders = None
    anime_hash = ""
    anime_cache = None

    cross_bank: tuple = ()

    def __init__(self, stem, references, negative_groups=(), *, device="cuda",
                 cache_enabled=True, encoders=None, progress=None,
                 tertiary_encoder=None, quaternary_encoder=None, raw_stem=None, anime_encoders=None,
                 role_key=None, geometry_guard=None):
        from ..config import ASSET_ROOT, CACHE_ROOT, SV_MODEL, CAMPLUS_MODEL
        from ..speaker import SpeakerVerifier, CAMPlusVerifier
        self.stem = Path(stem)
        self.references = sorted(map(Path, references))
        if geometry_guard is not None:
            self._geometry_guard = geometry_guard
        self._geometry_cache = {}
        from ..model_store import reference_key
        self.role_key = role_key or reference_key(file_digest(p) for p in self.references)
        self.negative_groups = sorted(
            [sorted(map(Path, group)) for group in negative_groups if group],
            key=lambda group: tuple(str(p.resolve()) for p in group),
        )
        self.progress = progress or (lambda message: None)
        paths = [self.stem, SV_MODEL, CAMPLUS_MODEL, *self.references,
                 *(p for group in self.negative_groups for p in group)]
        self.inputs = [{"path": str(Path(p).resolve()), "sha256": file_digest(Path(p))} for p in paths]
        self.source_hash = self.inputs[0]["sha256"]
        import torch
        implementation = (inspect.getsource(type(self).embed) + inspect.getsource(type(self)._file_embedding)
                          + file_digest(Path(__file__).parents[1] / "speaker.py")
                          + file_digest(Path(__file__).parents[1] / "audio.py")
                          + torch.__version__ + str(device))
        self.encoding_version = hashlib.sha256(implementation.encode()).hexdigest()
        self.model_hash = hashlib.sha256((self.inputs[1]["sha256"] + self.inputs[2]["sha256"]
                                         + self.encoding_version).encode()).hexdigest()
        self.cache = EmbeddingCache(CACHE_ROOT / "cluster_embeddings", enabled=cache_enabled)
        self.owns_encoders = encoders is None
        self.encoders = encoders or (SpeakerVerifier(device=device), CAMPlusVerifier(device=device))
        self.waveform = None
        self.anime_encoders = anime_encoders
        self.owns_anime = anime_encoders is None
        self.anime_hash = ""
        self.anime_cache = None
        self._init_anime(ASSET_ROOT / "models", CACHE_ROOT,
                         cache_enabled=cache_enabled, device=device)
        # Auxiliary models have independent cache namespaces and weight digests.
        self.tertiary = tertiary_encoder
        self.owns_tertiary = False
        self.tertiary_hash = ""
        self.tertiary_cache = None
        self.tertiary_negatives = []
        if self.tertiary is not None:
            from ..config import WAVLM_SV_MODEL
            weight = WAVLM_SV_MODEL / "pytorch_model.bin"
            digest = file_digest(weight) if weight.exists() else "external"
            self.tertiary_hash = hashlib.sha256((digest + getattr(self.tertiary, "feature_version", "")).encode()).hexdigest()
            self.tertiary_cache = EmbeddingCache(CACHE_ROOT / "wavlm_embeddings",
                                                 enabled=cache_enabled)
        self.quaternary = quaternary_encoder
        self.quaternary_floor = 0.0
        # Aligned raw audio can recover speech distorted by source separation.
        self.raw_stem = Path(raw_stem) if raw_stem else None
        self.raw_hash = ""
        self.raw_cache = None
        self.raw_waveform = None
        self._physiology_probe = None
        if self.raw_stem is not None and self.raw_stem.exists():
            self.raw_hash = file_digest(self.raw_stem)[:16]
            self.raw_cache = EmbeddingCache(CACHE_ROOT / "raw_embeddings",
                                            enabled=cache_enabled)
        for encoder in (*self.encoders, *(self.anime_encoders or ()), self.tertiary, self.quaternary):
            shared_cache = getattr(encoder, 'feature_cache', None)
            if shared_cache is not None:
                shared_cache.enabled = cache_enabled
        # Encoders own persistent content-addressed vectors. Interval caches
        # remain read-compatible with old runs, and memoize new pairs in RAM.
        for cache, models in ((self.cache,self.encoders), (self.raw_cache,self.encoders),
                              (self.anime_cache,self.anime_encoders or ()),
                              (self.tertiary_cache,(self.tertiary,))):
            if cache is not None and models and all(getattr(e,'feature_cache',None) is not None for e in models):
                cache.writable = False
        self.profile, self.negatives = self._profiles()
        self.bootstrap_provenance = None

    def bank_context(self):
        return {"encoding_version": self.encoding_version,
                "models": [item["sha256"] for item in self.inputs[1:3]],
                "anime_model_hash": self.anime_hash,
                "references": sorted(file_digest(p) for p in self.references),
                "negative_groups": sorted(sorted(file_digest(p) for p in group)
                                          for group in self.negative_groups)}

    def close(self):
        if self.owns_encoders:
            for encoder in self.encoders:
                encoder.close()
        if self.owns_tertiary and self.tertiary is not None and hasattr(self.tertiary, "close"):
            self.tertiary.close()
        if self.owns_anime:
            for encoder in self.anime_encoders or ():
                encoder.close()
        self.anime_encoders = None

    def wespeaker_score(self, start: float, end: float):
        """候选区间对参考质心的 WeSpeaker 相似度；无第四通道时返回 None。"""
        if self.quaternary is None:
            return None
        import torch
        from ..audio import load_mono
        if getattr(self, "_quaternary_waveform", None) is None:
            self._quaternary_waveform = load_mono(self.stem, 16000)
        clip = self._quaternary_waveform[
            max(0, round(float(start) * 16000)):round(float(end) * 16000)]
        if clip.numel() < 8000:
            return None
        embedding = self.quaternary.embeddings_from_waveforms([clip])[0]
        return float(embedding @ self.quaternary_centroid)

    def wespeaker_arbitrate(self, accepted, rejected) -> int:
        """Demote rescued candidates rejected by the independent WeSpeaker encoder."""
        from .cluster_gate import RESCUE_KEYS
        if self.quaternary is None:
            return 0
        demoted = []
        for candidate in list(accepted):
            if not any(candidate.diagnostics.get(key) for key in RESCUE_KEYS):
                continue
            # 若候选在主/副双通道上均显著高于参考下限（无争议强匹配），豁免 WeSpeaker 降级
            profile = getattr(self, "profile", None)
            if profile is not None:
                p_sc = float(getattr(candidate, "speaker_score", 0.0) or 0.0)
                s_sc = float(candidate.diagnostics.get("camplus_score") or 0.0)
                if (p_sc >= profile.primary_reference_floor + 0.12
                        and s_sc >= profile.secondary_reference_floor + 0.10):
                    continue
            score = self.wespeaker_score(candidate.start, candidate.end)
            if score is None or score >= self.quaternary_floor:
                continue
            candidate.diagnostics["wespeaker_arbitration"] = round(score, 5)
            candidate.diagnostics["wespeaker_floor"] = round(self.quaternary_floor, 5)
            candidate.accepted = False
            candidate.reject_reason = "簇级身份未通过：wespeaker_arbitration"
            candidate.diagnostics["cluster_tier"] = "B"
            candidate.diagnostics["cluster_reason"] = "wespeaker_arbitration"
            demoted.append(candidate)
        if demoted:
            removed_ids = {id(c) for c in demoted}
            accepted[:] = [c for c in accepted if id(c) not in removed_ids]
            rejected.extend(demoted)
        return len(demoted)

    def wespeaker_island_evidence(self, start: float, end: float):
        """Require independent utterance support before extending its islands.

        Original candidate flags and coarse scores cannot exempt newly added
        audio. Missing or non-finite evidence leaves the old proposal intact.
        """
        score = self.wespeaker_score(start, end)
        floor = getattr(self, 'quaternary_floor', None)
        score = float(score) if score is not None and np.isfinite(score) else None
        floor = float(floor) if floor is not None and np.isfinite(floor) else None
        return {'accepted': score is not None and floor is not None and score >= floor,
                'score': score, 'floor': floor, 'method': 'wespeaker_continuous_parent'}

    def _file_embedding(self, path):
        key = cache_key(file_digest(path), 0, 0, self.model_hash)
        return self.cache.get_or_compute(key, lambda: tuple(
            e.embedding_from_file(path).detach().cpu().numpy() for e in self.encoders
        ))

    def _file_embedding3(self, path):
        key = cache_key(file_digest(path), 0, 0, self.tertiary_hash)
        # EmbeddingCache 固定存双向量，这里复用同一键结构存单向量。
        pair = self.tertiary_cache.get_or_compute(key, lambda: self._tertiary_from_file(path))
        return pair[0]

    def _tertiary_from_file(self, path):
        import torch
        from ..audio import load_mono
        vector = self.tertiary._embedding_from_waveform(
            load_mono(Path(path), 16000)).detach().cpu().numpy()
        return (vector, vector)

    def _init_anime(self, models_root, cache_root, *, cache_enabled=True, device="cpu"):
        from .anime_embedding import AnimeSpeakerOnnx, _MODEL_FILES
        paths = [Path(models_root) / _MODEL_FILES[variant] for variant in ("char", "va")]
        if not all(path.is_file() for path in paths):
            return
        encoders = list(self.anime_encoders or ())
        try:
            for variant in (() if encoders else ("char", "va")):
                encoders.append(AnimeSpeakerOnnx(models_root, variant, device=device))
        except Exception:
            for encoder in encoders:
                encoder.close()
            raise
        self.anime_encoders = tuple(encoders)
        model_digests = [file_digest(path) for path in paths]
        implementation = (file_digest(Path(__file__).with_name("anime_embedding.py"))
                          + inspect.getsource(type(self).embed_anime)
                          + inspect.getsource(type(self)._file_embedding_anime))
        self.anime_hash = hashlib.sha256("|".join([
            *model_digests, implementation, str(device), "sample-exact-16000-v2",
        ]).encode()).hexdigest()
        self.anime_cache = EmbeddingCache(Path(cache_root) / "anime_embeddings",
                                          enabled=cache_enabled)
        self.inputs.extend({"path": str(path.resolve()), "sha256": digest}
                           for path, digest in zip(paths, model_digests))
        self.progress("已启用动漫角色与声优声纹模型")

    def _file_embedding_anime(self, path):
        from ..audio import load_mono
        key = cache_key(file_digest(path), 0, 0, self.anime_hash)
        def compute():
            waveform = load_mono(Path(path), 16000)
            return tuple(encoder.encode(waveform, sample_rate=16000)
                         for encoder in self.anime_encoders)
        return self.anime_cache.get_or_compute(key, compute)

    @staticmethod
    def _reference_keep_mask(primary, secondary, anime_va=None):
        """Remove only references rejected by every available identity channel."""
        count = len(primary)
        keep = np.ones(count, dtype=bool)
        if count < 4:
            return keep

        def leave_one_out_medians(matrix):
            similarities = matrix @ matrix.T
            return np.array([np.median(np.delete(row, index))
                             for index, row in enumerate(similarities)])

        outliers = ((leave_one_out_medians(primary) < .38)
                    & (leave_one_out_medians(secondary) < .22))
        if anime_va is not None:
            outliers &= leave_one_out_medians(anime_va) < .15
        cleaned = ~outliers
        return cleaned if int(cleaned.sum()) >= 2 else keep

    def _profiles(self):
        if not self.references:
            raise ValueError("Cluster analysis requires target references")
        vectors = [self._file_embedding(p) for p in self.references]
        p, s = [np.stack([v[channel] for v in vectors]) for channel in (0, 1)]
        p_unit = np.stack([unit(row) for row in p])
        s_unit = np.stack([unit(row) for row in s])
        anime = None
        if self.anime_encoders is not None:
            vectors = [self._file_embedding_anime(path) for path in self.references]
            anime = tuple(np.stack([unit(v[channel]) for v in vectors]) for channel in (0, 1))
        keep_mask = self._reference_keep_mask(
            p_unit, s_unit, anime[1] if anime is not None else None)
        p_clean, s_clean = p_unit[keep_mask], s_unit[keep_mask]
        pc, sc = unit(p_clean.mean(axis=0)), unit(s_clean.mean(axis=0))
        self.purified_reference_indexes = tuple(
            int(i) for i, ok in enumerate(keep_mask) if ok)
        profile = TargetProfile(
            pc, sc,
            float(np.clip(np.quantile(p_clean @ pc, 0.10) - 0.22, 0.52, 0.68)),
            float(np.clip(np.quantile(s_clean @ sc, 0.10) - 0.26, 0.48, 0.64)),
            primary_references=p_clean, secondary_references=s_clean,
        )
        target_views = {"primary": p_clean, "secondary": s_clean}
        if self.tertiary is not None:
            tv = np.stack([unit(self._file_embedding3(path)) for path in self.references])[keep_mask]
            tc = unit(tv.mean(axis=0))
            profile = replace(
                profile,
                tertiary_centroid=tc,
                tertiary_reference_floor=float(np.clip(
                    np.quantile(tv @ tc, 0.10) - 0.06, 0.74, 0.88)),
                tertiary_references=tv,
            )
            target_views["tertiary"] = tv
        if self.quaternary is not None:
            import torch
            from ..audio import load_mono
            clean_paths = [path for path, keep in zip(self.references, keep_mask) if keep]
            qv = self.quaternary.embeddings_from_waveforms(
                [load_mono(path, 16000) for path in clean_paths])
            qv = torch.nn.functional.normalize(qv, dim=1)
            self.quaternary_centroid = torch.nn.functional.normalize(qv.mean(dim=0), dim=0)
            self.quaternary_floor = float(np.clip(
                (qv @ self.quaternary_centroid).min().item() - .06, .58, .66))
            profile = replace(profile,
                quaternary_centroid=self.quaternary_centroid.detach().cpu().numpy(),
                quaternary_reference_floor=self.quaternary_floor)
        if anime is not None:
            char, va = (matrix[keep_mask] for matrix in anime)
            cc, vc = unit(char.mean(axis=0)), unit(va.mean(axis=0))
            profile = replace(profile,
                anime_char_centroid=cc, anime_va_centroid=vc,
                anime_char_reference_floor=float(np.clip(np.quantile(char @ cc, .10) - .30, .35, .55)),
                anime_va_reference_floor=float(np.clip(np.quantile(va @ vc, .10) - .35, .25, .50)),
                anime_char_references=char, anime_va_references=va)
        negatives = []
        self.exclusion_audit = {}
        self.exclusion_references = {}
        self.tertiary_negatives = []
        for group in self.negative_groups:
            v = [self._file_embedding(path) for path in group]
            # Stable role identity derives from files, not enumeration order.
            role = hashlib.sha256("|".join(sorted(file_digest(path) for path in group)).encode()).hexdigest()[:16]
            views = {name: np.stack([unit(pair[channel]) for pair in v])
                     for channel, name in enumerate(("primary", "secondary"))}
            negatives.append((role, unit(views["primary"].mean(axis=0)),
                              unit(views["secondary"].mean(axis=0))))
            if self.tertiary is not None:
                views["tertiary"] = np.stack([unit(self._file_embedding3(path)) for path in group])
                self.tertiary_negatives.append((role, unit(views["tertiary"].mean(axis=0))))
            self.exclusion_references[role] = views
            self.exclusion_audit[role] = audit_exclusion_views(target_views, views)
        return profile, negatives

    def _certification_profile(self):
        """Calibrate short/core certificates with leave-one-out reference scores."""
        references = self.profile.primary_references
        if references is None or len(references) < 3:
            return self.profile
        indexes = getattr(self, "purified_reference_indexes", None)
        if indexes is not None and len(indexes) < 3:
            return self.profile
        references = np.asarray(references)
        scores = []
        for index, reference in enumerate(references):
            others = np.delete(references, index, axis=0)
            scores.append(profile_score(reference, unit(others.mean(axis=0)), others))
        floor = float(np.clip(np.quantile(scores, .10) - .22, .52, .68))
        return replace(self.profile, primary_reference_floor=floor)

    def embed(self, start, end):
        # Use the same exact cuts in offline and pipeline analysis.
        a, b = canonical_interval(start, end)
        key = cache_key(self.source_hash, a, b, self.model_hash)
        def compute():
            from ..audio import load_mono
            if self.waveform is None:
                self.waveform = load_mono(self.stem, 16000)
            clip = self.waveform[max(0, round(a * 16000)):round(b * 16000)]
            return tuple(e._embedding_from_waveform(clip).detach().cpu().numpy() for e in self.encoders)
        return self.cache.get_or_compute(key, compute)

    def prosody(self, start, end):
        from .physiology import prosody_signature
        from ..audio import load_mono
        if not hasattr(self, '_prosody_cache'):
            self._prosody_cache = {}
        key = canonical_interval(start,end)
        if key not in self._prosody_cache:
            if self.waveform is None:
                self.waveform = load_mono(self.stem,16000)
            self._prosody_cache[key] = prosody_signature(
                self.waveform[round(key[0]*16000):round(key[1]*16000)])
        return self._prosody_cache[key]

    def relative_window_energy(self, start, end, parent_start, parent_end):
        """Silence in a speech turn must not acquire identity from model bias."""
        from ..audio import load_mono
        if self.waveform is None:
            self.waveform = load_mono(self.stem,16000)
        if not hasattr(self, '_energy_cache'):
            self._energy_cache = {}
        def rms(a,b):
            key = (round(a*16000),round(b*16000))
            if key not in self._energy_cache:
                clip = self.waveform[key[0]:key[1]]
                self._energy_cache[key] = float(clip.square().mean().sqrt()) if clip.numel() else 0.
            return self._energy_cache[key]
        energy = rms(start,end)
        return energy / max(1e-6,rms(parent_start,parent_end))

    def reference_prosody(self):
        from .physiology import prosody_signature
        from ..audio import load_mono
        if not hasattr(self, '_reference_prosody'):
            self._reference_prosody = [value for p in self.references
                if (value := prosody_signature(load_mono(p,16000))) is not None]
        return self._reference_prosody

    def prefetch(self, spans):
        """Batch new pyramid windows without changing the per-window cache keys."""
        if not getattr(self, 'encoders', None) or not hasattr(self.cache, 'get_many'):
            return
        from ..audio import load_mono
        if self.waveform is None:
            self.waveform = load_mono(self.stem, 16000)
        spans = list(dict.fromkeys(canonical_interval(a,b) for a,b in spans))
        keys = [cache_key(self.source_hash,a,b,self.model_hash) for a,b in spans]
        cached = self.cache.get_many(keys)
        missing = [(key,a,b) for key,(a,b) in zip(keys,spans) if key not in cached]
        for first in range(0,len(missing),24):
            batch=missing[first:first+24]
            waves=[self.waveform[round(a*16000):round(b*16000)] for _,a,b in batch]
            values=[encoder._embeddings_from_waveforms(waves,batch_size=24).detach().cpu().numpy()
                    for encoder in self.encoders]
            self.cache.put_many([(key,(values[0][i],values[1][i])) for i,(key,_,_) in enumerate(batch)])
        if self.anime_encoders and self.anime_cache is not None:
            from .pyramid_batching import anime_batch
            anime_keys = [cache_key(self.source_hash,a,b,self.anime_hash) for a,b in spans]
            found = self.anime_cache.get_many(anime_keys)
            missing = [(key,a,b) for key,(a,b) in zip(anime_keys,spans) if key not in found]
            for first in range(0,len(missing),24):
                batch = missing[first:first+24]
                waves = [self.waveform[round(a*16000):round(b*16000)] for _,a,b in batch]
                pairs = anime_batch(self.anime_encoders,waves)
                self.anime_cache.put_many([(key,pair) for (key,_,_),pair in zip(batch,pairs) if pair is not None])

    def embed3(self, start, end):
        """第三通道（WavLM）嵌入；独立内容寻址缓存，强化后自动重算。"""
        if self.tertiary is None:
            return None
        a, b = canonical_interval(start, end)
        key = cache_key(self.source_hash, a, b, self.tertiary_hash)
        def compute():
            import torch
            from ..audio import load_mono
            if self.waveform is None:
                self.waveform = load_mono(self.stem, 16000)
            clip = self.waveform[max(0, round(a * 16000)):round(b * 16000)]
            vector = self.tertiary._embedding_from_waveform(clip).detach().cpu().numpy()
            return (vector, vector)
        return self.tertiary_cache.get_or_compute(key, compute)[0]

    def embed_raw(self, start, end):
        """原轨（Raw）双通道嵌入；独立缓存命名空间，读取失败返回 None。"""
        if self.raw_cache is None:
            return None
        a, b = canonical_interval(start, end)
        key = cache_key(self.raw_hash, a, b, self.model_hash)
        def compute():
            import torch
            from ..audio import load_mono
            if self.raw_waveform is None:
                self.raw_waveform = load_mono(self.raw_stem, 16000)
            clip = self.raw_waveform[max(0, round(a * 16000)):round(b * 16000)]
            if clip.numel() < 5600:
                raise ValueError("原轨区间过短")
            return tuple(e._embedding_from_waveform(clip).detach().cpu().numpy()
                         for e in self.encoders)
        try:
            return self.raw_cache.get_or_compute(key, compute)
        except (OSError, RuntimeError, ValueError):
            self.raw_cache = None  # 原轨不可读/不可对齐时整段禁用，不影响主流程
            return None

    def embed_anime(self, start: float, end: float) -> tuple[np.ndarray, np.ndarray] | None:
        """Read only the requested audio interval on a cache miss."""
        if self.anime_encoders is None or self.anime_cache is None:
            return None
        from .anime_embedding import InsufficientSpeech
        a, b = canonical_interval(start, end)
        if b - a < .2:
            return None
        key = cache_key(self.source_hash, a, b, self.anime_hash)
        def compute():
            import torch
            if self.waveform is not None:
                clip = self.waveform[max(0, round(a * 16000)):round(b * 16000)]
            else:
                import soundfile as sf
                with sf.SoundFile(str(self.stem)) as source:
                    first = min(len(source), max(0, round(a * source.samplerate)))
                    last = min(len(source), max(first, round(b * source.samplerate)))
                    source.seek(first)
                    samples = source.read(last - first, dtype="float32", always_2d=True)
                    source_rate = source.samplerate
                clip = torch.from_numpy(samples.mean(axis=1))
                if source_rate != 16000 and clip.numel():
                    from torchaudio.functional import resample
                    clip = resample(clip, source_rate, 16000)
            return tuple(encoder.encode(clip, sample_rate=16000) for encoder in self.anime_encoders)
        try:
            return self.anime_cache.get_or_compute(key, compute)
        except InsufficientSpeech:
            return None

    def physiology_evidence(self, start, end):
        """Measure separation distortion without adding a speaker-identity vote."""
        if self.raw_stem is None:
            return None
        if self._physiology_probe is None:
            from .physiology_runtime import PhysiologyProbe
            try:
                root = self.cache.root.parent / "physiology"
                self._physiology_probe = PhysiologyProbe(
                    self.stem, self.raw_stem, self.source_hash, root,
                    enabled=getattr(self.cache, "enabled", True))
            except (OSError, ValueError):
                return {"available": False, "reason": "paired_audio_unavailable",
                        "reliability": 0., "artifact_confidence": 0.}
        return self._physiology_probe.measure(start, end)

    def vocal_geometry_evidence(self, start, end):
        from ..extraction_guards import VocalTractGeometryGuard
        from .physiology import vocal_tract_frames
        from ..audio import load_mono
        if not hasattr(self, '_geometry_guard'):
            self._geometry_guard = VocalTractGeometryGuard(self.references)
            self._geometry_cache = {}
        key = canonical_interval(start,end)
        if key not in self._geometry_cache:
            if self.waveform is None:
                self.waveform = load_mono(self.stem,16000)
            self._geometry_cache[key] = self._geometry_guard.assess_frames(vocal_tract_frames(
                self.waveform[round(start*16000):round(end*16000)]))
        return self._geometry_cache[key]

    def physiology_diagnostics(self):
        if self._physiology_probe is not None:
            return self._physiology_probe.diagnostics()
        return {"enabled": self.raw_stem is not None, "measured_spans": 0}

    def anime_scores(self, start: float, end: float, profile=None) -> tuple[float, float] | None:
        """Return support measured against the current target references."""
        profile = profile or self.profile
        if profile.anime_char_centroid is None or profile.anime_va_centroid is None:
            return None
        vectors = self.embed_anime(start, end)
        if vectors is None:
            return None
        return (profile_score(vectors[0], profile.anime_char_centroid, profile.anime_char_references),
                profile_score(vectors[1], profile.anime_va_centroid, profile.anime_va_references))

    def anime_decision(self, scores, primary, secondary, tertiary=None, profile=None):
        """Return positive support for a register change without rejecting existing evidence."""
        if scores is None:
            return {"available": False, "strong": False, "register_shift": False}
        profile = profile or self.profile
        char, va = scores
        cf, vf = profile.anime_char_reference_floor, profile.anime_va_reference_floor
        strong = char >= cf + .05 and va >= vf + .05
        register_shift = (primary >= profile.primary_reference_floor + .04
                          and char >= cf + .04 and va >= vf + .04)
        return {"available": True, "strong": strong, "register_shift": register_shift}

    def scores(self, p, s, profile=None):
        profile = profile or self.profile
        target = (profile_score(p, profile.primary_centroid, profile.primary_references),
                  profile_score(s, profile.secondary_centroid, profile.secondary_references))
        negative = [(role, float(unit(p) @ pc), float(unit(s) @ sc))
                    for role, pc, sc in self.negatives]
        return target, negative

    def evidence(self, spans):
        turns = []
        use_tertiary = self.tertiary is not None
        for index, (start, end) in enumerate(spans):
            usable = end - start >= .35 - 1e-9
            p, s = self.embed(start, end) if usable else (np.zeros(192), np.zeros(512))
            t = self.embed3(start, end) if (usable and use_tertiary) else None
            scores, others = self.scores(p, s) if usable else ((0., 0.), [])
            nearest = max(others, key=lambda row: row[1]) if others else (None, -1., -1.)
            anchor = None
            if usable:
                if (scores[0] >= self.profile.primary_reference_floor and
                    scores[1] >= self.profile.secondary_reference_floor and
                    all(scores[0] - o[1] >= 0.06 and scores[1] - o[2] >= 0.06 for o in others)):
                    anchor = "target"
                elif (nearest[1] >= 0.55 and nearest[2] >= 0.42 and
                      nearest[1] - scores[0] >= 0.06 and nearest[2] - scores[1] >= 0.06):
                    anchor = nearest[0]
            turns.append(TurnEvidence(index, float(start), float(end), max(0, end-start-0.04),
                                      p, s, nearest[1] if others else None, nearest[0],
                                      usable=usable, anchor_role=anchor, tertiary=t))
            if (index + 1) % 25 == 0:
                self.progress(f"簇嵌入 {index + 1}/{len(spans)}")
        return turns

    def tertiary_score(self, turn):
        """轮次对第三通道参考质心的相似度；无第三通道时返回 None。"""
        if self.tertiary is None or turn.tertiary is None:
            return None
        return float(unit(turn.tertiary) @ self.profile.tertiary_centroid)

    def analyze(self, spans, *, margin=0.06, temporal=True, keep_narration=True, require_dominant=True):
        turns = self.evidence(spans)
        clusters = cluster_turns(turns)
        exclusions = [(role, p) for role, p, _ in self.negatives]
        secondaries = [(role, s) for role, _, s in self.negatives]
        result = assign_target(turns, clusters, self.profile,
                               exclusions, secondaries,
                               exclusion_audit=getattr(self, "exclusion_audit", None),
                               exclusion_tertiary_centroids=self.tertiary_negatives,
                               margin_floor=margin,
                               require_dominant=require_dominant)
        result = self._apply_tertiary_promotion(turns, result)
        result = self._apply_raw_crosscheck(turns, result)
        local = {}
        target = result.target()
        if keep_narration and target is not None:
            membership = list(result.membership)
            narration = []
            for turn in turns:
                if not self._rescue_eligible(turn) or result.cluster_of[turn.index] != -1:
                    continue
                scores, negative = self.scores(turn.primary, turn.secondary)
                if (scores[0] >= self.profile.primary_reference_floor and
                    scores[1] >= self.profile.secondary_reference_floor and
                    all(n[1] < 0.40 and n[2] < 0.40 for n in negative)):
                    membership[turn.index] = True
                    narration.append(turn.index)
            result = replace(result, membership=tuple(membership), diagnostics={
                **result.diagnostics, "narration_candidates": narration})
        if temporal:
            result, local = self.verify_temporal(turns, result, margin=margin)
        return turns, result, local

    def _apply_tertiary_promotion(self, turns, result):
        """Recover any usable register with independent WavLM/anime support."""
        if self.tertiary is None:
            return result
        floor = self.profile.tertiary_reference_floor
        membership = list(result.membership)
        borderline = (list(result.borderline_membership) if len(result.borderline_membership) == len(turns)
                      else [False] * len(turns))
        promoted = []
        tertiary_scores = dict(result.diagnostics.get("tertiary_scores") or {})
        by_index = {turn.index: turn for turn in turns}
        tertiary_rivals = []
        for cluster in result.clusters:
            if (cluster.cluster_id == result.target_cluster_id
                    or cluster.cluster_id in result.target_cluster_ids
                    or not eligible_competitor(cluster, turns)):
                continue
            vectors = [unit(by_index[index].tertiary) for index in cluster.members
                       if index in by_index and self._rescue_eligible(by_index[index])
                       and by_index[index].tertiary is not None]
            if vectors:
                tertiary_rivals.append((set(cluster.members), unit(np.mean(vectors, axis=0))))
        for turn in turns:
            if not self._rescue_eligible(turn):
                continue
            score = self.tertiary_score(turn)
            if score is None:
                continue
            tertiary_scores[turn.index] = round(score, 5)
            if membership[turn.index]:
                continue
            beaten_by_negative = any(
                score - float(unit(turn.tertiary) @ nc) < 0.02
                for _, nc in self.tertiary_negatives)
            beaten_by_negative |= any(turn.index not in members
                and score - float(unit(turn.tertiary) @ centroid) < .02
                for members, centroid in tertiary_rivals)
            generic, _ = self.scores(turn.primary, turn.secondary)
            beaten_by_negative |= self._rescue_competitor_wins(turn, result, turn.primary, turn.secondary, generic, turns)
            anime = self.anime_scores(turn.start, turn.end) if not borderline[turn.index] else None
            invariant = anime is not None and (
                anime[0] >= self.profile.anime_char_reference_floor + .04
                or anime[1] >= self.profile.anime_va_reference_floor + .04)
            if (score >= floor + 0.02 and not beaten_by_negative
                    and (borderline[turn.index] or invariant)):
                membership[turn.index] = True
                borderline[turn.index] = False
                promoted.append(turn.index)
        diagnostics = {**result.diagnostics, "tertiary_scores": tertiary_scores,
                       "tertiary_floor": floor, "tertiary_promoted": promoted}
        return replace(result, membership=tuple(membership),
                       borderline_membership=tuple(borderline), diagnostics=diagnostics)

    def _apply_raw_crosscheck(self, turns, result):
        """Recover all usable separated turns using aligned raw-audio support."""
        if self.raw_cache is None:
            return result
        membership = list(result.membership)
        borderline = list(result.borderline_membership) if (
            len(result.borderline_membership) == len(turns)) else [False] * len(turns)
        rescued = []
        rescue_scores = {}
        physiology = {}
        for turn in turns:
            if membership[turn.index] or not self._rescue_eligible(turn):
                continue
            raw = self.embed_raw(turn.start, turn.end)
            if raw is None:
                continue
            raw_scores, _ = self.scores(raw[0], raw[1])
            if (raw_scores[0] >= self.profile.primary_reference_floor + 0.04
                    and raw_scores[1] >= self.profile.secondary_reference_floor + 0.02
                    and not self._rescue_competitor_wins(turn, result, raw[0], raw[1], raw_scores, turns)):
                membership[turn.index] = True
                borderline[turn.index] = False
                rescued.append(turn.index)
                rescue_scores[turn.index] = [round(v, 4) for v in raw_scores]
                physiology[str(turn.index)] = self.physiology_evidence(turn.start, turn.end)
        if not rescued:
            return result
        diagnostics = {**result.diagnostics,
                       "raw_rescued_turns": rescued,
                       "raw_rescue_physiology": physiology,
                       "raw_rescue_scores": {str(i): rescue_scores[i] for i in rescued}}
        return replace(result, membership=tuple(membership),
                       borderline_membership=tuple(borderline), diagnostics=diagnostics)

    def _rescue_eligible(self, turn):
        """Singing, overlap and explicit exclusion intervals cannot be rescued."""
        return turn.usable and not any(
            min(turn.end, end) > max(turn.start, start)
            for start, end in getattr(self, 'temporal_blocked_spans', ()))

    def _rescue_competitor_wins(self, turn, result, primary, secondary, scores, turns):
        """A rescue cannot override a positively identified independent speaker."""
        _, rivals = self.scores(primary, secondary)
        rivals += [(str(c.cluster_id), float(unit(primary) @ c.primary_centroid),
                    float(unit(secondary) @ c.secondary_centroid)) for c in result.clusters
                   if turn.index not in c.members and c.cluster_id not in result.target_cluster_ids
                   and c.cluster_id != result.target_cluster_id and eligible_competitor(c, turns)]
        return any(p >= self.profile.primary_reference_floor and s >= self.profile.secondary_reference_floor
                   and p >= scores[0] - .02 and s >= scores[1] - .02 for _, p, s in rivals)

    def verify_temporal(self, turns, candidates, *, margin=.06, bank=(), previous=None):
        """Verify local speech, then revisit uncertain windows with clean episode evidence."""
        from .temporal_mining import TemporalMiner

        return TemporalMiner(self, turns, candidates, margin, bank, previous).run()

    def provenance(self):
        code = [Path(__file__), Path(__file__).with_name("identity_cluster.py"),
                Path(__file__).with_name("anime_embedding.py"),
                Path(__file__).with_name("temporal_mining.py"),
                *[Path(__file__).with_name(name + ".py") for name in (
                    "identity_trajectory", "episode_manifold", "scene_dialogue",
                    "exclusion_reliability", "candidate_topology", "physiology", "physiology_runtime",
                    "feature_pyramid", "pyramid_batching", "score_calibration", "identity_heatmap")],
                Path(__file__).with_name("embedding_cache.py"),
                Path(__file__).with_name("cluster_reporting.py"),
                Path(__file__).with_name("cluster_gate.py"),
                Path(__file__).with_name("bootstrap_bank.py"),
                Path(__file__).with_name("bootstrap_training.py")]
        return {"inputs": self.inputs, "code": [{"path": str(p), "sha256": file_digest(p)} for p in code],
                "reference_paths": list(map(str, self.references)),
                "exclusion_groups": [list(map(str, group)) for group in self.negative_groups],
                "source_sha256": self.source_hash, "encoding_version": self.encoding_version,
                "cache_hits": self.cache.hits, "cache_misses": self.cache.misses,
                "purified_reference_indexes": list(getattr(self, "purified_reference_indexes", ())),
                "anime": {"enabled": self.anime_encoders is not None,
                          "hash": self.anime_hash or None,
                          "cache_hits": self.anime_cache.hits if self.anime_cache else 0,
                          "cache_misses": self.anime_cache.misses if self.anime_cache else 0},
                "tertiary": {"enabled": self.tertiary is not None,
                             "hash": self.tertiary_hash or None,
                             "cache_hits": self.tertiary_cache.hits if self.tertiary_cache else 0,
                             "cache_misses": self.tertiary_cache.misses if self.tertiary_cache else 0},
                "bootstrap": self.bootstrap_provenance,
                "physiology": self.physiology_diagnostics(),
                "cross_bank_size": len(self.cross_bank)}
