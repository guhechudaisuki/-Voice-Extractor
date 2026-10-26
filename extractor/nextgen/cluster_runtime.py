"""Shared offline/live cluster analysis; no human labels enter this module."""
from __future__ import annotations

import hashlib
import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np

from .embedding_cache import EmbeddingCache, cache_key, file_digest
from .identity_cluster import (TargetProfile, TurnEvidence, cluster_turns,
                               assign_target, profile_score)


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def window_spans(start, end, width=1.8, hop=0.9):
    if end - start <= width:
        return [(start, end)]
    starts = list(np.arange(start, end - width, hop)) + [end - width]
    return [(float(a), float(a + width)) for a in starts]


# 连续干净子区间的最短时长：短于该值的碎片不进入 accepted_spans。
MIN_CLEAN_SUBSPAN = 1.15
# 整轮放行的窗口通过率与首尾窗口要求。
WHOLE_TURN_PASS_RATIO = 0.80
# 长句内部换气平滑：夹在两个高分通过窗口之间的微跌窗口视为同人换气。
BREATH_DIP_TOLERANCE = 0.03
BREATH_NEIGHBOR_FLOOR = 0.06
# 窗口级 Cohort 竞争守卫：窗口对目标簇联盟的余量必须压过最强非目标簇。
# 这是拦截"混入长句里的相似女声窗口"的核心防线（AS-Norm 思想）。
WINDOW_COHORT_PRIMARY = 0.055
WINDOW_COHORT_SECONDARY = 0.02
WINDOW_COHORT_STRONG = 0.10
# 短句弱副通道降级：时长 ≤ 该值且副通道贴近下限的会员轮次降入 Tier B，
# 由多通道仲裁（pipeline 的救援严审对应轮次级同类规则）。
SHORT_TURN_SECONDS = 3.0
WEAK_SECONDARY_SLACK = 0.10


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

    auto_negatives: list = []
    cross_bank: tuple = ()

    def __init__(self, stem, references, negative_groups=(), *, device="cuda",
                 cache_enabled=True, encoders=None, progress=None,
                 tertiary_encoder=None, quaternary_encoder=None, raw_stem=None):
        from ..config import CACHE_ROOT, SV_MODEL, CAMPLUS_MODEL
        from ..speaker import SpeakerVerifier, CAMPlusVerifier
        self.stem = Path(stem)
        self.references = sorted(map(Path, references))
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
        # ── 第三通道（强化后的 WavLM）─────────────────────────
        # 主/副通道缓存与哈希保持原样（历史 .npz 100% 命中）；WavLM 使用独立
        # 轻量缓存，权重原地强化后自动失效重算，互不干扰。
        self.tertiary = tertiary_encoder
        self.owns_tertiary = False
        self.tertiary_hash = ""
        self.tertiary_cache = None
        self.tertiary_negatives = []
        if self.tertiary is not None:
            from ..config import WAVLM_SV_MODEL
            weight = WAVLM_SV_MODEL / "pytorch_model.bin"
            digest = file_digest(weight) if weight.exists() else "external"
            self.tertiary_hash = hashlib.sha256(digest.encode()).hexdigest()[:16]
            self.tertiary_cache = EmbeddingCache(CACHE_ROOT / "wavlm_embeddings",
                                                 enabled=cache_enabled)
        # ── 第四通道（WeSpeaker-ResNet34-LM）仲裁 ──────────────
        # 只对带高危救援标记且进入 Tier A 的候选做否决票：异构大间隔模型
        # 与前三个模型的相关性低，是防止救援路径假阳性混入主输出的最后一道
        # 多模型投票。参考分低于下限（相对参考自评分）→ 降级 Tier B。
        self.quaternary = quaternary_encoder
        self.quaternary_floor = 0.0
        if self.quaternary is not None:
            import torch as _torch
            from ..audio import load_mono
            ref_embeddings = self.quaternary.embeddings_from_waveforms(
                [load_mono(path, 16000) for path in self.references])
            self.quaternary_centroid = _torch.nn.functional.normalize(
                ref_embeddings.mean(dim=0), dim=0)
            self_scores = ref_embeddings @ self.quaternary_centroid
            # 上限封顶 0.66：防止同质化平静参考把自评分顶到 0.78+ 从而误杀同角色的情绪/对话句
            self.quaternary_floor = float(min(0.66, max(0.58, self_scores.min().item() - 0.06)))
        # ── 双轨（Stem+Raw）边缘复核 ──────────────────────────
        # UVR 剥 BGM 常把轻声齿音/尾音当乐器抹掉，导致真目标在分离轨上掉分。
        # 对分离轨副通道贴线的边缘轮次，用时间对齐的原轨嵌入交叉复核：
        # 原轨双通道显著达标 → 判定 UVR 伪影误杀并救回（只救不杀）。
        self.raw_stem = Path(raw_stem) if raw_stem else None
        self.raw_hash = ""
        self.raw_cache = None
        self.raw_waveform = None
        if self.raw_stem is not None and self.raw_stem.exists():
            self.raw_hash = file_digest(self.raw_stem)[:16]
            self.raw_cache = EmbeddingCache(CACHE_ROOT / "raw_embeddings",
                                            enabled=cache_enabled)
        self.profile, self.negatives = self._profiles()
        self.bootstrap_provenance = None

    def _load_auto_negatives(self):
        """加载用户标注沉淀的负例原型库（模型强化时自动更新）。

        余量因子 0.75：学习型原型比人工排除参考更噪，簇级资格判定按
        margin_floor * 0.75 生效；不参与窗口级否决（避免召回回退）。
        """
        import numpy as np
        from ..config import WAVLM_SV_MODEL
        path = Path(WAVLM_SV_MODEL) / "auto_negatives.npz"
        if not path.is_file():
            return []
        try:
            with np.load(path) as data:
                primary = np.asarray(data["primary"], dtype=np.float64)
                secondary = np.asarray(data["secondary"], dtype=np.float64)
        except (OSError, ValueError, KeyError):
            return []
        if primary.ndim != 2 or secondary.ndim != 2 or not len(primary):
            return []
        roles = [(f"auto_neg_{i}", unit(primary[i]), unit(secondary[i]))
                 for i in range(min(len(primary), len(secondary)))]
        self.progress(f"已加载习得排除原型 {len(roles)} 个（来自历史标注）")
        return roles

    def bank_context(self):
        return {"encoding_version": self.encoding_version,
                "models": [item["sha256"] for item in self.inputs[1:3]],
                "references": sorted(file_digest(p) for p in self.references),
                "negative_groups": sorted(sorted(file_digest(p) for p in group)
                                          for group in self.negative_groups)}

    def close(self):
        if self.owns_encoders:
            for encoder in self.encoders:
                encoder.close()
        if self.owns_tertiary and self.tertiary is not None and hasattr(self.tertiary, "close"):
            self.tertiary.close()

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
        """四模型仲裁：带高危救援标记的 Tier A 候选需通过 WeSpeaker 否决票。

        参考分 < 下限（参考自评分最低值 - 0.05）→ 降入 Tier B 边缘池。
        只降级救援路径候选（其余 Tier A 不受影响），返回降级数量。
        """
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

    def _profiles(self):
        if not self.references:
            raise ValueError("Cluster analysis requires target references")
        vectors = [self._file_embedding(p) for p in self.references]
        p, s = [np.stack([v[channel] for v in vectors]) for channel in (0, 1)]
        pc, sc = unit(p.mean(axis=0)), unit(s.mean(axis=0))
        profile = TargetProfile(
            pc, sc,
            float(np.clip(np.quantile(p @ pc, 0.10) - 0.22, 0.52, 0.68)),
            float(np.clip(np.quantile(s @ sc, 0.10) - 0.26, 0.48, 0.64)),
            primary_references=p, secondary_references=s,
        )
        if self.tertiary is not None:
            tv = np.stack([self._file_embedding3(path) for path in self.references])
            profile = replace(
                profile,
                tertiary_centroid=unit(tv.mean(axis=0)),
                tertiary_reference_floor=float(np.clip(
                    np.quantile(tv @ unit(tv.mean(axis=0)), 0.10) - 0.06, 0.74, 0.88)),
            )
        negatives = []
        for group in self.negative_groups:
            v = [self._file_embedding(path) for path in group]
            # Stable role identity derives from files, not enumeration order.
            role = hashlib.sha256("|".join(sorted(file_digest(path) for path in group)).encode()).hexdigest()[:16]
            negatives.append((role, unit(np.mean([x[0] for x in v], axis=0)),
                              unit(np.mean([x[1] for x in v], axis=0))))
        self.tertiary_negatives = []
        if self.tertiary is not None:
            for group in self.negative_groups:
                tv = np.stack([self._file_embedding3(path) for path in group])
                role = hashlib.sha256("|".join(sorted(file_digest(path) for path in group)).encode()).hexdigest()[:16]
                self.tertiary_negatives.append((role, unit(tv.mean(axis=0))))
        return profile, negatives

    def embed(self, start, end):
        # Use the same exact cuts in offline and pipeline analysis.
        a, b = float(start) + 0.02, float(end) - 0.02
        key = cache_key(self.source_hash, a, b, self.model_hash)
        def compute():
            from ..audio import load_mono
            if self.waveform is None:
                self.waveform = load_mono(self.stem, 16000)
            clip = self.waveform[max(0, round(a * 16000)):round(b * 16000)]
            return tuple(e._embedding_from_waveform(clip).detach().cpu().numpy() for e in self.encoders)
        return self.cache.get_or_compute(key, compute)

    def embed3(self, start, end):
        """第三通道（WavLM）嵌入；独立内容寻址缓存，强化后自动重算。"""
        if self.tertiary is None:
            return None
        a, b = float(start) + 0.02, float(end) - 0.02
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
        a, b = float(start) + 0.02, float(end) - 0.02
        key = cache_key(self.raw_hash, a, b, self.model_hash)
        def compute():
            import torch
            from ..audio import load_mono
            if self.raw_waveform is None:
                self.raw_waveform = load_mono(self.raw_stem, 16000)
            clip = self.raw_waveform[max(0, round(a * 16000)):round(b * 16000)]
            if clip.numel() < 8000:
                raise ValueError("原轨区间过短")
            return tuple(e._embedding_from_waveform(clip).detach().cpu().numpy()
                         for e in self.encoders)
        try:
            return self.raw_cache.get_or_compute(key, compute)
        except (OSError, RuntimeError, ValueError):
            self.raw_cache = None  # 原轨不可读/不可对齐时整段禁用，不影响主流程
            return None

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
            usable = end - start - 0.04 >= 0.55
            p, s = self.embed(start, end) if usable else (np.zeros(192), np.zeros(512))
            t = self.embed3(start, end) if (usable and use_tertiary) else None
            scores, others = self.scores(p, s)
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
        auto = self._load_auto_negatives()
        exclusions = [(role, p) for role, p, _ in self.negatives] + [
            (role, p) for role, p, _ in auto]
        secondaries = [(role, s) for role, _, s in self.negatives] + [
            (role, s) for role, _, s in auto]
        scales = [1.0] * len(self.negatives) + [0.75] * len(auto)
        result = assign_target(turns, clusters, self.profile,
                               exclusions, secondaries,
                               exclusion_scales=scales,
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
                if not turn.usable or result.cluster_of[turn.index] != -1:
                    continue
                scores, negative = self.scores(turn.primary, turn.secondary)
                if (scores[0] >= self.profile.primary_reference_floor and
                    scores[1] >= self.profile.secondary_reference_floor and
                    all(n[1] < 0.40 and n[2] < 0.40 for n in negative)):
                    membership[turn.index] = True
                    narration.append(turn.index)
            result = replace(result, membership=tuple(membership), diagnostics={
                **result.diagnostics, "narration_candidates": narration})
        if temporal and target is not None:
            result, local = self.verify_temporal(turns, result, margin=margin)
        return turns, result, local

    def _apply_tertiary_promotion(self, turns, result):
        """WavLM 第三通道边缘晋升：强匹配的边缘轮次升为核心成员。

        用户在 UI 里强化过的 WavLM 在这里真正参与门控：边缘池轮次若对强化后
        的参考质心显著高分（且不贴近任何排除角色），则晋升进核心成员，随后的
        滑窗核验仍照常执行。没有第三通道时原样返回。
        """
        if self.tertiary is None or not result.borderline_membership:
            return result
        floor = self.profile.tertiary_reference_floor
        membership = list(result.membership)
        borderline = list(result.borderline_membership)
        promoted = []
        tertiary_scores = dict(result.diagnostics.get("tertiary_scores") or {})
        for turn in turns:
            if not borderline[turn.index] or membership[turn.index]:
                continue
            score = self.tertiary_score(turn)
            if score is None:
                continue
            tertiary_scores[turn.index] = round(score, 5)
            beaten_by_negative = any(
                score - float(unit(turn.tertiary) @ nc) < 0.02
                for _, nc in self.tertiary_negatives)
            if score >= floor + 0.02 and not beaten_by_negative:
                membership[turn.index] = True
                borderline[turn.index] = False
                promoted.append(turn.index)
        diagnostics = {**result.diagnostics, "tertiary_scores": tertiary_scores,
                       "tertiary_floor": floor, "tertiary_promoted": promoted}
        return replace(result, membership=tuple(membership),
                       borderline_membership=tuple(borderline), diagnostics=diagnostics)

    def _apply_raw_crosscheck(self, turns, result):
        """Stem+Raw 双轨边缘复核：原轨显著达标 → UVR 伪影误杀的救回。

        仅作用于分离轨副通道贴线的边缘轮次（不碰核心成员，只救不杀）：
        原轨双通道同时高出参考下限时，判定该轮次是 UVR 伪影受害者，
        升为核心成员并持有原轨证书（verify_temporal 不再对其跑分离轨滑窗）。
        """
        if self.raw_cache is None:
            return result
        membership = list(result.membership)
        borderline = list(result.borderline_membership) if (
            len(result.borderline_membership) == len(turns)) else [False] * len(turns)
        rescued = []
        rescue_scores = {}
        for turn in turns:
            if not borderline[turn.index] or membership[turn.index] or not turn.usable:
                continue
            scores, _negatives = self.scores(turn.primary, turn.secondary)
            if scores[1] >= self.profile.secondary_reference_floor + 0.04:
                continue  # 分离轨副通道并不弱，不是 UVR 伪影的形状
            raw = self.embed_raw(turn.start, turn.end)
            if raw is None:
                continue
            raw_scores, _ = self.scores(raw[0], raw[1])
            if (raw_scores[0] >= self.profile.primary_reference_floor + 0.04
                    and raw_scores[1] >= self.profile.secondary_reference_floor + 0.02):
                membership[turn.index] = True
                borderline[turn.index] = False
                rescued.append(turn.index)
                rescue_scores[turn.index] = [round(v, 4) for v in raw_scores]
        if not rescued:
            return result
        diagnostics = {**result.diagnostics,
                       "raw_rescued_turns": rescued,
                       "raw_rescue_scores": {str(i): rescue_scores[i] for i in rescued}}
        return replace(result, membership=tuple(membership),
                       borderline_membership=tuple(borderline), diagnostics=diagnostics)

    def verify_temporal(self, turns, candidates, *, margin=.06, bank=(), previous=None):
        """Verify fixed candidates; expanded references never select a cluster.

        窗口核验对目标簇联盟（全部声线子簇质心）进行；长句内部换气微跌平滑；
        不再要求父轮次 100% 窗口全过——连续通过的窗口合并为干净子区间
        （≥ MIN_CLEAN_SUBSPAN），供候选句级门控裁切使用。
        Earlier certificates remain valid against their recorded bank.
        """
        from .bootstrap_bank import independent_vectors, merge_profile, evidence_spans
        target_union = candidates.target_clusters()
        if not target_union:
            return candidates, {}
        union_ids = {c.cluster_id for c in target_union}
        # 冒认者队列 = "不像目标"的簇。达标簇（含未入盟的次级声线簇）属于
        # 目标家族；参考分达到强参考水平的簇即使未达标也不得作为竞争对象
        # ——否则真目标窗口会被同说话人的另一个子簇误杀。
        family_ids = set(candidates.diagnostics.get("qualified_cluster_ids") or ()) | union_ids
        family_ids |= {
            c.cluster_id for c in candidates.clusters
            if c.primary_target_score >= 0.66 and c.secondary_target_score >= 0.60
        }
        non_union = [c for c in candidates.clusters if c.cluster_id not in family_ids]

        def cohort_ok(p, s):
            """窗口必须在竞争余量上压过最强非目标簇（Cohort AS-Norm）。

            双通道之一显著占优即可：主通道占优（cm_p 达标且 cm_s 不崩），
            或副通道显著占优（cm_s >= 0.10 且 cm_p 非负）。
            """
            if not non_union:
                return True
            u_p, u_s = unit(p), unit(s)
            tcp = max(float(u_p @ c.primary_centroid) for c in target_union)
            tcs = max(float(u_s @ c.secondary_centroid) for c in target_union)
            othp = max(float(u_p @ c.primary_centroid) for c in non_union)
            oths = max(float(u_s @ c.secondary_centroid) for c in non_union)
            cm_p, cm_s = tcp - othp, tcs - oths
            return ((cm_p >= WINDOW_COHORT_PRIMARY and cm_s >= WINDOW_COHORT_SECONDARY)
                    or (cm_p >= 0.0 and cm_s >= WINDOW_COHORT_STRONG)
                    or cm_p >= WINDOW_COHORT_STRONG + 0.045)

        def analyzer_self_embed(a, b):
            p_e, s_e = self.embed(a, b)
            p_prof = profile_score(unit(p_e), self.profile.primary_centroid,
                                   self.profile.primary_references)
            s_prof = profile_score(unit(s_e), self.profile.secondary_centroid,
                                   self.profile.secondary_references)
            return p_prof, s_prof

        def turn_cohort_margin(turn):
            """轮次整体对目标簇联盟的竞争余量；无非目标簇时返回 +inf。"""
            if not non_union:
                return float("inf")
            u_p = unit(turn.primary)
            tcp = max(float(u_p @ c.primary_centroid) for c in target_union)
            othp = max(float(u_p @ c.primary_centroid) for c in non_union)
            return tcp - othp

        membership = list(candidates.membership)
        borderline = list(candidates.borderline_membership) if (
            len(candidates.borderline_membership) == len(turns)) else [False] * len(turns)
        # 原轨证书轮次：UVR 伪影受害者已由原轨双通道复核救回，分离轨滑窗
        # 对其没有意义（分离伪影正是被救的原因），直接持有整轮区间。
        raw_rescued = set(candidates.diagnostics.get("raw_rescued_turns") or ())
        local, supports = {}, {}
        diagnostics = {**candidates.diagnostics,
            "profile_floors": {
                "primary": self.profile.primary_reference_floor,
                "secondary": self.profile.secondary_reference_floor},
        }
        turn_clean_spans = {}
        demoted = []
        cross_rescued = []

        def check(profile, *, strict=False):
            def window_ok(a, b, p, s):
                u_p, u_s = unit(p), unit(s)
                p_prof = profile_score(u_p, profile.primary_centroid,
                                       profile.primary_references)
                s_prof = profile_score(u_s, profile.secondary_centroid,
                                       profile.secondary_references)
                profile_ok = (p_prof >= profile.primary_reference_floor - .02
                              and s_prof >= profile.secondary_reference_floor - .02)
                # 质心一致分支允许情绪声线在簇内漂移，但仍需软参考下限兜底：
                # 远离参考的窗口（如混入的他人插话）不能只靠贴近簇质心放行。
                # 软兜底维持 -0.10：实测把副通道收紧到 -0.045 会把大量安静
                # 真目标窗口（s_prof 0.40~0.46）一并误杀、 Tier A 崩掉 1/4，
                # 而目标泄漏段会经裁切子段存活，收益为零。该类泄漏改由
                # WeSpeaker 四模型仲裁否决。扩充参考（bank）核验一律 strict。
                if not strict:
                    coherent = any(
                        float(u_p @ c.primary_centroid) >= c.primary_median - .20
                        and float(u_s @ c.secondary_centroid) >= profile.secondary_reference_floor - .02
                        for c in target_union)
                    if (coherent
                            and p_prof >= profile.primary_reference_floor - .10
                            and s_prof >= profile.secondary_reference_floor - .10):
                        profile_ok = True
                score, negative = self.scores(p, s, profile)
                # Stage 18 双通道互补排除余量：允许主通道极强时副通道贴线（容差 0.03），
                # 防止混合排除组（如含目标自身背景音的杂谈簇）因副通道微差 0.01 一票否决真目标。
                margins_ok = all(
                    (score[0] - n[1] >= margin and score[1] - n[2] >= margin)
                    or (score[0] - n[1] >= margin + 0.06 and score[1] - n[2] >= margin - 0.01)
                    or (score[0] - n[1] >= margin - 0.01 and score[1] - n[2] >= margin + 0.06)
                    for n in negative)
                return bool(profile_ok and margins_ok and cohort_ok(p, s)), score

            rows = []
            for a, b, p, s in windows:
                passes, score = window_ok(a, b, p, s)
                cm_p = cm_s = None
                if non_union:
                    tcp = max(float(unit(p) @ c.primary_centroid) for c in target_union)
                    tcs = max(float(unit(s) @ c.secondary_centroid) for c in target_union)
                    othp = max(float(unit(p) @ c.primary_centroid) for c in non_union)
                    oths = max(float(unit(s) @ c.secondary_centroid) for c in non_union)
                    cm_p, cm_s = round(tcp - othp, 4), round(tcs - oths, 4)
                rows.append({"span": [a, b], "primary": score[0],
                             "secondary": score[1], "accepted": bool(passes),
                             "cm_p": cm_p, "cm_s": cm_s})
            # 长句内部换气平滑：两侧高分通过、自身仅微跌且不偏向排除角色的
            # 窗口视为同人换气。
            neighbor_floor = profile.primary_reference_floor + BREATH_NEIGHBOR_FLOOR
            for k in range(1, len(rows) - 1):
                if rows[k]["accepted"]:
                    continue
                left, mid, right = rows[k - 1], rows[k], rows[k + 1]
                if not (left["accepted"] and right["accepted"]):
                    continue
                if (min(left["primary"], right["primary"]) >= neighbor_floor
                        and mid["primary"] >= min(left["primary"], right["primary"]) - BREATH_DIP_TOLERANCE
                        and mid["secondary"] >= profile.secondary_reference_floor - .12):
                    mid["accepted"] = True
                    mid["smoothed"] = True
            return rows

        def extract_spans(turn, rows):
            """整轮通过率高且首尾干净 → 整轮；否则合并连续通过窗口为子区间。"""
            accepted = [row["accepted"] for row in rows]
            if not accepted or not any(accepted):
                return []
            if (sum(accepted) / len(accepted) >= WHOLE_TURN_PASS_RATIO
                    and accepted[0] and accepted[-1]):
                return [[turn.start, turn.end]]
            spans, run = [], None
            for row, ok in zip(rows, accepted):
                if ok:
                    run = [row["span"][0], row["span"][1]] if run is None else [run[0], row["span"][1]]
                elif run is not None:
                    if run[1] - run[0] >= MIN_CLEAN_SUBSPAN:
                        spans.append(run)
                    run = None
            if run is not None and run[1] - run[0] >= MIN_CLEAN_SUBSPAN:
                spans.append(run)
            return spans

        for turn in turns:
            key = str(turn.index)
            if not membership[turn.index]:
                continue
            if turn.index in raw_rescued:
                turn_clean_spans[key] = [[turn.start, turn.end]]
                local[key] = []
                continue
            previous_demoted = set(
                (previous[0].diagnostics.get("short_weak_demoted") or ())
                if previous else ())
            if key in {str(i) for i in previous_demoted}:
                # 短句弱副通道降级在自举轮次间保持：扩充参考不得把已被
                # 降级的轮次重新拉回主输出。
                membership[turn.index] = False
                borderline[turn.index] = True
                demoted.append(turn.index)
                local[key] = previous[1].get(key, [])
                turn_clean_spans[key] = []
                continue
            if previous and previous[0].membership[turn.index]:
                local[key] = previous[1][key]
                if key in previous[0].diagnostics.get("bootstrap_supports", {}):
                    supports[key] = previous[0].diagnostics["bootstrap_supports"][key]
                clean = previous[0].diagnostics.get("turn_clean_spans", {}).get(key)
                if clean:
                    turn_clean_spans[key] = clean
                continue
            windows = [(a, b, *self.embed(a, b)) for a, b in window_spans(turn.start, turn.end)]

            rows = check(self.profile)
            if bank and not all(row["accepted"] for row in rows):
                independent = independent_vectors(bank, self.source_hash, turn.start, turn.end)
                if independent:
                    expanded = check(merge_profile(self.profile, independent), strict=True)
                    if all(row["accepted"] for row in expanded):
                        supports[key] = evidence_spans(independent)
                        rows = [{**row, "reference": "bootstrap"} for row in expanded]
            if not all(row["accepted"] for row in rows) and self.cross_bank:
                # Stage 12：跨集高纯度共享池。与单文件自举同机制——合并剖面
                # 一半权重锚在用户原始参考（merge_profile 保持参考下限不变），
                # 且要求全部窗口在合并剖面下通过才放行，绝不降低任何阈值。
                cross_profile = merge_profile(self.profile, list(self.cross_bank))
                expanded = check(cross_profile, strict=True)
                if all(row["accepted"] for row in expanded):
                    rows = [{**row, "reference": "cross_episode"} for row in expanded]
                    cross_rescued.append(turn.index)
            clean = extract_spans(turn, rows)
            if clean and turn.duration <= SHORT_TURN_SECONDS:
                # Stage 17：日常对话台词常在 1.2s~2.9s 之间。若短句具备以下任一强目标证据，
                # 则豁免短句降级，保留其 clean 区间进入 Tier A。
                turn_scores, _ = self.scores(turn.primary, turn.secondary)
                p_slack = turn_scores[0] - self.profile.primary_reference_floor
                slack = turn_scores[1] - self.profile.secondary_reference_floor
                partial_windows = any(
                    span != [turn.start, turn.end] for span in clean)
                t_score = self.tertiary_score(turn)
                t_strong = (t_score is not None
                            and t_score >= self.profile.tertiary_reference_floor + 0.03
                            and slack >= -0.02)
                accepted_rows = [r for r in rows if r.get("accepted")]
                cm_s_strong = bool(accepted_rows) and all(
                    isinstance(r.get("cm_s"), (int, float)) and r["cm_s"] >= 0.25
                    and isinstance(r.get("cm_p"), (int, float)) and r["cm_p"] >= 0.12
                    for r in accepted_rows)
                # Stage 17 豁免前先过局部分类器否决：高相似负例短句只有
                # 局部分类器能区分，其他通道都会给高分。
                strong_short_dialogue = (
                    p_slack >= 0.12 and cm_s_strong and t_strong
                )
                if not strong_short_dialogue and (
                        slack < WEAK_SECONDARY_SLACK / 2
                        or (slack < WEAK_SECONDARY_SLACK
                            and (turn_cohort_margin(turn) < WINDOW_COHORT_PRIMARY
                                 or partial_windows))):
                    borderline[turn.index] = True
                    demoted.append(turn.index)
                    clean = []
                    for row in rows:
                        row["accepted"] = False
                        row["demoted"] = True
            if not clean:
                # 簇级会员但全部窗口未过：不直接丢弃，进 Tier B 待试听。
                borderline[turn.index] = True
            elif clean and len(clean) == 1 and turn.duration <= SHORT_TURN_SECONDS + 1.5:
                # Stage 19：边界窗口精细剥离——当轮次仅提取出 1 个高分 1.8s 干净
                # 子段且其双通道分数 >= floor + 0.06 时，保留该干净核心段不被
                # partial_windows 降级逻辑抹杀（脏的那半秒已经在 extract_spans 里切掉了）。
                core_span = clean[0]
                core_a, core_b = core_span
                core_p, core_s = analyzer_self_embed(core_a, core_b)
                if (core_p >= self.profile.primary_reference_floor + 0.06
                        and core_s >= self.profile.secondary_reference_floor + 0.04):
                    pass  # 干净核心段有效，保留 clean 不降级
                else:
                    borderline[turn.index] = True
            turn_clean_spans[key] = clean
            membership[turn.index] = bool(clean)
            local[key] = rows
            if (turn.index + 1) % 25 == 0:
                self.progress(f"簇内局部核验 {turn.index + 1}/{len(turns)}")

        accepted_spans = []
        for turn in turns:
            if membership[turn.index]:
                spans = turn_clean_spans.get(str(turn.index)) or [[turn.start, turn.end]]
                accepted_spans.extend(spans)
        diagnostics["temporal_rejected_count"] = sum(candidates.membership) - sum(membership)
        diagnostics["accepted_spans"] = sorted(accepted_spans)
        diagnostics["turn_clean_spans"] = turn_clean_spans
        if demoted:
            diagnostics["short_weak_demoted"] = demoted
        if cross_rescued:
            diagnostics["cross_episode_rescued_turns"] = cross_rescued
        if bank:
            diagnostics["bootstrap_supports"] = supports
        return replace(candidates, membership=tuple(membership),
                       borderline_membership=tuple(borderline),
                       diagnostics=diagnostics), local

    def provenance(self):
        code = [Path(__file__), Path(__file__).with_name("identity_cluster.py"),
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
                "tertiary": {"enabled": self.tertiary is not None,
                             "hash": self.tertiary_hash or None,
                             "cache_hits": self.tertiary_cache.hits if self.tertiary_cache else 0,
                             "cache_misses": self.tertiary_cache.misses if self.tertiary_cache else 0},
                "bootstrap": self.bootstrap_provenance,
                "cross_bank_size": len(self.cross_bank)}
