"""Cluster episode turns and select target speakers using independent channel evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

from .exclusion_reliability import margin_consensus

Channel = Literal["primary", "secondary"]


def cohort_as_norm_z(vec: np.ndarray, target_centroids: Sequence[np.ndarray],
                     impostor_centroids: Sequence[np.ndarray], top_k: int = 5) -> float | None:
    """Return a probe-side cohort Z-score with a variance floor of .04."""
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    def normalized(value):
        value = np.asarray(value, dtype=np.float64)
        if value.ndim != 1 or not value.size or not np.isfinite(value).all():
            raise ValueError("cohort embeddings must be finite non-empty vectors")
        norm = float(np.linalg.norm(value))
        if norm <= 1e-12:
            raise ValueError("cohort embeddings must be non-zero")
        return value / norm
    probe = normalized(vec)
    targets = [normalized(v) for v in target_centroids]
    impostors = [normalized(v) for v in impostor_centroids]
    if any(v.shape != probe.shape for v in [*targets, *impostors]):
        raise ValueError("cohort embedding dimensions differ")
    if not targets or not impostors:
        return None
    target = max(float(probe @ v) for v in targets)
    cohort = np.sort([float(probe @ v) for v in impostors])[-top_k:]
    return float((target - cohort.mean()) / max(float(cohort.std()), .04))


def cohort_z_probability(z: float | None) -> float:
    """Calibrated confidence, with neutral .5 for unavailable cohort evidence."""
    if z is None:
        return .5
    if not np.isfinite(z):
        raise ValueError("cohort z must be finite")
    return float(1. / (1. + np.exp(-np.clip((z - 1.5) * 1.2, -30., 30.))))


def cohort_as_norm_score(vec: np.ndarray, target_centroids: Sequence[np.ndarray],
                        impostor_centroids: Sequence[np.ndarray], top_k: int = 5) -> float:
    """Return sigmoid((z - 1.5) * 1.2); see cohort_as_norm_z for the Z-score."""
    return cohort_z_probability(cohort_as_norm_z(vec, target_centroids, impostor_centroids, top_k))

# Primary affinity drives clustering; the secondary channel can veto a merge.
LINK_PRIMARY = 0.50
SECONDARY_VETO_FLOOR = 0.05

# Retained as API defaults; acoustic support, not sample count, qualifies a role.
MIN_TARGET_CLUSTER_MEMBERS = 1
MIN_CLUSTER_MEMBERS = MIN_TARGET_CLUSTER_MEMBERS
MIN_CLUSTER_SPEECH_SECONDS = 0.0


def eligible_competitor(cluster, turns):
    """Target discovery is permissive; a rival needs independent speech support."""
    if len(cluster.members) >= 2 and cluster.total_speech_seconds >= 2.5:
        return True
    members = set(cluster.members)
    return any(t.index in members and t.anchor_role not in (None, 'target') for t in turns)

# Target similarity must exceed the nearest excluded role by this margin.
TARGET_MARGIN_FLOOR = 0.06

# Member scores are compared with the cluster median and reference floors.
CLUSTER_OUTLIER_DROP = 0.18

# Average linkage needs a separate minimum-cohesion check.
CLUSTER_COHESION_FLOOR = 0.30

# Compatible voice registers can join the same target family.
UNION_PRIMARY_AFFINITY = 0.72
UNION_SECONDARY_AFFINITY = 0.70
# Strong reference matches still need minimum cross-cluster affinity.
UNION_REFERENCE_BONUS = 0.10
UNION_MARGIN_BONUS = 0.04
UNION_REFERENCE_PRIMARY_AFFINITY = 0.55
UNION_REFERENCE_SECONDARY_AFFINITY = 0.50
# Absolute reference support can establish a distant voice register.
UNION_STRONG_PRIMARY = 0.66
UNION_STRONG_SECONDARY = 0.60
# 与核心簇质心低于该亲和的达标簇是"另一位说话人"，仅在分数咬得很紧时仲裁。
RIVAL_PRIMARY_AFFINITY = 0.68
RIVAL_SECONDARY_AFFINITY = 0.65
RIVAL_SCORE_GAP = 0.015

# Non-target clusters provide the episode cohort.
COHORT_MARGIN_FLOOR = 0.01
# 竞争守卫：主通道明显败给非目标簇且副通道也不占优 → 判为混入外人。
COHORT_PRIMARY_GUARD = 0.01
COHORT_SECONDARY_GUARD = 0.02
# 边缘召回：未达核心准入但差距有限，供 Tier B 捡漏与滑窗复核。
BORDERLINE_SLACK = 0.10
BORDERLINE_COHORT_MARGIN = 0.0
BORDERLINE_SECONDARY_SLACK = 0.12

# 历史诊断接口保留此阈值；实际合并约束使用双通道确认的 anchor_role，
# 只禁止不同已知身份合并，同一排除角色允许形成自己的簇。
EXCLUSION_LINK_FLOOR = 0.55


@dataclass(frozen=True)
class TurnEvidence:
    """一个说话轮次的全部身份证据。index 即序号，全局稳定不可变。"""

    index: int
    start: float
    end: float
    speech_seconds: float
    primary: np.ndarray
    secondary: np.ndarray
    exclusion_max: float | None = None
    exclusion_role: str | None = None
    single_turn_accepted: bool = False
    usable: bool = True
    # Assigned only from independent target/exclusion profiles. Unknown remains None.
    anchor_role: str | None = None
    # 可选第三通道（强化后的 WavLM）证据；离线回放与救援仲裁使用。
    tertiary: np.ndarray | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class Cluster:
    """一个簇 = 程序认为的一个说话人。"""

    cluster_id: int
    members: tuple[int, ...]
    primary_centroid: np.ndarray
    secondary_centroid: np.ndarray
    primary_cohesion: float
    primary_median: float
    total_speech_seconds: float
    target_score: float = -1.0
    primary_target_score: float = -1.0
    secondary_target_score: float = -1.0
    nearest_other_score: float = -1.0
    nearest_other_role: str | None = None
    target_margin: float = -1.0
    # 副通道对同一最近排除角色的余量（与 target_margin 成对比较）。
    secondary_target_margin: float = -1.0
    is_target: bool = False
    outliers: tuple[int, ...] = ()
    posterior_primary: np.ndarray | None = None
    posterior_secondary: np.ndarray | None = None
    prior_weight: float = 0.


@dataclass(frozen=True)
class AssignResult:
    """整集的身份判决结果。"""

    clusters: tuple[Cluster, ...]
    target_cluster_id: int | None
    membership: tuple[bool, ...]
    cluster_of: tuple[int, ...]
    reason: str
    outliers: tuple[int, ...] = ()
    diagnostics: dict = field(default_factory=dict)
    # 同一目标角色的全部声线子簇（日常 / 独白 / 情绪）。
    target_cluster_ids: tuple[int, ...] = ()
    # 属于目标簇家族但置信度处于边缘的轮次（Tier B 捡漏池）。
    borderline_membership: tuple[bool, ...] = ()

    def target(self) -> Cluster | None:
        if self.target_cluster_id is None:
            return None
        return next(
            (c for c in self.clusters if c.cluster_id == self.target_cluster_id),
            None,
        )

    def target_clusters(self) -> tuple[Cluster, ...]:
        wanted = set(self.target_cluster_ids) or (
            {self.target_cluster_id} if self.target_cluster_id is not None else set()
        )
        return tuple(c for c in self.clusters if c.cluster_id in wanted)


@dataclass(frozen=True)
class TargetProfile:
    """目标声纹参考。可以来自用户文件，也可以来自自举库。"""

    primary_centroid: np.ndarray
    secondary_centroid: np.ndarray
    primary_reference_floor: float
    secondary_reference_floor: float
    source: str = "user_reference"
    # Unit reference rows preserve distinct voice registers.
    primary_references: np.ndarray | None = None
    secondary_references: np.ndarray | None = None
    # 可选第三通道（强化后的 WavLM）质心与准入下限。
    tertiary_centroid: np.ndarray | None = None
    tertiary_reference_floor: float = 0.72
    quaternary_centroid: np.ndarray | None = None
    quaternary_reference_floor: float = 0.58
    anime_char_centroid: np.ndarray | None = None
    anime_va_centroid: np.ndarray | None = None
    anime_char_reference_floor: float = 0.45
    anime_va_reference_floor: float = 0.35
    anime_char_references: np.ndarray | None = None
    anime_va_references: np.ndarray | None = None
    tertiary_references: np.ndarray | None = None


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(vector).all() or norm <= 1e-12:
        raise ValueError("嵌入向量长度为零")
    return vector / norm


def _stack(vectors: Sequence[np.ndarray]) -> np.ndarray:
    """把一组向量堆成 [n, d] 的单位向量矩阵。"""
    if not vectors:
        raise ValueError("需要至少一个向量")
    matrix = np.stack([np.asarray(v, dtype=np.float64).reshape(-1) for v in vectors])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms <= 1e-12] = 1.0
    return matrix / norms


def _unit_rows(matrix: np.ndarray | None) -> np.ndarray | None:
    """把参考矩阵规整为单位行向量；行数不足时返回 None（退化为质心打分）。"""
    if matrix is None:
        return None
    rows = np.asarray(matrix, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] < 2 or not np.isfinite(rows).all():
        return None
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if float(norms.min()) <= 1e-12:
        return None
    return rows / norms


def profile_score(vector: np.ndarray, centroid: np.ndarray,
                  references: np.ndarray | None = None) -> float:
    """Use the stronger of centroid similarity and the discounted top-two reference mean."""
    u = _unit(vector)
    c_score = float(u @ _unit(centroid))
    rows = _unit_rows(references)
    if rows is None:
        return c_score
    dots = np.sort(rows @ u)
    top2_mean = float(dots[-2:].mean())
    return max(c_score, top2_mean - 0.02)


def trimmed_centroid(
    vectors: np.ndarray,
    trim_ratio: float = 0.10,
    min_size: int = 10,
    drop_floor: float = 0.25,
) -> np.ndarray:
    """Trim low-affinity members from sufficiently large clusters before averaging."""
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("质心需要二维非空矩阵")
    centroid = _unit(matrix.mean(axis=0))
    if matrix.shape[0] < min_size or trim_ratio <= 0.0:
        return centroid
    scores = matrix @ centroid
    cutoff = max(float(np.quantile(scores, trim_ratio)), drop_floor)
    keep = scores >= cutoff
    if int(keep.sum()) < 2:
        return centroid
    return _unit(matrix[keep].mean(axis=0))


def _exclusion_blocks(left: TurnEvidence, right: TurnEvidence) -> bool:
    """Only distinct reliable identities are forbidden; same-role turns may cluster."""
    return (left.anchor_role is not None and right.anchor_role is not None
            and left.anchor_role != right.anchor_role)


def reference_posterior(centroid, reference, speech_seconds, members, prior_seconds=6.):
    """vMF mean direction with a finite reference pseudo-count.

    Call only after qualification on unshrunk acoustic evidence. Cap per-turn
    mass so one long, possibly mixed sentence cannot overwhelm the prior.
    """
    mass = min(max(0., speech_seconds), max(0, members) * 3.)
    weight = prior_seconds / (prior_seconds + mass)
    return _unit(weight * _unit(reference) + (1. - weight) * _unit(centroid)), weight


def agglomerative(
    primary: np.ndarray,
    secondary: np.ndarray,
    *,
    link_primary: float,
    veto_floor: float,
    blocked: np.ndarray | None = None,
) -> list[list[int]]:
    """Average linkage with secondary-channel and symmetric identity constraints."""
    count = primary.shape[0]
    members: list[list[int]] = [[i] for i in range(count)]
    score = primary @ primary.T
    cross = secondary @ secondary.T

    # 相似度与禁令分别维护：合并时累加相似度、传播禁令，便于单独检查约束。
    eye = np.eye(count, dtype=bool)
    total = np.where(eye, 0.0, score)
    if blocked is not None:
        blocked = np.asarray(blocked, dtype=bool)
        if blocked.shape != (count, count) or not np.array_equal(blocked, blocked.T):
            raise ValueError("禁止合并矩阵必须为对称的 n × n 矩阵")
        forbidden = blocked.copy()
    else:
        forbidden = np.zeros((count, count), dtype=bool)
    pairs = np.where(eye, 0.0, 1.0)
    floor = np.where(eye, np.inf, cross)

    active = list(range(count))
    while len(active) > 1:
        positions = np.ix_(active, active)
        mean = total[positions] / np.maximum(pairs[positions], 1.0)
        mean[forbidden[positions]] = -np.inf
        np.fill_diagonal(mean, -np.inf)
        flat = int(np.argmax(mean))
        i_pos, j_pos = divmod(flat, len(active))
        if i_pos == j_pos or float(mean[i_pos, j_pos]) < link_primary:
            break

        left, right = active[i_pos], active[j_pos]
        if left > right:
            left, right = right, left

        # 副通道的否决在后续簇合并中传播，但不修改原始相似度。
        if float(floor[left, right]) < veto_floor:
            forbidden[left, right] = forbidden[right, left] = True
            continue

        members[left].extend(members[right])
        total[left] += total[right]
        pairs[left] += pairs[right]
        floor[left] = np.minimum(floor[left], floor[right])
        forbidden[left] |= forbidden[right]
        total[:, left] = total[left]
        pairs[:, left] = pairs[left]
        floor[:, left] = floor[left]
        forbidden[:, left] = forbidden[left]
        active.remove(right)

    return [members[i] for i in active]


def cluster_turns(
    turns: Sequence[TurnEvidence],
    *,
    link_primary: float = LINK_PRIMARY,
    veto_floor: float = SECONDARY_VETO_FLOOR,
    min_members: int = MIN_CLUSTER_MEMBERS,
    min_speech_seconds: float = MIN_CLUSTER_SPEECH_SECONDS,
    cohesion_floor: float = CLUSTER_COHESION_FLOOR,
) -> tuple[Cluster, ...]:
    """Cluster usable turns without assigning a target identity."""
    usable = [turn for turn in turns if turn.usable]
    if not usable:
        return ()

    # 先归一化：点积即余弦。输入的向量不保证是单位向量，直接点积会把长度
    # 差异当成相似度差异。
    primary = _stack([turn.primary for turn in usable])
    secondary = _stack([turn.secondary for turn in usable])
    count = len(usable)

    blocked = np.zeros((count, count), dtype=bool)
    flagged = np.array([turn.anchor_role is not None for turn in usable])
    if flagged.any():
        for i in range(count):
            if not flagged[i]:
                continue
            for j in range(i + 1, count):
                if flagged[j] and _exclusion_blocks(usable[i], usable[j]):
                    blocked[i, j] = blocked[j, i] = True

    grouped = agglomerative(
        primary, secondary,
        link_primary=link_primary, veto_floor=veto_floor,
        blocked=blocked,
    )

    clusters: list[Cluster] = []
    next_id = 0
    for member_indexes in grouped:
        indexes = list(member_indexes)
        while indexes:
            centroid = trimmed_centroid(primary[indexes])
            scores = primary[indexes] @ centroid
            if float(scores.min()) >= cohesion_floor:
                break
            indexes.pop(int(np.argmin(scores)))
        if len(indexes) < min_members:
            continue

        members = tuple(sorted(usable[i].index for i in indexes))
        speech_seconds = sum(usable[i].speech_seconds for i in indexes)
        if speech_seconds < min_speech_seconds:
            continue

        primary_centroid = trimmed_centroid(primary[indexes])
        secondary_centroid = trimmed_centroid(secondary[indexes])
        cohesion = float((primary[indexes] @ primary_centroid).min())
        clusters.append(
            Cluster(
                cluster_id=next_id,
                members=members,
                primary_centroid=primary_centroid,
                secondary_centroid=secondary_centroid,
                primary_cohesion=round(cohesion, 6),
                primary_median=round(
                    float(np.median(primary[indexes] @ primary_centroid)), 6,
                ),
                total_speech_seconds=round(speech_seconds, 4),
            )
        )
        next_id += 1

    return tuple(
        sorted(clusters, key=lambda c: c.total_speech_seconds, reverse=True)
    )


def assign_target(
    turns: Sequence[TurnEvidence],
    clusters: Sequence[Cluster],
    profile: TargetProfile,
    exclusion_centroids: Sequence[tuple[str, np.ndarray]] = (),
    exclusion_secondary_centroids: Sequence[tuple[str, np.ndarray]] = (),
    *,
    exclusion_scales: Sequence[float] | None = None,
    exclusion_audit: dict | None = None,
    exclusion_tertiary_centroids: Sequence[tuple[str, np.ndarray]] = (),
    margin_floor: float = TARGET_MARGIN_FLOOR,
    require_dominant: bool = True,
    outlier_drop: float = CLUSTER_OUTLIER_DROP,
) -> AssignResult:
    """Select compatible target clusters and retain borderline turns for local verification."""
    membership = [False] * len(turns)
    borderline = [False] * len(turns)
    if any(turn.index != position for position, turn in enumerate(turns)):
        raise ValueError("轮次 index 必须按输入顺序连续编号为 0..n-1")
    cluster_of = [-1] * len(turns)
    for cluster in clusters:
        for index in cluster.members:
            if 0 <= index < len(cluster_of):
                cluster_of[index] = cluster.cluster_id

    if not clusters:
        return AssignResult(
            clusters=(), target_cluster_id=None, membership=tuple(membership),
            cluster_of=tuple(cluster_of), reason="no_cluster",
            borderline_membership=tuple(borderline),
        )

    target_centroid = _unit(profile.primary_centroid)
    target_centroid_secondary = _unit(profile.secondary_centroid)
    primary_refs = _unit_rows(profile.primary_references)
    secondary_refs = _unit_rows(profile.secondary_references)
    exclusion_centroids = [
        (role, _unit(centroid)) for role, centroid in exclusion_centroids
    ]
    secondary_by_role = {
        role: _unit(centroid) for role, centroid in exclusion_secondary_centroids
    }
    tertiary_by_role = {
        role: _unit(centroid) for role, centroid in exclusion_tertiary_centroids
    }
    tertiary_by_cluster = {}
    if profile.tertiary_centroid is not None and tertiary_by_role:
        for cluster in clusters:
            vectors = [turns[index].tertiary for index in cluster.members
                       if turns[index].usable and turns[index].tertiary is not None]
            if vectors:
                tertiary_by_cluster[cluster.cluster_id] = _unit(_stack(vectors).mean(axis=0))
    exclusion_decisions = {}

    scales = list(exclusion_scales) if exclusion_scales is not None else [1.0] * len(exclusion_centroids)
    if len(scales) != len(exclusion_centroids):
        raise ValueError("排除余量因子数量必须与排除角色一致")

    def cluster_exclusion_ok(cluster: Cluster) -> bool:
        """Retain strict matches; uncertain exclusion views need independent support."""
        for (role, centroid), scale in zip(exclusion_centroids, scales):
            p_gap = (cluster.primary_target_score
                     - float(cluster.primary_centroid @ centroid))
            if p_gap >= margin_floor * scale:
                continue
            secondary = secondary_by_role.get(role)
            if secondary is None:
                # 无副通道证据（与旧口径一致）：主通道贴线不得靠兜底放行。
                return False
            s_gap = (cluster.secondary_target_score
                     - float(cluster.secondary_centroid @ secondary))
            if (p_gap >= margin_floor * scale - 0.02
                    and s_gap >= margin_floor * scale + 0.02):
                continue
            gaps = {"primary": p_gap, "secondary": s_gap}
            tertiary = tertiary_by_cluster.get(cluster.cluster_id)
            if tertiary is not None and role in tertiary_by_role:
                gaps["tertiary"] = (profile_score(
                    tertiary, profile.tertiary_centroid, profile.tertiary_references)
                    - float(tertiary @ tertiary_by_role[role]))
            verdict = margin_consensus(gaps, (exclusion_audit or {}).get(role),
                                       margin_floor * scale)
            exclusion_decisions[f"{cluster.cluster_id}:{role}"] = verdict
            if verdict["accepted"]:
                continue
            return False
        return True

    def dual_margin_ok(cluster: Cluster, floor: float) -> bool:
        """双通道排除余量：主通道达标，或主通道贴线且副通道显著占优。"""
        if cluster.nearest_other_role is None:
            return True
        if cluster.target_margin >= floor:
            return True
        return (cluster.target_margin >= floor - 0.02
                and cluster.secondary_target_margin >= floor + 0.02)

    def rescore(cluster: Cluster, *, is_target: bool = False,
                outliers: tuple[int, ...] = ()) -> Cluster:
        # Each channel must pass its own floor; margins compare the same model.
        primary_score = profile_score(
            cluster.primary_centroid, target_centroid, primary_refs)
        secondary_score = profile_score(
            cluster.secondary_centroid, target_centroid_secondary, secondary_refs)
        target_score = min(primary_score, secondary_score)
        nearest_score, nearest_role = -1.0, None
        nearest_secondary = -1.0
        for role, centroid in exclusion_centroids:
            value = float(cluster.primary_centroid @ centroid)
            if value > nearest_score:
                nearest_score, nearest_role = value, role
                secondary_centroid = secondary_by_role.get(role)
                nearest_secondary = (
                    float(cluster.secondary_centroid @ secondary_centroid)
                    if secondary_centroid is not None else -1.0)
        posterior_p = posterior_s = None
        prior_weight = 0.
        if is_target:
            posterior_p, prior_weight = reference_posterior(
                cluster.primary_centroid, target_centroid, cluster.total_speech_seconds, len(cluster.members))
            posterior_s, _ = reference_posterior(
                cluster.secondary_centroid, target_centroid_secondary, cluster.total_speech_seconds, len(cluster.members))
        return Cluster(
            cluster_id=cluster.cluster_id,
            members=cluster.members,
            primary_centroid=cluster.primary_centroid,
            secondary_centroid=cluster.secondary_centroid,
            primary_cohesion=cluster.primary_cohesion,
            primary_median=cluster.primary_median,
            total_speech_seconds=cluster.total_speech_seconds,
            target_score=round(target_score, 6),
            primary_target_score=round(primary_score, 6),
            secondary_target_score=round(secondary_score, 6),
            nearest_other_score=round(nearest_score, 6),
            nearest_other_role=nearest_role,
            target_margin=(
                round(primary_score - nearest_score, 6)
                if nearest_role is not None else -1.0
            ),
            secondary_target_margin=(
                round(secondary_score - nearest_secondary, 6)
                if nearest_role is not None and nearest_secondary > -1.0 else -1.0
            ),
            is_target=is_target,
            outliers=outliers,
            posterior_primary=posterior_p, posterior_secondary=posterior_s,
            prior_weight=prior_weight,
        )

    scored = [rescore(cluster) for cluster in clusters]

    qualified = [
        c for c in scored
        if c.primary_target_score >= profile.primary_reference_floor
        and c.secondary_target_score >= profile.secondary_reference_floor
        and cluster_exclusion_ok(c)
    ]

    diagnostics = {
        "cluster_count": len(scored),
        "qualified_count": len(qualified),
        "largest_cluster_id": scored[0].cluster_id if scored else None,
        "exclusion_reliability": exclusion_audit or {},
        "exclusion_consensus": exclusion_decisions,
    }
    if qualified:
        # 供滑窗 Cohort 守卫区分"目标家族簇"与真正的冒认者簇。
        diagnostics["qualified_cluster_ids"] = [c.cluster_id for c in qualified]

    if not qualified:
        return AssignResult(
            clusters=tuple(scored), target_cluster_id=None, membership=tuple(membership),
            cluster_of=tuple(cluster_of), reason="no_qualifying_cluster",
            diagnostics=diagnostics, borderline_membership=tuple(borderline),
        )

    by_id = {c.cluster_id: c for c in qualified}

    def same_family(candidate: Cluster, core: Cluster) -> bool:
        """同人声线子簇判定：高质心亲和，或对参考集双通道强匹配直达。"""
        p_affinity = float(candidate.primary_centroid @ core.primary_centroid)
        s_affinity = float(candidate.secondary_centroid @ core.secondary_centroid)
        margin_ok = cluster_exclusion_ok(candidate)
        if (p_affinity >= UNION_PRIMARY_AFFINITY
                and s_affinity >= UNION_SECONDARY_AFFINITY and margin_ok):
            return True
        strong_reference = (candidate.primary_target_score >= UNION_STRONG_PRIMARY
                            and candidate.secondary_target_score >= UNION_STRONG_SECONDARY
                            and dual_margin_ok(candidate, margin_floor + UNION_MARGIN_BONUS))
        if strong_reference:
            return True
        return (p_affinity >= UNION_REFERENCE_PRIMARY_AFFINITY
                and s_affinity >= UNION_REFERENCE_SECONDARY_AFFINITY
                and candidate.primary_target_score >= profile.primary_reference_floor + UNION_REFERENCE_BONUS
                and candidate.secondary_target_score >= profile.secondary_reference_floor + UNION_REFERENCE_BONUS
                and dual_margin_ok(candidate, margin_floor + UNION_MARGIN_BONUS))

    best = max(qualified, key=lambda c: c.target_score)

    # Resolve similarly scored, low-affinity clusters before forming the family.
    def is_rival(candidate: Cluster) -> bool:
        if candidate.cluster_id == best.cluster_id:
            return False
        if candidate.target_score < best.target_score - RIVAL_SCORE_GAP:
            return False
        p_affinity = float(candidate.primary_centroid @ best.primary_centroid)
        s_affinity = float(candidate.secondary_centroid @ best.secondary_centroid)
        return p_affinity < RIVAL_PRIMARY_AFFINITY or s_affinity < RIVAL_SECONDARY_AFFINITY

    rivals = [c for c in qualified if is_rival(c)]
    rival_ids: list[int] = []
    if rivals:
        contenders = [best, *rivals]
        winner = max(
            contenders,
            key=lambda c: (c.primary_target_score + c.secondary_target_score) / 2.0,
        )
        rival_ids = sorted(c.cluster_id for c in contenders if c.cluster_id != winner.cluster_id)
        best = winner
    diagnostics["rival_cluster_ids"] = rival_ids

    target_ids = {best.cluster_id}
    union_details = []
    for candidate in qualified:
        if candidate.cluster_id == best.cluster_id:
            continue
        if same_family(candidate, best):
            target_ids.add(candidate.cluster_id)
            union_details.append({
                "cluster_id": candidate.cluster_id,
                "members": len(candidate.members),
                "primary_affinity": round(float(
                    candidate.primary_centroid @ best.primary_centroid), 4),
                "secondary_affinity": round(float(
                    candidate.secondary_centroid @ best.secondary_centroid), 4),
            })
    target_cluster_ids = tuple(sorted(target_ids))
    diagnostics["target_cluster_ids"] = list(target_cluster_ids)
    diagnostics["union_clusters"] = union_details

    largest_qualified = max(qualified, key=lambda c: c.total_speech_seconds)
    diagnostics["largest_qualified_cluster_id"] = largest_qualified.cluster_id
    if require_dominant and largest_qualified.cluster_id not in target_ids:
        # The caller requires the longest qualified cluster to belong to the target family.
        diagnostics["dominance_conflict"] = True
        return AssignResult(
            clusters=tuple(scored), target_cluster_id=None, membership=tuple(membership),
            cluster_of=tuple(cluster_of), reason="dominance_conflict",
            diagnostics=diagnostics, borderline_membership=tuple(borderline),
        )

    # Exclude independently supported target registers from the impostor cohort.
    family_ids = {c.cluster_id for c in qualified} | set(target_cluster_ids)
    family_ids |= {
        c.cluster_id for c in scored
        if c.primary_target_score >= UNION_STRONG_PRIMARY
        and c.secondary_target_score >= UNION_STRONG_SECONDARY
    }
    non_target_clusters = [
        c for c in scored
        if c.cluster_id not in family_ids and eligible_competitor(c, turns)
    ]
    other_primary = [c.primary_centroid for c in non_target_clusters]
    other_secondary = [c.secondary_centroid for c in non_target_clusters]

    union_members = set()
    union_by_id = {cid: by_id[cid] for cid in target_cluster_ids if cid in by_id}
    for cluster in union_by_id.values():
        union_members.update(cluster.members)

    # 各目标簇的稳健中位数基准（簇内成员对自身质心的得分）。
    floors: dict[int, tuple[float, float]] = {}
    for cid, cluster in union_by_id.items():
        p_scores = [float(_unit(turns[i].primary) @ cluster.primary_centroid)
                    for i in cluster.members if i < len(turns) and turns[i].usable]
        s_scores = [float(_unit(turns[i].secondary) @ cluster.secondary_centroid)
                    for i in cluster.members if i < len(turns) and turns[i].usable]
        if not p_scores or not s_scores:
            continue
        p_median, s_median = float(np.median(p_scores)), float(np.median(s_scores))
        floors[cid] = (
            max(p_median - outlier_drop, profile.primary_reference_floor - 0.03),
            max(s_median - outlier_drop - 0.02, profile.secondary_reference_floor - 0.03),
        )

    core_count = 0
    outlier_indexes: list[int] = []
    cohort_stats: list[dict] = []
    for turn in turns:
        if turn.index not in union_members or not turn.usable:
            continue
        u_primary, u_secondary = _unit(turn.primary), _unit(turn.secondary)
        own = union_by_id.get(cluster_of[turn.index])
        p_tc = float(u_primary @ own.primary_centroid) if own is not None else -1.0
        s_tc = float(u_secondary @ own.secondary_centroid) if own is not None else -1.0
        p_prof = profile_score(u_primary, target_centroid, primary_refs)
        s_prof = profile_score(u_secondary, target_centroid_secondary, secondary_refs)
        p_eff, s_eff = max(p_tc, p_prof), max(s_tc, s_prof)

        max_other_p = max((float(u_primary @ c) for c in other_primary), default=-1.0)
        max_other_s = max((float(u_secondary @ c) for c in other_secondary), default=-1.0)
        cohort_margin_p = p_eff - max_other_p

        p_floor, s_floor = floors.get(
            cluster_of[turn.index],
            (profile.primary_reference_floor - 0.03, profile.secondary_reference_floor - 0.03),
        )
        # 竞争守卫：主通道明显败给某个非目标簇、副通道也无优势 → 混入外人。
        cohort_defeat = (max_other_p > p_eff + COHORT_PRIMARY_GUARD
                         and max_other_s > s_eff - COHORT_SECONDARY_GUARD)
        if p_eff >= p_floor and s_eff >= s_floor and not cohort_defeat:
            membership[turn.index] = True
            core_count += 1
        elif (cohort_defeat
              and p_eff >= p_floor - BORDERLINE_SLACK
              and s_eff >= s_floor - BORDERLINE_SECONDARY_SLACK):
            # 竞争守卫判为"更像某个非目标簇"的混疑轮次：交给 Tier B 试听，
            # 不直接丢弃（其中不少是情绪声线真目标）。
            borderline[turn.index] = True
            outlier_indexes.append(turn.index)
            cohort_stats.append({
                "index": turn.index, "tier": "borderline_defeat",
                "p_eff": round(p_eff, 4), "cohort_margin_p": round(cohort_margin_p, 4),
            })
        elif (p_eff >= p_floor - BORDERLINE_SLACK
              and cohort_margin_p >= BORDERLINE_COHORT_MARGIN
              and s_eff >= s_floor - BORDERLINE_SECONDARY_SLACK):
            borderline[turn.index] = True
            outlier_indexes.append(turn.index)
            cohort_stats.append({
                "index": turn.index, "tier": "borderline",
                "p_eff": round(p_eff, 4), "cohort_margin_p": round(cohort_margin_p, 4),
            })
        else:
            outlier_indexes.append(turn.index)
            cohort_stats.append({
                "index": turn.index, "tier": "dropped",
                "p_eff": round(p_eff, 4), "cohort_margin_p": round(cohort_margin_p, 4),
            })

    marked = tuple(sorted(outlier_indexes))
    diagnostics["outlier_count"] = len(marked)
    diagnostics["core_member_count"] = core_count
    diagnostics["borderline_count"] = sum(borderline)
    diagnostics["cohort_cluster_count"] = len(non_target_clusters)
    if cohort_stats:
        diagnostics["cohort_membership"] = cohort_stats

    # Keep near matches available for independent tertiary and raw-audio checks.
    near_miss_turns = []
    for turn in turns:
        if (turn.usable and not membership[turn.index] and not borderline[turn.index]
                and cluster_of[turn.index] not in target_ids
                and cluster_of[turn.index] != -1):
            p_near = profile_score(turn.primary, target_centroid, primary_refs)
            s_near = profile_score(turn.secondary, target_centroid_secondary, secondary_refs)
            if (p_near >= profile.primary_reference_floor + 0.03
                    and s_near >= profile.secondary_reference_floor - 0.20):
                borderline[turn.index] = True
                near_miss_turns.append(turn.index)
    if near_miss_turns:
        diagnostics["near_miss_cluster_turns"] = near_miss_turns
        diagnostics["borderline_count"] = sum(borderline)
    final_clusters = tuple(
        rescore(
            cluster,
            is_target=cluster.cluster_id in target_ids,
            outliers=tuple(i for i in marked if cluster_of[i] == cluster.cluster_id),
        )
        for cluster in scored
    )
    return AssignResult(
        clusters=final_clusters,
        target_cluster_id=best.cluster_id,
        membership=tuple(membership),
        cluster_of=tuple(cluster_of),
        reason="target_cluster_found",
        outliers=marked,
        diagnostics=diagnostics,
        target_cluster_ids=target_cluster_ids,
        borderline_membership=tuple(borderline),
    )
