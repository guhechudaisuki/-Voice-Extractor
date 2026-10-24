"""整集说话轮次的簇级身份判决。

现有流程按单个片段判决身份：拿一段音频对参考集算相似度、比阈值、过。单段
相似度的方差很大，0.4 秒的短句和 3 秒的完整句即使同一人分数也差得远，换一集
录音条件变了分数还会整体漂移。用一个高方差的量做逐段硬判决，阈值高一点就漏，
低一点就进错人。

本模块把判决粒度换成"人"：先把整集的说话轮次聚成若干簇（每个簇近似一个人），
再用目标声纹去匹配**簇统计量**，最后整簇接受或整簇拒绝。一个不在排除集里的
相似女声会自然形成自己的簇，只要它不是目标簇，它的每一段都被同一句话拒掉。

为什么不用单链：真实嵌入上实测过，单链会一穿到底。第三集 254 个
说话轮次的主通道两两余弦中位数是 0.369、1% 分位是 0.038，任意两人之间都存在
足够高的边，于是哪怕阈值取到 0.50 也会把全部 254 个并成 1 个簇。单链的传递性
对说话人嵌入是危险的：A 像 B、B 像 C 不代表 A 像 C。目前使用主通道平均连接，
副通道最低配对相似度及排除角色约束否决合并；这不保证每个簇都是单一人物。

设计约束：

* 纯函数，无 IO、无模型、无全局状态 —— 因此可以被完整单测覆盖，不必跑整集。
* 只用整数索引引用轮次，不用对象引用。调用方维持 index -> 对象 的映射。
* ERes2NetV2 192 维决定亲和度，CAM++ 512 维提供独立否决。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

Channel = Literal["primary", "secondary"]

# 合并阈值。**主通道决定能否合并，副通道只做否决**。
#
# 实测（第三集 254 个轮次）：主通道两两余弦中位 0.369、99% 分位 0.769；
# 副通道中位 0.177、99% 分位 0.694。副通道整体低一大截，如果要求"两通道都
# 达标"，它就成了主要约束——同一人跨情绪、跨音量的轮次会因为它给分不够而
# 拒绝合并，结果是 254 个轮次被碎成 60 个簇。一集动画没有 60 个说话人。
#
# 0.50 是第三集开发扫描的起点；簇数随锚点约束而变化，不能用簇数本身证明身份正确。
LINK_PRIMARY = 0.50
# 副通道地板：低于此值视为强烈反对，禁止合并。取值明显低于副通道中位数
# 相关的同人区间，只拦"两个模型都认为不像"的情况。
SECONDARY_VETO_FLOOR = 0.05

# 簇的最小规模。一集里的女性龙套可能只有一两句，把它们当成"人"没有意义，
# 更可能是嵌入噪声。低于此规模的簇不参与目标簇竞争。
MIN_CLUSTER_MEMBERS = 3
MIN_CLUSTER_SPEECH_SECONDS = 6.0

# 目标簇判定的余量下限：目标分数减去最强排除角色的分数。
# 0.06 是开发集验证的起点，需结合双通道绝对下限及整轮局部一致性，不能单独作证。
TARGET_MARGIN_FLOOR = 0.06

# 簇内成员与目标分数的最大允许落差。超出者标记为离群并不输出。
# 0.18 沿用 SpeakerVerifier.verify_waveform 里 p20 >= threshold - 0.18 的量级。
CLUSTER_OUTLIER_DROP = 0.18

# 簇内一致性下限。当前平均连接不保证最低成员的相似度，故需另外检查。
CLUSTER_COHESION_FLOOR = 0.30

# ── 目标簇联盟（Target Cluster Union）─────────────────────
# 同一角色在一集里常分裂成"日常 / 独白 / 情绪"多个声线子簇。质心亲和达到
# 以下双通道阈值即视为同声线家族，直接并入目标簇集合，不再触发整集弃权。
UNION_PRIMARY_AFFINITY = 0.72
UNION_SECONDARY_AFFINITY = 0.70
# 双通道对参考集同时强匹配的子簇即使与核心簇亲和略低也可并入，但仍需
# 保持最低质心亲和——亲和过低说明是另一位说话人，参考分再高也不并。
UNION_REFERENCE_BONUS = 0.10
UNION_MARGIN_BONUS = 0.04
UNION_REFERENCE_PRIMARY_AFFINITY = 0.55
UNION_REFERENCE_SECONDARY_AFFINITY = 0.50
# 绝对强参考直达：参考分达到绝对高位（与真实参考下限尺度对齐）时豁免
# 亲和度要求——同一角色的独白/情绪声线在嵌入空间可能漂得很远。
UNION_STRONG_PRIMARY = 0.66
UNION_STRONG_SECONDARY = 0.60
# 与核心簇质心低于该亲和的达标簇是"另一位说话人"，仅在分数咬得很紧时仲裁。
RIVAL_PRIMARY_AFFINITY = 0.68
RIVAL_SECONDARY_AFFINITY = 0.65
RIVAL_SCORE_GAP = 0.015

# Cohort AS-Norm：以本集非目标簇为天然冒认者队列。目标分数高出最强非目标
# 簇至少该余量即视为具备竞争排他性。
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
    # 多原型参考：各段参考音频自身的单位向量 [M, D]。均值质心会把不同声线
    # 参考中和成四不像，Top-K 原型匹配允许"极像某两段特定声线参考"的子簇。
    primary_references: np.ndarray | None = None
    secondary_references: np.ndarray | None = None
    # 可选第三通道（强化后的 WavLM）质心与准入下限。
    tertiary_centroid: np.ndarray | None = None
    tertiary_reference_floor: float = 0.72


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
    """对目标参考集的多原型打分。

    全局质心分会稀释特定声线子簇；与最匹配的 Top-2 参考原型的均值分取较大值
    （带 0.02 折减防单参考噪声），两者哪个高算哪个。
    """
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
    """算质心前先丢掉偏离最远的那一档成员。

    一个簇里混进一两个"像目标但其实是别人"的段时，普通均值会把质心拖向他们；
    质心一旦被拖走，后续所有成员的分都会下降，形成自证循环。参考集构建里用
    leave-one-out 中位数解决同一个问题，这里用修剪均值。

    修剪只在成员够多（min_size）时才做，而且只丢"确实偏离"的成员
    （分数低于 drop_floor）。纯按分位数截断会把排在最末的正常成员也丢掉——
    20 个成员的簇里固定丢 10% 就是丢 2 个好人。
    """
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


def agglomerative(
    primary: np.ndarray,
    secondary: np.ndarray,
    *,
    link_primary: float,
    veto_floor: float,
    blocked: np.ndarray | None = None,
) -> list[list[int]]:
    """平均连接的凝聚聚类：主通道决定合并，副通道只否决。

    簇间亲和度用**平均连接**（所有成员对的平均）而不是完全连接。完全连接只由
    "最远一对"决定，目标簇里混进一个情绪波动大的轮次就会把整个簇拆开——实测
    254 个轮次被碎成 91 个簇。平均连接让多数成员投票，同时因为要求平均相似度
    达标，不会像单链那样把不相似的人串起来。

    副通道（CAM++）不参与"该不该合并"的判断，只在它强烈反对时禁止合并。它
    整体给分比主通道低一大截（中位 0.177 对 0.369），若与主通道并列要求达标，
    它会成为主要约束，导致同一人的轮次被拆开。

    blocked 是形状 [n, n] 的禁止合并矩阵（排除角色闸门）。
    """
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
    """把一集的说话轮次聚成"人"。只做聚类，不做目标判定。

    分离聚类与目标判定，是为了让"聚类好不好"和"目标选对没选对"能分别评估。
    """
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
    margin_floor: float = TARGET_MARGIN_FLOOR,
    require_dominant: bool = True,
    outlier_drop: float = CLUSTER_OUTLIER_DROP,
) -> AssignResult:
    """挑出目标簇联盟，并给出每个轮次的最终归属。

    这是判决的唯一出口，方向是**双向**的：目标簇内的接受、簇外的拒绝，
    包括那些已经被单段判决接受、但不属于目标簇的段。

    与旧版的三个关键差别：
    1. 同一角色的多个声线子簇（日常 / 独白 / 情绪）并入 ``target_cluster_ids``，
       不再因"两个达标簇"而把整集弃权（旧 ``ambiguous_target_cluster`` 自杀）。
    2. 簇内离群基准从 ``max(scores) - 0.18`` 改为**中位数基准 + 参考下限**，
       情绪句、感叹词不再被一句贴近质心的标准音连坐。
    3. 引入 Cohort AS-Norm：本集全部非目标簇是天然冒认者队列，轮次必须在
       竞争余量上压过最强非目标簇才进核心成员，否则落入边缘池。
    """
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
        )

    scored = [rescore(cluster) for cluster in clusters]

    qualified = [
        c for c in scored
        if c.primary_target_score >= profile.primary_reference_floor
        and c.secondary_target_score >= profile.secondary_reference_floor
        and dual_margin_ok(c, margin_floor)
    ]

    diagnostics = {
        "cluster_count": len(scored),
        "qualified_count": len(qualified),
        "largest_cluster_id": scored[0].cluster_id if scored else None,
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
        margin_ok = dual_margin_ok(candidate, margin_floor)
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

    # 真正的对手是"另一位不同声线的说话人"：与核心簇亲和低且分数咬得极紧。
    # 此时按双通道平均分仲裁出优胜簇，绝不再把整集清零。
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
        # 目标是主角，她在一集里的语音时长应当最长。判定出的目标簇（含联盟）
        # 不是最大的簇，本身就是一个可疑信号，此时保守处理而不是硬选。
        diagnostics["dominance_conflict"] = True
        return AssignResult(
            clusters=tuple(scored), target_cluster_id=None, membership=tuple(membership),
            cluster_of=tuple(cluster_of), reason="dominance_conflict",
            diagnostics=diagnostics, borderline_membership=tuple(borderline),
        )

    # ── Cohort AS-Norm：以"不像目标"的簇作为冒认者队列 ─────────
    # 达标簇（含未入盟次级声线簇）属于目标家族；参考分达到强参考水平的簇
    # 即使被排除角色贴近而未达标，也不得作为竞争对象——否则真目标轮次会被
    # 同说话人的另一个子簇"竞争"掉。
    family_ids = {c.cluster_id for c in qualified} | set(target_cluster_ids)
    family_ids |= {
        c.cluster_id for c in scored
        if c.primary_target_score >= UNION_STRONG_PRIMARY
        and c.secondary_target_score >= UNION_STRONG_SECONDARY
    }
    non_target_clusters = [
        c for c in scored
        if c.cluster_id not in family_ids and len(c.members) >= MIN_CLUSTER_MEMBERS
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

    # 达标但未入盟的次级簇（或未聚类的独立旁白）：双通道对参考集达标的
    # 轮次进内部边缘池。该池是 tertiary/raw 晋升通道的输入（晋升后即离开
    # 边缘池），导出侧的 Tier B 瘦身由 cluster_gate 的互斥 + NMS + 上限
    # 完成——在这里收紧会连带收窄晋升通道，实测净伤 Tier A 召回。
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
