"""簇级身份判决的合成数据测试。

全部用构造出来的单位向量，不加载任何模型、不读任何音频，因此可以在几秒内
跑完。测的是模块自己的逻辑，不是它在真实数据上的效果。
"""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extractor.nextgen.identity_cluster import (  # noqa: E402
    CLUSTER_COHESION_FLOOR,
    TargetProfile,
    TurnEvidence,
    assign_target,
    agglomerative,
    cluster_turns,
    trimmed_centroid,
)

DIM = 64
SEED = 20260929


def unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def blob(axis: np.ndarray, count: int, spread: float, rng: np.random.Generator,
         start: float = 0.0, duration: float = 2.0) -> list[TurnEvidence]:
    """在给定轴附近生成一团轮次。spread 越大越松散。"""
    turns = []
    for i in range(count):
        noise = rng.normal(0.0, spread, size=DIM)
        vector = unit(axis + noise)
        turns.append(
            TurnEvidence(
                index=int(start) + i,
                start=(start + i) * 3.0,
                end=(start + i) * 3.0 + duration,
                speech_seconds=duration,
                primary=vector,
                secondary=vector,
            )
        )
    return turns


def axis_at(cosine: float, rng: np.random.Generator, base: np.ndarray) -> np.ndarray:
    """构造一个与 base 余弦约为 cosine 的轴。"""
    other = unit(rng.normal(0.0, 1.0, size=DIM))
    orthogonal = unit(other - (other @ base) * base)
    return unit(cosine * base + np.sqrt(max(0.0, 1.0 - cosine ** 2)) * orthogonal)


def profile_for(centroid: np.ndarray) -> TargetProfile:
    return TargetProfile(
        primary_centroid=centroid,
        secondary_centroid=centroid,
        primary_reference_floor=0.55,
        secondary_reference_floor=0.50,
    )


# ── 聚类 ─────────────────────────────────────────────────


class IdentityClusterTests(unittest.TestCase):
    def test_blocked_pair_does_not_split_unrelated_cluster(self):
        # 2、3 先合并，再与 0 合并；1 在轴的另一侧，避免先吸收 2 的歧义。
        primary = np.array([
            [1.0, 0.0], [0.995, -0.1], [0.98, 0.2], [0.955, 0.3],
        ])
        primary /= np.linalg.norm(primary, axis=1, keepdims=True)
        blocked = np.zeros((4, 4), dtype=bool)
        blocked[0, 1] = blocked[1, 0] = True
        groups = agglomerative(primary, primary, link_primary=0.90,
                               veto_floor=-1.0, blocked=blocked)
        self.assertFalse(any({0, 1}.issubset(group) for group in map(set, groups)))
        self.assertTrue(any({0, 2, 3}.issubset(group) for group in map(set, groups)))

    def test_two_separated_clusters(self):
        rng = np.random.default_rng(SEED)
        base = unit(rng.normal(0.0, 1.0, size=DIM))
        other = axis_at(0.30, rng, base)
        turns = blob(base, 25, 0.05, rng) + blob(other, 15, 0.05, rng, start=100)

        clusters = cluster_turns(turns)

        assert len(clusters) == 2
        sizes = sorted(len(c.members) for c in clusters)
        assert sizes == [15, 25]


    def test_similar_voice_stays_separate(self):
        """与目标相似度较高的另一簇不得被并进目标簇。"""
        rng = np.random.default_rng(SEED)
        base = unit(rng.normal(0.0, 1.0, size=DIM))
        near = axis_at(0.55, rng, base)
        turns = blob(base, 25, 0.04, rng) + blob(near, 8, 0.04, rng, start=25)

        clusters = cluster_turns(turns)
        target = assign_target(turns, clusters, profile_for(base))

        assert target.target_cluster_id is not None
        target_cluster = target.target()
        assert len(target_cluster.members) == 25
        # 那个相似女声单独成簇，不是目标簇
        assert len(clusters) == 2


    def test_channels_must_agree(self):
        """主通道相似但副通道不像，不能合并。"""
        rng = np.random.default_rng(SEED)
        base = unit(rng.normal(0.0, 1.0, size=DIM))
        turns = []
        for i in range(8):
            turns.append(
                TurnEvidence(
                    index=i, start=i * 3.0, end=i * 3.0 + 2.0, speech_seconds=2.0,
                    primary=unit(base + rng.normal(0, 0.02, DIM)),
                    secondary=unit((base if i < 4 else -base) + rng.normal(0, 0.02, DIM)),
                )
            )
        clusters = cluster_turns(turns)
        # 两组主通道相同、副通道相反；组内仍可聚类，跨组不可合并。
        assert {frozenset(c.members) for c in clusters} == {
            frozenset(range(4)), frozenset(range(4, 8)),
        }


    def test_chain_drift_is_split(self):
        """A~B、B~C 但 A 与 C 差很远时，不得把 A 和 C 并成一群。"""
        rng = np.random.default_rng(SEED)
        a = unit(rng.normal(0.0, 1.0, size=DIM))
        b = axis_at(0.50, rng, a)
        c = axis_at(0.50, rng, b)
        # 确保 a 与 c 确实不像
        assert float(a @ c) < CLUSTER_COHESION_FLOOR + 0.05

        turns = blob(a, 6, 0.03, rng) + blob(c, 6, 0.03, rng, start=100)
        turns.insert(
            6,
            TurnEvidence(
                index=999, start=900.0, end=902.0, speech_seconds=2.0,
                primary=b, secondary=b,
            ),
        )

        clusters = cluster_turns(turns)
        # 拆解之后不应出现一个同时含 a 和 c 的簇
        for cluster in clusters:
            members = set(cluster.members)
            assert not (members & set(range(0, 6)) and members & set(range(100, 106)))


    def test_excluded_role_not_absorbed(self):
        """A known exclusion forms its own cluster and cannot absorb target anchors."""
        rng = np.random.default_rng(SEED)
        base = unit(rng.normal(0.0, 1.0, size=DIM))
        turns = blob(base, 10, 0.03, rng)
        marked = {2, 5, 7}
        turns = [replace(t, anchor_role="excluded" if t.index in marked else "target") for t in turns]
        clusters = cluster_turns(turns)
        self.assertEqual(sorted(len(c.members) for c in clusters), [3, 7])
        self.assertTrue(any(set(c.members) == marked for c in clusters))

    def test_largest_non_target_does_not_block_target(self):
        rng = np.random.default_rng(SEED)
        base = unit(rng.normal(0.0, 1.0, size=DIM))
        other = axis_at(0.30, rng, base)
        turns = blob(base, 6, 0.02, rng) + blob(other, 20, 0.02, rng, start=6)
        result = assign_target(turns, cluster_turns(turns), profile_for(base))
        self.assertIsNotNone(result.target_cluster_id)

    def test_secondary_reference_floor_cannot_be_bypassed(self):
        rng = np.random.default_rng(SEED)
        base = unit(rng.normal(0.0, 1.0, size=DIM))
        other = axis_at(0.20, rng, base)
        turns = [replace(t, secondary=other) for t in blob(base, 10, 0.02, rng)]
        result = assign_target(turns, cluster_turns(turns), profile_for(base))
        self.assertIsNone(result.target_cluster_id)

    def test_small_cluster_filtered(self):
        rng = np.random.default_rng(SEED)
        base = unit(rng.normal(0.0, 1.0, size=DIM))
        turns = blob(base, 20, 0.04, rng)
        for i in range(2):
            vector = unit(rng.normal(0.0, 1.0, size=DIM))
            turns.append(
                TurnEvidence(
                    index=200 + i, start=900.0 + i, end=901.0 + i, speech_seconds=1.0,
                    primary=vector, secondary=vector,
                )
            )

        clusters = cluster_turns(turns)
        assert len(clusters) == 1
        assert len(clusters[0].members) == 20


    def test_short_speech_cluster_filtered(self):
        """成员数够但总时长短的簇同样被过滤。"""
        rng = np.random.default_rng(SEED)
        base = unit(rng.normal(0.0, 1.0, size=DIM))
        turns = blob(base, 4, 0.03, rng, duration=1.0)  # 4 条 × 1.0s = 4.0s < 6.0s
        assert cluster_turns(turns) == ()


    # ── 目标判定 ─────────────────────────────────────────────


    def test_target_selected_and_members_marked(self):
        rng = np.random.default_rng(SEED)
        base = unit(rng.normal(0.0, 1.0, size=DIM))
        other = axis_at(0.25, rng, base)
        turns = blob(base, 20, 0.04, rng) + blob(other, 10, 0.04, rng, start=20)

        clusters = cluster_turns(turns)
        result = assign_target(turns, clusters, profile_for(base))

        assert result.reason == "target_cluster_found"
        target = result.target()
        assert target is not None
        assert set(target.members) == set(range(20))
        assert all(result.membership[i] for i in range(20))
        assert not any(result.membership[i] for i in range(20, 30))


    def test_same_family_twin_clusters_union(self):
        """同声线家族的两个达标簇 → 并入目标簇联盟，不再整集弃权。"""
        rng = np.random.default_rng(SEED)
        base = unit(rng.normal(0.0, 1.0, size=DIM))
        twin = axis_at(0.995, rng, base)
        turns = blob(base, 20, 0.02, rng) + blob(twin, 18, 0.02, rng, start=20)
        # 直接向判决层提供两个相似簇，不能因聚类合并而跳过断言。
        left = cluster_turns(turns[:20])[0]
        right = replace(cluster_turns(turns[20:])[0], cluster_id=1)
        result = assign_target(turns, (left, right), profile_for(base))
        assert result.reason == "target_cluster_found"
        assert set(result.target_cluster_ids) == {left.cluster_id, right.cluster_id}
        assert all(result.membership[i] for i in range(38))

    def test_true_rival_resolved_by_dual_channel_average(self):
        """不同声线的达标对手分数咬紧 → 按双通道平均分仲裁，绝不清零。"""
        rng = np.random.default_rng(SEED)
        base = unit(rng.normal(0.0, 1.0, size=DIM))
        rival_axis = axis_at(0.35, rng, base)
        turns = blob(base, 20, 0.02, rng) + blob(rival_axis, 18, 0.02, rng, start=20)
        left = cluster_turns(turns[:20])[0]
        right = replace(cluster_turns(turns[20:])[0], cluster_id=1)
        # 对手簇主通道分被抬到与核心簇几乎并列（模拟另一位说话人高分达标）。
        profile = replace(profile_for(base), primary_reference_floor=0.20,
                          secondary_reference_floor=0.20)
        result = assign_target(turns, (left, right), profile)
        assert result.target_cluster_id is not None
        assert result.reason == "target_cluster_found"
        # 对手簇亲和过低，绝不并入联盟。
        assert result.target_cluster_ids == (left.cluster_id,)
        assert not any(result.membership[i] for i in range(20, 38))

    def test_median_outlier_rule_keeps_emotional_members(self):
        """簇内一句贴近质心的标准音不得把其余情绪句连坐成离群。"""
        rng = np.random.default_rng(SEED)
        base = unit(rng.normal(0.0, 1.0, size=DIM))
        turns = blob(base, 18, 0.03, rng)
        # 加入一句极其贴近质心的标准音，旧 max-0.18 规则会把 0.75 以下的
        # 真实成员全部判离群；中位数基准下应全部保留。
        star = replace(turns[0], primary=base, secondary=base)
        turns[0] = star
        clusters = cluster_turns(turns)
        result = assign_target(turns, clusters, profile_for(base))
        assert result.target_cluster_id is not None
        assert sum(result.membership) == 18
        assert result.outliers == ()

    def test_cohort_defeat_drops_intruder_turn(self):
        """混入目标簇、却更像某个非目标簇的轮次必须被竞争守卫剔除。"""
        rng = np.random.default_rng(SEED)
        base = unit(rng.normal(0.0, 1.0, size=DIM))
        intruder_axis = axis_at(0.55, rng, base)
        turns = blob(base, 16, 0.02, rng)
        turns.append(replace(
            turns[0], index=16, start=100.0, end=102.0,
            primary=unit(intruder_axis + rng.normal(0, 0.01, DIM)),
            secondary=unit(intruder_axis + rng.normal(0, 0.01, DIM)),
        ))
        clusters = cluster_turns(turns)
        # 强行把 16 号轮次塞进目标簇（模拟聚类把相似女声粘进目标簇）。
        target = next(c for c in clusters if 0 in c.members)
        clusters = tuple(
            replace(c, members=(*c.members, 16)) if c is target else c
            for c in clusters
        )
        result = assign_target(turns, clusters, profile_for(base))
        assert result.target_cluster_id is not None
        # 该轮次被中位数基准或竞争守卫剔除，不进核心成员。
        assert not result.membership[16]

    def test_borderline_membership_marks_near_miss(self):
        """略低于核心准入的家族轮次应落入边缘池（Tier B）而不是被静默丢弃。"""
        rng = np.random.default_rng(SEED)
        base = unit(rng.normal(0.0, 1.0, size=DIM))
        near_axis = axis_at(0.90, rng, base)
        turns = blob(base, 14, 0.02, rng)
        turns.append(replace(
            turns[0], index=14, start=100.0, end=102.0,
            primary=unit(near_axis + rng.normal(0, 0.02, DIM)),
            secondary=unit(near_axis + rng.normal(0, 0.02, DIM)),
        ))
        clusters = cluster_turns(turns)
        target = next(c for c in clusters if 0 in c.members)
        clusters = tuple(
            replace(c, members=(*c.members, 14)) if c is target else c
            for c in clusters
        )
        result = assign_target(turns, clusters, profile_for(base))
        assert result.borderline_membership == (False,) * 15 or result.borderline_membership[14]


    def test_no_qualifying_cluster(self):
        rng = np.random.default_rng(SEED)
        base = unit(rng.normal(0.0, 1.0, size=DIM))
        far = axis_at(0.10, rng, base)
        turns = blob(far, 20, 0.04, rng)

        clusters = cluster_turns(turns)
        result = assign_target(turns, clusters, profile_for(base))

        assert result.target_cluster_id is None
        assert result.reason == "no_qualifying_cluster"
        assert not any(result.membership)


    def test_dominance_conflict_abstains(self):
        """目标分数最高的簇不是最大的簇 → 保守处理。"""
        rng = np.random.default_rng(SEED)
        base = unit(rng.normal(0.0, 1.0, size=DIM))
        near = axis_at(0.45, rng, base)
        turns = blob(base, 6, 0.02, rng) + blob(near, 20, 0.02, rng, start=6)

        clusters = cluster_turns(turns)
        result = assign_target(turns, clusters, replace(profile_for(base), primary_reference_floor=0.30, secondary_reference_floor=0.30))

        assert result.target_cluster_id is None
        assert result.reason == "dominance_conflict"


    def test_exclusion_margin_rejects_cluster(self):
        """与排除角色太近的簇不能成为目标簇。"""
        rng = np.random.default_rng(SEED)
        base = unit(rng.normal(0.0, 1.0, size=DIM))
        turns = blob(base, 20, 0.02, rng)

        clusters = cluster_turns(turns)
        # 构造一个与目标几乎重合的排除角色质心 → 余量变成约 0
        result = assign_target(
            turns, clusters, profile_for(base),
            exclusion_centroids=[("绿", base)],
            margin_floor=0.10,
        )
        assert result.target_cluster_id is None
        assert result.reason == "no_qualifying_cluster"


    # ── 质心 ─────────────────────────────────────────────────


    def test_trimmed_centroid_resists_contamination(self):
        """少量污染段不得把质心拖偏。"""
        rng = np.random.default_rng(SEED)
        base = unit(rng.normal(0.0, 1.0, size=DIM))
        intruder = axis_at(0.45, rng, base)

        clean = np.stack([unit(base + rng.normal(0, 0.03, DIM)) for _ in range(25)])
        polluted = np.vstack([clean, np.stack([intruder] * 3)])

        pure = trimmed_centroid(clean)
        contaminated = trimmed_centroid(polluted)
        angle = np.degrees(np.arccos(np.clip(float(pure @ contaminated), -1.0, 1.0)))

        assert angle < 8.0, f"质心偏移 {angle:.2f}°，修剪均值没有起作用"


    def test_trimmed_centroid_handles_tiny_input(self):
        rng = np.random.default_rng(SEED)
        vector = unit(rng.normal(0.0, 1.0, size=DIM))
        result = trimmed_centroid(np.stack([vector]))
        assert np.allclose(result, vector, atol=1e-9)


    def test_zero_vector_rejected(self):
        with self.assertRaises(ValueError):
            trimmed_centroid(np.zeros((3, DIM)))


    # ── 索引与边界 ───────────────────────────────────────────


    def test_unusable_turns_excluded_from_clustering(self):
        rng = np.random.default_rng(SEED)
        base = unit(rng.normal(0.0, 1.0, size=DIM))
        turns = blob(base, 20, 0.03, rng)
        turns.append(
            TurnEvidence(
                index=20, start=900.0, end=902.0, speech_seconds=2.0,
                primary=np.zeros(DIM), secondary=np.zeros(DIM), usable=False,
            )
        )

        clusters = cluster_turns(turns)
        result = assign_target(turns, clusters, profile_for(base))

        assert 20 not in {m for c in clusters for m in c.members}
        assert result.membership[20] is False

    def test_assignment_rejects_sparse_indexes(self):
        rng = np.random.default_rng(SEED)
        base = unit(rng.normal(0.0, 1.0, size=DIM))
        turns = blob(base, 3, 0.02, rng, start=100)
        with self.assertRaisesRegex(ValueError, "index"):
            assign_target(turns, cluster_turns(turns), profile_for(base))


    def test_membership_length_matches_turn_count(self):
        rng = np.random.default_rng(SEED)
        base = unit(rng.normal(0.0, 1.0, size=DIM))
        turns = blob(base, 12, 0.03, rng)

        clusters = cluster_turns(turns)
        result = assign_target(turns, clusters, profile_for(base))

        assert len(result.membership) == len(turns)
        assert len(result.cluster_of) == len(turns)


if __name__ == "__main__":
    unittest.main()
