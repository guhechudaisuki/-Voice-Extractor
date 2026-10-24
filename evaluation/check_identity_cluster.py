"""簇级判决的不依赖 pytest 自检。

项目自带的运行时里没有 pytest，而 tests/test_identity_cluster.py 需要它。
本脚本用合成数据跑一遍核心断言，只要有 numpy 就能用。

    python evaluation\check_identity_cluster.py

关于合成数据的一个教训：第一版把簇内散布设成 0.04（64 维随机向量上的
高斯扰动），簇内相似度接近 1，于是"不同簇被链在一起"这种失败根本无法复现，
自检全绿而真实数据上一个簇都没分开。现在用真实嵌入的统计量来标定合成数据。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extractor.nextgen.identity_cluster import (  # noqa: E402
    TargetProfile,
    TurnEvidence,
    assign_target,
    cluster_turns,
    trimmed_centroid,
)

DIM = 64
SEED = 20260929
RESULTS: list[tuple[bool, str, str]] = []


def unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def axis_at(cosine: float, rng: np.random.Generator, base: np.ndarray) -> np.ndarray:
    """构造一个与 base 余弦约为 cosine 的轴。"""
    other = unit(rng.normal(0.0, 1.0, size=DIM))
    orthogonal = unit(other - (other @ base) * base)
    return unit(cosine * base + np.sqrt(max(0.0, 1.0 - cosine ** 2)) * orthogonal)


def blob(axis, count, spread, rng, start=0, duration=2.0, secondary_axis=None):
    """生成一团轮次。

    index 从 start 连续递增，调用方必须保证整个序列的 index 是 0..n-1，
    因为 membership / cluster_of 是按 index 索引的。
    """
    turns = []
    for i in range(count):
        vector = unit(axis + rng.normal(0.0, spread, size=DIM))
        second = (
            vector if secondary_axis is None
            else unit(secondary_axis + rng.normal(0.0, spread, size=DIM))
        )
        turns.append(
            TurnEvidence(
                index=start + i,
                start=start + i * 3.0,
                end=start + i * 3.0 + duration,
                speech_seconds=duration,
                primary=vector,
                secondary=second,
            )
        )
    return turns


def profile_for(centroid: np.ndarray) -> TargetProfile:
    return TargetProfile(
        primary_centroid=centroid,
        secondary_centroid=centroid,
        primary_reference_floor=0.55,
        secondary_reference_floor=0.50,
    )


def calibrate_spread(path: Path, dim: int = DIM) -> float:
    """从真实嵌入反推一个让"同人相似度"落在实测范围的散布值。

    直接量真实向量的离散度没有意义（192 维和 64 维不可比）。有意义的量是
    **同一簇内两两余弦**：用每个轮次与它最近邻的相似度作估计。

    反解用数值二分而不是解析式。解析式假设"基准轴 + 各向同性噪声"，但噪声
    与基准正交分量的相对幅度强烈依赖维度，192 维上的实测余弦套到 64 维会
    系统性低估相似度（上一版算出 0.817，导致合成数据里同一簇都连不起来）。
    把生成器当黑盒、直接二分到余弦对上，就没有这个换算误差。
    """
    payload = np.load(path, allow_pickle=False)
    usable = payload["usable"]
    matrix = payload["primary"][usable]
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    similarity = matrix @ matrix.T
    np.fill_diagonal(similarity, -np.inf)
    target_cosine = float(np.median(similarity.max(axis=1)))
    print(f"实测：每段与最近邻余弦的中位数为 {target_cosine:.3f}")

    # 余弦随散布单调下降，直接二分。
    rng = np.random.default_rng(SEED)
    base = unit(rng.normal(0.0, 1.0, size=dim))
    low, high = 0.0, 2.0
    for _ in range(30):
        middle = (low + high) / 2
        samples = np.stack(
            [unit(base + rng.normal(0.0, middle, size=dim)) for _ in range(300)]
        )
        cosine = float(np.median(samples @ unit(base)))
        # 余弦太大说明噪声不够 → 需要更大的散布
        if cosine > target_cosine:
            low = middle
        else:
            high = middle
    spread = (low + high) / 2
    print(f"反解：64 维合成数据用散布 {spread:.3f} 可重现该相似度")
    return round(spread, 3)


def check(name: str, condition: bool, detail: str = "") -> None:
    RESULTS.append((bool(condition), name, detail))
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibrate", type=Path,
        help="从真实嵌入 npz 读出簇内散布，用它标定合成数据（可选）",
    )
    args = parser.parse_args()

    spread = 0.32
    if args.calibrate and args.calibrate.is_file():
        spread = calibrate_spread(args.calibrate)
        print(f"已从 {args.calibrate.name} 标定：簇内散布 {spread:.3f}\n")

    print(f"簇级判决自检（合成数据，散布 {spread:.3f}，不加载模型）\n")

    # 1. 两簇分离
    rng = np.random.default_rng(SEED)
    base = unit(rng.normal(0.0, 1.0, size=DIM))
    other = axis_at(0.30, rng, base)
    turns = blob(base, 25, spread, rng) + blob(other, 15, spread, rng, start=25)
    clusters = cluster_turns(turns)
    check("两簇分离", len(clusters) == 2, f"聚出 {len(clusters)} 个簇")

    # 2. 相似女声不并入。断言的是"不混"这个结构性质，不是精确成员数：
    # 平均连接下簇边界本来就有 1-2 个成员的浮动，钉死数字会让测试变脆。
    rng = np.random.default_rng(SEED)
    base = unit(rng.normal(0.0, 1.0, size=DIM))
    near = axis_at(0.55, rng, base)
    turns = blob(base, 25, spread, rng) + blob(near, 8, spread, rng, start=25)
    clusters = cluster_turns(turns)
    result = assign_target(turns, clusters, profile_for(base))
    target = result.target()
    base_set, near_set = set(range(25)), set(range(25, 33))
    target_members = set(target.members) if target else set()
    leaked = target_members & near_set
    check(
        "相似女声不混进目标簇",
        result.target_cluster_id is not None
        and len(clusters) == 2
        and not leaked
        and len(target_members & base_set) >= 23,
        f"簇数 {len(clusters)}，目标簇含目标轮次 "
        f"{len(target_members & base_set)}/25，误含相似女声 {len(leaked)}",
    )

    # 3. 双通道必须一致：主通道是同一个人，副通道各说各话
    rng = np.random.default_rng(SEED)
    base = unit(rng.normal(0.0, 1.0, size=DIM))
    turns = [
        TurnEvidence(
            index=i, start=i * 3.0, end=i * 3.0 + 2.0, speech_seconds=2.0,
            primary=unit(base + rng.normal(0, 0.05, DIM)),
            secondary=unit(rng.normal(0.0, 1.0, size=DIM)),
        )
        for i in range(6)
    ]
    check("主通道像、副通道不像 → 不合并", cluster_turns(turns) == ())

    # 4. 链式漂移被拆解
    rng = np.random.default_rng(SEED)
    a = unit(rng.normal(0.0, 1.0, size=DIM))
    b = axis_at(0.50, rng, a)
    c = axis_at(0.50, rng, b)
    left = blob(a, 6, spread, rng)
    middle = TurnEvidence(index=6, start=900.0, end=902.0, speech_seconds=2.0,
                          primary=b, secondary=b)
    right = blob(c, 6, spread, rng, start=7)
    turns = left + [middle] + right
    clusters = cluster_turns(turns)
    merged = any(
        (set(cl.members) & {t.index for t in left})
        and (set(cl.members) & {t.index for t in right})
        for cl in clusters
    )
    check(
        "链式漂移被拆解（A 与 C 不同簇）",
        not merged,
        f"a·c = {float(a @ c):.3f}",
    )

    # 5. 孤立向量不被吸进主簇
    rng = np.random.default_rng(SEED)
    base = unit(rng.normal(0.0, 1.0, size=DIM))
    turns = blob(base, 20, spread, rng)
    for i in range(2):
        vector = unit(rng.normal(0.0, 1.0, size=DIM))
        turns.append(
            TurnEvidence(index=20 + i, start=900.0 + i, end=901.0 + i,
                         speech_seconds=1.0, primary=vector, secondary=vector)
        )
    clusters = cluster_turns(turns)
    base_set = set(range(20))
    absorbed = {
        member for cl in clusters for member in cl.members
        if member in {20, 21} and (set(cl.members) & base_set)
    }
    biggest = max((len(set(cl.members) & base_set) for cl in clusters), default=0)
    check(
        "孤立向量不被吸进主簇",
        not absorbed and biggest >= 18,
        f"主簇含 {biggest}/20 个目标轮次，被吸入的孤立向量 {len(absorbed)} 个",
    )

    # 6. 总时长不足的簇被过滤
    rng = np.random.default_rng(SEED)
    base = unit(rng.normal(0.0, 1.0, size=DIM))
    check("总时长不足的簇被过滤", cluster_turns(blob(base, 4, spread, rng, duration=1.0)) == ())

    # 7. 目标选中、簇外双向拒绝。关键性质是"簇外一个都不能放进"，
    # 簇内因边界浮动允许少量出入。
    rng = np.random.default_rng(SEED)
    base = unit(rng.normal(0.0, 1.0, size=DIM))
    other = axis_at(0.25, rng, base)
    turns = blob(base, 20, spread, rng) + blob(other, 10, spread, rng, start=20)
    clusters = cluster_turns(turns)
    result = assign_target(turns, clusters, profile_for(base))
    inside = sum(result.membership[i] for i in range(20))
    outside = sum(result.membership[i] for i in range(20, 30))
    check(
        "目标簇选中且簇外全部拒绝",
        result.reason == "target_cluster_found" and inside >= 18 and outside == 0,
        f"簇内 {inside}/20，簇外误判 {outside}/10",
    )

    # 8. 无簇达标 → 保守档
    rng = np.random.default_rng(SEED)
    base = unit(rng.normal(0.0, 1.0, size=DIM))
    far = axis_at(0.10, rng, base)
    turns = blob(far, 20, spread, rng)
    result = assign_target(turns, cluster_turns(turns), profile_for(base))
    check(
        "无簇达标 → 不输出",
        result.target_cluster_id is None and not any(result.membership),
        result.reason,
    )

    # 9. dominance 冲突 → 保守档
    rng = np.random.default_rng(SEED)
    base = unit(rng.normal(0.0, 1.0, size=DIM))
    near = axis_at(0.62, rng, base)
    turns = blob(base, 6, spread, rng) + blob(near, 20, spread, rng, start=6)
    result = assign_target(turns, cluster_turns(turns), profile_for(base))
    check(
        "目标分数最高的簇不是最大簇 → 不输出",
        result.target_cluster_id is None,
        result.reason,
    )

    # 10. 排除余量不足 → 拒绝
    rng = np.random.default_rng(SEED)
    base = unit(rng.normal(0.0, 1.0, size=DIM))
    turns = blob(base, 20, spread, rng)
    result = assign_target(
        turns, cluster_turns(turns), profile_for(base),
        exclusion_centroids=[("绿", base)], margin_floor=0.10,
    )
    check("与排除角色几乎重合 → 不选为目标簇", result.target_cluster_id is None, result.reason)

    # 11. 修剪均值抗污染
    rng = np.random.default_rng(SEED)
    base = unit(rng.normal(0.0, 1.0, size=DIM))
    intruder = axis_at(0.45, rng, base)
    clean = np.stack([unit(base + rng.normal(0, 0.05, DIM)) for _ in range(25)])
    polluted = np.vstack([clean, np.stack([intruder] * 3)])
    pure = trimmed_centroid(clean)
    dirty = trimmed_centroid(polluted)
    angle = float(np.degrees(np.arccos(np.clip(float(pure @ dirty), -1.0, 1.0))))
    check("修剪均值抗污染（偏移 < 8°）", angle < 8.0, f"实际偏移 {angle:.2f}°")

    # 12. unusable 轮次不参与
    rng = np.random.default_rng(SEED)
    base = unit(rng.normal(0.0, 1.0, size=DIM))
    turns = blob(base, 20, spread, rng)
    turns.append(
        TurnEvidence(index=20, start=900.0, end=902.0, speech_seconds=2.0,
                     primary=np.zeros(DIM), secondary=np.zeros(DIM), usable=False)
    )
    clusters = cluster_turns(turns)
    result = assign_target(turns, clusters, profile_for(base))
    check(
        "unusable 轮次不进入聚类也不被接受",
        20 not in {m for cl in clusters for m in cl.members}
        and result.membership[20] is False,
    )

    passed = sum(1 for ok, _, _ in RESULTS if ok)
    total = len(RESULTS)
    print(f"\n{passed}/{total} 通过")
    if passed != total:
        print("\n失败项：")
        for ok, name, detail in RESULTS:
            if not ok:
                print(f"  - {name}" + (f"  ({detail})" if detail else ""))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
