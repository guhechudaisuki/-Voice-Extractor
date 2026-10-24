"""扫描主通道阈值与副通道否决地板，看簇结构怎么变。

聚类用平均连接：主通道决定能否合并，副通道只否决明显不该合并的配对。
本脚本复用已导出的 npz，不重算嵌入。

    python evaluation\sweep_cluster_linkage.py --embeddings work\embeddings\5bd1f3.npz
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
    SECONDARY_VETO_FLOOR,
    agglomerative,
)

# 一集动画的合理说话人数量级。用来判断碎得是否过分。
SANE_CLUSTER_RANGE = (5, 40)


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    usable = payload["usable"]
    primary, secondary = payload["primary"][usable], payload["secondary"][usable]
    primary = primary / np.linalg.norm(primary, axis=1, keepdims=True)
    norms = np.linalg.norm(secondary, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return primary, secondary / norms


def describe(groups: list[list[int]], total: int) -> str:
    sizes = sorted((len(g) for g in groups), reverse=True)
    if not sizes:
        return "无簇"
    head = ", ".join(str(s) for s in sizes[:6])
    tail = len(sizes) - 6
    suffix = f" …(+{tail})" if tail > 0 else ""
    flag = ""
    if not SANE_CLUSTER_RANGE[0] <= len(sizes) <= SANE_CLUSTER_RANGE[1]:
        flag = "  ← 簇数超出合理范围" if len(sizes) > SANE_CLUSTER_RANGE[1] else "  ← 簇数过少"
    return f"{len(sizes):>4} 簇: {head}{suffix}  最大占比 {sizes[0] / total:.0%}{flag}"


def distribution(label: str, matrix: np.ndarray) -> None:
    count = matrix.shape[0]
    off = ~np.eye(count, dtype=bool)
    values = matrix[off]
    print(f"{label}两两余弦分布:")
    for name, value in (
        ("最小", values.min()),
        ("1%", np.quantile(values, 0.01)),
        ("中位", np.median(values)),
        ("99%", np.quantile(values, 0.99)),
        ("最大", values.max()),
    ):
        print(f"  {name:>4}: {value:+.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument(
        "--thresholds", type=float, nargs="*",
        default=[0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
    )
    parser.add_argument(
        "--veto-floors", type=float, nargs="*",
        default=[-0.10, 0.00, 0.05, 0.10, 0.20],
    )
    args = parser.parse_args()

    primary, secondary = load(args.embeddings)
    count = primary.shape[0]
    print(f"可用轮次: {count}")
    print(f"主通道 {primary.shape[1]} 维，副通道 {secondary.shape[1]} 维\n")

    distribution("主通道", primary @ primary.T)
    print()
    distribution("副通道", secondary @ secondary.T)

    print(f"\n【主阈值扫描】副通道否决地板固定 {SECONDARY_VETO_FLOOR:.2f}")
    for value in args.thresholds:
        groups = agglomerative(
            primary, secondary,
            link_primary=value, veto_floor=SECONDARY_VETO_FLOOR,
        )
        print(f"  {value:.2f}  {describe(groups, count)}")

    print("\n【否决地板扫描】主阈值固定 0.55")
    for floor in args.veto_floors:
        groups = agglomerative(
            primary, secondary, link_primary=0.55, veto_floor=floor,
        )
        print(f"  {floor:+.2f}  {describe(groups, count)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
