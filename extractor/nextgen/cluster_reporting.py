"""Shared report schema for offline and pipeline cluster decisions."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

def load_review(path: Path | None, *, source_sha256: str | None = None) -> list[dict]:
    """读用户试听结论。

    真实文件用的是 excluded_clips[]，每条有 span 与 reason；部分错误的那条
    额外带 negative_span。这里统一成 [{span, kind}] 列表，不依赖 reason 的文字。
    """
    if path is None:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if source_sha256 is not None and data.get("source_stem_sha256") != source_sha256:
        raise ValueError("试听标签与当前人声轨 SHA256 不一致，不能用于验收")
    rows = data.get("excluded_clips") or data.get("flagged_outputs") or []
    out: list[dict] = []
    for row in rows:
        span = row.get("span") or [row.get("start"), row.get("end")]
        if not span or len(span) != 2 or span[0] is None or span[1] is None:
            continue
        # 带 negative_span 的是"部分非目标"：整句有一半是别人。
        kind = row.get("kind") or (
            "mixed" if row.get("negative_span") else "entire_other"
        )
        out.append(
            {
                "span": (float(span[0]), float(span[1])),
                "kind": kind,
                "ordinal": row.get("ordinal"),
                "negative_span": row.get("negative_span"),
            }
        )
    return out


def review_clip(row: dict, turns, result) -> dict:
    """每条试听标签只计一次；覆盖不全不能当作成功挡下。"""
    span = row.get("negative_span") or row["span"]
    start, end = map(float, span)
    if not np.isfinite([start, end]).all() or end <= start:
        raise ValueError("试听标签必须包含有限且递增的时间区间")
    matches, intervals = [], []
    outliers = set(result.outliers)
    for turn in turns:
        left, right = max(start, turn.start), min(end, turn.end)
        if right <= left:
            continue
        intervals.append((left, right))
        matches.append({
            "index": turn.index,
            "span": [turn.start, turn.end],
            "overlap_seconds": round(right - left, 6),
            "cluster_id": result.cluster_of[turn.index],
            "accepted_by_cluster": bool(result.membership[turn.index]),
            "is_outlier": turn.index in outliers,
        })

    covered, cursor = 0.0, start
    for left, right in sorted(intervals):
        covered += max(0.0, right - max(left, cursor))
        cursor = max(cursor, right)
    accepted_spans = result.diagnostics.get("accepted_spans")
    retained = (any(min(end, b) > max(start, a) + 1e-6 for a, b in accepted_spans)
                if accepted_spans is not None else any(turn["accepted_by_cluster"] for turn in matches))
    if result.target_cluster_id is None:
        status = "withheld_no_target"
    elif retained:
        status = "retained"
    elif covered < end - start - 1e-6 or (row["kind"] == "mixed" and not row.get("negative_span")):
        status = "unresolved"
    elif any(turn["is_outlier"] for turn in matches):
        status = "rejected_with_outlier"
    else:
        status = "rejected_elsewhere"
    return {
        "ordinal": row.get("ordinal"), "span": list(row["span"]),
        "kind": row["kind"], "evaluated_span": [start, end],
        "coverage_ratio": round(covered / (end - start), 6),
        "status": status, "in_target_cluster": retained, "turns": matches,
    }


def summarize(result, turns, review: list[dict]) -> dict:
    """把判决结果整理成可读报告，重点是"错人的段落在哪个簇"。"""
    clusters = []
    for cluster in result.clusters:
        members = [
            {
                "index": index,
                "span": [
                    round(turns[index].start, 3), round(turns[index].end, 3),
                ],
                "speech_seconds": round(turns[index].speech_seconds, 3),
                "accepted_by_cluster": bool(
                    result.membership[index] and index in set(cluster.members)
                ),
            }
            for index in cluster.members
        ]
        clusters.append(
            {
                "cluster_id": cluster.cluster_id,
                "member_count": len(cluster.members),
                "total_speech_seconds": cluster.total_speech_seconds,
                "primary_cohesion": cluster.primary_cohesion,
                "primary_median": cluster.primary_median,
                "target_score": cluster.target_score,
                "primary_target_score": cluster.primary_target_score,
                "secondary_target_score": cluster.secondary_target_score,
                "nearest_other_role": cluster.nearest_other_role,
                "nearest_other_score": cluster.nearest_other_score,
                "target_margin": cluster.target_margin,
                "is_target": cluster.is_target,
                "outliers": list(cluster.outliers),
                "members": members,
            }
        )

    flagged: dict[str, list[dict]] = {}
    for row in review:
        flagged.setdefault(row["kind"], []).append(review_clip(row, turns, result))

    # 整句错人与局部污染分别计数，二者都不能视作可发布的正确输出。
    identity = flagged.get("entire_other", [])
    inside = [row for row in identity if row["status"] == "retained"]
    outlier = [row for row in identity if row["status"] == "rejected_with_outlier"]
    outside = [row for row in identity if row["status"] == "rejected_elsewhere"]
    unresolved = [row for row in identity if row["status"] == "unresolved"]
    withheld = [row for row in identity if row["status"] == "withheld_no_target"]
    boundary = flagged.get("mixed", [])
    boundary_retained = sum(row["status"] == "retained" for row in boundary)
    boundary_unresolved = sum(row["status"] == "unresolved" for row in boundary)
    boundary_withheld = sum(row["status"] == "withheld_no_target" for row in boundary)
    verdict = _verdict(len(inside), len(identity))
    if result.target_cluster_id is None:
        verdict = "no_target_cluster"
    elif unresolved or boundary_unresolved:
        verdict = "incomplete_review_coverage"
    elif verdict == "review_errors_blocked" and boundary_retained:
        verdict = "identity_blocked_boundary_remaining"

    return {
        "schema_version": 3,
        "evaluation_scope": "identity_before_stt_not_human_acceptance",
        "reason": result.reason,
        "target_cluster_id": result.target_cluster_id,
        "diagnostics": result.diagnostics,
        "cluster_count": len(result.clusters),
        "accepted_turn_count": sum(result.membership),
        "clusters": clusters,
        "review_check": {
            "whole_wrong_count": len(identity),
            "whole_wrong_inside_target": len(inside),
            "whole_wrong_elsewhere": len(outside),
            "whole_wrong_caught_by_outlier_rule": len(outlier),
            "whole_wrong_unresolved": len(unresolved),
            "whole_wrong_withheld_no_target": len(withheld),
            "mixed_clips_in_target": boundary_retained,
            "mixed_clips_unresolved": boundary_unresolved,
            "mixed_clips_withheld_no_target": boundary_withheld,
            "mixed_clips_total": len(boundary),
            "verdict": verdict,
            "notes": (
                "按人工试听片段计数，不按轮次计数。mixed 只检查已标注的负例区间；"
                "覆盖不足单列 unresolved。簇内保留不等于最终导出，零目标不算成功。"
                "此结果不测召回，也不是未见集泛化证据。"
            ),
            "details": flagged,
        },
    }


def _verdict(inside: int, total: int) -> str:
    if total == 0:
        return "no_review_overlap"
    if inside == 0:
        return "review_errors_blocked"
    if inside <= max(1, total // 2):
        return "partially_solved"
    return "not_solved"


