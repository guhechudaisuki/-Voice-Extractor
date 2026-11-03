"""Shared report schema for offline and pipeline cluster decisions."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

def load_review(path: Path | None, *, source_sha256: str | None = None) -> list[dict]:
    """Normalize excluded_clips and optional negative_span labels into span/kind rows."""
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
        # negative_span 标记句内污染，而非整句身份错误。
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


def _member_angular_geometry(turns, indexes):
    """Store exact pairwise angles without exporting the encoder vectors."""
    count = len(indexes)
    if count < 2:
        return None
    distances = np.zeros((count, count), dtype=np.float64)
    supports = np.zeros((count, count), dtype=np.int32)
    channels = []
    for channel in ("primary", "secondary", "tertiary"):
        valid, vectors = [], []
        dimension = None
        for position, index in enumerate(indexes):
            raw = getattr(turns[index], channel, None)
            if raw is None:
                continue
            vector = np.asarray(raw, dtype=np.float64)
            if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
                continue
            norm = float(np.linalg.norm(vector))
            if norm == 0 or (dimension is not None and vector.size != dimension):
                continue
            dimension = vector.size
            valid.append(position)
            vectors.append(vector / norm)
        if len(valid) < 2:
            continue
        matrix = np.stack(vectors)
        angles = np.arccos(np.clip(matrix @ matrix.T, -1., 1.)) / np.pi
        selected = np.ix_(valid, valid)
        distances[selected] += angles
        supports[selected] += 1
        channels.append(channel)
    if not channels:
        return None
    upper = np.triu_indices(count, 1)
    return {
        "method": "mean_multiview_angular_distance",
        "member_indexes": list(indexes),
        "channels": channels,
        "upper_triangle": [round(float(total / support), 6) if support else None
                           for total, support in zip(distances[upper], supports[upper])],
    }


def summarize(result, turns, review: list[dict]) -> dict:
    """把判决结果整理成可读报告，重点是"错人的段落在哪个簇"。"""
    clusters = []
    def normalized(vector):
        vector = np.asarray(vector, dtype=np.float64)
        return vector / max(float(np.linalg.norm(vector)), 1e-12)

    centroids = {c.cluster_id: (normalized(c.primary_centroid), normalized(c.secondary_centroid))
                 for c in result.clusters}

    def member_geometry(index, cluster):
        turn = turns[index]
        p, s = normalized(turn.primary), normalized(turn.secondary)
        own_pc, own_sc = centroids[cluster.cluster_id]
        own_p, own_s = float(p @ own_pc), float(s @ own_sc)
        others = [c for c in result.clusters if c.cluster_id != cluster.cluster_id
                  and not (cluster.is_target and c.is_target)]
        nearest_p = max(((float(p @ centroids[c.cluster_id][0]), c.cluster_id) for c in others),
                        default=None)
        nearest_s = max(((float(s @ centroids[c.cluster_id][1]), c.cluster_id) for c in others),
                        default=None)
        margin_p = own_p - nearest_p[0] if nearest_p else None
        margin_s = own_s - nearest_s[0] if nearest_s else None
        return {"primary_cluster_score": round(own_p, 5),
                "secondary_cluster_score": round(own_s, 5),
                "distinctiveness_margin": round(min(margin_p, margin_s), 5) if nearest_p else None,
                "cluster_margin_primary": round(margin_p, 5) if nearest_p else None,
                "cluster_margin_secondary": round(margin_s, 5) if nearest_s else None,
                "nearest_cluster_primary": nearest_p[1] if nearest_p else None,
                "nearest_cluster_secondary": nearest_s[1] if nearest_s else None}
    for cluster in result.clusters:
        members = [
            {
                "index": index,
                "span": [
                    round(turns[index].start, 3), round(turns[index].end, 3),
                ],
                "speech_seconds": round(turns[index].speech_seconds, 3),
                **member_geometry(index, cluster),
                "accepted_by_cluster": bool(
                    result.membership[index] and index in set(cluster.members)
                ),
            }
            for index in cluster.members
        ]
        # Discovery needs geometric outliers separately from target-membership outliers.
        secondary_median = float(np.median([m["secondary_cluster_score"] for m in members])) if members else 0.
        discovery_outliers = [m["index"] for m in members
                              if m["primary_cluster_score"] < cluster.primary_median - .18
                              or m["secondary_cluster_score"] < secondary_median - .18]
        margins = [m["distinctiveness_margin"] for m in members if m["distinctiveness_margin"] is not None]
        clusters.append(
            {
                "cluster_id": cluster.cluster_id,
                "member_count": len(cluster.members),
                "total_speech_seconds": cluster.total_speech_seconds,
                "reference_prior_weight": cluster.prior_weight,
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
                "discovery_outliers": discovery_outliers,
                "distinctiveness_margin": float(np.median(margins)) if margins else None,
                "member_acoustic_geometry": _member_angular_geometry(turns, cluster.members),
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


