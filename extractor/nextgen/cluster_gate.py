"""Map exact candidate intervals to canonical turns without allowing later rescue bypasses."""
from __future__ import annotations
from dataclasses import replace


HARD_VETOES = ("structural_hard_reject", "excluded_role_rejected",
               "final_same_speaker_internal_discard", "final_identity_boundary_discard",
               "multi_model_anchor_decisive_boundary", "boundary_suppressed_recovery")

# 高危救援路径标记：这些片段绕过了正常声纹下限，进入主输出（Tier A）前必须
# 通过更严的多通道复核；第 1~3 集已知错人段几乎全部带有这些标记且低分。
RESCUE_KEYS = ("locator_recovery", "target_locator_recovery", "tertiary_locator_recovery",
               "classifier_target_rescue", "wavlm_local_rescue", "wavlm_rescue",
               "subtitle_sentence_restored", "experimental_island_rescued_child")

# 带救援标记的候选进入 Tier A 的窗口最低分硬下限。
RESCUE_WINDOW_FLOOR = 0.62
RESCUE_SECONDARY_SLACK = 0.04
# 无救援标记但内部存在低分窗口的候选同样不得进 Tier A（实证阈值）。
WEAK_WINDOW_FLOOR = 0.56
# 救援标记候选被簇层晋升（Tier B → Tier A）时的更高窗口下限。
PROMOTION_RESCUE_WINDOW_FLOOR = 0.75
# 晋升候选要求其重叠轮次的滑窗通过率不低于该值（低通过率说明父轮次是
# 混合长句，裁切子段机制才是正确的救赎路径，而不是整句晋升）。
PROMOTION_TURN_PASS_RATIO = 0.5

# 上游因"疑似混合说话人"保守撤下的候选：不整句晋升，但允许与已验证干净
# 区间求交集裁出干净子段（≥ min_output_seconds）。
MIXED_CROP_REASONS = ("最终局部身份冲突", "疑似混合说话人")
# 局部声学分类器否决但与簇级成员证据冲突的候选：两个证据源打架，交给
# Tier B 人工仲裁（绝不进主输出）。
CLASSIFIER_CONFLICT_REASON = "局部声学分类器"

# 候选句级窗口核验的参数。
CANDIDATE_WINDOW_OVERLAP = 0.45
CANDIDATE_WINDOW_PASS_RATIO = 0.80
CANDIDATE_SPAN_TOLERANCE = 0.06
CANDIDATE_COVERAGE_RATIO = 0.85


def _secondary_hint(candidate):
    for key in ("global_target_secondary_score", "secondary_score",
                "secondary_speaker_score", "secondary_cluster_score",
                "camplus_score"):
        value = candidate.diagnostics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _window_min(candidate):
    """候选的窗口最低分；<=0 视为无窗口证据（不参与弱窗判定）。"""
    value = float(getattr(candidate, "window_min_score", 0.0) or 0.0)
    return value if value > 0.0 else None


def _rescue_tier(candidate, secondary_floor):
    """高危救援候选严审：低副通道分或低窗口分 → 只能进 Tier B。"""
    window_min = _window_min(candidate)
    if not any(candidate.diagnostics.get(key) for key in RESCUE_KEYS):
        if window_min is not None and window_min < WEAK_WINDOW_FLOOR:
            return "B"
        return "A"
    secondary = _secondary_hint(candidate)
    weak_secondary = (secondary is not None and secondary_floor is not None
                      and secondary < secondary_floor + RESCUE_SECONDARY_SLACK)
    weak_window = window_min is not None and window_min < RESCUE_WINDOW_FLOOR
    return "B" if (weak_secondary or weak_window) else "A"


def candidate_membership(candidate, turns, result, *, local=None, min_seconds=1.2):
    """候选句级核验：候选只为自己区间的声纹证据负责，不再被父轮次连坐。

    返回 ``(allowed, cluster_ids, reason, tier)``；tier 为 "A"（主输出）、
    "B"（边缘待捡漏）或 None（彻底拒绝）。
    """
    overlaps = [t for t in turns if min(t.end, candidate.end) > max(t.start, candidate.start)]
    ids = sorted({result.cluster_of[t.index] for t in overlaps})
    if result.target_cluster_id is None or not overlaps:
        return False, ids, "no_target_or_coverage", None
    floors = result.diagnostics.get("profile_floors") or {}
    secondary_floor = floors.get("secondary")

    accepted_spans = result.diagnostics.get("accepted_spans")
    if accepted_spans is not None:
        allowed = any(a <= candidate.start + CANDIDATE_SPAN_TOLERANCE
                      and b >= candidate.end - CANDIDATE_SPAN_TOLERANCE
                      for a, b in accepted_spans)
        reason = "target_cluster" if allowed else ""
        if not allowed:
            # 交集占候选时长 ≥85%（且 ≥ min_seconds）也算覆盖：边界微偏或
            # 头尾被裁掉少量杂音的候选句不应整句连坐。
            duration = max(candidate.duration, 1e-6)
            coverage = max(
                (min(candidate.end, b) - max(candidate.start, a)
                 for a, b in accepted_spans),
                default=0.0,
            )
            if coverage >= min_seconds and coverage >= CANDIDATE_COVERAGE_RATIO * duration:
                allowed, reason = True, "target_cluster_partial_coverage"
        if not allowed and local:
            # 候选句专属窗口：只看与候选区间重叠 ≥0.45s 的滑窗。
            rows = [
                row
                for group in local.values()
                if _group_overlaps_candidate(group, candidate)
                for row in group
                if isinstance(row, dict) and row.get("span") and (
                    min(row["span"][1], candidate.end)
                    - max(row["span"][0], candidate.start) >= CANDIDATE_WINDOW_OVERLAP)
            ]
            if rows:
                accepted = [bool(row.get("accepted")) for row in rows]
                ratio = sum(accepted) / len(accepted)
                if all(accepted) or (ratio >= CANDIDATE_WINDOW_PASS_RATIO
                                     and accepted[0] and accepted[-1]):
                    allowed, reason = True, "target_cluster_windows"
        if not allowed:
            border = len(result.borderline_membership) > 0 and any(
                result.borderline_membership[t.index] for t in overlaps)
            tier = "B" if border else None
            return False, ids, "local_identity_conflict", tier
    else:
        if any(not result.membership[t.index] for t in overlaps):
            border = len(result.borderline_membership) > 0 and any(
                result.borderline_membership[t.index] for t in overlaps)
            return False, ids, "non_target_or_impure_turn", ("B" if border else None)
        cursor = candidate.start
        for turn in sorted(overlaps, key=lambda t: t.start):
            if turn.start > cursor + 0.02:
                return False, ids, "uncovered_interval", None
            cursor = max(cursor, turn.end)
        if cursor < candidate.end - 0.02:
            return False, ids, "uncovered_interval", None

    tier = _rescue_tier(candidate, secondary_floor)
    if tier == "B":
        return False, ids, "rescue_strict_review", "B"
    return True, ids, "target_cluster", "A"


def _group_overlaps_candidate(group, candidate):
    """local 里每个键的窗口组只要有一条与候选重叠就参与核验。"""
    return isinstance(group, list) and any(
        isinstance(row, dict) and row.get("span")
        and min(row["span"][1], candidate.end) > max(row["span"][0], candidate.start)
        for row in group
    )


def _turn_pass_ratio(candidate, turns, local):
    """候选重叠轮次中最低的滑窗通过率；无窗口证据时返回 None。"""
    ratios = []
    for turn in turns:
        if min(turn.end, candidate.end) <= max(turn.start, candidate.start):
            continue
        rows = local.get(str(turn.index)) if local else None
        if not rows:
            continue
        accepted = [bool(row.get("accepted")) for row in rows if isinstance(row, dict)]
        if accepted:
            ratios.append(sum(accepted) / len(accepted))
    return min(ratios) if ratios else None


def apply_cluster_gate(accepted, rejected, turns, result, *, mode, min_seconds=1.2,
                       blocked_spans=(), local=None):
    if mode == "off":
        return {"removed": 0, "promoted": 0, "cropped": 0, "borderline": 0}
    removed, promoted, cropped, borderline = [], [], [], []
    originally_accepted = list(accepted)
    floors = result.diagnostics.get("profile_floors") or {}
    secondary_floor = floors.get("secondary")
    accepted_spans = result.diagnostics.get("accepted_spans") or []

    def make_crop(parent, start, end, reason, *, strict_parent=False):
        """从已验证干净区间裁出子句；子段继承父段的救援/弱窗审查。

        """
        child = replace(parent, start=start, end=end, accepted=True,
                        reject_reason="", diagnostics=dict(parent.diagnostics))
        child.diagnostics.update(cluster_cropped=True, cluster_identity=True,
                                 cluster_would_accept=True,
                                 cluster_parent_span=[parent.start, parent.end],
                                 cluster_reason=reason,
                                 cluster_tier="A")
        child.diagnostics["acoustic_identity_audit"] = {
            "span": [start, end], "passed": True, "method": "cluster_temporal_consensus"}
        window_min = _window_min(parent)
        rescue_marked = any(parent.diagnostics.get(k) for k in RESCUE_KEYS)
        if _rescue_tier(parent, secondary_floor) == "B" or (strict_parent and (
                window_min is None
                or (rescue_marked and window_min < PROMOTION_RESCUE_WINDOW_FLOOR))):
            # 父段证据不足（救援标记 + 弱副通道/弱窗，或整体弱窗；混合嫌疑
            # 父段还要求窗口证据齐全、救援标记者窗口分更高）：子段只进
            # Tier B 待人工/多模型仲裁，不进主输出。
            child.accepted = False
            child.reject_reason = "簇级身份未通过：rescue_strict_review"
            child.diagnostics["cluster_tier"] = "B"
            child.diagnostics["cluster_removed"] = False
            borderline.append(child)
            return None
        cropped.append(child)
        return child

    for candidate in [*originally_accepted, *rejected]:
        allowed, ids, reason, tier = candidate_membership(
            candidate, turns, result, local=local, min_seconds=min_seconds)
        hard = (any(candidate.diagnostics.get(k) for k in HARD_VETOES) or
                any(min(candidate.end, b) > max(candidate.start, a) for a, b in blocked_spans))
        allowed = allowed and not hard
        candidate.diagnostics.update(cluster_ids=ids, cluster_would_accept=allowed,
                                     cluster_mode=mode, cluster_reason=reason,
                                     cluster_target_id=result.target_cluster_id)
        if mode == "audit":
            continue
        if any(candidate is item for item in originally_accepted):
            if allowed and tier == "A":
                # 门控前已接受、簇层也放行的整句候选：若其所在轮次的滑窗通过率
                # 很低（混合长句），整句证据不足，降入 Tier B（干净子段另由
                # 裁切机制保留）。
                ratio = _turn_pass_ratio(candidate, turns, local)
                if ratio is not None and ratio < PROMOTION_TURN_PASS_RATIO:
                    candidate.accepted = False
                    candidate.reject_reason = "簇级身份未通过：low_window_pass_ratio"
                    candidate.diagnostics["cluster_removed"] = True
                    candidate.diagnostics["cluster_tier"] = "B"
                    removed.append(candidate)
                    borderline.append(candidate)
                else:
                    candidate.diagnostics["cluster_tier"] = "A"
            else:
                # 门控前已被上游流水线接受的候选被簇层拦下：默认进 Tier B
                # 捡漏池（上游已验证、簇层无法确认 → 交给人工试听）；仅当簇层
                # 有明确反向证据（重叠轮次全部属于非目标簇）或硬否决时才彻底
                # 撤下。
                keep_tier = ("B" if (tier == "B"
                                     or _is_borderline(candidate, turns, result)
                                     or _has_family_overlap(candidate, turns, result))
                             else None)
                candidate.accepted = False
                candidate.reject_reason = "簇级身份未通过：" + reason
                candidate.diagnostics["cluster_removed"] = True
                candidate.diagnostics["cluster_tier"] = keep_tier or "rejected"
                removed.append(candidate)
                if keep_tier == "B":
                    borderline.append(candidate)
                if not hard:
                    for a, b in accepted_spans:
                        start, end = max(candidate.start, a), min(candidate.end, b)
                        if end - start >= min_seconds:
                            make_crop(candidate, start, end, reason)
        elif tier == "B" and not hard:
            # 边缘召回池（Tier B）：保留在 rejected 里，由导出层写入 _borderline/
            # 供前端 1 分钟捡漏；绝不混入主输出。
            candidate.diagnostics["cluster_tier"] = "B"
            borderline.append(candidate)
        elif (candidate.reject_reason == "声纹匹配不足" and not hard):
            # Tier B → Tier A 晋升：要求救援标记候选窗口分更高，且父轮次
            # 滑窗通过率达标（低通过率的混合长句走裁切路径而非整句晋升）。
            ratio = _turn_pass_ratio(candidate, turns, local)
            rescue_marked = any(candidate.diagnostics.get(k) for k in RESCUE_KEYS)
            window_min = _window_min(candidate)
            promotable = (
                allowed and candidate.duration >= min_seconds
                and (not rescue_marked or window_min is None
                     or window_min >= PROMOTION_RESCUE_WINDOW_FLOOR)
                and (ratio is None or ratio >= PROMOTION_TURN_PASS_RATIO))
            if promotable and not any(
                    min(candidate.end, a.end) > max(candidate.start, a.start)
                    for a in [*(c for c in accepted if all(c is not r for r in removed)),
                              *promoted, *cropped]):
                candidate.accepted = True
                candidate.reject_reason = ""
                candidate.diagnostics["cluster_tier"] = "A"
                promoted.append(candidate)
            elif promotable or _has_family_overlap(candidate, turns, result):
                # 晋升证据不足但与目标簇家族重叠 → Tier B 待试听。
                candidate.diagnostics["cluster_tier"] = "B"
                borderline.append(candidate)
        elif (not hard
              and any(key in candidate.reject_reason for key in MIXED_CROP_REASONS)):
            # 混合嫌疑候选：不整句晋升，只裁出与已验证干净区间的交集子段。
            # 父段已被上游怀疑过，裁切审查从严格（窗口证据必须齐全）。
            for a, b in accepted_spans:
                start, end = max(candidate.start, a), min(candidate.end, b)
                if end - start >= min_seconds:
                    make_crop(candidate, start, end, "mixed_trimmed", strict_parent=True)
        elif (not hard and CLASSIFIER_CONFLICT_REASON in candidate.reject_reason
              and _has_strong_family_overlap(candidate, turns, result)):
            # 局部声学分类器说非目标、簇级证据说是成员：冲突交人工仲裁。
            candidate.diagnostics["cluster_tier"] = "B"
            candidate.diagnostics["cluster_reason"] = "classifier_conflict"
            borderline.append(candidate)
        if allowed:
            candidate.diagnostics["cluster_identity"] = True
            candidate.diagnostics["acoustic_identity_audit"] = {
                "span": [candidate.start, candidate.end], "passed": True,
                "method": "cluster_temporal_consensus",
            }
    if mode == "enforce":
        removed_ids, promoted_ids = {id(c) for c in removed}, {id(c) for c in promoted}
        kept = [c for c in accepted if id(c) not in removed_ids] + promoted
        unique_crops = []
        for child in sorted(cropped, key=lambda c: (-c.duration, c.start)):
            if not any(min(child.end, c.end) > max(child.start, c.start) for c in [*kept, *unique_crops]):
                unique_crops.append(child)
        cropped = unique_crops
        accepted[:] = kept + cropped
        rejected[:] = [c for c in rejected if id(c) not in promoted_ids] + removed
    return {"removed": len(removed), "promoted": len(promoted),
            "cropped": len(cropped), "borderline": len(borderline)}


def _deduplicate_tier_a(candidates, min_overlap=0.15):
    """消除 Tier A 内部互相重叠或包裹的冗余片段（父子重复导出）。

    优先保留：1) 非救援标记段优于救援段；2) 窗口最低分无明显短板时保留
    更完整的长句；3) 若长句窗口最低分比干净子句低 0.06 以上（长句带了
    杂音尾巴），用高纯度子句替换污染长句。
    """
    ordered = sorted(
        candidates,
        key=lambda c: (
            0 if any(c.diagnostics.get(k) for k in RESCUE_KEYS) else 1,
            round(c.duration, 3),
            float(getattr(c, "speaker_score", 0.0) or 0.0),
        ),
        reverse=True,
    )
    selected = []
    for cand in ordered:
        conflict = False
        for i, existing in enumerate(selected):
            ov = min(cand.end, existing.end) - max(cand.start, existing.start)
            if ov <= min_overlap:
                continue
            c_wmin = _window_min(cand) or 0.0
            e_wmin = _window_min(existing) or 0.0
            if (cand.duration < existing.duration
                    and c_wmin >= e_wmin + 0.06 and ov >= 0.8 * cand.duration):
                # 高纯度短子句替换带杂音尾巴的长父句。
                selected[i] = cand
            conflict = True
            break
        if not conflict:
            selected.append(cand)
    return sorted(selected, key=lambda c: (c.start, c.end))


def _is_borderline(candidate, turns, result):
    if len(result.borderline_membership) != len(turns):
        return False
    return any(
        result.borderline_membership[t.index]
        for t in turns
        if min(t.end, candidate.end) > max(t.start, candidate.start))


def _has_strong_family_overlap(candidate, turns, result):
    """候选与目标簇联盟的成员/边缘轮次直接重叠（强家族证据）。"""
    for turn in turns:
        if min(turn.end, candidate.end) <= max(turn.start, candidate.start):
            continue
        if len(result.membership) == len(turns) and result.membership[turn.index]:
            return True
        if len(result.borderline_membership) == len(turns) and result.borderline_membership[turn.index]:
            return True
    return False


def _has_family_overlap(candidate, turns, result):
    """候选与目标家族证据重叠。

    成员轮次 / 边缘轮次 / 未聚类旁白轮次，以及"灰簇"（簇质心过了主通道
    参考下限但未达标的次级簇）——灰簇不是明确反向证据，只能降级待试听。
    """
    target_ids = set(result.target_cluster_ids) or (
        {result.target_cluster_id} if result.target_cluster_id is not None else set())
    qualified = set(result.diagnostics.get("qualified_cluster_ids") or ())
    primary_floor = (result.diagnostics.get("profile_floors") or {}).get("primary")
    cluster_scores = {
        c.cluster_id: c.primary_target_score for c in result.clusters}
    for turn in turns:
        if min(turn.end, candidate.end) <= max(turn.start, candidate.start):
            continue
        if len(result.membership) == len(turns) and result.membership[turn.index]:
            return True
        if len(result.borderline_membership) == len(turns) and result.borderline_membership[turn.index]:
            return True
        cluster_id = result.cluster_of[turn.index]
        if cluster_id in target_ids | qualified | {-1}:
            return True
        if (primary_floor is not None and cluster_id in cluster_scores
                and cluster_scores[cluster_id] >= primary_floor):
            return True
    return False
