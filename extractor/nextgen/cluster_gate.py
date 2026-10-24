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


def _cluster_windows_strong(local, primary_floor, secondary_floor,
                            a: float, b: float) -> bool:
    """1.8s 簇滑窗权威豁免（Stage 13）：候选区间在 ClusterAnalyzer 自身的
    1.8s 滑窗下全窗通过、双通道强匹配、且对非目标簇无竞争劣势时，豁免上游
    0.6s 超短窗的瞬时抖动（男声/轻声独白在 0.6s 窗内极易被一个爆破音打落）。
    """
    if not local:
        return False
    rows = [
        row
        for group in local.values() if isinstance(group, list)
        for row in group
        if isinstance(row, dict) and row.get("span")
        and min(row["span"][1], b) - max(row["span"][0], a) >= CANDIDATE_WINDOW_OVERLAP
    ]
    if not rows or not all(bool(row.get("accepted")) for row in rows):
        return False
    p_floor = (primary_floor or 0.56) + 0.02
    s_floor = (secondary_floor or 0.50) + 0.02
    min_p = min(float(row.get("primary", 0.0)) for row in rows)
    min_s = min(float(row.get("secondary", 0.0)) for row in rows)
    mean_p = sum(float(row.get("primary", 0.0)) for row in rows) / len(rows)
    mean_s = sum(float(row.get("secondary", 0.0)) for row in rows) / len(rows)
    # 任一窗口的竞争余量为负 → 不豁免（防救援错人）。
    if any(isinstance(row.get("cm_s"), (int, float)) and row["cm_s"] < 0.01
           for row in rows):
        return False
    if any(isinstance(row.get("cm_p"), (int, float)) and row["cm_p"] < 0.08
           for row in rows):
        return False
    return (min_p >= p_floor and min_s >= s_floor
            and mean_p >= p_floor + 0.03 and mean_s >= s_floor + 0.03)


def _local_classifier_multi_veto(candidate, start: float | None = None,
                                 end: float | None = None) -> bool:
    """局部分类器双强负窗否决：候选区间内出现 >= 2 个跌破 reject_threshold
    （约 -4.24）的 0.5s 强负分窗口 → 证据表明区间内混有他人语音。

    降入 Tier B（不整句否决）。实测 Ep1–3 全量扫描：该规则对真目标 0 误伤
    （单窗换气负分不触发），唯独精准命中 Ep3 [550.64,555.74] 的 4 个强负窗
    （最低 -12.40）。不要求两个负窗连续紧挨。
    """
    audit = candidate.diagnostics.get("local_classifier_audit")
    if not isinstance(audit, dict):
        return False
    windows = audit.get("windows") or []
    threshold = audit.get("reject_threshold")
    if not isinstance(threshold, (int, float)):
        threshold = -4.24
    a = candidate.start if start is None else start
    b = candidate.end if end is None else end
    neg_count = sum(
        1 for w in windows
        if isinstance(w, dict) and w.get("voiced")
        and isinstance(w.get("score"), (int, float))
        and min(float(w.get("end", 0)), b) - max(float(w.get("start", 0)), a) >= 0.25
        and float(w["score"]) <= threshold)
    return neg_count >= 2


def _window_min(candidate):
    """候选的窗口最低分；<=0 视为无窗口证据（不参与弱窗判定）。"""
    value = float(getattr(candidate, "window_min_score", 0.0) or 0.0)
    return value if value > 0.0 else None


def _rescue_tier(candidate, secondary_floor, *, local=None, primary_floor=None,
                 start=None, end=None):
    """高危救援候选严审：低副通道分或低窗口分 → 只能进 Tier B。

    1.8s 簇滑窗全过且双通道强匹配时（_cluster_windows_strong）豁免 0.6s
    短窗抖动，直接放行 Tier A（Stage 13 权威豁免）。
    """
    if _local_classifier_multi_veto(candidate):
        return "B"
    if _cluster_windows_strong(local, primary_floor, secondary_floor,
                               candidate.start if start is None else start,
                               candidate.end if end is None else end):
        return "A"
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

    tier = _rescue_tier(candidate, secondary_floor, local=local,
                        primary_floor=floors.get("primary"))
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
    primary_floor = floors.get("primary")
    accepted_spans = result.diagnostics.get("accepted_spans") or []
    # Stage 13：相邻干净子区间合并（间隔 <=0.35s 且间隙不碰阻塞区间）。
    # 上游把长轮次切成碎片时，连续同簇窗口各自 <1.2s 无法借尸还魂；
    # 合并后即可成为完整 Tier A 产出。
    merged_accepted_spans: list[list[float]] = []
    merged_constituents: list[list[tuple[float, float]]] = []
    for span_a, span_b in sorted(accepted_spans):
        if (merged_accepted_spans and span_a - merged_accepted_spans[-1][1] <= 0.35
                and not any(min(span_a, mark[1]) - max(merged_accepted_spans[-1][1], mark[0]) > 0.02
                            for mark in blocked_spans)):
            merged_accepted_spans[-1][1] = max(merged_accepted_spans[-1][1], span_b)
            merged_constituents[-1].append((span_a, span_b))
            continue
        merged_accepted_spans.append([span_a, span_b])
        merged_constituents.append([(span_a, span_b)])

    def make_crop(parent, start, end, reason, *, strict_parent=False):
        """从已验证干净区间裁出子句；子段继承父段的救援/弱窗审查。

        """
        child = replace(parent, start=start, end=end, accepted=True,
                        reject_reason="", diagnostics=dict(parent.diagnostics))
        if _local_classifier_multi_veto(child, start, end):
            # 裁切子段区间内仍有 >=2 个局部分类器强负窗：主干里混着他人
            # 语音，降入 Tier B 待试听（0 误伤实测见函数注释）。
            child.accepted = False
            child.reject_reason = "簇级身份未通过：local_classifier_multi_veto"
            child.diagnostics["cluster_tier"] = "B"
            child.diagnostics["cluster_reason"] = "local_classifier_multi_veto"
            borderline.append(child)
            return None
        child.diagnostics.update(cluster_cropped=True, cluster_identity=True,
                                 cluster_would_accept=True,
                                 cluster_parent_span=[parent.start, parent.end],
                                 cluster_reason=reason,
                                 cluster_tier="A")
        child.diagnostics["acoustic_identity_audit"] = {
            "span": [start, end], "passed": True, "method": "cluster_temporal_consensus"}
        window_min = _window_min(parent)
        rescue_marked = any(parent.diagnostics.get(k) for k in RESCUE_KEYS)
        strong_windows = _cluster_windows_strong(
            local, floors.get("primary"), secondary_floor, start, end)
        rescue_b = _rescue_tier(parent, secondary_floor, local=local,
                                primary_floor=floors.get("primary"),
                                start=start, end=end) == "B"
        strict_b = strict_parent and (window_min is None
                or (rescue_marked and window_min < PROMOTION_RESCUE_WINDOW_FLOOR))
        if not strong_windows and (rescue_b or strict_b):
            # 父段证据不足（救援标记 + 弱副通道/弱窗，或整体弱窗；混合嫌疑
            # 父段还要求窗口证据齐全、救援标记者窗口分更高）：子段只进
            # Tier B 待人工/多模型仲裁，不进主输出。1.8s 簇滑窗全过且双通道
            # 强匹配时豁免（Stage 13 权威豁免）。
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
                    for a, b in merged_accepted_spans:
                        start, end = max(candidate.start, a), min(candidate.end, b)
                        if end - start >= min_seconds:
                            make_crop(candidate, start, end, reason)
        elif tier == "B" and not hard:
            # 边缘召回池（Tier B）：保留在 rejected 里，由导出层写入 _borderline/
            # 供前端 1 分钟捡漏；绝不混入主输出。
            candidate.diagnostics["cluster_tier"] = "B"
            borderline.append(candidate)
        elif (not hard and candidate.reject_reason in (
                "声纹匹配不足", *MIXED_CROP_REASONS, "说话回合过短")):
            # Tier B → Tier A 晋升：要求救援标记候选窗口分更高，且父轮次
            # 滑窗通过率达标（低通过率的混合长句走裁切路径而非整句晋升）。
            ratio = _turn_pass_ratio(candidate, turns, local)
            rescue_marked = any(candidate.diagnostics.get(k) for k in RESCUE_KEYS)
            window_min = _window_min(candidate)
            promotable = (
                allowed and candidate.duration >= min_seconds
                and candidate.reject_reason == "声纹匹配不足"
                and (not rescue_marked or window_min is None
                     or window_min >= PROMOTION_RESCUE_WINDOW_FLOOR)
                and (ratio is None or ratio >= PROMOTION_TURN_PASS_RATIO))
            whole_promoted = False
            if promotable and not any(
                    min(candidate.end, a.end) > max(candidate.start, a.start)
                    for a in [*(c for c in accepted if all(c is not r for r in removed)),
                              *promoted, *cropped]):
                candidate.accepted = True
                candidate.reject_reason = ""
                candidate.diagnostics["cluster_tier"] = "A"
                promoted.append(candidate)
                whole_promoted = True
            if not whole_promoted:
                # 未能整句晋升（常见于首/尾 1 个换气窗口未过的 4~12 秒高分
                # 长句）：与已验证干净区间求交集裁出干净主干升 Tier A。
                # 双重门槛：父轮次滑窗通过率 >= PROMOTION_TURN_PASS_RATIO
                # （低通过率混合长句的"干净区间"边界不可信，实测 Ep2
                # [228.44,230.54] 借此混入），加上 make_crop 的救援/弱窗/
                # 双强负窗审查——混音主干会被降入 Tier B 而不是混进主输出。
                croppable = ratio is None or ratio >= PROMOTION_TURN_PASS_RATIO
                cropped_any = False
                if croppable:
                    for a, b in merged_accepted_spans:
                        start, end = max(candidate.start, a), min(candidate.end, b)
                        if end - start >= min_seconds:
                            child = make_crop(candidate, start, end, "promoted_subspan",
                                              strict_parent=False)
                            if child is not None:
                                cropped_any = True
                if not cropped_any and (promotable
                                        or _has_family_overlap(candidate, turns, result)):
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
        accepted[:] = _deduplicate_tier_a([*kept, *cropped])
        rejected[:] = [c for c in rejected if id(c) not in promoted_ids] + removed
        _refine_tier_b(borderline, accepted)
        # Stage 13 孤儿实体化：合并后的已认证干净区间若未有任何 Tier A 候选
        # 覆盖（上游碎片化/说话回合过短导致），直接合成 Tier A 候选交给
        # STT——ClusterAnalyzer 的 1.8s 窗口认证本身就是产出凭据。
        from ..types import CandidateSentence
        tier_a_spans = [(c.start, c.end) for c in accepted]
        synthesized = 0
        for merged_index, (span_start, span_end) in enumerate(merged_accepted_spans):
            if span_end - span_start < min_seconds:
                continue
            if any(min(span_end, mark_b) - max(span_start, mark_a) > 0.02
                   for mark_a, mark_b in blocked_spans):
                continue
            constituents = merged_constituents[merged_index] if merged_index < len(merged_constituents) else []
            # 合成凭据检查：accepted 窗口运行段本身已过 floors/margins/cohort，
            # 此处只复核分数地板与竞争余量（跨边缘的 rejected 邻窗不计入，
            # 否则 0.9s hop 下任何 1.8s span 都会因相邻窗被误杀）。
            # Stage 13：合成区间必须继承重叠候选的局部分类器审计——已知
            # 负例多数带强负窗（0 误伤实测见函数注释）。
            class _SpanShim:
                start, end = span_start, span_end
                diagnostics = {"local_classifier_audit": None}
            for pool_candidate in [*rejected, *originally_accepted]:
                if (min(pool_candidate.end, span_end) - max(pool_candidate.start, span_start)
                        < 0.45 * min(pool_candidate.duration or 1e-6, span_end - span_start)):
                    continue
                audit = pool_candidate.diagnostics.get("local_classifier_audit")
                if isinstance(audit, dict) and audit.get("windows"):
                    _SpanShim.diagnostics["local_classifier_audit"] = audit
                    break
            if _local_classifier_multi_veto(_SpanShim, span_start, span_end):
                continue
            constituent_ok = True
            for s_a, s_b in constituents:
                rows = [
                    row for group in (local or {}).values() if isinstance(group, list)
                    for row in group
                    if isinstance(row, dict) and row.get("accepted")
                    and row.get("span")
                    and min(row["span"][1], s_b) - max(row["span"][0], s_a) >= 0.45
                ]
                if not rows:
                    constituent_ok = False
                    break
                if min(float(row["primary"]) for row in rows) < (primary_floor or 0.56):
                    constituent_ok = False
                    break
                if min(float(row["secondary"]) for row in rows) < (secondary_floor or 0.50):
                    constituent_ok = False
                    break
                if any(isinstance(row.get("cm_s"), (int, float)) and row["cm_s"] < 0.01
                       for row in rows):
                    constituent_ok = False
                    break
                if any(isinstance(row.get("cm_p"), (int, float)) and row["cm_p"] < 0.08
                       for row in rows):
                    constituent_ok = False
                    break
            if not constituent_ok:
                continue
            if _covered_ratio(span_start, span_end, tier_a_spans) >= 0.3:
                continue
            accepted.append(CandidateSentence(
                span_start, span_end, "", accepted=True,
                diagnostics={
                    "cluster_span_synthesized": True,
                    "cluster_tier": "A",
                    "cluster_identity": True,
                    "cluster_would_accept": True,
                    "cluster_mode": mode,
                    "cluster_target_id": result.target_cluster_id,
                    "acoustic_identity_audit": {
                        "span": [span_start, span_end], "passed": True,
                        "method": "cluster_span_synthesis"},
                }))
            tier_a_spans.append((span_start, span_end))
            synthesized += 1
        if synthesized:
            # 合成 span 来自不同轮次的合并区间，互相之间也可能包含——
            # 再跑一次包含去重保证 Tier A 零重叠。
            accepted[:] = _deduplicate_tier_a(accepted)
            accepted.sort(key=lambda c: (c.start, c.end))
    refined_b = sum(
        1 for c in borderline if c.diagnostics.get("cluster_tier") == "B")
    return {"removed": len(removed), "promoted": len(promoted),
            "cropped": len(cropped), "borderline": refined_b,
            "synthesized": sum(1 for c in accepted
                               if c.diagnostics.get("cluster_span_synthesized"))}


def _deduplicate_tier_a(candidates, containment=0.85):
    """Tier A 内部父子重复消解：仅当短片段被长片段包裹 >= 85% 自身时长时
    去重（保留更完整的长句）。不设数量上限、不按分数替换——相邻独立短句
    与 Tier A 时间轴覆盖均不受影响。"""
    ordered = sorted(candidates, key=lambda c: (-c.duration, c.start))
    selected: list = []
    for cand in ordered:
        if any(min(cand.end, e.end) - max(cand.start, e.start)
               >= containment * max(1e-6, cand.duration)
               for e in selected):
            continue
        selected.append(cand)
    return sorted(selected, key=lambda c: (c.start, c.end))


def _tier_b_rank(candidate) -> float:
    """Tier B 捡漏优先级：上游已接受的候选（完整流水线验证过、只在簇层
    无法确认）显著优先于从未被任何环节接受的候选。"""
    upstream = 0.8 if candidate.diagnostics.get("cluster_removed") else 0.0
    return (upstream
            + 0.5 * float(getattr(candidate, "speaker_score", 0.0) or 0.0)
            + 0.3 * float(_secondary_hint(candidate) or 0.0)
            + 0.2 * float(_window_min(candidate) or 0.0))


def _covered_ratio(span_start, span_end, spans) -> float:
    """[span_start, span_end] 被 spans 时间轴覆盖的时长比例。"""
    duration = max(1e-6, span_end - span_start)
    covered, cursor = 0.0, span_start
    for left, right in sorted(spans):
        if right <= cursor or left >= span_end:
            continue
        covered += max(0.0, min(span_end, right) - max(cursor, left))
        cursor = max(cursor, right)
    return min(covered / duration, 1.0)


def _refine_tier_b(borderline, accepted, tier_a_coverage=0.85,
                   mutual_containment=0.80):
    """Tier B 无损精炼：互斥 + 包含去重 + 置信度排序，绝不设数量上限。

    仅隐藏两类纯重复碎片：已被 Tier A 覆盖 >= 85% 时长（covered_by_tier_a），
    或与更高优先级 Tier B 候选重叠 >= 80% 自身时长（tier_b_nms_suppressed）。
    其余全部保留并按 _tier_b_rank 降序排列——Tier A ∪ Tier B 的时间轴
    覆盖面积与精炼前完全一致，覆盖率一分不掉。
    """
    if not borderline:
        return
    tier_a_spans = [(a.start, a.end) for a in accepted]
    survivors = []
    for candidate in borderline:
        if _covered_ratio(candidate.start, candidate.end, tier_a_spans) >= tier_a_coverage:
            candidate.diagnostics["cluster_tier"] = "covered_by_tier_a"
            continue
        survivors.append(candidate)
    ranked = sorted(survivors, key=_tier_b_rank, reverse=True)
    kept: list = []
    for candidate in ranked:
        if any(min(candidate.end, k.end) - max(candidate.start, k.start)
               >= mutual_containment * max(1e-6, candidate.duration)
               for k in kept):
            candidate.diagnostics["cluster_tier"] = "tier_b_nms_suppressed"
            continue
        candidate.diagnostics["tier_b_rank_score"] = round(_tier_b_rank(candidate), 4)
        kept.append(candidate)


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
