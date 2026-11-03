"""Map exact candidate intervals to canonical turns without allowing later rescue bypasses."""
from __future__ import annotations
from dataclasses import replace
import math

from .identity_cluster import cohort_z_probability
from .identity_trajectory import IdentityTrajectory
from .candidate_topology import CandidateTopology
from ..candidate_lattice import CandidateLattice
from ..recall import certificate_matches, subtract_intervals, MIN_TARGET_SECONDS


HARD_VETOES = ("structural_hard_reject", "excluded_role_rejected",
               "final_same_speaker_internal_discard", "final_identity_boundary_discard",
               "multi_model_anchor_decisive_boundary", "boundary_suppressed_recovery")

RESCUE_KEYS = ("locator_recovery", "target_locator_recovery", "tertiary_locator_recovery",
               "classifier_target_rescue", "wavlm_local_rescue", "wavlm_rescue",
               "subtitle_sentence_restored", "experimental_island_rescued_child")

RESCUE_WINDOW_FLOOR = 0.62
RESCUE_SECONDARY_SLACK = 0.04
WEAK_WINDOW_FLOOR = 0.56
PROMOTION_RESCUE_WINDOW_FLOOR = 0.75
PROMOTION_TURN_PASS_RATIO = 0.5

MIXED_CROP_REASONS = ("最终局部身份冲突", "疑似混合说话人")
CLASSIFIER_CONFLICT_REASON = "局部声学分类器"

CANDIDATE_WINDOW_OVERLAP = 0.45
CANDIDATE_WINDOW_PASS_RATIO = 0.80
CANDIDATE_SPAN_TOLERANCE = 0.06
CANDIDATE_COVERAGE_RATIO = 0.85


def _secondary_hint(candidate):
    for key in ("global_target_secondary_score", "secondary_score",
                "secondary_speaker_score", "secondary_cluster_score",
                "camplus_score"):
        value = candidate.diagnostics.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            return float(value)
    return None


def _interval_rows(local, start, end):
    """Collect windows that overlap the candidate's identity-bearing interval."""
    rows = [row for group in (local or {}).values() if isinstance(group, list)
            for row in group if isinstance(row, dict) and row.get("span")
            and min(row["span"][1], end) - max(row["span"][0], start)
            >= min(CANDIDATE_WINDOW_OVERLAP, end - start)]
    return sorted(rows, key=lambda row: (row["span"][0], row["span"][1]))


def _calibrate_candidate(candidate, turns, result, local):
    """Calibrate available evidence before ranking; absent channels carry no weight."""
    def finite(value):
        return isinstance(value, (int, float)) and math.isfinite(value)
    def lookup(mapping, index):
        return mapping.get(str(index), mapping.get(index))
    def average(values):
        values = [(v, w) for v, w in values if finite(v) and w > 0]
        return sum(v*w for v,w in values) / sum(w for _,w in values) if values else None
    overlaps = [(t, min(t.end, candidate.end) - max(t.start, candidate.start)) for t in turns]
    overlaps = [(t, w) for t, w in overlaps if w > 0]
    rows = [(r, min(r["span"][1], candidate.end) - max(r["span"][0], candidate.start))
            for r in _interval_rows(local, candidate.start, candidate.end)]
    reference = result.diagnostics.get("turn_reference_scores") or {}
    def reference_score(channel):
        local_score = average([(r.get(channel), w) for r, w in rows])
        if local_score is not None:
            return local_score
        return average([((lookup(reference, t.index) or {}).get(channel), w) for t,w in overlaps])
    primary = getattr(candidate, "speaker_score", None)
    if not finite(primary) or primary <= 0 or candidate.diagnostics.get("cluster_cropped"):
        primary = reference_score("primary")
    secondary = _secondary_hint(candidate)
    if secondary is None or candidate.diagnostics.get("cluster_cropped"):
        secondary = reference_score("secondary")
    tertiary = candidate.diagnostics.get("wavlm_score")
    tertiary_source = "candidate"
    if not finite(tertiary) or tertiary <= 0:
        tertiary = average([(r.get("tertiary"), w) for r, w in rows])
        tertiary_source = "local_windows"
        if tertiary is None:
            mapping = result.diagnostics.get("tertiary_scores") or {}
            tertiary = average([(lookup(mapping, t.index), w) for t,w in overlaps])
            tertiary_source = "overlapping_turns" if tertiary is not None else "unavailable"
    anime_char = reference_score("anime_char_score")
    anime_va = reference_score("anime_va_score")
    floors = result.diagnostics.get("profile_floors") or {}
    def sigmoid(value, floor, scale):
        return 1 / (1 + math.exp(-max(-30., min(30., (value-floor)/scale))))
    components = []
    for value, floor, scale, weight in (
        (primary, floors.get("primary", .58), .06, .30),
        (secondary, floors.get("secondary", .51), .06, .25),
        (tertiary, result.diagnostics.get("tertiary_floor", .75), .04, .20),
        (anime_char, floors.get("anime_char", .45), .07, .20),
        (anime_va, floors.get("anime_va", .45), .07, .20)):
        if finite(value):
            components.append((sigmoid(value, floor, scale), weight))
    cohort = result.diagnostics.get("turn_cohort_z") or {}
    for name, weight in (("cohort_z_primary", .15), ("cohort_z_secondary", .10)):
        z = average([((lookup(cohort, t.index) or {}).get(name), w) for t,w in overlaps])
        if z is not None:
            components.append((cohort_z_probability(z), weight))
    fusion = sum(v*w for v,w in components) / sum(w for _,w in components) if components else 0.
    calibrated_llrs = [(r.get('manifold_log_odds'), w) for r,w in rows
        if sum(item.get('method') == 'as_norm_gaussian_llr' for item in
               (r.get('identity_evidence', {}).get('calibration') or {}).values()) >= 2]
    episode_llr = average(calibrated_llrs)
    if episode_llr is not None:
        fusion = sigmoid(episode_llr, 0., 1.)
    candidate.diagnostics["calibrated_fusion_score"] = round(fusion, 6)
    candidate.diagnostics["fusion_evidence"] = {"primary": primary, "secondary": secondary,
        "tertiary": tertiary, "tertiary_source": tertiary_source,
        "anime_char_score": anime_char, "anime_va_score": anime_va,
        "method": "episode_as_norm_llr" if episode_llr is not None else "reference_sigmoid_and_topk_cohort_z",
        "episode_llr": episode_llr}
    for name in ("cm_p", "cm_s"):
        margins = [r[name] for r,_ in rows if finite(r.get(name))]
        if margins:
            candidate.diagnostics["cohort_" + name] = min(margins)
    candidate.diagnostics["tier_b_rank_score"] = round(_tier_b_rank(candidate), 6)


def _cluster_windows_strong(local, primary_floor, secondary_floor,
                            a: float, b: float, *, allow_anime=True) -> bool:
    """Require complete local coverage with strong identity and cohort margins."""
    if not local:
        return False
    rows = _interval_rows(local, a, b)
    if not rows or not all(bool(row.get("accepted")) for row in rows):
        return False
    if not allow_anime and _covered_ratio(a, b, [row["span"] for row in rows]) < .85:
        return False
    p_floor = (primary_floor or 0.56) + 0.02
    s_floor = (secondary_floor or 0.50) + 0.02
    min_p = min(float(row.get("primary", 0.0)) for row in rows)
    min_s = min(float(row.get("secondary", 0.0)) for row in rows)
    mean_p = sum(float(row.get("primary", 0.0)) for row in rows) / len(rows)
    mean_s = sum(float(row.get("secondary", 0.0)) for row in rows) / len(rows)
    if any(isinstance(row.get("cm_s"), (int, float)) and row["cm_s"] < 0.01
           for row in rows):
        return False
    if any(isinstance(row.get("cm_p"), (int, float)) and row["cm_p"] < 0.08
           for row in rows):
        return False
    if allow_anime and all(row.get("anime_register_shift") and not row.get("hard_veto") for row in rows):
        return (min_p >= (primary_floor or .56) + .04
                and min_s >= max(.20, (secondary_floor or .50) - .25))
    return (min_p >= p_floor and min_s >= s_floor
            and mean_p >= p_floor + 0.03 and mean_s >= s_floor + 0.03)


def _local_classifier_multi_veto(candidate, start: float | None = None,
                                 end: float | None = None, *, strong_windows=False) -> bool:
    """Reject sustained negative classifier evidence inside the requested interval."""
    audit = candidate.diagnostics.get("local_classifier_audit")
    if not isinstance(audit, dict):
        return False
    a = candidate.start if start is None else start
    b = candidate.end if end is None else end
    parent_a, parent_b = candidate.diagnostics.get("cluster_parent_span", (candidate.start, candidate.end))
    allow_near = (parent_b - parent_a <= 3.0 or not strong_windows
                  or b - a >= .75 * (parent_b - parent_a))
    return _classifier_audit_veto(audit, a, b, allow_near=allow_near)


def _classifier_audit_veto(audit, a, b, *, allow_near=True):
    windows = audit.get("windows") or []
    threshold = audit.get("reject_threshold")
    if not isinstance(threshold, (int, float)):
        threshold = -4.24
    relevant = [w for w in windows
        if isinstance(w, dict) and w.get("voiced")
        and isinstance(w.get("score"), (int, float))
        and min(float(w.get("end", 0)), b) - max(float(w.get("start", 0)), a) >= 0.25]
    neg_count = sum(float(w["score"]) <= threshold for w in relevant)
    near_count = sum(float(w["score"]) <= threshold + 1. for w in relevant)
    return neg_count >= 2 or (allow_near and bool(audit.get("rejected")) and near_count >= 3
                             and near_count >= .75 * len(relevant))


def _window_min(candidate):
    """候选的窗口最低分；<=0 视为无窗口证据（不参与弱窗判定）。"""
    value = float(getattr(candidate, "window_min_score", 0.0) or 0.0)
    return value if math.isfinite(value) and value > 0.0 else None


def _meets_export_length(start, end, minimum, result):
    if end - start >= minimum - 1e-9:
        return True
    if certificate_matches(start, end, result.diagnostics.get("certified_short_spans", ())):
        return True
    if minimum > 1.2 or end - start < 1.10 - 1e-9:
        return False
    return certificate_matches(start, end, [span
        for spans in result.diagnostics.get("clean_core_certificates", {}).values() for span in spans])


def _synthesis_regions(spans, occupied, minimum, result, trajectory):
    """Subtract existing exports, then assess each remaining interval on its own."""
    for start, end in spans:
        pieces = [(start, end)]
        for a, b in sorted(occupied, key=lambda span: (float(span[0]), float(span[1]))):
            pieces = [(x, y) for left, right in pieces
                      for x, y in ((left, min(right, a)), (max(left, b), right))
                      if y > x]
        for a, b in pieces:
            if _meets_export_length(a, b, minimum, result) and trajectory.assess(a, b)["accepted"]:
                yield a, b


def _join_short_remainders(candidates, spans, minimum, trajectory):
    """Keep a certified sentence's short residual audio attached to its export."""
    for candidate in candidates:
        for start, end in spans:
            if not (start <= candidate.start < candidate.end <= end):
                continue
            left = candidate.start - start
            right = end - candidate.end
            a = start if 0. < left < minimum else candidate.start
            b = end if 0. < right < minimum else candidate.end
            if (a, b) == (candidate.start, candidate.end):
                continue
            if any(other is not candidate and min(b, other.end) > max(a, other.start)
                   for other in candidates):
                continue
            evidence = trajectory.assess(a, b)
            if evidence["accepted"]:
                candidate.start, candidate.end = a, b
                candidate.diagnostics["identity_trajectory"] = evidence
                candidate.diagnostics["cluster_short_remainder_joined"] = True
                candidate.diagnostics["acoustic_identity_audit"] = {
                    "span": [a, b], "passed": True, "method": "interval_evidence"}


def _rescue_tier(candidate, secondary_floor, *, local=None, primary_floor=None,
                 start=None, end=None, trajectory=None):
    """Use interval evidence when available, otherwise retain the upstream review policy."""
    if trajectory is not None:
        evidence = trajectory.assess(candidate.start if start is None else start,
                                     candidate.end if end is None else end)
        if evidence["observed_fraction"] > 0:
            return "A" if evidence["accepted"] else "B"
    strong_windows = _cluster_windows_strong(local, primary_floor, secondary_floor,
        candidate.start if start is None else start, candidate.end if end is None else end)
    if (_local_classifier_multi_veto(candidate, start, end, strong_windows=strong_windows)
            or candidate.diagnostics.get("cluster_source_identity_conflict")):
        return "B"
    if strong_windows:
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


def candidate_membership(candidate, turns, result, *, local=None, min_seconds=1.2, trajectory=None):
    """Return local acceptance, cluster IDs, reason and output tier."""
    overlaps = [t for t in turns if min(t.end, candidate.end) > max(t.start, candidate.start)]
    ids = sorted({result.cluster_of[t.index] for t in overlaps})
    if not overlaps or (result.target_cluster_id is None
                        and not result.diagnostics.get("reference_mining")):
        return False, ids, "no_target_or_coverage", None
    floors = result.diagnostics.get("profile_floors") or {}
    secondary_floor = floors.get("secondary")

    accepted_spans = result.diagnostics.get("trajectory_spans", result.diagnostics.get("accepted_spans"))
    if accepted_spans is not None:
        allowed = any(a <= candidate.start + CANDIDATE_SPAN_TOLERANCE
                      and b >= candidate.end - CANDIDATE_SPAN_TOLERANCE
                      for a, b in accepted_spans)
        reason = "target_cluster" if allowed else ""
        if not allowed:
            duration = max(candidate.duration, 1e-6)
            coverage = max(
                (min(candidate.end, b) - max(candidate.start, a)
                 for a, b in accepted_spans),
                default=0.0,
            )
            if trajectory is None and coverage >= min_seconds and coverage >= CANDIDATE_COVERAGE_RATIO * duration:
                allowed, reason = True, "target_cluster_partial_coverage"
        if not allowed and local and trajectory is None:
            rows = _interval_rows(local, candidate.start, candidate.end)
            if rows:
                accepted = [bool(row.get("accepted")) for row in rows]
                ratio = sum(accepted) / len(accepted)
                covered = _covered_ratio(candidate.start, candidate.end,
                    [r["span"] for r in rows if r.get("accepted")])
                if covered >= CANDIDATE_COVERAGE_RATIO and (all(accepted) or (
                        ratio >= CANDIDATE_WINDOW_PASS_RATIO and accepted[0] and accepted[-1])):
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
                        primary_floor=floors.get("primary"), trajectory=trajectory)
    if tier == "B":
        return False, ids, "rescue_strict_review", "B"
    return True, ids, "target_cluster", "A"


def _turn_pass_ratio(candidate, turns, local):
    """候选重叠轮次中最低的滑窗通过率；无窗口证据时返回 None。"""
    ratios = []
    for turn in turns:
        if min(turn.end, candidate.end) <= max(turn.start, candidate.start):
            continue
        rows = local.get(str(turn.index)) if local else None
        if not rows:
            continue
        accepted = [bool(row.get("accepted")) for row in rows if isinstance(row, dict)
                    and row.get('micro_state') != 'uncertain']
        if accepted:
            ratios.append(sum(accepted) / len(accepted))
    return min(ratios) if ratios else None


def _positive_fine_review(candidate, evidence):
    """Keep a well-observed classifier/coarse disagreement available for review.

    Fine classifier support alone never establishes target identity. It must
    not disappear merely because the coarse cluster has no target membership.
    Sparse positive frames, contamination and forbidden spans remain excluded.
    """
    return bool(
        candidate.duration >= MIN_TARGET_SECONDS - 1e-9
        and not evidence.get('accepted')
        and evidence.get('blocked_fraction', 0.) == 0.
        and not evidence.get('fine_conflict')
        and not evidence.get('occlusion', {}).get('localized_contamination')
        and evidence.get('fine_covered_fraction', 0.) >= .75
        and evidence.get('positive_fraction', 0.) >= .75
        and evidence.get('fine_mean', 0.) > 2. * evidence.get('fine_error', 0.))


def apply_cluster_gate(accepted, rejected, turns, result, *, mode, min_seconds=1.2,
                       blocked_spans=(), local=None):
    if mode == "off":
        return {"removed": 0, "promoted": 0, "cropped": 0, "borderline": 0}
    removed, promoted, cropped, borderline = [], [], [], []
    originally_accepted = list(accepted)
    source_pool = [*originally_accepted, *rejected]
    hard_source_spans = [(c.start, c.end) for c in source_pool
                         if any(c.diagnostics.get(k) for k in HARD_VETOES)]
    floors = result.diagnostics.get("profile_floors") or {}
    secondary_floor = floors.get("secondary")
    primary_floor = floors.get("primary")
    trajectory = IdentityTrajectory(local, floors, source_pool, blocked_spans)
    accepted_spans = result.diagnostics.get("accepted_spans") or []
    mergeable_spans = [*accepted_spans, *result.diagnostics.get("certified_short_spans", [])]
    target_ids = set(result.target_cluster_ids) or (
        {result.target_cluster_id} if result.target_cluster_id is not None else set())
    non_target_turns = [(t.start, t.end) for t in turns
                        if result.cluster_of[t.index] not in target_ids
                        and result.cluster_of[t.index] != -1 and not result.membership[t.index]]
    # A mixed parent is not a veto on independently supported children.
    supported = [tuple(row['span']) for rows in (local or {}).values() for row in rows
                 if row.get('accepted') and not row.get('hard_veto')
                 and not row.get('identity_evidence', {}).get('exclusion_evidence', {}).get('conflict')]
    positive_union = []
    for a, b in sorted(set(supported)):
        if positive_union and a <= positive_union[-1][1]:
            positive_union[-1][1] = max(positive_union[-1][1], b)
        else:
            positive_union.append([a, b])
    supported = [part for a, b in positive_union
                 for part in trajectory.decode(a, b, MIN_TARGET_SECONDS)]
    non_target_turns = subtract_intervals(non_target_turns, supported)
    mergeable_spans.extend(supported)
    certified = result.diagnostics.setdefault('certified_short_spans', [])
    certified = result.diagnostics['certified_short_spans'] = list(certified)
    certified.extend([a, b] for a, b in supported if b - a < min_seconds)
    topology = CandidateTopology(source_pool, local, min_seconds)
    merged, _ = topology.merge(mergeable_spans, [*blocked_spans, *non_target_turns])
    short_spans = list(result.diagnostics.get('certified_short_spans', ()))
    short_spans += [(a,b) for a,b in merged if b-a < min_seconds
                   and _meets_export_length(a,b,min_seconds,result)]
    lattice = CandidateLattice(trajectory, merged)
    merged_accepted_spans = lattice.decode(min_seconds, short_spans) if local else merged
    result.diagnostics["trajectory_spans"] = [list(span) for span in merged_accepted_spans]
    result.diagnostics["trajectory_decoding"] = {
        "input_spans": len(mergeable_spans), "output_spans": len(merged_accepted_spans),
        "method": "candidate_lattice_identity_dp"}

    def make_crop(parent, start, end, reason, *, strict_parent=False):
        """Create a child using the identity evidence inside its own boundaries."""
        child = replace(parent, start=start, end=end, accepted=True,
                        reject_reason="", diagnostics=dict(parent.diagnostics))
        child.diagnostics["cluster_parent_span"] = [parent.start, parent.end]
        evidence = trajectory.assess(start, end)
        strong_windows = evidence["accepted"]
        child.diagnostics["identity_trajectory"] = evidence
        child.diagnostics["cluster_source_identity_conflict"] = bool(
            evidence["observed_fraction"] > 0 and not strong_windows)
        if child.diagnostics["cluster_source_identity_conflict"]:
            child.accepted = False
            child.reject_reason = "簇级身份未通过：interval_identity_conflict"
            child.diagnostics["cluster_tier"] = "B"
            child.diagnostics["cluster_reason"] = "interval_identity_conflict"
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
        rescue_b = _rescue_tier(child, secondary_floor, local=local,
                                primary_floor=floors.get("primary"),
                                start=start, end=end, trajectory=trajectory) == "B"
        strict_b = strict_parent and (window_min is None
                or (rescue_marked and window_min < PROMOTION_RESCUE_WINDOW_FLOOR))
        if not strong_windows and (rescue_b or strict_b):
            child.accepted = False
            child.reject_reason = "簇级身份未通过：rescue_strict_review"
            child.diagnostics["cluster_tier"] = "B"
            child.diagnostics["cluster_removed"] = False
            borderline.append(child)
            return None
        cropped.append(child)
        return child

    for candidate in [*originally_accepted, *rejected]:
        evidence = trajectory.assess(candidate.start, candidate.end)
        candidate.diagnostics["identity_trajectory"] = evidence
        candidate.diagnostics["cluster_source_identity_conflict"] = bool(
            evidence["observed_fraction"] > 0 and not evidence["accepted"])
        allowed, ids, reason, tier = candidate_membership(
            candidate, turns, result, local=local, min_seconds=min_seconds, trajectory=trajectory)
        hard = (any(candidate.diagnostics.get(k) for k in HARD_VETOES) or
                any(min(candidate.end, b) > max(candidate.start, a) for a, b in blocked_spans))
        allowed = allowed and not hard
        if (mode == 'enforce' and not hard and not allowed
                and not any(min(candidate.end, b) > max(candidate.start, a)
                            for a, b in hard_source_spans)
                and _positive_fine_review(candidate, evidence)):
            tier, reason = 'B', 'fine_identity_disagreement'
            candidate.diagnostics['review_only'] = True
            candidate.diagnostics['cluster_identity'] = False
            candidate.diagnostics.pop('acoustic_identity_audit', None)
            candidate.diagnostics['review_evidence_source'] = {
                'method': 'candidate_classifier_and_trajectory',
                'classifier_model_sha256': (candidate.diagnostics.get(
                    'local_classifier_audit') or {}).get('model_sha256'),
                'confirmed_target': False,
            }
        candidate.diagnostics.update(cluster_ids=ids, cluster_would_accept=allowed,
                                     cluster_mode=mode, cluster_reason=reason,
                                     cluster_target_id=result.target_cluster_id)
        if mode == "audit":
            continue
        if any(candidate is item for item in originally_accepted):
            if allowed and tier == "A":
                ratio = _turn_pass_ratio(candidate, turns, local)
                if evidence["observed_fraction"] == 0 and ratio is not None and ratio < PROMOTION_TURN_PASS_RATIO:
                    candidate.accepted = False
                    candidate.reject_reason = "簇级身份未通过：low_window_pass_ratio"
                    candidate.diagnostics["cluster_removed"] = True
                    candidate.diagnostics["cluster_tier"] = "B"
                    removed.append(candidate)
                    borderline.append(candidate)
                else:
                    candidate.diagnostics["cluster_tier"] = "A"
            else:
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
                        if _meets_export_length(start, end, min_seconds, result):
                            make_crop(candidate, start, end, reason)
        elif tier == "B" and not hard:
            candidate.diagnostics["cluster_tier"] = "B"
            borderline.append(candidate)
        elif (not hard and candidate.reject_reason in (
                "声纹匹配不足", *MIXED_CROP_REASONS, "说话回合过短")):
            ratio = _turn_pass_ratio(candidate, turns, local)
            rescue_marked = any(candidate.diagnostics.get(k) for k in RESCUE_KEYS)
            window_min = _window_min(candidate)
            promotable = (
                allowed and _meets_export_length(candidate.start, candidate.end, min_seconds, result)
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
                croppable = bool(merged_accepted_spans)
                cropped_any = False
                if croppable:
                    for a, b in merged_accepted_spans:
                        start, end = max(candidate.start, a), min(candidate.end, b)
                        if _meets_export_length(start, end, min_seconds, result):
                            child = make_crop(candidate, start, end, "promoted_subspan",
                                              strict_parent=False)
                            if child is not None:
                                cropped_any = True
                # A promoted subspan must not hide the parent's unreviewed
                # remainder. Fully covered duplicates are removed by Tier B NMS.
                if ((not cropped_any and promotable)
                        or _has_family_overlap(candidate, turns, result)):
                    candidate.diagnostics["cluster_tier"] = "B"
                    borderline.append(candidate)
        elif (not hard
              and any(key in candidate.reject_reason for key in MIXED_CROP_REASONS)):
            for a, b in accepted_spans:
                start, end = max(candidate.start, a), min(candidate.end, b)
                if _meets_export_length(start, end, min_seconds, result):
                    make_crop(candidate, start, end, "mixed_trimmed", strict_parent=True)
        elif (not hard and CLASSIFIER_CONFLICT_REASON in candidate.reject_reason
              and _has_strong_family_overlap(candidate, turns, result)):
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
        _join_short_remainders(accepted, merged_accepted_spans, min_seconds, trajectory)
        rejected[:] = [c for c in rejected if id(c) not in promoted_ids] + removed
        from ..types import CandidateSentence
        tier_a_spans = [(c.start, c.end) for c in accepted]
        synthesized = 0
        for span_start, span_end in _synthesis_regions(
                merged_accepted_spans, tier_a_spans, min_seconds, result, trajectory):
            accepted.append(CandidateSentence(
                span_start, span_end, "", accepted=True,
                diagnostics={
                    "cluster_span_synthesized": True,
                    "identity_trajectory": trajectory.assess(span_start, span_end),
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
            accepted[:] = _deduplicate_tier_a(accepted)
            accepted.sort(key=lambda c: (c.start, c.end))
    if mode == 'enforce' and local:
        # Preserve review access to audio where independently positive meso
        # evidence disagrees with a resolved micro boundary. Never promote it.
        from ..types import CandidateSentence
        rows = [r for group in local.values() if isinstance(group,list)
                for r in group if isinstance(r,dict) and r.get('span')]
        coarse = [r['span'] for r in rows if r.get('scale') != 'micro'
                  and r.get('accepted') and r.get('identity_evidence',{}).get('accepted')]
        forbidden = [*blocked_spans, *hard_source_spans]
        for row in rows:
            if row.get('micro_state') != 'other':
                continue
            a,b = row['span']
            if (b-a < .4-1e-9 or not any(x <= a+1e-6 and y >= b-1e-6 for x,y in coarse)
                    or any(min(b,y)>max(a,x) for x,y in forbidden)
                    or _covered_ratio(a,b,[(c.start,c.end) for c in [*accepted,*borderline]]) >= 1.-1e-6):
                continue
            borderline.append(CandidateSentence(a,b,'',accepted=False,
                reject_reason='多尺度声纹证据冲突，待复核',diagnostics={
                    'cluster_tier':'B','cluster_reason':'multiscale_disagreement',
                    'cluster_mode':mode,'review_only':True}))
    for c in {id(c): c for c in [*accepted, *rejected, *borderline]}.values():
        _calibrate_candidate(c, turns, result, local)
        if local:
            from .identity_heatmap import heatmap
            c.diagnostics['identity_heatmap'] = heatmap(trajectory, c.start, c.end)
    if mode == "enforce":
        _refine_tier_b(borderline, accepted)
        rejected_ids = {id(c) for c in rejected}
        rejected.extend(c for c in borderline if id(c) not in rejected_ids)
        rejected[:] = list({id(c): c for c in rejected}.values())
        rejected.sort(key=lambda c: (c.diagnostics.get("cluster_tier") == "B",
                                    c.diagnostics["tier_b_rank_score"]), reverse=True)
    refined_b = sum(c.diagnostics.get("cluster_tier") == "B" for c in rejected)
    return {"removed": len(removed), "promoted": len(promoted),
            "cropped": len(cropped), "borderline": refined_b,
            "synthesized": sum(1 for c in accepted
                               if c.diagnostics.get("cluster_span_synthesized"))}


def _tier_b_rank(candidate) -> float:
    """Prefer calibrated identity evidence, penalizing clear impostor victories."""
    diag = candidate.diagnostics
    cm_p, cm_s = diag.get("cohort_cm_p", diag.get("cm_p")), diag.get("cohort_cm_s", diag.get("cm_s"))
    loses = ((isinstance(cm_p, (int, float)) and cm_p < 0)
             or (isinstance(cm_s, (int, float)) and cm_s < -.05))
    def score(value):
        return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else 0.
    return (.65 * score(diag.get("calibrated_fusion_score"))
            + .15 * score(getattr(candidate, "speaker_score", 0))
            + .05 * score(_secondary_hint(candidate))
            + .15 * score(_window_min(candidate))
            + (.05 if diag.get("cluster_removed") else 0) - (.6 if loses else 0))


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
    """Suppress fully covered duplicates while retaining unique audio."""
    if not borderline:
        return
    tier_a_spans = [(a.start, a.end) for a in accepted]
    survivors = []
    for candidate in borderline:
        if _covered_ratio(candidate.start, candidate.end, tier_a_spans) >= max(tier_a_coverage, 1.-1e-6):
            candidate.diagnostics["cluster_tier"] = "covered_by_tier_a"
            continue
        survivors.append(candidate)
    ranked = sorted(survivors, key=_tier_b_rank, reverse=True)
    kept: list = []
    for candidate in ranked:
        if any(min(candidate.end, k.end) - max(candidate.start, k.start)
               >= max(mutual_containment, 1.-1e-6) * max(1e-6, candidate.duration)
               for k in kept):
            candidate.diagnostics["cluster_tier"] = "tier_b_nms_suppressed"
            continue
        candidate.diagnostics["tier_b_rank_score"] = round(_tier_b_rank(candidate), 4)
        kept.append(candidate)


def _deduplicate_tier_a(candidates, min_overlap=0.15):
    """Resolve overlapping exports, preferring complete clean intervals."""
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
        conflicts = [(i, e) for i, e in enumerate(selected)
                     if min(cand.end, e.end)-max(cand.start, e.start) > min_overlap]
        conflict = bool(conflicts)
        for i, existing in conflicts:
            ov = min(cand.end, existing.end) - max(cand.start, existing.start)
            if ov <= min_overlap:
                continue
            c_wmin = _window_min(cand) or 0.0
            e_wmin = _window_min(existing) or 0.0
            if (len(conflicts) == 1 and cand.duration < existing.duration
                    and c_wmin >= e_wmin + 0.06 and ov >= 0.8 * cand.duration):
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
    """Include uncertain clusters when retaining candidates for review."""
    target_ids = set(result.target_cluster_ids) or (
        {result.target_cluster_id} if result.target_cluster_id is not None else set())
    qualified = set(result.diagnostics.get("qualified_cluster_ids") or ())
    primary_floor = (result.diagnostics.get("profile_floors") or {}).get("primary")
    cluster_scores = {
        c.cluster_id: c.primary_target_score for c in result.clusters}
    # Formerly unclustered short/singleton turns remain reviewable. Gaining a
    # cluster ID is not independent evidence that a voice belongs to a rival.
    sparse_ids = {c.cluster_id for c in result.clusters
                  if len(c.members) < 3 or c.total_speech_seconds < 6.}
    for turn in turns:
        if min(turn.end, candidate.end) <= max(turn.start, candidate.start):
            continue
        if len(result.membership) == len(turns) and result.membership[turn.index]:
            return True
        if len(result.borderline_membership) == len(turns) and result.borderline_membership[turn.index]:
            return True
        cluster_id = result.cluster_of[turn.index]
        if cluster_id in target_ids | qualified | sparse_ids | {-1}:
            return True
        if (primary_floor is not None and cluster_id in cluster_scores
                and cluster_scores[cluster_id] >= primary_floor):
            return True
    return False
