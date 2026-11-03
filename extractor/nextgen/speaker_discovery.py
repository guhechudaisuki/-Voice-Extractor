"""Episode speaker discovery and diverse, acoustically ranked reference seeds."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import math
from statistics import mean, median, pstdev

from ..subtitles import SubtitleCue, read_subtitles


@dataclass(frozen=True)
class DiscoveredSpeaker:
    cluster_id: int
    member_count: int
    total_speech_seconds: float
    purity_score: float
    recommended_seeds: tuple[dict, ...]
    sample_texts: tuple[str, ...]
    distinctiveness_margin: float = 0.0
    is_target: bool = False
    is_mixed: bool = False
    quality_flags: tuple[str, ...] = ()


def _number(value, default=0.):
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else default


def _clip(value):
    return max(0., min(1., value))


def _outlier_ids(cluster):
    return {str(i) for i in [*(cluster.get("outliers") or []),
                            *(cluster.get("discovery_outliers") or [])]}


def _member_windows(cluster, member):
    """Read own-cluster certificates; legacy target windows apply only to target clusters."""
    own = member.get("own_cluster_windows")
    if own is None:
        mapping = cluster.get("own_cluster_windows") or {}
        own = mapping.get(str(member["index"]), mapping.get(member["index"]))
    if isinstance(own, list):
        return own, "own_cluster_windows"
    if cluster.get("is_target") or member.get("accepted_by_cluster"):
        mapping = cluster.get("local_windows") or {}
        rows = mapping.get(str(member["index"]), mapping.get(member["index"], []))
        return rows if isinstance(rows, list) else [], "target_local_windows"
    return [], "member_geometry_only"


def _cluster_quality(cluster):
    cohesion = _number(cluster.get("primary_cohesion"))
    center = _number(cluster.get("primary_median"))
    count = max(1, cluster.get("member_count", len(cluster.get("members", []))))
    outlier_ratio = len(_outlier_ids(cluster)) / count
    margins = [_number(m["distinctiveness_margin"]) for m in cluster.get("members", [])
               if isinstance(m.get("distinctiveness_margin"), (int, float))
               and math.isfinite(m["distinctiveness_margin"])]
    distinctiveness = median(margins) if margins else 0.
    flags = []
    if cohesion < .40:
        flags.append("low_cohesion")
    if outlier_ratio > .25:
        flags.append("many_outliers")
    if margins and distinctiveness < -.02:
        flags.append("competing_clusters")
    failures = []
    for member in cluster.get("members", []):
        rows, source = _member_windows(cluster, member)
        for row in rows:
            if not isinstance(row, dict):
                continue
            if source == "own_cluster_windows":
                failures.append(not bool(row.get("accepted")))
                continue
            # Own-cluster purity requires competing-speaker evidence, not a target-floor miss.
            cm_p, cm_s = row.get("cm_p"), row.get("cm_s")
            if isinstance(cm_p, (int, float)) and isinstance(cm_s, (int, float)):
                failures.append(not row.get("accepted") and (cm_p < -.02 or cm_s < -.05))
            elif not row.get("demoted"):
                failures.append(not bool(row.get("accepted")))
    failure_ratio = mean(failures) if failures else 0.
    if failure_ratio > .45:
        flags.append("unstable_local_identity")
    mixed = bool(flags)
    purity = _clip((.5 * cohesion + .5 * center) * (1 - min(.8, outlier_ratio))
                   * (1 - .4 * failure_ratio))
    if mixed:
        purity *= .65
    return round(purity, 4), distinctiveness, mixed, tuple(flags)


def compute_purity_score(cluster: dict) -> float:
    return _cluster_quality(cluster)[0]


def _fold_cues(subtitles):
    """Tolerate unfurled bilingual cues at equal times as well as folded SubtitleCue."""
    cues = []
    for cue in sorted((c for c in subtitles if c.kind == "speech"), key=lambda c: (c.start, c.end)):
        if (cues and abs(cue.start-cues[-1].start) <= .06
                and abs(cue.end-cues[-1].end) <= .06):
            text = "\n".join(dict.fromkeys((cues[-1].text, cue.text)))
            cues[-1] = replace(cues[-1], text=text)
        else:
            cues.append(cue)
    return cues


def _seed_signature(member, rows):
    """Keep score geometry separate from optional speaker embeddings."""
    fields = {
        "primary": "primary_cluster_score", "secondary": "secondary_cluster_score",
        "cm_p": "cluster_margin_primary", "cm_s": "cluster_margin_secondary",
        "anime_char_score": "anime_char_score", "anime_va_score": "anime_va_score",
        "tertiary": "tertiary_score",
    }
    signature, lower_bounds = {}, {}
    for channel, member_key in fields.items():
        values = [float(row[channel]) for row in rows
                  if isinstance(row.get(channel), (int, float))
                  and math.isfinite(row[channel])]
        fallback = member.get(member_key)
        if fallback is None and channel in ("cm_p", "cm_s"):
            fallback = member.get("distinctiveness_margin")
        if not values and isinstance(fallback, (int, float)) and math.isfinite(fallback):
            values = [float(fallback)]
        if values:
            signature[channel] = median(values)
            lower_bounds[channel] = min(values)
    embeddings = {}
    for channel in ("primary", "secondary", "anime_char", "anime_va", "tertiary"):
        vector = member.get(channel + "_embedding")
        if not isinstance(vector, (list, tuple)) or not vector:
            continue
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in vector):
            continue
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            embeddings[channel] = tuple(value / norm for value in vector)
    return signature, lower_bounds, embeddings


def _rank_seed_evidence(seeds):
    """Compare channels within a role; raw scores from different models are not comparable."""
    distributions = {}
    for seed in seeds:
        for channel, value in seed["identity_lower_bounds"].items():
            distributions.setdefault(channel, []).append(value)
    for seed in seeds:
        ranks = {}
        for channel, value in seed["identity_lower_bounds"].items():
            peers = distributions[channel]
            ranks[channel] = (sum(other < value for other in peers)
                              + sum(other == value for other in peers) / 2) / len(peers)
        # Correlated channels vote as families, so adding a model is not extra proof.
        families = [("primary", "secondary"), ("cm_p", "cm_s"),
                    ("anime_char_score", "anime_va_score"), ("tertiary",)]
        evidence = [mean(ranks[channel] for channel in family if channel in ranks)
                    for family in families if any(channel in ranks for channel in family)]
        if evidence:
            seed["seed_quality"] = (seed["seed_quality"] + mean(evidence)) / 2
            seed["score"] = seed["seed_quality"]
        seed["identity_channel_ranks"] = ranks


def _reported_distances(geometry):
    """Read the packed upper triangle emitted by the cluster report."""
    if not isinstance(geometry, dict) or geometry.get("method") != "mean_multiview_angular_distance":
        return {}
    indexes, values = geometry.get("member_indexes"), geometry.get("upper_triangle")
    if not isinstance(indexes, list) or not isinstance(values, list):
        return {}
    indexes = [str(index) for index in indexes]
    if len(set(indexes)) != len(indexes) or len(values) != len(indexes) * (len(indexes) - 1) // 2:
        return {}
    distances = {}
    cursor = 0
    for position, left in enumerate(indexes):
        for right in indexes[position + 1:]:
            value = values[cursor]
            cursor += 1
            if isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1:
                distances[left, right] = distances[right, left] = float(value)
    return distances


def _diverse_seeds(seeds, maximum, geometry=None):
    """Quality-weighted farthest-point sampling in time and available acoustic views."""
    _rank_seed_evidence(seeds)
    centers = sorted({(seed["start"] + seed["end"]) / 2 for seed in seeds})
    gaps = [right - left for left, right in zip(centers, centers[1:])]
    time_scale = median(gaps) if gaps else 1.
    values = {}
    for seed in seeds:
        for channel, value in seed["identity_signature"].items():
            values.setdefault(channel, []).append(value)
    scales = {channel: pstdev(observations) for channel, observations in values.items()}
    reported = _reported_distances(geometry)
    acoustic_members = {index for pair in reported for index in pair}

    def acoustic_distance(left, right):
        pair = (str(left["turn_index"]), str(right["turn_index"]))
        if pair in reported:
            return reported[pair]
        vectors = []
        for channel, a in left["_embeddings"].items():
            b = right["_embeddings"].get(channel)
            if b is not None and len(a) == len(b):
                cosine = max(-1., min(1., sum(x * y for x, y in zip(a, b))))
                vectors.append(math.acos(cosine) / math.pi)
        if vectors:
            return mean(vectors)
        a, b = left["identity_signature"], right["identity_signature"]
        differences = [abs(a[channel] - b[channel]) / scale
                       for channel, scale in scales.items()
                       if scale > 0 and channel in a and channel in b]
        return mean(value / (1 + value) for value in differences) if differences else None

    def distance(left, right):
        gap = abs((left["start"] + left["end"] - right["start"] - right["end"]) / 2)
        temporal = gap / (gap + time_scale)
        acoustic = acoustic_distance(left, right)
        return temporal if acoustic is None else (temporal + acoustic) / 2

    selected = []
    remaining = list(seeds)
    while remaining and len(selected) < maximum:
        def score(seed):
            novelty = min((distance(seed, previous) for previous in selected), default=0.)
            return seed["seed_quality"] * (1 + novelty)
        best = max(remaining, key=lambda seed: (score(seed), seed["seed_quality"], -seed["start"]))
        best["selection_score"] = round(score(best), 5)
        selected.append(best)
        remaining = [seed for seed in remaining if seed["turn_index"] != best["turn_index"]
                     and min(seed["end"], best["end"]) <= max(seed["start"], best["start"])]
    for seed in selected:
        embeddings = seed.pop("_embeddings")
        seed["diversity_evidence"] = (
            "cluster_embedding_angles" if str(seed["turn_index"]) in acoustic_members else
            "speaker_embeddings" if embeddings else
            "identity_score_signature" if seed["identity_signature"] else "temporal_only")
        seed["seed_quality"] = round(seed["seed_quality"], 5)
        seed["score"] = round(seed["score"], 5)
    return tuple(selected)


def select_pure_seeds(cluster: dict, subtitles: list[SubtitleCue] = (),
                      max_seeds: int = 4) -> tuple[dict, ...]:
    """Select complete, stable own-cluster seeds with scene diversity."""
    if max_seeds <= 0:
        return ()
    outliers = _outlier_ids(cluster)
    cues = _fold_cues(subtitles)
    scored = []
    for member in cluster.get("members", []):
        if str(member["index"]) in outliers:
            continue
        start, end = map(float, member["span"])
        if not math.isfinite(start+end) or end <= start:
            continue
        overlapping = [c for c in cues if min(c.end, end)-max(c.start, start) > .05]
        contained = [c for c in overlapping if c.start >= start and c.end <= end]
        # Only contained, non-overlapping cues can describe complete seed lines.
        if overlapping:
            intervals = [(c.start, c.end, c.text, len(overlapping) == 1)
                         for c in contained if not any(
                             c is not other and min(c.end, other.end)-max(c.start, other.start) > .06
                             for other in overlapping)]
        else:
            intervals = [(start, end, "", False)]
        margin = member.get("distinctiveness_margin")
        if margin is not None and (not isinstance(margin, (int, float))
                                   or not math.isfinite(margin) or margin < 0):
            continue
        group, evidence_source = _member_windows(cluster, member)
        for a, b, text, single in intervals:
            duration = b-a
            if not 1.8 <= duration <= 6.:
                continue
            rows = [r for r in group if isinstance(r, dict) and r.get("span")
                    and min(r["span"][1], b)-max(r["span"][0], a) >= .45]
            if rows and not all(r.get("accepted") and not r.get("smoothed") for r in rows):
                continue
            if any(not isinstance(r.get("primary"), (int, float))
                   or not math.isfinite(r["primary"])
                   or (r.get("secondary") is not None and (
                       not isinstance(r["secondary"], (int, float))
                       or not math.isfinite(r["secondary"]))) for r in rows):
                continue
            covered, cursor = 0., a
            for left, right in sorted(r["span"] for r in rows):
                covered += max(0., min(b, right)-max(cursor, left))
                cursor = max(cursor, right)
            coverage = min(1., covered/duration)
            fully_verified = bool(rows) and coverage >= .98
            primary_scores = [_number(r.get("primary")) for r in rows]
            secondary_scores = [_number(r["secondary"]) for r in rows if r.get("secondary") is not None]
            minimum = (min(primary_scores) if primary_scores
                       else _number(member.get("primary_cluster_score"), .5))
            secondary_minimum = (min(secondary_scores) if secondary_scores
                                 else _number(member.get("secondary_cluster_score"), minimum))
            variation = max(pstdev(primary_scores) if primary_scores else 0.,
                            pstdev(secondary_scores) if secondary_scores else 0.)
            if variation > .12:
                continue
            row_margins = [min(r["cm_p"], r["cm_s"]) for r in rows
                           if isinstance(r.get("cm_p"), (int, float))
                           and isinstance(r.get("cm_s"), (int, float))
                           and math.isfinite(r["cm_p"]) and math.isfinite(r["cm_s"])]
            exclusion = min(row_margins) if row_margins else margin
            if exclusion is not None and exclusion < 0:
                continue
            duration_score = 1. if 2 <= duration <= 5.5 else .65
            stable = _clip(min(minimum, secondary_minimum) - 2*variation)
            margin_score = _clip(.5 + 2*_number(exclusion)) if exclusion is not None else .35
            quality = (.15*duration_score + .35*stable + .2*margin_score
                       + (.15 if single else .06 if text else 0)
                       + (.15 if fully_verified else .03 if margin is not None else 0))
            signature, lower_bounds, embeddings = _seed_signature(member, rows)
            scored.append({"start": round(a, 3), "end": round(b, 3), "text": text,
                "turn_index": member["index"], "seed_quality": round(quality, 5),
                "score": round(quality, 5), "distinctiveness_margin": exclusion,
                "window_minimum": minimum, "window_std": variation,
                "secondary_window_minimum": secondary_minimum,
                "all_windows_pass": True if fully_verified else None,
                "window_coverage_ratio": coverage if rows else None,
                "evidence_source": evidence_source,
                "identity_signature": signature, "identity_lower_bounds": lower_bounds,
                "_embeddings": embeddings,
                "single_complete_cue": single})
    return _diverse_seeds(scored, max_seeds, cluster.get("member_acoustic_geometry"))


def discover_episode_speakers(cluster_report: dict, subtitle_path: Path | None = None,
                              max_seeds: int = 4, min_speech_seconds: float = 8.) -> list[DiscoveredSpeaker]:
    subtitles = []
    if subtitle_path and Path(subtitle_path).is_file():
        try:
            subtitles = read_subtitles(Path(subtitle_path))
        except (ValueError, OSError):
            pass
    results = []
    target_ids = set(cluster_report.get("diagnostics", {}).get("target_cluster_ids", []))
    target_ids.add(cluster_report.get("target_cluster_id"))
    for raw in cluster_report.get("clusters", []):
        cluster = {**raw, "is_target": bool(raw.get("is_target") or raw["cluster_id"] in target_ids),
                   "local_windows": raw.get("local_windows", cluster_report.get("local_windows", {})),
                   "own_cluster_windows": raw.get("own_cluster_windows", cluster_report.get("own_cluster_windows", {}))}
        count = cluster.get("member_count", len(cluster.get("members", [])))
        seconds = cluster.get("total_speech_seconds", 0.)
        if count < 3 or seconds < min_speech_seconds:
            continue
        purity, distinctiveness, mixed, flags = _cluster_quality(cluster)
        seeds = select_pure_seeds(cluster, subtitles, max_seeds)
        results.append(DiscoveredSpeaker(cluster["cluster_id"], count, seconds, purity,
            seeds, tuple(s["text"] for s in seeds if s["text"]), distinctiveness,
            cluster["is_target"], mixed, flags))
    results.sort(key=lambda s: (s.is_mixed, -s.purity_score,
                               -s.distinctiveness_margin, -s.total_speech_seconds, s.cluster_id))
    return results


def export_speaker_seeds(stem_path: Path, cluster_report: dict, output_dir: Path,
                         subtitle_path: Path | None = None, max_clusters: int = 8) -> dict[int, list[Path]]:
    """Export pure roles; mixed clusters stay visible but are not auto-exported."""
    if max_clusters <= 0:
        return {}
    speakers = [s for s in discover_episode_speakers(cluster_report, subtitle_path)
                if not s.is_mixed and s.recommended_seeds][:max_clusters]
    if not speakers:
        return {}
    from ..audio import write_clip
    result = {}
    for speaker in speakers:
        directory = Path(output_dir) / f"cluster_{speaker.cluster_id:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, seed in enumerate(speaker.recommended_seeds):
            path = directory / f"seed_{i:02d}.wav"
            write_clip(Path(stem_path), path, seed["start"], seed["end"],
                       sample_rate=16000, normalize_level=True)
            paths.append(path)
        result[speaker.cluster_id] = paths
    return result


def export_discovered_speaker_seeds(stem_path: Path, cluster_report: dict, output_dir: Path,
                                    subtitle_path: Path | None = None) -> dict[str, list[Path]]:
    """Existing launcher API, retaining its string keys and all-role limit."""
    exports = export_speaker_seeds(stem_path, cluster_report, output_dir, subtitle_path,
                                   max_clusters=len(cluster_report.get("clusters", [])))
    return {f"cluster_{key:02d}": paths for key, paths in exports.items()}
