"""全片角色簇纯度评估与种子推荐器（Stage 24-26）。

给定一集的 cluster_report.json，自动对全部说话人簇进行单人纯度评分与
代表性独白 Cue 挑选，替代手工切参考的粗糙做法。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..subtitles import SubtitleCue, read_subtitles


@dataclass(frozen=True)
class DiscoveredSpeaker:
    cluster_id: int
    member_count: int
    total_speech_seconds: float
    purity_score: float
    recommended_seeds: tuple[dict, ...]
    sample_texts: tuple[str, ...]


def compute_purity_score(cluster: dict) -> float:
    """单人纯度分 = 0.5 * primary_cohesion + 0.5 * primary_median。"""
    return round(0.5 * cluster.get("primary_cohesion", 0.0)
                 + 0.5 * cluster.get("primary_median", 0.0), 4)


def select_pure_seeds(
    cluster: dict,
    subtitles: list[SubtitleCue] = (),
    max_seeds: int = 4,
) -> tuple[dict, ...]:
    """挑选最贴近簇质心、远离其他簇、无重叠的 Top-N 纯净单句切片。"""
    outliers = set(cluster.get("outliers", []))
    members = [
        m for m in cluster.get("members", [])
        if m["index"] not in outliers and m.get("accepted_by_cluster", False)
    ]
    scored = []
    for m in members:
        start, end = m["span"]
        duration = end - start
        if not (1.8 <= duration <= 5.0):
            continue
        if 2.0 <= duration <= 4.5:
            duration_score = 1.0
        elif duration < 2.0:
            duration_score = (duration - 1.8) / 0.2
        else:
            duration_score = max(0.0, (5.0 - duration) / 0.5)
        has_single_cue = False
        cue_text = ""
        if subtitles:
            overlapping = [
                c for c in subtitles
                if c.kind == "speech"
                and min(c.end, end) - max(c.start, start) > 0.5 * duration
            ]
            if len(overlapping) == 1 and overlapping[0].end - overlapping[0].start <= duration + 0.3:
                has_single_cue = True
                cue_text = overlapping[0].text
        proximity = min(1.0, m.get("speech_seconds", 0) / 5.0)
        score = duration_score * 0.4 + proximity * 0.3 + (0.3 if has_single_cue else 0.0)
        scored.append((score, start, end, m["index"], has_single_cue, cue_text))
    scored.sort(key=lambda x: -x[0])
    seeds = []
    for score, start, end, idx, has_cue, text in scored[:max_seeds]:
        seeds.append({
            "start": round(start, 3), "end": round(end, 3),
            "score": round(score, 4),
            "text": text,
            "turn_index": idx,
        })
    return tuple(seeds)


def discover_episode_speakers(
    cluster_report: dict,
    subtitle_path: Path | None = None,
    max_seeds: int = 4,
    min_speech_seconds: float = 8.0,
) -> list[DiscoveredSpeaker]:
    """对全集所有说话人簇进行纯度评估与代表句推荐。"""
    subtitles: list[SubtitleCue] = []
    if subtitle_path and Path(subtitle_path).is_file():
        try:
            subtitles = read_subtitles(Path(subtitle_path))
        except (ValueError, OSError):
            pass
    results = []
    for cluster in cluster_report.get("clusters", []):
        if cluster.get("member_count", 0) < 3:
            continue
        if cluster.get("total_speech_seconds", 0) < min_speech_seconds:
            continue
        purity = compute_purity_score(cluster)
        seeds = select_pure_seeds(cluster, subtitles, max_seeds)
        sample_texts = tuple(s["text"] for s in seeds if s.get("text"))
        results.append(DiscoveredSpeaker(
            cluster_id=cluster["cluster_id"],
            member_count=cluster["member_count"],
            total_speech_seconds=cluster["total_speech_seconds"],
            purity_score=purity,
            recommended_seeds=seeds,
            sample_texts=sample_texts,
        ))
    results.sort(key=lambda s: -s.total_speech_seconds)
    return results


def export_discovered_speaker_seeds(
    stem_path: Path,
    cluster_report: dict,
    output_dir: Path,
    subtitle_path: Path | None = None,
) -> dict[str, list[Path]]:
    """一键切出各角色簇的标准纯净参考包。"""
    from ..audio import write_clip

    speakers = discover_episode_speakers(cluster_report, subtitle_path)
    stem_path = Path(stem_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, list[Path]] = {}
    for speaker in speakers:
        key = f"cluster_{speaker.cluster_id:02d}"
        cluster_dir = output_dir / key
        cluster_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, seed in enumerate(speaker.recommended_seeds):
            path = cluster_dir / f"seed_{i:02d}.wav"
            write_clip(stem_path, path, seed["start"], seed["end"],
                       sample_rate=16000, normalize_level=True)
            paths.append(path)
        if paths:
            result[key] = paths
    return result
