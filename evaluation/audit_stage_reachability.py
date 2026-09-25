"""Measure where historically reviewed speech remains in a stage trace.

Presence at a stage is a geometric upper bound, not proof of correct identity or
audio quality. This deliberately avoids claiming complete episode recall.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_reviewed_manifest import covered_seconds, intervals


STAGES = (
    "initial_vad",
    "subtitle_assisted_vad",
    "atomic_speech_islands",
    "clean_speech_islands",
    "speaker_turns",
    "target_locator_proposals",
    "identity_accepted_before_stt",
    "final_accepted",
)
SERIAL_STAGES = tuple(
    name for name in STAGES if name != "target_locator_proposals"
)
BLOCKING_STAGES = (
    "pre_uvr_singing_mask",
    "singing_blocked_islands",
    "overlap_blocked_islands",
)


def records(stage: dict) -> list[dict]:
    return [{"start": start, "end": end} for start, end in stage.get("spans", [])]


def clipped_union(
    span: tuple[float, float], covering: list[dict],
) -> list[tuple[float, float]]:
    """Clip and union stage ranges so overlaps cannot inflate coverage."""
    pieces = sorted(
        (max(span[0], float(item["start"])), min(span[1], float(item["end"])))
        for item in covering
        if min(span[1], float(item["end"])) > max(span[0], float(item["start"]))
    )
    merged: list[tuple[float, float]] = []
    for begin, end in pieces:
        if merged and begin <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((begin, end))
    return merged


def subtract_intervals(
    left: list[tuple[float, float]], right: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Return intervals in a normalized left set absent from normalized right."""
    result: list[tuple[float, float]] = []
    for begin, end in left:
        cursor = begin
        for blocked_begin, blocked_end in right:
            if blocked_end <= cursor:
                continue
            if blocked_begin >= end:
                break
            if blocked_begin > cursor:
                result.append((cursor, min(blocked_begin, end)))
            cursor = max(cursor, blocked_end)
            if cursor >= end:
                break
        if cursor < end:
            result.append((cursor, end))
    return result


def rounded_intervals(spans: list[tuple[float, float]]) -> list[list[float]]:
    return [[round(begin, 5), round(end, 5)] for begin, end in spans]


def summarize(reviewed: dict, audit: dict) -> dict:
    baselines = intervals(reviewed, True)
    stages = audit.get("stages", {})
    rows = []
    for index, old in enumerate(baselines, start=1):
        span = (float(old["start"]), float(old["end"]))
        duration = span[1] - span[0]
        coverage = {}
        transitions = {}
        previous_name = None
        previous_intervals: list[tuple[float, float]] = []
        for name in STAGES:
            if name in stages:
                current_intervals = clipped_union(span, records(stages[name]))
                covered = sum(end - begin for begin, end in current_intervals)
                coverage[name] = {
                    "seconds": round(covered, 5),
                    "fraction": round(covered / duration, 5) if duration > 0 else 0.0,
                    "missing_intervals": rounded_intervals(
                        subtract_intervals([span], current_intervals)
                    ),
                }
                # The target locator proposes an additional recovery route;
                # it is not a gate between speaker turns and identity matching.
                if name not in SERIAL_STAGES:
                    continue
                if previous_name is not None:
                    lost = subtract_intervals(previous_intervals, current_intervals)
                    recovered = subtract_intervals(current_intervals, previous_intervals)
                    transitions[name] = {
                        "from_stage": previous_name,
                        "lost_intervals": rounded_intervals(lost),
                        "lost_seconds": round(sum(end - begin for begin, end in lost), 5),
                        "recovered_intervals": rounded_intervals(recovered),
                        "recovered_seconds": round(
                            sum(end - begin for begin, end in recovered), 5
                        ),
                    }
                previous_name = name
                previous_intervals = current_intervals
        blockers = {}
        for name in BLOCKING_STAGES:
            if name in stages:
                blockers[name] = round(covered_seconds(span, records(stages[name])), 5)
        rows.append({
            "reviewed_index": index,
            "reviewed_span": list(span),
            "stage_coverage": coverage,
            "stage_transitions": transitions,
            "blocked_overlap_seconds": blockers,
        })
    return {
        "warning": "Stage coverage and serial transitions are geometric only; missing time can be silence. Target locator proposals are a parallel recovery route, not a required gate. Stages may recover candidates later and historical clips are not a complete ground-truth set.",
        "reviewed_clip_count": len(rows),
        "stage_counts": {name: stages[name]["count"] for name in stages},
        "reviewed_coverage": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reviewed_manifest", type=Path)
    parser.add_argument("stage_audit", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reviewed = json.loads(args.reviewed_manifest.read_text(encoding="utf-8"))
    audit = json.loads(args.stage_audit.read_text(encoding="utf-8"))
    report = summarize(reviewed, audit)
    contents = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(contents, encoding="utf-8")
    else:
        print(contents)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
