"""Compare a new episode run with historically reviewed output by source time.

This reports geometric coverage and new candidates. It does not certify a new
clip's speaker, internal purity, acoustic completeness, or whole-episode recall.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def intervals(manifest: dict, accepted: bool) -> list[dict]:
    return sorted(
        (
            item for item in manifest.get("sentences", [])
            if item.get("accepted") is accepted
        ),
        key=lambda item: (item["start"], item["end"]),
    )


def overlap(left: tuple[float, float], right: tuple[float, float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def covered_seconds(span: tuple[float, float], covering: list[dict]) -> float:
    clipped = sorted(
        (max(span[0], float(item["start"])), min(span[1], float(item["end"])))
        for item in covering
        if overlap(span, (float(item["start"]), float(item["end"]))) > 0
    )
    if not clipped:
        return 0.0
    total = 0.0
    begin, end = clipped[0]
    for next_begin, next_end in clipped[1:]:
        if next_begin <= end:
            end = max(end, next_end)
        else:
            total += end - begin
            begin, end = next_begin, next_end
    return total + end - begin


def summarize(reviewed: dict, candidate: dict) -> dict:
    baseline = intervals(reviewed, True)
    accepted = intervals(candidate, True)
    rejected = intervals(candidate, False)
    rows = []
    for index, old in enumerate(baseline, start=1):
        span = (float(old["start"]), float(old["end"]))
        duration = span[1] - span[0]
        adjacent_accepted = [
            item for item in accepted
            if overlap(span, (float(item["start"]), float(item["end"]))) > 0
        ]
        adjacent_rejected = [
            item for item in rejected
            if overlap(span, (float(item["start"]), float(item["end"]))) > 0
        ]
        rows.append({
            "reviewed_index": index,
            "reviewed_span": list(span),
            "reviewed_audio": old.get("audio_file", ""),
            "reviewed_seconds": round(duration, 5),
            "new_time_covered_seconds": round(covered_seconds(span, accepted), 5),
            "new_time_coverage_fraction": round(
                covered_seconds(span, accepted) / duration, 5,
            ) if duration > 0 else 0.0,
            "new_accepted_spans": [
                [item["start"], item["end"]] for item in adjacent_accepted
            ],
            "overlapping_rejections": [
                {"span": [item["start"], item["end"]], "reason": item.get("reject_reason", "")}
                for item in adjacent_rejected
            ],
        })
    outside = [
        item for item in accepted
        if not any(
            overlap((float(item["start"]), float(item["end"])),
                    (float(old["start"]), float(old["end"]))) > 0
            for old in baseline
        )
    ]
    return {
        "warning": "Geometry only: all new clips still require identity, completeness, and contamination audit.",
        "reviewed_clip_count": len(baseline),
        "new_clip_count": len(accepted),
        "reviewed_clip_duration_seconds": round(sum(row["reviewed_seconds"] for row in rows), 5),
        "reviewed_time_covered_by_new_seconds": round(
            sum(row["new_time_covered_seconds"] for row in rows), 5,
        ),
        "new_clips_without_reviewed_time": [
            {"span": [item["start"], item["end"]], "audio_file": item.get("audio_file", "")}
            for item in outside
        ],
        "reviewed_coverage": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reviewed_manifest", type=Path)
    parser.add_argument("new_manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    old = json.loads(args.reviewed_manifest.read_text(encoding="utf-8"))
    new = json.loads(args.new_manifest.read_text(encoding="utf-8"))
    report = summarize(old, new)
    contents = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(contents, encoding="utf-8")
    else:
        print(contents)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
