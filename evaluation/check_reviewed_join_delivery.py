"""Fast geometric regression for reviewed same-target sides and known changes.

This is a development gate, not an episode recall estimate: historical review
does not cover every utterance and interval coverage does not prove audio quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def assess(manifest: dict, cases: list[dict], *, source_offset: float = 0.0,
           tolerance: float = 0.05) -> dict:
    accepted = [row for row in manifest["sentences"] if row["accepted"]]
    rows = []
    for case in cases:
        left = [float(value) - source_offset for value in case["left"]]
        right = [float(value) - source_offset for value in case["right"]]
        def covered(part: list[float]) -> bool:
            return any(
                float(row["start"]) <= part[0] + tolerance
                and float(row["end"]) >= part[1] - tolerance
                for row in accepted
            )
        left_covered = covered(left)
        right_covered = covered(right)
        crossed_change = any(
            float(row["start"]) <= left[1] - tolerance
            and float(row["end"]) >= right[0] + tolerance
            for row in accepted
        )
        passed = (left_covered and right_covered) if case["same_target"] else not crossed_change
        rows.append({
            "id": case["id"], "expected_same_target": bool(case["same_target"]),
            "left_covered": left_covered, "right_covered": right_covered,
            "crossed_known_change": crossed_change, "passed": passed,
        })
    return {
        "warning": "Geometric coverage only; known reviewed development cases, not complete acoustic truth or blind-test recall.",
        "passed": sum(row["passed"] for row in rows),
        "total": len(rows),
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("join_cases.json"))
    parser.add_argument("--source-offset", type=float, default=0.0)
    parser.add_argument("--ids", nargs="*", default=None)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    if args.ids is not None:
        requested = set(args.ids)
        cases = [case for case in cases if case["id"] in requested]
        if {case["id"] for case in cases} != requested:
            parser.error("An --ids value does not exist in the case file")
    report = assess(
        json.loads(args.manifest.read_text(encoding="utf-8")),
        cases,
        source_offset=args.source_offset,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
