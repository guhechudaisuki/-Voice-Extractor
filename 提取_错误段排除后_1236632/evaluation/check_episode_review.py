"""Fast, coarse regression for the user-reviewed first-episode exports.

This does not prove that a new clip is clean: the mixed clips have no manually
marked sample-accurate change points. It only rejects unchanged bad exports and
checks coverage of the older, manually reviewed clean baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def accepted_spans(manifest: dict) -> list[tuple[float, float]]:
    return [
        (float(row["start"]), float(row["end"]))
        for row in manifest["sentences"]
        if row["accepted"]
    ]


def covered_seconds(span: tuple[float, float], outputs: list[tuple[float, float]]) -> float:
    start, end = span
    intersections = sorted(
        (max(start, left), min(end, right))
        for left, right in outputs
        if min(end, right) > max(start, left)
    )
    covered = 0.0
    cursor = start
    for left, right in intersections:
        covered += max(0.0, right - max(cursor, left))
        cursor = max(cursor, right)
    return covered


def check(manifest: dict, review: dict, frozen: dict, *, edge_tolerance: float = 0.15,
          minimum_baseline_coverage: float = 0.85) -> dict:
    outputs = accepted_spans(manifest)
    frozen_outputs = accepted_spans(frozen)
    cases = []
    for case in review["flagged_outputs"]:
        start, end = map(float, case["span"])
        if case["kind"] == "entire_other":
            bad = [
                [left, right] for left, right in outputs
                if min(end, right) - max(start, left) > edge_tolerance
            ]
        elif case["kind"] == "mixed":
            # Without human-marked change points, this can only detect that
            # the original contaminated interval survives substantially intact.
            bad = [
                [left, right] for left, right in outputs
                if left <= start + edge_tolerance and right >= end - edge_tolerance
            ]
        else:
            raise ValueError(f"Unknown review kind: {case['kind']}")
        cases.append({"index": case["index"], "kind": case["kind"],
                      "passed": not bad, "still_bad_exports": bad})

    baseline = []
    for index, span in enumerate(frozen_outputs, 1):
        ratio = covered_seconds(span, outputs) / (span[1] - span[0])
        baseline.append({"index": index, "span": list(span),
                         "coverage": round(ratio, 4),
                         "passed": ratio >= minimum_baseline_coverage})

    return {
        "warning": "A pass is only a coarse regression, not a clean-audio verdict; user listening is still required.",
        "known_bad_blocked": sum(row["passed"] for row in cases),
        "known_bad_total": len(cases),
        "baseline_preserved": sum(row["passed"] for row in baseline),
        "baseline_total": len(baseline),
        "cases": cases,
        "baseline_failures": [row for row in baseline if not row["passed"]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--review", type=Path,
                        default=Path(__file__).with_name("episode1_user_review_20260926.json"))
    parser.add_argument("--frozen", type=Path,
                        default=Path("output/20260825_174500_batch_d90730/batch_manifest.json"))
    args = parser.parse_args()
    report = check(
        json.loads(args.manifest.read_text(encoding="utf-8")),
        json.loads(args.review.read_text(encoding="utf-8")),
        json.loads(args.frozen.read_text(encoding="utf-8")),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if (report["known_bad_blocked"] == report["known_bad_total"]
                 and report["baseline_preserved"] == report["baseline_total"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
