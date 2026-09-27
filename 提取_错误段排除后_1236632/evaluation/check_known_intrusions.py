"""Fail if delivered clips cover known, independently flagged other-voice audio.

The case file is only a development counterexample; it is never an inference
rule and does not establish full-episode correctness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def check(manifest: dict, cases: list[dict], *, source_offset: float = 0.0) -> dict:
    if "sentences" in manifest:
        accepted_spans = [
            (float(row["start"]), float(row["end"]))
            for row in manifest["sentences"] if row["accepted"]
        ]
    elif "clips" in manifest:
        accepted_spans = [
            tuple(map(float, row["span"])) for row in manifest["clips"]
        ]
    else:
        raise ValueError("manifest must contain sentences or clips")
    rows = []
    for case in cases:
        start, end = (float(value) - source_offset for value in case["other_audio_span"])
        threshold = float(case["minimum_contaminated_seconds"])
        contamination = [
            {
                "accepted_span": [row_start, row_end],
                "overlap_seconds": round(max(0.0, min(end, row_end)
                                              - max(start, row_start)), 5),
            }
            for row_start, row_end in accepted_spans
            if min(end, row_end) - max(start, row_start) >= threshold
        ]
        rows.append({"id": case["id"], "passed": not contamination,
                     "contaminating_outputs": contamination})
    return {"passed": sum(row["passed"] for row in rows), "total": len(rows), "cases": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--cases", type=Path,
                        default=Path(__file__).with_name("known_intrusion_probe_20260925.json"))
    parser.add_argument("--source-offset", type=float, default=0.0)
    parser.add_argument("--case-id", action="append",
                        help="Only check the named counterexample (repeatable)")
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    if args.case_id:
        selected = set(args.case_id)
        cases = [case for case in cases if case["id"] in selected]
        unknown = selected - {case["id"] for case in cases}
        if unknown:
            parser.error(f"unknown case-id: {', '.join(sorted(unknown))}")
    report = check(
        json.loads(args.manifest.read_text(encoding="utf-8")),
        cases,
        source_offset=args.source_offset,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
