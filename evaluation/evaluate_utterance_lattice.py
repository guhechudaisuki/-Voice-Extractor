"""Measure reachability and known-change risk of a proposal lattice.

Cases are a small development set. A matched interval is only a geometric
candidate, not a correctly identified or clean sentence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def assess_reachability(
    cases: dict, lattice: dict, *,
    start_tolerance_seconds: float = 0.05,
    end_tolerance_seconds: float = 0.15,
) -> dict:
    if start_tolerance_seconds < 0 or end_tolerance_seconds < 0:
        raise ValueError("Endpoint tolerances cannot be negative")
    sample_rate = int(lattice["analysis_sample_rate"])
    if sample_rate <= 0:
        raise ValueError("Invalid analysis sample rate")
    proposals = lattice["proposals"]
    ids = [item["source_id"] for item in proposals]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source IDs in proposal lattice")
    for item in proposals:
        left, right = item["start_sample"], item["end_sample"]
        if (type(left) is not int or type(right) is not int
                or not 0 <= left < right):
            raise ValueError(f"Invalid sample interval for {item['source_id']}")
    results = []
    for case in cases["cases"]:
        start = float(case["left"][0])
        end = float(case["right"][1])
        if not (math.isfinite(start) and math.isfinite(end)
                and 0 <= start < end):
            raise ValueError(f"Invalid case interval for {case['id']}")
        matching = [
            item for item in proposals
            if abs(item["start_sample"] / sample_rate - start)
            <= start_tolerance_seconds + 1e-7
            and abs(item["end_sample"] / sample_rate - end)
            <= end_tolerance_seconds + 1e-7
        ]
        matching.sort(key=lambda item: (
            abs(item["start_sample"] / sample_rate - start)
            + abs(item["end_sample"] / sample_rate - end),
            len(item["island_indexes"]),
        ))
        best = matching[0] if matching else None
        # A tolerance match can still clip a final phoneme. Compare on the
        # analysis sample grid, allowing only half a sample for decimal input.
        half_sample = 0.5 / sample_rate + 1e-9
        enclosing = [
            item for item in matching
            if item["start_sample"] / sample_rate <= start + half_sample
            and item["end_sample"] / sample_rate >= end - half_sample
        ]
        row = {
            "id": case["id"],
            "historical_same_target_pair": case.get("same_target"),
            "reviewed_span": [start, end],
            "candidate_found": best is not None,
            "matching_candidate_count": len(matching),
            "reviewed_span_enclosed_by_any_candidate": bool(enclosing),
            "enclosing_candidate_count": len(enclosing),
            "candidate": None,
        }
        if best is not None:
            found_start = best["start_sample"] / sample_rate
            found_end = best["end_sample"] / sample_rate
            row["candidate"] = {
                "source_id": best["source_id"],
                "start": round(found_start, 5),
                "end": round(found_end, 5),
                "start_error_seconds": round(found_start - start, 5),
                "end_error_seconds": round(found_end - end, 5),
                "missing_start_seconds": round(max(0.0, found_start - start), 5),
                "missing_end_seconds": round(max(0.0, end - found_end), 5),
                "extra_start_seconds": round(max(0.0, start - found_start), 5),
                "extra_end_seconds": round(max(0.0, found_end - end), 5),
                "reviewed_span_enclosed": best in enclosing,
                "island_indexes": best["island_indexes"],
                "subtitle_cue_indexes": best.get("subtitle_cue_indexes", []),
                "identity_state": best["identity_state"],
            }
        results.append(row)
    positive = [row for row in results if row["historical_same_target_pair"] is True]
    negative = [row for row in results if row["historical_same_target_pair"] is False]
    return {
        "warning": (
            "Candidate reachability is not target recall or accuracy. Known-change "
            "candidates are contamination risks and must not be exported based "
            "on geometry or whole-span speaker similarity alone. Enclosing a "
            "reviewed time interval is not proof of a complete spoken phrase."
        ),
        "start_tolerance_seconds": start_tolerance_seconds,
        "end_tolerance_seconds": end_tolerance_seconds,
        "positive_development_cases": len(positive),
        "positive_candidates_found": sum(row["candidate_found"] for row in positive),
        "positive_reviewed_spans_enclosed": sum(
            row["reviewed_span_enclosed_by_any_candidate"] for row in positive
        ),
        "known_change_cases": len(negative),
        "known_change_candidates_found": sum(row["candidate_found"] for row in negative),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("lattice", type=Path)
    parser.add_argument("--start-tolerance-seconds", type=float, default=0.05)
    parser.add_argument("--end-tolerance-seconds", type=float, default=0.15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = assess_reachability(
        json.loads(args.cases.read_text(encoding="utf-8")),
        json.loads(args.lattice.read_text(encoding="utf-8")),
        start_tolerance_seconds=args.start_tolerance_seconds,
        end_tolerance_seconds=args.end_tolerance_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"LATTICE_REACHABILITY={args.output} "
        f"POSITIVE={report['positive_candidates_found']}/{report['positive_development_cases']} "
        f"ENCLOSED={report['positive_reviewed_spans_enclosed']}/{report['positive_development_cases']} "
        f"KNOWN_CHANGE={report['known_change_candidates_found']}/{report['known_change_cases']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
