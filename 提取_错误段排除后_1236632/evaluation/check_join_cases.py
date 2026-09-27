"""Test the current same-speaker join gate on reviewed local audio pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_reviewed_manifest import covered_seconds, intervals


CASES = Path(__file__).with_name("join_cases.json")


def _pair_key(left: list[float], right: list[float]) -> tuple[float, ...]:
    return tuple(round(float(value), 5) for value in (*left, *right))


def validate_positive_evidence(cases: dict, reviewed: dict) -> None:
    historically_accepted = intervals(reviewed, True)
    for case in cases["cases"]:
        if not case["same_target"]:
            continue
        for side in ("left", "right"):
            start, end = map(float, case[side])
            if end <= start or covered_seconds((start, end), historically_accepted) < end - start - 0.03:
                raise ValueError(f"{case['id']} {side} exceeds historical reviewed target coverage")


def assess(
    probe: dict,
    cases: dict,
    reviewed: dict,
    predictions: dict[str, bool] | None = None,
) -> dict:
    validate_positive_evidence(cases, reviewed)
    by_span = {
        _pair_key(row["left"], row["right"]): row
        for row in probe["pairs"]
    }
    rows = []
    for case in cases["cases"]:
        key = _pair_key(case["left"], case["right"])
        if key not in by_span:
            raise ValueError(f"Probe is missing {case['id']}: {key}")
        if predictions is not None and case["id"] not in predictions:
            raise ValueError(f"Predictions are missing case {case['id']}")
        row = by_span[key]
        # These are the present production join floors. Requiring whole-span
        # acceptance models the following formal verifier, but not every other
        # production veto; this is a targeted diagnostic, not an output oracle.
        current_gate = (
            row["gap_seconds"] <= 0.85
            and row["primary_pair_similarity"] >= 0.76
            and row["secondary_pair_similarity"] >= 0.64
            and row["whole_result"]["accepted_by_whole_span_verifier"]
            and not row["whole_result"]["excluded_role_rejected"]
        )
        predicted = bool(predictions[case["id"]]) if predictions is not None else bool(current_gate)
        rows.append({
            "id": case["id"],
            "expected_same_target": bool(case["same_target"]),
            "current_join_gate": bool(current_gate),
            "evaluated_join": predicted,
            "passed": bool(predicted) == bool(case["same_target"]),
        })
    return {
        "warning": "Episode-1 development cases only; not complete identity ground truth or a deployable join policy.",
        "passed": sum(row["passed"] for row in rows),
        "total": len(rows),
        "results": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", type=Path)
    parser.add_argument("--reviewed-manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, help="New policy decisions as {case_id: boolean}")
    args = parser.parse_args()
    predictions = json.loads(args.predictions.read_text(encoding="utf-8")) if args.predictions else None
    if predictions is not None and (
        not isinstance(predictions, dict)
        or any(type(value) is not bool for value in predictions.values())
    ):
        parser.error("--predictions must be an object of boolean decisions")
    report = assess(
        json.loads(args.probe.read_text(encoding="utf-8")),
        json.loads(CASES.read_text(encoding="utf-8")),
        json.loads(args.reviewed_manifest.read_text(encoding="utf-8")),
        predictions,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
