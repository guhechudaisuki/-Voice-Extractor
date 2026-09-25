"""Evaluate an offline full-utterance join hypothesis against fixed evidence.

This is not a deployment threshold. It tests a structural proposal: both
speaker parts must independently favor the target in a domain model, the
whole utterance must pass the existing verifier, and a confirmed local change
vetoes a join. Labels are only used after decisions for evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def target_wins(side: dict) -> bool:
    """Compare robust target reference score with the strongest other group."""
    negatives = side.get("negative_group_max")
    if not negatives:
        return False
    return float(side["target_median"]) > max(float(score) for score in negatives)


def assess_episode(
    cases: dict, whole_probe: dict, boundary_stem: dict, boundary_raw: dict,
    char_probe: dict, va_probe: dict,
) -> dict:
    joined = {}
    for name, report, field in (
        ("whole", whole_probe, "pairs"),
        ("stem_boundary", boundary_stem, "cases"),
        ("raw_boundary", boundary_raw, "cases"),
        ("char", char_probe, "cases"),
        ("va", va_probe, "cases"),
    ):
        rows = report[field]
        by_id = {row["id"]: row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError(f"Duplicate case IDs in {name}")
        joined[name] = by_id
    results = []
    for case in cases["cases"]:
        case_id = case["id"]
        if not all(case_id in records for records in joined.values()):
            raise ValueError(f"Missing evidence for {case_id}")
        whole = joined["whole"][case_id]
        stem_boundary = joined["stem_boundary"][case_id]
        raw_boundary = joined["raw_boundary"][case_id]
        char = joined["char"][case_id]
        va = joined["va"][case_id]
        for name, record in joined.items():
            if name == "whole":
                continue
            if bool(record[case_id]["historical_same_target_pair"]) != bool(case["same_target"]):
                raise ValueError(f"Case label mismatch for {case_id} in {name}")
        side_witnesses = {
            side: {
                "char": target_wins(char[side]),
                "va": target_wins(va[side]),
            }
            for side in ("left", "right")
        }
        decision = (
            float(whole["gap_seconds"]) <= 0.85
            and whole["whole_result"]["accepted_by_whole_span_verifier"]
            and not whole["whole_result"]["excluded_role_rejected"]
            and not stem_boundary["join_change_detected"]
            and not raw_boundary["join_change_detected"]
            and all(any(witness.values()) for witness in side_witnesses.values())
        )
        results.append({
            "id": case_id,
            "expected_same_target": bool(case["same_target"]),
            "join_proposed": bool(decision),
            "passed": bool(decision) == bool(case["same_target"]),
            "side_witnesses": side_witnesses,
            "stem_change": bool(stem_boundary["join_change_detected"]),
            "raw_change": bool(raw_boundary["join_change_detected"]),
        })
    return {
        "passed": sum(row["passed"] for row in results),
        "total": len(results),
        "results": results,
    }


def assess_reference_stress(
    boundaries: dict, char_competition: dict, va_competition: dict,
) -> dict:
    left = boundaries["rows"]
    char = char_competition["rows"]
    va = va_competition["rows"]
    if not len(left) == len(char) == len(va):
        raise ValueError("Reference stress reports have different pair counts")
    results = []
    for boundary, char_row, va_row in zip(left, char, va):
        if len({boundary["kind"], char_row["kind"], va_row["kind"]}) != 1:
            raise ValueError("Reference stress pair ordering differs")
        if bool(boundary["expected_same_group"]) != bool(char_row["expected_same_group"]):
            raise ValueError("Reference stress truth differs")
        side_witnesses = {
            side: {
                "char": bool(char_row[side]["open_set_competition"]["target_wins"]),
                "va": bool(va_row[side]["open_set_competition"]["target_wins"]),
            }
            for side in ("left", "right")
        }
        decision = (
            not boundary["join_change_detected"]
            and all(any(witness.values()) for witness in side_witnesses.values())
        )
        expected = bool(boundary["expected_same_group"])
        results.append({
            "kind": boundary["kind"],
            "expected_same_group": expected,
            "join_proposed": bool(decision),
            "passed": bool(decision) == expected,
            "side_witnesses": side_witnesses,
        })
    return {
        "passed": sum(row["passed"] for row in results),
        "total": len(results),
        "same_group_recovered": sum(
            row["expected_same_group"] and row["join_proposed"] for row in results
        ),
        "same_group_total": sum(row["expected_same_group"] for row in results),
        "different_group_false_joins": sum(
            not row["expected_same_group"] and row["join_proposed"] for row in results
        ),
        "results": results,
    }


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_dir", type=Path)
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("join_cases.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = args.evidence_dir
    report = {
        "warning": "Both episode and constructed reference cases are development-only. Scores were inspected before this hypothesis; no independent natural-dialogue validation yet.",
        "episode": assess_episode(
            read(args.cases),
            read(directory / "join_probe_corrected.json"),
            read(directory / "join_boundary_stem_probe.json"),
            read(directory / "join_boundary_raw_probe.json"),
            read(directory / "anime_char_stem_probe.json"),
            read(directory / "anime_va_stem_probe.json"),
        ),
        "reference_stress": assess_reference_stress(
            read(directory / "reference_join_boundary_probe.json"),
            read(directory / "reference_char_competition_probe.json"),
            read(directory / "reference_va_competition_probe.json"),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"JOIN_HYPOTHESIS={args.output} EPISODE={report['episode']['passed']}/{report['episode']['total']} REFERENCE={report['reference_stress']['passed']}/{report['reference_stress']['total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
