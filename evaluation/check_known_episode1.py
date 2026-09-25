"""Check narrow, time-indexed episode-1 regressions against a batch manifest.

This is deliberately not a whole-episode recall or precision evaluator. A pass
does not certify the identity or completeness of unreviewed output clips.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASES = Path(__file__).with_name("known_episode1_cases.json")


def accepted_spans(manifest: dict) -> list[tuple[float, float]]:
    return [
        (float(item["start"]), float(item["end"]))
        for item in manifest.get("sentences", [])
        if item.get("accepted") is True
    ]


def assess(manifest: dict, cases: dict) -> list[dict]:
    spans = accepted_spans(manifest)
    results = []
    for case in cases["cases"]:
        kind = case["kind"]
        if kind == "must_not_cover_reviewed_mixed_clip":
            mixed_start, mixed_end = case["span"]
            containing = [
                (start, end) for start, end in spans
                if start <= mixed_start + 0.02 and end >= mixed_end - 0.02
            ]
            passed = not containing
            evidence = {"containing_mixed_clip": containing, "reviewed_span": case["span"]}
        elif kind == "must_cover_target_tail":
            core_start, core_end = case["target_core"]
            tail_start, tail_end = case["tail_probe"]
            next_start = case["next_speaker_starts_no_earlier_than"]
            candidates = [
                (start, end) for start, end in spans
                if start <= core_start + 0.10 and end >= core_end - 0.10
            ]
            passed = any(
                end >= tail_end - 0.10 and end <= next_start + 0.02
                for _start, end in candidates
            )
            evidence = {"matching_core_spans": candidates, "tail_probe": [tail_start, tail_end]}
        elif kind == "must_not_cross_change":
            change = case["change_time"]
            left = case["left_context"]
            right = case["right_context"]
            crossing = [
                (start, end) for start, end in spans
                if start < change - left and end > change + right
            ]
            passed = not crossing
            evidence = {"crossing_spans": crossing, "change_time": change}
        else:
            raise ValueError(f"Unknown regression case: {kind}")
        results.append({"id": case["id"], "passed": passed, "evidence": evidence})
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Batch manifest to evaluate")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    results = assess(manifest, cases)
    print(json.dumps({"source": str(args.manifest), "results": results}, ensure_ascii=False, indent=2))
    return 0 if all(item["passed"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
