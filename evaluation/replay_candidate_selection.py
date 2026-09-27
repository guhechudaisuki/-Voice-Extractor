"""Replay cached local findings through the experimental proposal selector.

This is selection-only: no new acoustic decision, ASR or audio export. Human
review is never used as input to selection. Reported conflicts are model
conflicts, not ground truth that a person changed at an exact timestamp.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extractor.candidate_selection import RecoverySelector
from extractor.types import CandidateSentence


def candidate(row: dict) -> CandidateSentence:
    return CandidateSentence(**{key: value for key, value in row.items()
                               if key in CandidateSentence.__dataclass_fields__})


def record_local(selector: RecoverySelector, row: dict, *, passed: bool) -> None:
    info = row.get("diagnostics", {})
    if info.get("local_identity_evidence"):
        for evidence in info["local_identity_evidence"]:
            selector.observe(CandidateSentence(*evidence["span"], ""), evidence["state"])
    elif info.get("local_boundary_recovery") and info.get("recovery_part_index") is not None:
        # Legacy logs retained atomic scope but no explicit three-state record.
        state = ("other" if info.get("excluded_role_rejected")
                 else "target" if passed else "unresolved")
        selector.observe(candidate(row), state)


def replay(report: dict, proposals: list[dict]) -> dict:
    selector = RecoverySelector()
    for row in report["accepted_proposals"]:
        record_local(selector, row, passed=True)
    for row in report["rejected_proposals"]:
        record_local(selector, row, passed=False)
    selected = [candidate(row) for row in report["accepted_proposals"]]
    deferred = []
    start, end = report["source_turn"]
    tested = []
    for row in proposals:
        # Replaying only source-contained alternatives avoids pretending a local
        # report supplies complete evidence for a candidate outside its scope.
        if row["start"] < start or row["end"] > end:
            continue
        proposed = candidate(row)
        selector.install(proposed, selected, deferred)
        tested.append([proposed.start, proposed.end])
    return {
        "source_turn": [start, end], "tested_proposals": tested,
        "selected_proposals": [row.to_dict() for row in selected],
        "deferred_proposals": [row.to_dict() for row in deferred],
        "ledger": selector.to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("local_report", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    local = json.loads(args.local_report.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    # Accepted output and pre-STT locator recovery records are acoustic
    # proposals. Do not use rejected local fragments as whole-span approvals.
    proposals = [row for row in manifest["sentences"]
                 if row.get("accepted") or row.get("diagnostics", {}).get("target_locator_recovery")]
    reports = local.get("reports", [local])
    results = [replay(report, proposals) for report in reports]
    output = {
        "warning": "Offline selection replay; not acoustic verification, final output or accuracy.",
        "local_report": str(args.local_report), "manifest": str(args.manifest),
        "reports": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    for result in results:
        print("SELECTION", result["source_turn"],
              "retained", [(row["start"], row["end"]) for row in result["selected_proposals"]],
              "deferred", [(row["start"], row["end"]) for row in result["deferred_proposals"]])


if __name__ == "__main__":
    main()
