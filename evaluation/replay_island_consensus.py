"""Replay an independent local-other-voice consensus on cached model evidence.

Development-only abstention analysis. Human review is loaded only after every
decision has been made, and this script never writes an audio export. In
particular, a weak short target island is not automatically another person.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extractor.nextgen.purity_consensus import (  # noqa: E402
    local_other_witnesses, whole_identity_conflict,
)


def _find_span(rows: list[dict], span: list[float], key: str,
               tolerance: float = 0.05) -> dict | None:
    return next(
        (row for row in rows
         if abs(float(row[key][0]) - span[0]) <= tolerance
         and abs(float(row[key][1]) - span[1]) <= tolerance),
        None,
    )


def audit(manifest: dict, domain_whole: dict, domain_islands: dict,
          production_islands: dict) -> dict:
    accepted = [row for row in manifest["sentences"] if row["accepted"]]
    findings = []
    for index, candidate in enumerate(accepted, 1):
        span = [float(candidate["start"]), float(candidate["end"])]
        whole = _find_span(domain_whole["rows"], span, "source_span")
        main_whole = next(
            (row for row in production_islands["rows"]
             if row["kind"] == "whole" and _find_span([row], span, "span")),
            None,
        )
        reasons = []
        whole_conflict_unavailable = False
        if whole is not None and main_whole is not None:
            exclusion = main_whole.get("exclusion") or {}
            direct_margin = exclusion.get("excluded_primary_direct_margin")
            # Older saved anime-domain reports have only one decoding view.
            # The current production abstention requires two, so do not
            # fabricate a whole-clip conflict from incomplete evidence.
            alternate = whole.get("alternate_view_margins")
            whole_conflict_unavailable = alternate is None
            if (alternate is not None and whole_identity_conflict(
                whole["view_margins"], alternate, direct_margin,
            )):
                reasons.append({"kind": "whole_identity_conflict", "span": span})

        domain_row = _find_span(domain_islands["rows"], span, "accepted_span")
        if domain_row is not None:
            for island in domain_row["speech_islands"]:
                margins = island.get("view_margins")
                if not margins:
                    continue
                local = next(
                    (row for row in production_islands["rows"]
                     if row["kind"] == "island"
                     and _find_span([row], island["span"], "span", 0.02)),
                    None,
                )
                if local is None:
                    continue
                exclusion = local.get("exclusion") or {}
                witnesses = local_other_witnesses(margins, exclusion)
                if witnesses:
                    reasons.append({
                        "kind": "local_other_voice_consensus",
                        "span": list(map(float, island["span"])),
                        "witnesses": list(witnesses),
                    })
        findings.append({"index": index, "span": span,
                         "abstain": bool(reasons), "reasons": reasons,
                         "whole_conflict_unavailable": whole_conflict_unavailable})
    return {"warning": "Development-only abstention proposals; absent alternate decoding evidence disables whole-clip conflict. No audio has been repaired or certified.",
            "abstained": sum(row["abstain"] for row in findings),
            "total": len(findings), "findings": findings}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("whole", type=Path)
    parser.add_argument("islands", type=Path)
    parser.add_argument("production", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(
        json.loads(args.manifest.read_text(encoding="utf-8")),
        json.loads(args.whole.read_text(encoding="utf-8")),
        json.loads(args.islands.read_text(encoding="utf-8")),
        json.loads(args.production.read_text(encoding="utf-8")),
    )
    if args.review:
        review = json.loads(args.review.read_text(encoding="utf-8"))
        flagged = {int(row["index"]) for row in review["flagged_outputs"]}
        report["reviewed_known_bad_abstained"] = sum(
            row["abstain"] for row in report["findings"] if row["index"] in flagged
        )
        report["reviewed_known_bad_total"] = len(flagged)
        report["unlisted_abstentions"] = [
            row["index"] for row in report["findings"]
            if row["abstain"] and row["index"] not in flagged
        ]
    contents = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(contents, encoding="utf-8")
        print(f"Audited {report['total']} clips; abstained {report['abstained']}: {args.output}")
    else:
        print(contents)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
