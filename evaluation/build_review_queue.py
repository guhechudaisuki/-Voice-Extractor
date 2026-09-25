"""Build a source-time review queue without assigning machine labels as truth.

Each acoustic island gets a stable source-hash/sample-index key and links to
current decisions and historical listening evidence. Human fields stay blank.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_reviewed_manifest import covered_seconds, intervals, overlap


def _span(record: dict) -> tuple[float, float]:
    return float(record["start"]), float(record["end"])


def build_queue(
    reviewed: dict,
    current: dict,
    audit: dict,
    provenance: dict,
    *,
    sample_rate: int = 16000,
) -> dict:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    source_hash = provenance.get("inputs_sha256", {}).get("target")
    if not isinstance(source_hash, str) or re.fullmatch(r"[0-9a-fA-F]{64}", source_hash) is None:
        raise ValueError("evaluation provenance must contain the full target SHA-256")
    islands = audit.get("stages", {}).get("atomic_speech_islands", {}).get("spans")
    if not isinstance(islands, list):
        raise ValueError("stage audit has no atomic_speech_islands")
    accepted = intervals(current, True)
    rejected = intervals(current, False)
    historical = intervals(reviewed, True)
    rows = []
    for number, (start, end) in enumerate(islands, start=1):
        start, end = float(start), float(end)
        if end <= start:
            raise ValueError(f"Invalid island at index {number}")
        span = start, end
        touching_rejections = [
            item for item in rejected if overlap(span, _span(item)) > 0
        ]
        reasons = sorted({item.get("reject_reason") or "未知原因" for item in touching_rejections})
        reviewed_coverage = covered_seconds(span, historical)
        current_coverage = covered_seconds(span, accepted)
        rows.append({
            "source_id": f"{source_hash}:{round(start * sample_rate)}:{round(end * sample_rate)}",
            "island_index": number,
            "start": start,
            "end": end,
            "historical_reviewed_overlap_seconds": round(reviewed_coverage, 5),
            "current_accepted_overlap_seconds": round(current_coverage, 5),
            "current_rejection_reasons": reasons,
            "human_label": None,
            "human_speaker_boundary_intervals": None,
            "human_notes": None,
        })
    return {
        "schema_version": 1,
        "warning": (
            "Current decisions and historical overlap are review hints only. "
            "No machine output here is a ground-truth identity or completeness label. "
            "Confirm separately that a historical manifest comes from the same source."
        ),
        "source_sha256": source_hash,
        "sample_rate": sample_rate,
        "island_count": len(rows),
        "islands": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reviewed_manifest", type=Path)
    parser.add_argument("current_manifest", type=Path)
    parser.add_argument("stage_audit", type=Path)
    parser.add_argument("evaluation_provenance", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (
            args.reviewed_manifest,
            args.current_manifest,
            args.stage_audit,
            args.evaluation_provenance,
        )
    ]
    queue = build_queue(*inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(queue, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"REVIEW_QUEUE={args.output} ISLANDS={queue['island_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
