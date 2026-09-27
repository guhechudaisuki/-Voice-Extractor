"""Build a review listening package with user-flagged error regions excluded.

Reads a batch manifest plus the human review file, removes every accepted
export that overlaps a flagged span, and copies the remaining audio/text into
an explicitly unverified preview folder and ZIP. Review truth stays inside the
evaluation tooling; the extraction pipeline itself never sees it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]


def overlap(left: tuple[float, float], right: tuple[float, float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_manifest", type=Path)
    parser.add_argument("--review", type=Path,
                        default=Path(__file__).with_name("episode1_user_review_20260926.json"))
    parser.add_argument("--min-overlap", type=float, default=0.10,
                        help="Exports overlapping a flagged span by more than this are excluded")
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.batch_manifest.read_text(encoding="utf-8"))
    review = json.loads(args.review.read_text(encoding="utf-8"))
    flagged = [tuple(map(float, case["span"])) for case in review["flagged_outputs"]]

    batch_root = args.batch_manifest.parent
    accepted = [row for row in manifest["sentences"] if row["accepted"]]
    kept, removed = [], []
    for row in accepted:
        span = (float(row["start"]), float(row["end"]))
        hits = [list(flagged_span) for flagged_span in flagged
                if overlap(span, flagged_span) > args.min_overlap]
        (removed if hits else kept).append({
            "start": span[0], "end": span[1], "text": row.get("text", ""),
            "overlaps_flagged": hits, "row": row,
        })

    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    audio_dir = destination / "audio"
    text_dir = destination / "text"
    audio_dir.mkdir()
    text_dir.mkdir()

    entries = []
    for ordinal, item in enumerate(kept, 1):
        row = item["row"]
        basename = f"{ordinal:04d}_{row['start']:.2f}-{row['end']:.2f}"
        source_audio = batch_root / row["audio_file"]
        source_text = batch_root / row["text_file"] if row.get("text_file") else None
        audio_file = audio_dir / f"{basename}.wav"
        text_file = text_dir / f"{basename}.txt"
        audio_file.write_bytes(source_audio.read_bytes())
        text = Path(source_text).read_text(encoding="utf-8") if source_text and source_text.is_file() else row.get("text", "")
        text_file.write_text(text, encoding="utf-8")
        entries.append({
            "span": [row["start"], row["end"]],
            "audio_file": str(audio_file.relative_to(destination)),
            "text_file": str(text_file.relative_to(destination)),
            "stt": text,
            "in_frozen_baseline": row.get("diagnostics", {}).get("in_frozen_baseline"),
        })

    payload = {
        "status": "UNVERIFIED_PREVIEW_ERRORS_EXCLUDED",
        "source_manifest": str(args.batch_manifest),
        "review_file": str(args.review),
        "warning": (
            "Human-flagged error regions and every export overlapping them were "
            "removed. Remaining clips still need human listening; do not use as "
            "training audio or replace the stable version."
        ),
        "flagged_spans": [list(span) for span in flagged],
        "kept_count": len(kept),
        "removed_count": len(removed),
        "removed": [
            {"span": [item["start"], item["end"]],
             "text": item["text"],
             "overlaps_flagged": item["overlaps_flagged"]}
            for item in removed
        ],
        "clips": entries,
    }
    (destination / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    archive = destination.with_suffix(".zip")
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as zipped:
        for file in sorted(destination.rglob("*")):
            if file.is_file():
                zipped.write(file, file.relative_to(destination))
    print(f"PREVIEW={destination}")
    print(f"ZIP={archive}")
    print(f"KEPT={len(kept)} REMOVED={len(removed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
