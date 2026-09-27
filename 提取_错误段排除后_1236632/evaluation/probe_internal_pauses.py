"""Scan already accepted cached clips for VAD-missed UVR quiet gaps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from extractor.audio import load_mono  # noqa: E402
from extractor.nextgen.internal_pauses import locate_quiet_gaps  # noqa: E402
from extractor.pipeline import ExtractionPipeline  # noqa: E402
from extractor.types import TimeSpan  # noqa: E402


def run(work: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    options = manifest["options"]
    rows = []
    analysis_rate = 16000
    stem = load_mono(work / "stems" / "target_vocals.wav", analysis_rate)
    for turn in manifest["sentences"]:
        if not turn["accepted"]:
            continue
        start, end = float(turn["start"]), float(turn["end"])
        waveform = ExtractionPipeline._waveform_span(
            stem, TimeSpan(start, end)
        ).detach().cpu().numpy()
        gaps = locate_quiet_gaps(
            waveform, analysis_rate,
            lower_seconds=float(options["silence_min_seconds"]),
            upper_seconds=float(options["silence_split_seconds"]),
        )
        if gaps:
            rows.append({
                "accepted_span": [start, end],
                "gaps": [{
                    "span": [round(start + gap.start / analysis_rate, 5),
                             round(start + gap.end / analysis_rate, 5)],
                    "hard_split": gap.hard_split,
                    "left_db": gap.left_level_db,
                    "gap_db": gap.gap_level_db,
                    "right_db": gap.right_level_db,
                    "right_remainder_seconds": round(
                        end - (start + gap.end / analysis_rate), 5,
                    ),
                } for gap in gaps],
            })
    return {
        "warning": "Quiet gaps are candidate boundaries only; UVR can erase weak speech. No segment is cropped or accepted by this probe.",
        "accepted_count": int(manifest["accepted_count"]),
        "with_gap_count": len(rows),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.work, args.manifest)
    contents = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(contents, encoding="utf-8")
    else:
        print(contents)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
