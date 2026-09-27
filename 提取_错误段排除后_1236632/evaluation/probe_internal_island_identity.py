"""Score every speech island inside accepted clips; never change exports.

Quiet gaps are only boundary proposals. Short islands or contradictory model
votes remain unresolved rather than inheriting the whole clip's target label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_frozen_join_audio import build_bank, view_margins  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extractor.audio import load_mono  # noqa: E402
from extractor.pipeline import ExtractionPipeline  # noqa: E402
from extractor.types import TimeSpan  # noqa: E402


def speech_islands(start: float, end: float,
                   gaps: list[dict]) -> list[tuple[float, float]]:
    if not 0 <= start < end:
        raise ValueError("Invalid accepted clip span")
    cursor = start
    islands = []
    for gap in gaps:
        left, right = map(float, gap["span"])
        if left < cursor or right <= left or right > end:
            raise ValueError("Quiet gaps must be ordered within the clip")
        if left > cursor:
            islands.append((cursor, left))
        cursor = right
    if cursor < end:
        islands.append((cursor, end))
    return islands


def run(work: Path, pauses: dict) -> dict:
    bank = build_bank(work)
    stem = load_mono(work / "stems" / "target_vocals.wav", 16000)
    raw = load_mono(work / "target_normalized.wav", 16000)
    rows = []
    for row in pauses["rows"]:
        start, end = map(float, row["accepted_span"])
        parts = []
        for left, right in speech_islands(start, end, row["gaps"]):
            part = {"span": [left, right], "duration": round(right - left, 5)}
            if round((right - left) * 16000) < 3200:
                part["state"] = "unresolved_short"
            else:
                try:
                    verdict = bank.score(
                        stem=ExtractionPipeline._waveform_span(
                            stem, TimeSpan(left, right)
                        ),
                        raw=ExtractionPipeline._waveform_span(
                            raw, TimeSpan(left, right)
                        ),
                        sample_rate=16000,
                    )
                    part["state"] = verdict.state
                    part["view_margins"] = view_margins(verdict)
                except ValueError as error:
                    part["state"] = "unresolved_acoustic"
                    part["reason"] = str(error)
            parts.append(part)
        rows.append({"accepted_span": [start, end], "speech_islands": parts})
    return {
        "warning": "Development diagnosis only. Quiet gaps and model states are not human truth or export permission.",
        "reviewed_clips": len(rows),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("pauses", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.work, json.loads(args.pauses.read_text(encoding="utf-8")))
    contents = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(contents, encoding="utf-8")
    else:
        print(contents)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
