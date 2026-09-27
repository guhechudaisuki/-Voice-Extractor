"""Audit frozen anime embeddings on previously listened-to development clips.

This is an offline identity diagnostic, not a ground-truth generator or a
production acceptance gate. The reviewed first-episode clips are incomplete as
a recall denominator, and some clips may overlap the supplied references.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_frozen_join_audio import build_bank, view_margins  # noqa: E402
from extractor.audio import load_mono  # noqa: E402
from extractor.pipeline import ExtractionPipeline  # noqa: E402
from extractor.types import TimeSpan  # noqa: E402


def parse_other(value: str) -> tuple[str, float, float]:
    try:
        name, bounds = value.split("=", 1)
        start_text, end_text = bounds.split(":", 1)
        start, end = float(start_text), float(end_text)
        if not name or not 0 <= start < end:
            raise ValueError("Invalid name or source interval")
        return name, start, end
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Known other must use NAME=START:END with START < END"
        ) from error


def audit(work: Path, manifest: dict,
          known_other: list[tuple[str, float, float]]) -> dict:
    raw = work / "target_normalized.wav"
    stem = work / "stems/target_vocals.wav"
    if not raw.is_file() or not stem.is_file():
        raise FileNotFoundError("Cached paired target audio is required")
    raw_duration = sf.info(raw).duration
    stem_duration = sf.info(stem).duration
    if abs(raw_duration - stem_duration) > 0.05:
        raise ValueError("Original and UVR audio have different timelines")
    reviewed = [row for row in manifest["sentences"] if row.get("accepted")]
    if not reviewed:
        raise ValueError("Reviewed manifest contains no accepted clips")
    spans = [
        (f"reviewed_{index:04d}", float(row["start"]),
         float(row["end"]), "reviewed_target")
        for index, row in enumerate(reviewed, 1)
    ]
    spans.extend((name, start, end, "known_other")
                 for name, start, end in known_other)
    if any(not 0 <= start < end <= min(raw_duration, stem_duration)
           for _name, start, end, _kind in spans):
        raise ValueError("A reviewed source interval is outside the cached audio")
    bank = build_bank(work)
    stem_waveform = load_mono(stem, 16000)
    raw_waveform = load_mono(raw, 16000)
    rows = []
    for name, start, end, kind in spans:
        span = TimeSpan(start, end)
        verdict = bank.score(
            stem=ExtractionPipeline._waveform_span(stem_waveform, span),
            raw=ExtractionPipeline._waveform_span(raw_waveform, span),
            sample_rate=16000,
        )
        rows.append({
            "id": name,
            "kind": kind,
            "source_span": [start, end],
            "state": verdict.state,
            "view_margins": view_margins(verdict),
        })
    positives = [row for row in rows if row["kind"] == "reviewed_target"]
    negatives = [row for row in rows if row["kind"] == "known_other"]
    return {
        "schema": 1,
        "warning": "Development clips only; not an independent or complete identity/recall test.",
        "reviewed_target_supported": sum(
            row["state"] == "target_supported" for row in positives
        ),
        "reviewed_target_total": len(positives),
        "known_other_supported_as_target": sum(
            row["state"] == "target_supported" for row in negatives
        ),
        "known_other_total": len(negatives),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("reviewed_manifest", type=Path)
    parser.add_argument("--known-other", type=parse_other, action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(
        args.work,
        json.loads(args.reviewed_manifest.read_text(encoding="utf-8")),
        args.known_other,
    )
    content = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(content)
    else:
        print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
