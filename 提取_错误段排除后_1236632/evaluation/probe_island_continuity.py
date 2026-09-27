"""Compare adjacent quiet-gap speech islands with frozen anime encoders.

This is a diagnostic only: continuity scores on short speech cannot certify
identity, and neither this probe nor its output changes exported audio.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_frozen_join_audio import read_audio  # noqa: E402
from probe_internal_island_identity import speech_islands  # noqa: E402
from extractor.nextgen.anime_embedding import AnimeSpeakerOnnx  # noqa: E402


def run(work: Path, pauses: dict) -> dict:
    encoders = {
        variant: AnimeSpeakerOnnx(ROOT / "models", variant)
        for variant in ("char", "va")
    }
    sources = {
        "stem": work / "stems" / "target_vocals.wav",
        "raw": work / "target_normalized.wav",
    }
    cache: dict[tuple[str, str, float, float], np.ndarray | None] = {}

    def vector(variant: str, channel: str,
               span: tuple[float, float]) -> np.ndarray | None:
        key = (variant, channel, *span)
        if key not in cache:
            if round((span[1] - span[0]) * 16000) < 3200:
                cache[key] = None
            else:
                try:
                    cache[key] = encoders[variant].encode(
                        read_audio(sources[channel], *span), sample_rate=16000,
                    )
                except ValueError:
                    cache[key] = None
        return cache[key]

    rows = []
    for row in pauses["rows"]:
        start, end = map(float, row["accepted_span"])
        islands = speech_islands(start, end, row["gaps"])
        pairs = []
        for left, right in zip(islands, islands[1:]):
            scores = {}
            for variant in encoders:
                for channel in sources:
                    first = vector(variant, channel, left)
                    second = vector(variant, channel, right)
                    scores[f"{variant}_{channel}"] = (
                        round(float(first @ second), 6)
                        if first is not None and second is not None else None
                    )
            pairs.append({"left": list(left), "right": list(right),
                          "similarities": scores})
        rows.append({"accepted_span": [start, end], "pairs": pairs})
    return {
        "warning": "Development continuity evidence only. No threshold or export decision is implied.",
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
