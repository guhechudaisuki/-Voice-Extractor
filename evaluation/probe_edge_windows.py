"""Read-only, reference-conditioned evidence for the ends of cached clips.

Equal-length adjacent windows expose a short foreign tail that can disappear
inside a longer utterance embedding. This is a diagnostic, not a cut rule or
an export certificate. All windows are generated from clip geometry alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_frozen_join_audio import build_bank, view_margins  # noqa: E402
from extractor.audio import load_mono  # noqa: E402
from extractor.pipeline import ExtractionPipeline  # noqa: E402
from extractor.types import TimeSpan  # noqa: E402


def edge_windows(start: float, end: float,
                 lengths: tuple[float, ...] = (0.25, 0.35, 0.5, 0.75)) -> list[dict]:
    """Return unbiased trailing windows and preceding same-size controls."""
    if not 0 <= start < end:
        raise ValueError("Invalid clip span")
    windows = [{"kind": "whole", "span": [start, end]}]
    for length in lengths:
        if length < 0.20:
            raise ValueError("Anime identity windows require at least 0.20 seconds")
        if end - start < 2 * length:
            continue
        windows.append({"kind": f"before_tail_{length:g}",
                        "span": [end - 2 * length, end - length]})
        windows.append({"kind": f"tail_{length:g}",
                        "span": [end - length, end]})
    return windows


def _db(waveform: torch.Tensor) -> float:
    rms = torch.sqrt(torch.mean(waveform.square()) + 1e-12)
    return round(float(20.0 * torch.log10(rms + 1e-12)), 2)


def run(work: Path, manifest: dict, lengths: tuple[float, ...]) -> dict:
    if "clips" in manifest:
        clips = [(index, *map(float, row["span"]))
                 for index, row in enumerate(manifest["clips"], 1)]
    elif "sentences" in manifest:
        clips = [(index, float(row["start"]), float(row["end"]))
                 for index, row in enumerate(manifest["sentences"], 1)
                 if row["accepted"]]
    else:
        raise ValueError("Manifest must contain clips or sentences")
    bank = build_bank(work)
    stem = load_mono(work / "stems" / "target_vocals.wav", 16000)
    raw = load_mono(work / "target_normalized.wav", 16000)
    rows = []
    for index, start, end in clips:
        windows = []
        for window in edge_windows(start, end, lengths):
            left, right = window["span"]
            span = TimeSpan(left, right)
            stem_part = ExtractionPipeline._waveform_span(stem, span)
            raw_part = ExtractionPipeline._waveform_span(raw, span)
            verdict = bank.score(stem=stem_part, raw=raw_part,
                                 sample_rate=16000)
            windows.append({
                **window,
                "state": verdict.state,
                "view_margins": view_margins(verdict),
                "nearest_roles": {
                    f"{score.variant}_{score.channel}": score.nearest_role
                    for score in verdict.scores
                },
                "stem_db": _db(stem_part),
                "raw_db": _db(raw_part),
            })
        rows.append({"index": index, "span": [start, end], "windows": windows})
    return {
        "warning": "Development evidence only. Short-window speaker scores and energy are not reliable cut or export labels.",
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--lengths", type=float, nargs="+",
                        default=[0.25, 0.35, 0.5, 0.75])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.work, json.loads(args.manifest.read_text(encoding="utf-8")),
                 tuple(args.lengths))
    contents = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(contents, encoding="utf-8")
        print(f"Audited {len(report['rows'])} clip edges: {args.output}")
    else:
        print(contents)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
