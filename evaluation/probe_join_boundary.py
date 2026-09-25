"""Check whether existing multiscale change evidence separates join cases.

The result is development-only evidence. Absence of a detected change is not
proof of a single speaker and cannot, by itself, authorize an output join.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extractor.audio import write_clip
from extractor.speaker import CAMPlusVerifier, LocalSpeakerTurnSplitter, SpeakerVerifier
from extractor.types import TimeSpan


CASES = Path(__file__).with_name("join_cases.json")


def boundary_near_join(
    times: list[float], left_end: float, right_start: float,
    tolerance: float = 0.25,
) -> bool:
    return any(left_end - tolerance <= time <= right_start + tolerance for time in times)


def run(
    work_dir: Path, cases_path: Path, channel: str,
    *, single_context: float | None = None,
) -> dict:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))["cases"]
    return probe_cases(work_dir, cases, channel, single_context=single_context)


def probe_cases(
    work_dir: Path, cases: list[dict], channel: str,
    *, single_context: float | None = None,
) -> dict:
    if channel not in ("stem", "raw"):
        raise ValueError("channel must be stem or raw")
    if single_context is not None and single_context < 0.30:
        raise ValueError("single-context window must be at least 0.30 seconds")
    source = work_dir / (
        "stems/target_vocals.wav" if channel == "stem" else "target_normalized.wav"
    )
    primary = SpeakerVerifier()
    secondary = CAMPlusVerifier()
    splitter = LocalSpeakerTurnSplitter(primary, secondary=secondary)
    rows = []
    try:
        with TemporaryDirectory(prefix="join_boundary_", dir=work_dir) as temporary:
            for index, case in enumerate(cases):
                start = float(case["left"][0])
                end = float(case["right"][1])
                path = Path(temporary) / f"case_{index:02d}.wav"
                write_clip(source, path, start, end, sample_rate=16000)
                if single_context is None:
                    boundaries = splitter.detect_multiscale_speaker_boundaries(
                        path, [TimeSpan(0.0, end - start)],
                    )
                else:
                    boundaries = splitter.detect_speaker_boundaries(
                        path, [TimeSpan(0.0, end - start)],
                        context_seconds=single_context,
                        scan_hop_seconds=0.10,
                        primary_candidate_threshold=0.78,
                        minimum_separation_seconds=0.20,
                    )
                times = [start + boundary.time for boundary in boundaries]
                rows.append({
                    "id": case["id"],
                    "historical_same_target_pair": (
                        bool(case["same_target"])
                        if case.get("same_target") is not None else None
                    ),
                    "left": case["left"],
                    "right": case["right"],
                    "join_change_detected": boundary_near_join(
                        times, float(case["left"][1]), float(case["right"][0]),
                    ),
                    "boundaries": [
                        {"time": round(start + boundary.time, 5),
                         "confidence": boundary.confidence,
                         "scale_votes": boundary.scale_votes,
                         "primary_similarity": boundary.primary_similarity,
                         "secondary_similarity": boundary.secondary_similarity}
                        for boundary in boundaries
                    ],
                })
    finally:
        secondary.close()
        primary.close()
    return {
        "warning": "Development cases only. A missing boundary is not same-speaker evidence and does not authorize merging.",
        "channel": channel,
        "single_context_seconds": single_context,
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--island-scan", type=Path,
                        help="Instead probe all domain-witness adjacent islands, without assigning truth labels")
    parser.add_argument("--channel", choices=("stem", "raw"), default="stem")
    parser.add_argument("--single-context", type=float,
                        help="Use one short acoustic context instead of multiscale consensus")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.island_scan:
        scan = json.loads(args.island_scan.read_text(encoding="utf-8"))
        cases = [
            {
                "id": f"island_{pair['left_index']}_{pair['right_index']}",
                "left": pair["left"],
                "right": pair["right"],
                "same_target": None,
            }
            for pair in scan["pairs"] if pair["both_parts_have_domain_witness"]
        ]
        report = probe_cases(
            args.work_dir, cases, args.channel,
            single_context=args.single_context,
        )
    else:
        report = run(
            args.work_dir, args.cases, args.channel,
            single_context=args.single_context,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"JOIN_BOUNDARY_PROBE={args.output} CASES={len(report['cases'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
