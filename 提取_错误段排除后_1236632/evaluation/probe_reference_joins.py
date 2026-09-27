"""Stress-test multiscale join evidence using explicitly grouped references.

Reference grouping supplies only the expected same/different speaker relation.
These constructed transitions are not naturally occurring dialogue or blind
evidence of production precision.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from itertools import combinations, product
from pathlib import Path
from tempfile import TemporaryDirectory

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extractor.audio import load_mono
from extractor.speaker import CAMPlusVerifier, LocalSpeakerTurnSplitter, SpeakerVerifier
from extractor.types import TimeSpan


def select_pairs(
    targets: list[Path], others: list[Path],
    *, per_kind: int, seed: int,
) -> list[tuple[str, Path, Path, bool]]:
    if len(targets) < 2 or not others or per_kind < 1:
        raise ValueError("Need two target references, one other reference, and positive pair count")
    rng = random.Random(seed)
    same = list(combinations(targets, 2))
    target_other = list(product(targets, others))
    other_target = [(other, target) for target, other in target_other]
    result = []
    for kind, pairs, expected in (
        ("target_target", same, True),
        ("target_other", target_other, False),
        ("other_target", other_target, False),
    ):
        result.extend(
            (kind, left, right, expected)
            for left, right in rng.sample(pairs, min(per_kind, len(pairs)))
        )
    return result


def run(work_dir: Path, *, per_kind: int, seed: int, gap_seconds: float) -> dict:
    if gap_seconds < 0 or gap_seconds > 0.85:
        raise ValueError("Expected a gap inside the product's merge range")
    target_paths = sorted((work_dir / "reference_voice_clips").glob("*.wav"))
    negative_paths = sorted((work_dir / "negative_reference_voice_clips").glob("role_*/*.wav"))
    pairs = select_pairs(target_paths, negative_paths, per_kind=per_kind, seed=seed)
    primary = SpeakerVerifier()
    secondary = CAMPlusVerifier()
    splitter = LocalSpeakerTurnSplitter(primary, secondary=secondary)
    rows = []
    try:
        with TemporaryDirectory(prefix="reference_join_", dir=work_dir) as temporary:
            for index, (kind, left_path, right_path, expected_same) in enumerate(pairs):
                left = load_mono(left_path, 16000)
                right = load_mono(right_path, 16000)
                gap = torch.zeros(round(gap_seconds * 16000))
                joined = torch.cat((left, gap, right))
                join_start = left.numel() / 16000
                join_end = join_start + gap_seconds
                path = Path(temporary) / f"pair_{index:03d}.wav"
                sf.write(path, joined.numpy(), 16000, subtype="PCM_16")
                # Only the local transition is relevant. Keep full source
                # waveforms but restrict the detector's analysis span.
                analysis = TimeSpan(
                    max(0.0, join_start - 1.4),
                    min(joined.numel() / 16000, join_end + 1.4),
                )
                boundaries = splitter.detect_multiscale_speaker_boundaries(path, [analysis])
                near = [
                    boundary for boundary in boundaries
                    if join_start - 0.25 <= boundary.time <= join_end + 0.25
                ]
                rows.append({
                    "kind": kind,
                    "expected_same_group": expected_same,
                    "left_path": str(left_path.relative_to(work_dir)),
                    "right_path": str(right_path.relative_to(work_dir)),
                    "join_change_detected": bool(near),
                    "near_boundary_times": [round(item.time, 5) for item in near],
                })
                print(f"{index + 1}/{len(pairs)} {kind} change={bool(near)}", flush=True)
    finally:
        secondary.close()
        primary.close()
    return {
        "warning": "Constructed reference joins, not natural dialogue. Absence of a change cannot authorize an output join.",
        "gap_seconds": gap_seconds,
        "rows": rows,
        "summary": {
            kind: {
                "count": sum(row["kind"] == kind for row in rows),
                "join_change_detected": sum(
                    row["kind"] == kind and row["join_change_detected"] for row in rows
                ),
            }
            for kind in ("target_target", "target_other", "other_target")
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("--per-kind", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--gap-seconds", type=float, default=0.35)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        args.work_dir, per_kind=args.per_kind,
        seed=args.seed, gap_seconds=args.gap_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"REFERENCE_JOIN_PROBE={args.output} PAIRS={len(report['rows'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
