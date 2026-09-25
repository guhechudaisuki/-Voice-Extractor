"""Measure short-fragment identity evidence without changing extraction policy.

Historical reviewed outputs are development positives, not an exhaustive truth
set. Explicit known-other spans are development negatives. Neither these scores
nor a threshold fitted to them may authorize production output.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extractor.audio import load_mono  # noqa: E402
from extractor.speaker import DualSpeakerVerifier  # noqa: E402


def windows_for_span(
    start: float, end: float, length: float, hop: float,
) -> list[tuple[float, float]]:
    if not (0 <= start < end and length > 0 and hop > 0):
        raise ValueError("Invalid source span or window geometry")
    if end - start + 1e-6 < length:
        return []
    count = int(math.floor((end - start - length + 1e-6) / hop))
    starts = [start + index * hop for index in range(count + 1)]
    final = end - length
    if not starts or final - starts[-1] > 1e-5:
        starts.append(final)
    return [(round(value, 5), round(value + length, 5)) for value in starts]


def covered_fraction(
    start: float, end: float, speech_spans: list[tuple[float, float]],
) -> float:
    if not 0 <= start < end:
        raise ValueError("Invalid query span")
    clipped = sorted(
        (max(start, left), min(end, right))
        for left, right in speech_spans
        if min(end, right) > max(start, left)
    )
    merged: list[tuple[float, float]] = []
    for left, right in clipped:
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))
    return sum(right - left for left, right in merged) / (end - start)


def load_cases(manifest_path: Path, negatives_path: Path) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    negatives = json.loads(negatives_path.read_text(encoding="utf-8"))
    positive = [
        {
            "id": f"reviewed_{index:03d}",
            "label": "target",
            "span": [float(row["start"]), float(row["end"])],
        }
        for index, row in enumerate(manifest["sentences"], start=1)
        if row.get("accepted")
    ]
    if not positive:
        raise ValueError("Manifest has no accepted historical sentences")
    other = []
    for row in negatives["cases"]:
        start, end = map(float, row["span"])
        other.append({"id": str(row["id"]), "label": "other", "span": [start, end]})
    if not other:
        raise ValueError("No known-other spans were supplied")
    ids = [row["id"] for row in positive + other]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate development case ID")
    for case in positive + other:
        start, end = case["span"]
        if not 0 <= start < end:
            raise ValueError(f"Invalid span for {case['id']}")
    return positive + other


def _distribution(rows: list[dict], key: str) -> dict:
    values = torch.tensor([row[key] for row in rows], dtype=torch.float64)
    if not len(values):
        return {"count": 0}
    return {
        "count": len(values),
        "minimum": round(float(values.min()), 5),
        "p10": round(float(torch.quantile(values, 0.10)), 5),
        "median": round(float(torch.quantile(values, 0.50)), 5),
        "p90": round(float(torch.quantile(values, 0.90)), 5),
        "maximum": round(float(values.max()), 5),
    }


def run(
    work_dir: Path,
    manifest_path: Path,
    negatives_path: Path,
    stage_audit_path: Path,
    *,
    lengths: tuple[float, ...],
    hop: float,
    minimum_speech_fraction: float,
    channel: str,
) -> dict:
    if channel not in {"stem", "raw"}:
        raise ValueError("channel must be stem or raw")
    if not lengths or any(length < 0.30 for length in lengths):
        raise ValueError("Window lengths must be at least 0.30 seconds")
    if not 0 <= minimum_speech_fraction <= 1:
        raise ValueError("Speech coverage fraction must be between 0 and 1")
    if hop <= 0:
        raise ValueError("Window hop must be positive")
    cases = load_cases(manifest_path, negatives_path)
    stage = json.loads(stage_audit_path.read_text(encoding="utf-8"))
    speech_spans = [
        (float(start), float(end))
        for start, end in stage["stages"]["clean_speech_islands"]["spans"]
    ]
    source = work_dir / (
        "stems/target_vocals.wav" if channel == "stem" else "target_normalized.wav"
    )
    waveform = load_mono(source, 16000)
    duration = waveform.numel() / 16000
    rows: list[dict] = []
    queries: list[torch.Tensor] = []
    for case in cases:
        start, end = case["span"]
        if end > duration + 1e-5:
            raise ValueError(f"Case {case['id']} extends past source audio")
        for length in lengths:
            for left, right in windows_for_span(start, end, length, hop):
                first, last = round(left * 16000), round(right * 16000)
                if last <= first:
                    continue
                coverage = covered_fraction(left, right, speech_spans)
                rows.append({
                    "case_id": case["id"],
                    "label": case["label"],
                    "start": left,
                    "end": right,
                    "window_seconds": length,
                    "clean_speech_fraction": round(coverage, 5),
                })
                queries.append(waveform[first:last])
    if not rows:
        raise ValueError("No windows fit the supplied cases")
    reference_dir = work_dir / (
        "reference_voice_clips" if channel == "stem"
        else "reference_original_voice_clips"
    )
    references = sorted(reference_dir.glob("*.wav"))
    if len(references) < 2:
        raise ValueError("At least two target references are required")
    verifier = DualSpeakerVerifier()
    try:
        profile = verifier.build_profile(references, 0.68)
        secondary_profile = verifier._ensure_secondary(profile)
        assert verifier.secondary is not None
        primary_embeddings = verifier.primary._embeddings_from_waveforms(queries)
        secondary_embeddings = verifier.secondary._embeddings_from_waveforms(queries)
        for row, primary, secondary in zip(rows, primary_embeddings, secondary_embeddings):
            eres = float(primary @ profile.primary.centroid)
            camplus = float(secondary @ secondary_profile.centroid)
            row.update({
                "eres_centroid": round(eres, 5),
                "camplus_centroid": round(camplus, 5),
                "mean_centroid": round((eres + camplus) / 2, 5),
            })
    finally:
        verifier.close()
    summary = {}
    for length in lengths:
        kept = [
            row for row in rows
            if row["window_seconds"] == length
            and row["clean_speech_fraction"] >= minimum_speech_fraction
        ]
        summary[str(length)] = {
            label: {
                key: _distribution(
                    [row for row in kept if row["label"] == label], key,
                )
                for key in ("eres_centroid", "camplus_centroid", "mean_centroid")
            }
            for label in ("target", "other")
        }
    return {
        "warning": (
            "Development evidence only. Historical accepted clips are not a "
            "complete ground truth; short-window scores are not calibrated "
            "identity probabilities or production crop thresholds."
        ),
        "channel": channel,
        "reference_count": len(references),
        "source_duration_seconds": round(duration, 3),
        "case_count": len(cases),
        "window_count": len(rows),
        "minimum_clean_speech_fraction": minimum_speech_fraction,
        "summary": summary,
        "windows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("negatives", type=Path)
    parser.add_argument("stage_audit", type=Path)
    parser.add_argument("--window-seconds", type=float, nargs="+", default=[0.6, 0.9])
    parser.add_argument("--hop-seconds", type=float, default=0.3)
    parser.add_argument("--minimum-speech-fraction", type=float, default=0.8)
    parser.add_argument("--channel", choices=("stem", "raw"), default="stem")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        args.work_dir, args.manifest, args.negatives, args.stage_audit,
        lengths=tuple(args.window_seconds), hop=args.hop_seconds,
        minimum_speech_fraction=args.minimum_speech_fraction,
        channel=args.channel,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SHORT_SPAN_PROBE={args.output} WINDOWS={report['window_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
