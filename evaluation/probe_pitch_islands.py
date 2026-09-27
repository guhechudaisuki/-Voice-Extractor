"""Measure reference-relative pitch on quiet-gap speech islands for diagnosis.

Pitch is not speaker identity. This report cannot by itself accept or reject a
clip, especially when the island has too few reliably voiced frames.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import parselmouth

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_frozen_join_audio import read_audio  # noqa: E402
from probe_internal_island_identity import speech_islands  # noqa: E402


def pitch_stats(waveform: np.ndarray, sample_rate: int = 16000) -> dict:
    if sample_rate <= 0 or waveform.ndim != 1:
        raise ValueError("Expected mono audio and a positive sample rate")
    if waveform.size < round(0.12 * sample_rate):
        return {"voiced_frames": 0, "median_hz": None}
    pitch = parselmouth.Sound(
        np.asarray(waveform, dtype=np.float64),
        sampling_frequency=sample_rate,
    ).to_pitch(time_step=0.01, pitch_floor=65, pitch_ceiling=650)
    values = pitch.selected_array["frequency"]
    voiced = values[np.isfinite(values) & (values > 0)]
    return {
        "voiced_frames": int(voiced.size),
        "median_hz": round(float(np.median(voiced)), 2) if voiced.size else None,
    }


def run(work: Path, pauses: dict) -> dict:
    references = sorted((work / "reference_original_voice_clips").glob("*.wav"))
    reference_rows = [
        {"name": path.name, **pitch_stats(read_audio(path, 0.0, None).numpy())}
        for path in references
    ]
    target_medians = [
        row["median_hz"] for row in reference_rows
        if row["voiced_frames"] >= 5 and row["median_hz"] is not None
    ]
    reference_median = (
        round(float(np.median(target_medians)), 2) if target_medians else None
    )
    source = work / "target_normalized.wav"
    rows = []
    for row in pauses["rows"]:
        start, end = map(float, row["accepted_span"])
        parts = []
        for left, right in speech_islands(start, end, row["gaps"]):
            stats = pitch_stats(read_audio(source, left, right).numpy())
            if stats["median_hz"] is not None and reference_median is not None:
                stats["octaves_from_reference"] = round(
                    float(np.log2(stats["median_hz"] / reference_median)), 3,
                )
            parts.append({"span": [left, right], **stats})
        rows.append({"accepted_span": [start, end], "speech_islands": parts})
    return {
        "warning": "Research measurement only. Pitch can be undefined or octave-shifted and is not an identity verdict.",
        "reference_median_hz": reference_median,
        "references": reference_rows,
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
