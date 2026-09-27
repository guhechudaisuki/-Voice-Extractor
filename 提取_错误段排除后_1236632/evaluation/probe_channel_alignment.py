"""Measure actual raw/UVR timing offset on speech-rich development intervals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, correlate, correlation_lags, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extractor.audio import load_mono  # noqa: E402


CASES = Path(__file__).with_name("join_cases.json")
SAMPLE_RATE = 16000


def measure(raw: np.ndarray, stem: np.ndarray, start: float, end: float) -> dict:
    begin = round(start * SAMPLE_RATE)
    finish = round(end * SAMPLE_RATE)
    x = raw[begin:finish].astype(np.float64, copy=False)
    y = stem[begin:finish].astype(np.float64, copy=False)
    if len(x) != len(y) or len(x) < SAMPLE_RATE // 2:
        raise ValueError("Raw and UVR intervals are not time aligned or are too short")
    sos = butter(3, 120, btype="highpass", fs=SAMPLE_RATE, output="sos")
    x = sosfiltfilt(sos, x)
    y = sosfiltfilt(sos, y)
    x -= x.mean()
    y -= y.mean()
    maximum_lag = round(0.25 * SAMPLE_RATE)
    values = correlate(x, y, mode="full", method="fft")
    lags = correlation_lags(len(x), len(y), mode="full")
    allowed = np.abs(lags) <= maximum_lag
    values, lags = values[allowed], lags[allowed]
    best = int(np.argmax(values))
    correlation = values[best] / max(1e-9, np.linalg.norm(x) * np.linalg.norm(y))
    return {
        "start": start,
        "end": end,
        "offset_samples_raw_relative_to_stem": int(lags[best]),
        "offset_seconds_raw_relative_to_stem": round(float(lags[best]) / SAMPLE_RATE, 6),
        "normalized_peak_correlation": round(float(correlation), 5),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    work = args.work_dir.resolve()
    raw = load_mono(work / "target_normalized.wav", SAMPLE_RATE).numpy()
    stem = load_mono(work / "stems/target_vocals.wav", SAMPLE_RATE).numpy()
    if abs(len(raw) - len(stem)) > SAMPLE_RATE // 10:
        raise ValueError("Raw and UVR full-duration mismatch exceeds 0.1 seconds")
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    rows = []
    for case in cases:
        for side in ("left", "right"):
            start, end = case[side]
            rows.append({"case": case["id"], "side": side, **measure(raw, stem, start, end)})
    offsets = [row["offset_seconds_raw_relative_to_stem"] for row in rows]
    report = {
        "warning": "Only sampled development intervals; not a proof of alignment over all source media.",
        "sample_rate": SAMPLE_RATE,
        "median_offset_seconds": float(np.median(offsets)),
        "maximum_absolute_offset_seconds": float(np.max(np.abs(offsets))),
        "intervals": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ALIGNMENT_PROBE={args.output} INTERVALS={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
