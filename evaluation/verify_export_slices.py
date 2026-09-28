"""Verify manifest bounds, WAV content and the absence of stale audio exports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


def verify_exports(manifest_path: Path, stem: Path) -> dict:
    manifest_path, stem = Path(manifest_path), Path(stem)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("target_count", 1) != 1:
        raise ValueError("One vocal stem can verify only one target")
    root = manifest_path.parent.resolve()
    rows = []
    declared = set()
    with sf.SoundFile(str(stem)) as source:
        for ordinal, row in enumerate(
            (r for r in manifest["sentences"] if r["accepted"]), 1,
        ):
            audio = (root / row["audio_file"].replace("\\", "/")).resolve()
            if not audio.is_relative_to(root):
                raise ValueError("Audio path escapes the manifest directory")
            declared.add(audio)
            errors = []
            start, end = float(row["start"]), float(row["end"])
            if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end:
                raise ValueError("Invalid export time span")
            begin = round(start * source.samplerate)
            finish = round(end * source.samplerate)
            if finish > len(source):
                raise ValueError("Export span exceeds the vocal stem")
            source.seek(begin)
            expected = source.read(finish - begin, dtype="float32", always_2d=True).mean(axis=1)
            if source.samplerate != 16000:
                expected = torchaudio.functional.resample(
                    torch.from_numpy(expected), source.samplerate, 16000,
                ).numpy()
            actual, rate = sf.read(str(audio), dtype="float32", always_2d=True)
            actual = actual.mean(axis=1)
            if rate != 16000:
                errors.append("sample_rate")
            if abs(len(actual) - len(expected)) > 1:
                errors.append("sample_count")
            count = min(len(actual), len(expected))
            left = np.asarray(actual[:count], dtype=np.float64)
            right = np.asarray(expected[:count], dtype=np.float64)
            correlation = None
            max_residual = None
            if count and np.isfinite(left).all() and np.isfinite(right).all():
                left -= left.mean()
                right -= right.mean()
                denominator = np.linalg.norm(left) * np.linalg.norm(right)
                if denominator > 1e-10:
                    correlation = float(np.clip((left @ right) / denominator, -1, 1))
                    gain = float((left @ right) / (right @ right))
                    max_residual = float(np.max(np.abs(left - gain * right)))
            if correlation is None or correlation < 0.995:
                errors.append("content")
            # A global score can hide wrong quiet syllables behind a loud
            # matching prefix. Constant gain is the only export transform;
            # residuals must stay within a few PCM16 quantization steps.
            if max_residual is None or max_residual > 3.0 / 32768:
                errors.append("local_content")
            rows.append({
                "ordinal": ordinal, "span": [start, end],
                "audio_file": row["audio_file"], "samples": len(actual),
                "expected_samples": len(expected), "sample_rate": rate,
                "correlation": round(correlation, 8) if correlation is not None else None,
                "max_gain_adjusted_residual": max_residual,
                "errors": errors,
            })
    actual_files = {
        p.resolve() for p in root.rglob("*.wav")
        if "rejected_audio" not in p.relative_to(root).parts
    }
    unlisted = sorted(str(p.relative_to(root)) for p in actual_files - declared)
    return {
        "passed": bool(rows) and not unlisted and all(not row["errors"] for row in rows),
        "verified_count": len(rows),
        "failed_count": sum(bool(row["errors"]) for row in rows),
        "unlisted_audio": unlisted,
        "stem": str(stem.resolve()), "clips": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--stem", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = verify_exports(args.manifest, args.stem)
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "clips"}, ensure_ascii=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
