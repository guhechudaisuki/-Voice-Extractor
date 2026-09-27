"""Replay the production dual-channel speaker verifier on cached speech islands.

This only records model evidence. It neither changes outputs nor treats short
island scores as ground truth; use it to test whether a proposed final purity
rule would also reject previously reviewed target speech.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_internal_island_identity import speech_islands  # noqa: E402
from extractor.audio import load_mono  # noqa: E402
from extractor.pipeline import ExtractionPipeline, PipelineOptions  # noqa: E402
from extractor.speaker import DualSpeakerVerifier  # noqa: E402
from extractor.types import TimeSpan  # noqa: E402


def _groups(root: Path) -> list[list[Path]]:
    return [sorted(folder.glob("*.wav")) for folder in sorted(root.glob("role_*"))]


def run(work: Path, manifest: dict, pauses: dict, starts: list[float] | None) -> dict:
    options = PipelineOptions(**manifest["options"])
    verifier = DualSpeakerVerifier("cuda")
    try:
        profile = verifier.build_channel_profile(
            sorted((work / "reference_voice_clips").glob("*.wav")),
            sorted((work / "reference_original_voice_clips").glob("*.wav")),
            options.speaker_threshold,
        )
        exclusions = verifier.build_channel_exclusion_profiles(
            _groups(work / "negative_reference_voice_clips"), profile,
            raw_reference_groups=_groups(work / "negative_reference_original_voice_clips"),
        )
        threshold = max(options.speaker_threshold, profile.primary.suggested_threshold)
        stem_file = work / "stems" / "target_vocals.wav"
        raw_file = work / "target_normalized.wav"
        # Match the production decoder/resampler exactly. A per-span librosa
        # resample changes short-window embeddings enough to invalidate the
        # comparison with the saved production manifest.
        stem_waveform = load_mono(stem_file, 16000)
        raw_waveform = load_mono(raw_file, 16000)
        rows = []
        accepted = [row for row in manifest["sentences"] if row["accepted"]]
        for candidate in accepted:
            start, end = float(candidate["start"]), float(candidate["end"])
            if starts is not None and not any(
                abs(start - selected) < 0.05 for selected in starts
            ):
                continue
            pause_row = next(
                (row for row in pauses["rows"]
                 if abs(float(row["accepted_span"][0]) - start) < 0.05
                 and abs(float(row["accepted_span"][1]) - end) < 0.05),
                None,
            )
            parts = [((start, end), "whole")]
            if pause_row is not None:
                parts.extend((span, "island") for span in speech_islands(
                    start, end, pause_row["gaps"]
                ))
            for (left, right), kind in parts:
                span = TimeSpan(left, right)
                stem = ExtractionPipeline._waveform_span(stem_waveform, span)
                raw = ExtractionPipeline._waveform_span(raw_waveform, span)
                duration = right - left
                window = min(1.8, max(1.0, duration))
                hop = min(0.9, max(0.5, duration / 2))
                match = verifier.verify_dual_channel_waveform(
                    stem, raw, profile, threshold,
                    duration=duration,
                    window_seconds=window,
                    hop_seconds=hop,
                    clean_gate=True,
                    allow_raw_rescue=options.use_raw_speaker_rescue,
                    audit_raw_on_uvr_accept=True,
                    audit_raw_channel=True,
                )
                exclusion = verifier.exclusion_audit(match, profile, exclusions)
                rows.append({
                    "parent_span": [start, end],
                    "kind": kind,
                    "span": [left, right],
                    "accepted": match.accepted,
                    "tier": match.tier,
                    "stem_primary": round(match.primary.score, 5),
                    "stem_secondary": round(match.secondary.score, 5) if match.secondary else None,
                    "raw_primary": round(match.raw_primary.score, 5) if match.raw_primary else None,
                    "raw_secondary": round(match.raw_secondary.score, 5) if match.raw_secondary else None,
                    "exclusion": exclusion,
                })
        return {"warning": "Diagnostic model evidence only, not export permission.", "rows": rows}
    finally:
        verifier.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("pauses", type=Path)
    parser.add_argument("--starts", type=float, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.work, json.loads(args.manifest.read_text(encoding="utf-8")),
                 json.loads(args.pauses.read_text(encoding="utf-8")), args.starts)
    contents = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(contents, encoding="utf-8")
        print(f"Audited {len(report['rows'])} spans: {args.output}")
    else:
        print(contents)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
