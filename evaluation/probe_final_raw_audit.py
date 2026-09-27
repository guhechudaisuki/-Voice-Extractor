"""Recheck cached short-scene decisions with original-channel exclusion evidence.

This is an offline diagnostic. It never changes exported audio or STT files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extractor.audio import load_mono  # noqa: E402
from extractor.pipeline import ExtractionPipeline, PipelineOptions  # noqa: E402
from extractor.speaker import DualSpeakerVerifier  # noqa: E402
from extractor.types import TimeSpan  # noqa: E402


def _groups(root: Path) -> list[list[Path]]:
    return [sorted(folder.glob("*.wav")) for folder in sorted(root.glob("role_*"))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    work_dir = args.work_dir.resolve()
    manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
    options = PipelineOptions(**manifest["options"])
    stem = load_mono(work_dir / "stems" / "target_vocals.wav", 16000)
    raw = load_mono(work_dir / "target_normalized.wav", 16000)
    pipeline = ExtractionPipeline.__new__(ExtractionPipeline)
    pipeline.options = options
    pipeline._raw_target_waveform = raw
    pipeline._raw_blocked_spans = ()
    verifier = DualSpeakerVerifier("cuda")
    profile = verifier.build_channel_profile(
        sorted((work_dir / "reference_voice_clips").glob("*.wav")),
        sorted((work_dir / "reference_original_voice_clips").glob("*.wav")),
        options.speaker_threshold,
    )
    exclusions = verifier.build_channel_exclusion_profiles(
        _groups(work_dir / "negative_reference_voice_clips"),
        profile,
        raw_reference_groups=_groups(work_dir / "negative_reference_original_voice_clips"),
    )
    rows = []
    for candidate in manifest["sentences"]:
        old = candidate.get("diagnostics", {})
        if not candidate["accepted"] and not old.get("excluded_role_rejected"):
            continue
        span = TimeSpan(float(candidate["start"]), float(candidate["end"]))
        match = pipeline._verify_speaker_span(
            verifier, stem, span, profile,
            max(options.speaker_threshold, profile.primary.suggested_threshold),
            audit_raw=True,
        )
        exclusion = verifier.exclusion_audit(match, profile, exclusions)
        rows.append({
            "span": [span.start, span.end],
            "old_accepted": bool(candidate["accepted"]),
            "uvr_accepted": bool(match.diagnostics.get("uvr_accepted", match.accepted)),
            "raw_audit_available": bool(exclusion and exclusion.get("excluded_raw_available")),
            "raw_channel_skipped": bool(match.diagnostics.get("raw_channel_skipped")),
            "old_excluded": bool(old.get("excluded_role_rejected")),
            "new_excluded": bool(exclusion and exclusion.get("excluded_role_rejected")),
            "raw_negative_vote": bool(exclusion and exclusion.get("excluded_raw_vote")),
            "raw_primary_margin": exclusion.get("excluded_raw_primary_margin") if exclusion else None,
            "raw_secondary_margin": exclusion.get("excluded_raw_secondary_margin") if exclusion else None,
        })
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
