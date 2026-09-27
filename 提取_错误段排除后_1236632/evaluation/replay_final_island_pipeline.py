"""Run the actual experimental final-island pipeline method on cached exports.

No UVR, singing detector, whole episode transcription, or WAV export is rerun.
The replay isolates the final identity decision and writes a manifest-like
report suitable for the fast user-review regression.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extractor.audio import load_mono  # noqa: E402
from extractor.pipeline import ExtractionPipeline, PipelineOptions  # noqa: E402
from extractor.speaker import DualSpeakerVerifier  # noqa: E402
from extractor.types import CandidateSentence  # noqa: E402


def _groups(root: Path) -> list[list[Path]]:
    return [sorted(folder.glob("*.wav")) for folder in sorted(root.glob("role_*"))]


def run(work: Path, manifest: dict,
        starts: list[float] | None = None) -> dict:
    options = PipelineOptions(**manifest["options"])
    pipeline = ExtractionPipeline(options)
    stem = load_mono(work / "stems" / "target_vocals.wav", 16000)
    pipeline._raw_target_waveform = load_mono(work / "target_normalized.wav", 16000)
    references = sorted((work / "reference_voice_clips").glob("*.wav"))
    raw_references = sorted((work / "reference_original_voice_clips").glob("*.wav"))
    negatives = _groups(work / "negative_reference_voice_clips")
    raw_negatives = _groups(work / "negative_reference_original_voice_clips")
    accepted = [
        CandidateSentence(
            float(row["start"]), float(row["end"]), "",
            text=row.get("text", ""), diagnostics=dict(row.get("diagnostics") or {}),
        )
        for row in manifest["sentences"]
        if row["accepted"] and (
            starts is None or any(
                abs(float(row["start"]) - selected) < 0.05
                for selected in starts
            )
        )
    ]
    rejected: list[CandidateSentence] = []
    verifier = DualSpeakerVerifier("cuda")
    try:
        profile = verifier.build_channel_profile(
            references, raw_references, options.speaker_threshold,
        )
        exclusions = verifier.build_channel_exclusion_profiles(
            negatives, profile, raw_reference_groups=raw_negatives,
        )
        threshold = max(options.speaker_threshold, profile.primary.suggested_threshold)
        withheld, rescued = pipeline._experimental_audit_final_islands(
            accepted, rejected, verifier, profile, stem,
            work / "stems" / "target_vocals.wav",
            work / "target_normalized.wav", threshold,
            references, raw_references, negatives, raw_negatives, exclusions,
            [],
            progress=lambda _value, message: print(message, flush=True),
        )
    finally:
        verifier.close()
        pipeline._raw_target_waveform = None
    return {
        "warning": "Final identity-method replay only; no STT, song/overlap masks or audio delivery verification.",
        "withheld_parents": withheld,
        "rescued_children": rescued,
        "accepted_count": len(accepted),
        "sentences": [
            {"start": item.start, "end": item.end, "accepted": True,
             "diagnostics": item.diagnostics}
            for item in accepted
        ],
        "rejected": [
            {"start": item.start, "end": item.end,
             "reason": item.reject_reason,
             "diagnostics": item.diagnostics}
            for item in rejected
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--starts", type=float, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.work, json.loads(args.manifest.read_text(encoding="utf-8")),
                 args.starts)
    contents = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(contents, encoding="utf-8")
        print(f"WITHHELD={report['withheld_parents']} RESCUED={report['rescued_children']} "
              f"ACCEPTED={report['accepted_count']} OUTPUT={args.output}")
    else:
        print(contents)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
