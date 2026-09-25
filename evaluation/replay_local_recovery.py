"""Replay one cached acoustic recovery turn; no subtitles, labels, or ASR cuts.

Reports proposals only. It never exports accepted audio or infers human truth.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extractor.audio import load_mono
from extractor.pipeline import ExtractionPipeline, PipelineOptions
from extractor.speaker import DualSpeakerVerifier, SpeakerBoundary
from extractor.types import TimeSpan
from evaluation.probe_adjacent_pairs import clips


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--containing", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    evidence = next(row for row in manifest["sentences"]
                    if row["start"] <= args.containing < row["end"]
                    and row["diagnostics"].get("local_boundary_recovery")
                    and row["diagnostics"].get("original_turn_start") is not None)
    info = evidence["diagnostics"]
    span = TimeSpan(info["original_turn_start"], info["original_turn_end"])
    # Cached cuts were selected by the production acoustic scanner. Confidence
    # is a replay sentinel for its already-passed cutoff, not a fresh score.
    cuts = [SpeakerBoundary(time=t, primary_similarity=0, secondary_similarity=None, confidence=1)
            for t in info["recovery_boundaries"]]
    work = args.work_dir
    pipeline = ExtractionPipeline(PipelineOptions(**manifest["options"]))
    stage = json.loads((args.manifest.parent / "stage_audit.json").read_text(encoding="utf-8"))
    pipeline._raw_blocked_spans = tuple(
        TimeSpan(*pair) for name in ("singing_blocked_islands", "overlap_blocked_islands")
        for pair in stage["stages"].get(name, {}).get("spans", [])
    )
    stem = load_mono(work / "stems/target_vocals.wav", 16000)
    pipeline._raw_target_waveform = load_mono(work / "target_normalized.wav", 16000)
    verifier = DualSpeakerVerifier(pipeline.device)
    try:
        profile = verifier.build_channel_profile(clips(work / "reference_voice_clips"),
            clips(work / "reference_original_voice_clips"), pipeline.options.speaker_threshold)
        groups = sorted((work / "negative_reference_voice_clips").glob("role_*"))
        exclusions = verifier.build_channel_exclusion_profiles(
            [clips(group) for group in groups], profile,
            raw_reference_groups=[clips(work / "negative_reference_original_voice_clips" / group.name)
                                  for group in groups]) if groups else []
        result = pipeline._recover_target_segments(span, cuts, verifier, stem, profile,
            exclusions, max(pipeline.options.speaker_threshold, profile.primary.suggested_threshold),
            lambda _v, message: print(message, flush=True), 1, 1)
        if result is None:
            raise ValueError("Cached cuts did not produce recovery parts")
        accepted, rejected = result
        report = {"warning": "Local proposals only; not full-pipeline or human verification.",
                  "source_turn": [span.start, span.end], "cached_cuts": info["recovery_boundaries"],
                  "accepted_proposals": [candidate.to_dict() for candidate, _match in accepted],
                  "rejected_proposals": [candidate.to_dict() for candidate in rejected]}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("LOCAL_RECOVERY", [(c.start, c.end) for c, _m in accepted], flush=True)
    finally:
        verifier.close()


if __name__ == "__main__":
    main()
