"""Replay selected cached acoustic recovery turns; no subtitles or ASR cuts.

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
from evaluation.replay_candidate_selection import candidate as load_candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--containing", type=float, required=True, action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-consensus", action="store_true",
                        help="Test existing prototype recovery on scored local rejects; no export")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    plans = {}
    for point in args.containing:
        evidence = next((row for row in manifest["sentences"]
                         if row["diagnostics"].get("local_boundary_recovery")
                         and row["diagnostics"].get("original_turn_start") is not None
                         and row["diagnostics"]["original_turn_start"] <= point
                         < row["diagnostics"]["original_turn_end"]), None)
        if evidence is None:
            parser.error(f"No cached local recovery parent contains {point}")
        info = evidence["diagnostics"]
        key = (info["original_turn_start"], info["original_turn_end"])
        plans[key] = info["recovery_boundaries"]
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
        reports = []
        scored_parts = []
        rejected_parts = []
        for index, (key, times) in enumerate(plans.items(), start=1):
            span = TimeSpan(*key)
            # Already-selected acoustic cuts, not new boundary measurements.
            cuts = [SpeakerBoundary(time=t, primary_similarity=0,
                    secondary_similarity=None, confidence=1) for t in times]
            result = pipeline._recover_target_segments(span, cuts, verifier, stem, profile,
                exclusions, max(pipeline.options.speaker_threshold, profile.primary.suggested_threshold),
                lambda _v, message: print(message, flush=True), index, len(plans),
                scored_parts=scored_parts)
            if result is None:
                raise ValueError("Cached cuts did not produce recovery parts")
            accepted, rejected = result
            rejected_parts.extend(rejected)
            reports.append({
                "warning": "Local proposals only; not full-pipeline or human verification.",
                "source_turn": list(key), "cached_cuts": times,
                "accepted_proposals": [candidate.to_dict() for candidate, _match in accepted],
                "rejected_proposals": [candidate.to_dict() for candidate in rejected],
            })
            output = reports[0] if len(plans) == 1 else {"reports": reports,
                     "completed": len(reports), "total": len(plans)}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
            print("LOCAL_RECOVERY", key, [(c.start, c.end) for c, _m in accepted], flush=True)
        if args.probe_consensus:
            # Never allow the query's own parent/overlapping output to become
            # a prototype that "independently" confirms its children.
            anchors = [load_candidate(row) for row in manifest["sentences"]
                       if row.get("accepted") and not any(
                           min(row["end"], end) > max(row["start"], start)
                           for start, end in plans)]
            original_ids = {id(row) for row in anchors}
            local_ids = {id(row) for row in rejected_parts}
            eligible = [part for _span, part, match in scored_parts
                        if id(part) in local_ids and match is not None
                        and part.reject_reason == "声纹匹配不足"
                        and max(1.80, pipeline.options.min_sentence_seconds)
                        <= part.duration <= min(15.0, pipeline.options.max_sentence_seconds)]
            print("CONSENSUS_INPUT", "anchors", len(anchors), "newly_reachable_local_parts",
                  [(part.start, part.end) for part in eligible], flush=True)
            count = pipeline._promote_multimodel_target_subclusters(
                scored_parts, anchors, rejected_parts, verifier, profile,
                work / "stems/target_vocals.wav", stem, exclusions,
                lambda _v, message: print(message, flush=True),
            )
            promoted = [row for row in anchors if id(row) not in original_ids]
            output = {"reports": reports, "completed": len(reports), "total": len(plans),
                      "consensus_probe": {
                          "warning": "Offline diagnostic; not final audio or human verification.",
                          "promoted_count": count,
                          "promoted_local_parts": [row.to_dict() for row in promoted],
                          "evaluated_local_parts": [row.to_dict() for row in eligible],
                          "anchors_are_disjoint_from_query_parents": True,
                      }}
            args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
            print("LOCAL_CONSENSUS", [(row.start, row.end) for row in promoted], flush=True)
    finally:
        verifier.close()


if __name__ == "__main__":
    main()
