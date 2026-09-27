"""Propose clean sub-sentences around independently suspected voice intrusions.

Research only. This uses cached audio and exact production speaker decoding, but
does not certify sentence completeness, run STT, or alter the desktop pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extractor.nextgen.purity_consensus import target_runs  # noqa: E402


def run(work: Path, manifest: dict, domain_islands: dict,
        consensus: dict) -> dict:
    # Imports stay local so the pure geometric regression test needs no CUDA.
    from check_frozen_join_audio import build_bank
    from extractor.audio import load_mono
    from extractor.pipeline import ExtractionPipeline, PipelineOptions
    from extractor.speaker import DualSpeakerVerifier
    from extractor.types import TimeSpan

    options = PipelineOptions(**manifest["options"])
    bank = build_bank(work)
    verifier = DualSpeakerVerifier("cuda")
    try:
        profile = verifier.build_channel_profile(
            sorted((work / "reference_voice_clips").glob("*.wav")),
            sorted((work / "reference_original_voice_clips").glob("*.wav")),
            options.speaker_threshold,
        )
        negative_root = work / "negative_reference_voice_clips"
        negatives = [sorted(folder.glob("*.wav"))
                     for folder in sorted(negative_root.glob("role_*"))]
        raw_negative_root = work / "negative_reference_original_voice_clips"
        raw_negatives = [sorted(folder.glob("*.wav"))
                         for folder in sorted(raw_negative_root.glob("role_*"))]
        exclusions = verifier.build_channel_exclusion_profiles(
            negatives, profile, raw_reference_groups=raw_negatives,
        )
        stem_file = work / "stems" / "target_vocals.wav"
        raw_file = work / "target_normalized.wav"
        stem = load_mono(stem_file, 16000)
        raw = load_mono(raw_file, 16000)
        threshold = max(options.speaker_threshold, profile.primary.suggested_threshold)
        rows = []
        for finding in consensus["findings"]:
            if not finding["abstain"]:
                continue
            parent = finding["span"]
            domain = next(
                (row for row in domain_islands["rows"]
                 if abs(float(row["accepted_span"][0]) - parent[0]) < 0.05
                 and abs(float(row["accepted_span"][1]) - parent[1]) < 0.05),
                None,
            )
            suspect = {
                tuple(map(float, reason["span"]))
                for reason in finding["reasons"]
                if reason["kind"] == "local_other_voice_consensus"
            }
            proposals = target_runs(
                domain["speech_islands"] if domain else [], suspect,
                maximum_gap_seconds=options.silence_split_seconds,
            )
            children = []
            for left, right in proposals:
                child = {"span": [left, right], "duration": round(right - left, 5)}
                if right - left < options.min_output_seconds:
                    child["state"] = "too_short_for_current_export"
                    children.append(child)
                    continue
                span = TimeSpan(left, right)
                stem_part = ExtractionPipeline._waveform_span(stem, span)
                raw_part = ExtractionPipeline._waveform_span(raw, span)
                window = min(1.8, max(1.0, span.duration))
                hop = min(0.9, max(0.5, span.duration / 2))
                match = verifier.verify_dual_channel_waveform(
                    stem_part, raw_part, profile, threshold,
                    duration=span.duration, window_seconds=window,
                    hop_seconds=hop, clean_gate=True,
                    allow_raw_rescue=options.use_raw_speaker_rescue,
                    audit_raw_on_uvr_accept=True,
                    audit_raw_channel=True,
                )
                exclusion = verifier.exclusion_audit(match, profile, exclusions)
                domain_match = bank.score(
                    stem=stem_part,
                    raw=raw_part,
                    sample_rate=16000,
                )
                child.update({
                    "production_accepted": match.accepted,
                    "production_tier": match.tier,
                    "excluded": bool(exclusion and exclusion["excluded_role_rejected"]),
                    "domain_state": domain_match.state,
                    "state": (
                        "identity_supported_unverified_boundary"
                        if match.accepted
                        and not (exclusion and exclusion["excluded_role_rejected"])
                        and domain_match.state == "target_supported"
                        else "withhold_identity_unresolved"
                    ),
                })
                children.append(child)
            rows.append({"index": finding["index"], "parent_span": parent,
                         "suspect_islands": [list(span) for span in sorted(suspect)],
                         "children": children})
        return {
            "warning": "Development proposals only. Identity support is not sentence-completeness or clean-audio proof.",
            "parents": len(rows),
            "supported_children": sum(
                child["state"] == "identity_supported_unverified_boundary"
                for row in rows for child in row["children"]
            ),
            "rows": rows,
        }
    finally:
        verifier.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("domain_islands", type=Path)
    parser.add_argument("consensus", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(
        args.work,
        json.loads(args.manifest.read_text(encoding="utf-8")),
        json.loads(args.domain_islands.read_text(encoding="utf-8")),
        json.loads(args.consensus.read_text(encoding="utf-8")),
    )
    contents = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(contents, encoding="utf-8")
        print(f"Proposed {report['supported_children']} children from {report['parents']} parents: {args.output}")
    else:
        print(contents)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
