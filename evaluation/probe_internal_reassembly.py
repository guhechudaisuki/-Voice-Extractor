"""Audit short speech tails after VAD-missed pauses without changing exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_frozen_join_audio import build_bank, read_audio  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extractor.speaker import DualSpeakerVerifier  # noqa: E402


def _groups(root: Path) -> list[list[Path]]:
    return [sorted(folder.glob("*.wav")) for folder in sorted(root.glob("role_*"))]


def run(work: Path, manifest_path: Path, pauses_path: Path,
        *, maximum_tail_seconds: float = 0.8) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pauses = json.loads(pauses_path.read_text(encoding="utf-8"))
    with sf.SoundFile(work / "stems" / "target_vocals.wav") as source:
        source_duration = len(source) / source.samplerate
    bank = build_bank(work)
    stem_file = work / "stems" / "target_vocals.wav"
    raw_file = work / "target_normalized.wav"
    score_cache = {}
    def score(start: float, end: float):
        key = (round(start, 5), round(end, 5))
        if key not in score_cache:
            try:
                score_cache[key] = bank.score(
                    stem=read_audio(stem_file, start, end),
                    raw=read_audio(raw_file, start, end),
                    sample_rate=16000,
                )
            except ValueError:
                score_cache[key] = None
        return score_cache[key]
    cases = []
    for row in pauses["rows"]:
        accepted_start, accepted_end = map(float, row["accepted_span"])
        for gap in row["gaps"]:
            if float(gap["right_remainder_seconds"]) > maximum_tail_seconds:
                continue
            gap_start, gap_end = map(float, gap["span"])
            # Analysis may borrow at most 120 ms beyond the old output, never
            # extending the exported candidate or its claimed identity.
            tail_end = min(source_duration, accepted_end + 0.12,
                           max(accepted_end, gap_end + 0.28))
            left = score(accepted_start, gap_start)
            right = score(gap_end, tail_end)
            case = {
                "accepted_span": [accepted_start, accepted_end],
                "quiet_gap": [gap_start, gap_end],
                "tail_analysis_span": [gap_end, round(tail_end, 5)],
                "left_state": left.state if left else "unresolved",
                "tail_state": right.state if right else "unresolved",
                "possible_other_tail": bool(
                    left and right and left.state == "target_supported"
                    and right.state == "other_supported"
                ),
            }
            cases.append(case)
    recoverable = [row for row in cases if row["possible_other_tail"]]
    if recoverable:
        verifier = DualSpeakerVerifier("cuda")
        profile = verifier.build_channel_profile(
            sorted((work / "reference_voice_clips").glob("*.wav")),
            sorted((work / "reference_original_voice_clips").glob("*.wav")),
            float(manifest["options"]["speaker_threshold"]),
        )
        exclusions = verifier.build_channel_exclusion_profiles(
            _groups(work / "negative_reference_voice_clips"), profile,
            raw_reference_groups=_groups(work / "negative_reference_original_voice_clips"),
        )
        threshold = max(float(manifest["options"]["speaker_threshold"]),
                        profile.primary.suggested_threshold)
        for case in recoverable:
            a, _ = case["accepted_span"]
            gap_start = case["quiet_gap"][0]
            predecessors = [
                row for row in manifest["sentences"]
                if not row["accepted"] and row["reject_reason"] == "声纹匹配不足"
                and 0.2 <= a - float(row["end"]) <= 0.85
                and float(row["start"]) < a
            ]
            choices = [(a, "cropped_core")]
            choices.extend((float(row["start"]), "joined_predecessor")
                           for row in predecessors
                           if score(float(row["start"]), float(row["end"])) is not None
                           and score(float(row["start"]), float(row["end"])).state
                           == "target_supported")
            verified = []
            for start, source in choices:
                duration = gap_start - start
                match = verifier.verify_dual_channel_waveform(
                    read_audio(stem_file, start, gap_start),
                    read_audio(raw_file, start, gap_start),
                    profile, threshold, duration,
                    window_seconds=min(1.8, max(1.0, duration)),
                    hop_seconds=min(0.9, max(0.5, duration / 2)),
                    clean_gate=True, audit_raw_channel=True,
                )
                exclusion = verifier.exclusion_audit(match, profile, exclusions)
                verified.append({
                    "span": [start, gap_start], "source": source,
                    "whole_accepted": bool(match.accepted),
                    "excluded": bool(exclusion and exclusion.get("excluded_role_rejected")),
                    "tier": match.tier,
                })
            case["candidate_reassemblies"] = verified
    return {
        "warning": "Research only; local model labels and whole-span acceptance do not prove complete clean audio. No exports changed.",
        "short_tail_cases": len(cases),
        "possible_other_tails": len(recoverable),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("pauses", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.work, args.manifest, args.pauses)
    contents = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(contents, encoding="utf-8")
    else:
        print(contents)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
