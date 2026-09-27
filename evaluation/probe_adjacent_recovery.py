"""Audit adjacent physical speech islands with frozen local evidence.

This proposes no output. It reads one cached target and its stage audit,
avoiding singing/UVR/STT reruns. Both rejected sides remain discoverable;
the old accepted set is annotation, never a prerequisite for recall.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_frozen_join_audio import build_bank, read_audio
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extractor.pipeline import PipelineOptions  # noqa: E402
from extractor.speaker import DualSpeakerVerifier  # noqa: E402
from extractor.nextgen.domain_identity import DomainVerdict  # noqa: E402
from extractor.types import TimeSpan  # noqa: E402


def _groups(root: Path) -> list[list[Path]]:
    return [sorted(folder.glob("*.wav")) for folder in sorted(root.glob("role_*"))]


def candidate_pairs(stage: dict, *, gap_min: float,
                    gap_max: float) -> list[tuple[TimeSpan, TimeSpan]]:
    """Consider only consecutive, clean islands; never skip an intervening voice."""
    if not 0 <= gap_min <= gap_max:
        raise ValueError("Invalid adjacent-island gap range")
    stages = stage["stages"]
    islands = [TimeSpan(*row) for row in stages["clean_speech_islands"]["spans"]]
    if islands != sorted(islands, key=lambda span: (span.start, span.end)):
        raise ValueError("Clean speech islands must be sorted")
    if any(left.end > right.start for left, right in zip(islands, islands[1:])):
        raise ValueError("Clean speech islands overlap")
    blocked = [
        TimeSpan(*row)
        for name in ("pre_uvr_singing_mask", "residual_singing_evidence",
                     "singing_blocked_islands", "overlap_evidence",
                     "overlap_blocked_islands")
        for row in stages.get(name, {}).get("spans", [])
    ]
    return [
        (left, right)
        for left, right in zip(islands, islands[1:])
        if gap_min <= right.start - left.end <= gap_max
        and not any(
            min(right.end, mask.end) - max(left.start, mask.start) > 0.01
            for mask in blocked
        )
    ]


def parse_window(value: str) -> TimeSpan:
    try:
        start, end = (float(piece) for piece in value.split(":"))
        if not 0 <= start < end:
            raise ValueError("Invalid window")
        return TimeSpan(start, end)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("Window must use START:END seconds") from error


def local_windows(span: TimeSpan, *, sample_rate: int = 16000,
                  widths: tuple[float, ...] = (0.25, 0.60)) -> list[TimeSpan]:
    """Cover a proposed output with short, endpoint-anchored identity queries.

    Whole-utterance embeddings can hide a 0.x-second foreign voice. These
    windows are acoustic *veto probes*, not independently complete sentences.
    """
    if sample_rate <= 0 or not widths or any(width < 0.20 for width in widths):
        raise ValueError("Invalid local acoustic scan")
    start, end = round(span.start * sample_rate), round(span.end * sample_rate)
    seen: set[tuple[int, int]] = set()
    for width_seconds in widths:
        width = round(width_seconds * sample_rate)
        if end - start < width:
            continue
        hop = max(1, width // 2)
        starts = list(range(start, end - width + 1, hop))
        starts.append(end - width)
        seen.update((left, left + width) for left in starts)
    return [TimeSpan(left / sample_rate, right / sample_rate)
            for left, right in sorted(seen)]


def run(work: Path, manifest_path: Path, stage_path: Path, *, gap_min: float = 0.2,
        gap_max: float = 0.85, windows: tuple[TimeSpan, ...] = ()) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    pairs = candidate_pairs(stage, gap_min=gap_min, gap_max=gap_max)
    if windows:
        pairs = [
            (left, right) for left, right in pairs
            if any(window.start <= left.start and right.end <= window.end
                   for window in windows)
        ]
    bank = build_bank(work)
    stem_file = work / "stems" / "target_vocals.wav"
    raw_file = work / "target_normalized.wav"
    options = PipelineOptions(**manifest["options"])
    score_cache = {}
    issues = {}
    def score(span: TimeSpan):
        key = (span.start, span.end)
        if key not in score_cache:
            try:
                score_cache[key] = bank.score(
                    stem=read_audio(stem_file, *key),
                    raw=read_audio(raw_file, *key),
                    sample_rate=16000,
                )
            except ValueError as error:
                message = str(error)
                if message not in (
                    "Silent audio cannot provide speaker identity evidence",
                    "Expected at least 0.2 s of finite mono audio at 16 kHz",
                ):
                    raise
                # An inaudible or invalid physical island supplies no identity
                # evidence. It is never evidence for either target or other.
                score_cache[key] = DomainVerdict("unresolved", ())
                issues[key] = message
        return score_cache[key]
    old_accepted = [row for row in manifest["sentences"] if row["accepted"]]
    def previously_covered(span: TimeSpan) -> bool:
        return any(
            float(row["start"]) <= span.start + 0.05
            and float(row["end"]) >= span.end - 0.05
            for row in old_accepted
        )
    rows = []
    for left, right in pairs:
        left_result = score(left)
        right_result = score(right)
        rows.append({
            "left": [left.start, left.end],
            "right": [right.start, right.end],
            "left_old_accepted": previously_covered(left),
            "right_old_accepted": previously_covered(right),
            "left_state": left_result.state,
            "right_state": right_result.state,
            "left_acoustic_issue": issues.get((left.start, left.end)),
            "right_acoustic_issue": issues.get((right.start, right.end)),
            "both_target_supported": (
                left_result.state == "target_supported"
                and right_result.state == "target_supported"
            ),
        })
    supported = [row for row in rows if row["both_target_supported"]]
    if supported:
        verifier = DualSpeakerVerifier("cuda")
        profile = verifier.build_channel_profile(
            sorted((work / "reference_voice_clips").glob("*.wav")),
            sorted((work / "reference_original_voice_clips").glob("*.wav")),
            options.speaker_threshold,
        )
        exclusions = verifier.build_channel_exclusion_profiles(
            _groups(work / "negative_reference_voice_clips"),
            profile,
            raw_reference_groups=_groups(work / "negative_reference_original_voice_clips"),
        )
        threshold = max(options.speaker_threshold, profile.primary.suggested_threshold)
        for row in supported:
            start, end = row["left"][0], row["right"][1]
            duration = end - start
            match = verifier.verify_dual_channel_waveform(
                read_audio(stem_file, start, end),
                read_audio(raw_file, start, end),
                profile, threshold, duration,
                window_seconds=min(1.8, max(1.0, duration)),
                hop_seconds=min(0.9, max(0.5, duration / 2)),
                clean_gate=True,
                audit_raw_channel=True,
            )
            exclusion = verifier.exclusion_audit(match, profile, exclusions)
            row["whole_verified"] = bool(match.accepted)
            row["whole_tier"] = match.tier
            row["excluded"] = bool(exclusion and exclusion.get("excluded_role_rejected"))
            local_other = []
            local_unresolved = []
            if match.accepted and not row["excluded"]:
                for window in local_windows(TimeSpan(start, end)):
                    state = score(window).state
                    if state == "other_supported":
                        local_other.append([window.start, window.end])
                    elif state == "unresolved":
                        local_unresolved.append([window.start, window.end])
            row["local_other_windows"] = local_other
            row["local_unresolved_windows"] = local_unresolved
            # Subsecond anime embeddings also flag known true-target speech
            # as "other". Keep this contradiction visible for review, never
            # silently delete a recall proposal solely on that observation.
            row["local_purity_unresolved"] = bool(local_other or local_unresolved)
            row["whole_still_possible"] = bool(
                match.accepted and not row["excluded"]
            )
    return {
        "warning": "Development probe only. Local target support and a whole-span score do not prove speaker purity or complete boundaries. No clip is approved for export; change, overlap, singing and exact-edge checks remain.",
        "source": "consecutive_clean_speech_islands",
        "review_windows": [[window.start, window.end] for window in windows],
        "pair_count": len(rows),
        "both_target_supported_count": sum(row["both_target_supported"] for row in rows),
        "whole_still_possible_count": sum(row.get("whole_still_possible", False) for row in rows),
        "pairs": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("stage_audit", type=Path)
    parser.add_argument("--window", type=parse_window, action="append", default=[],
                        help="Analyze only pairs wholly within START:END seconds; repeatable")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.work, args.manifest, args.stage_audit,
                 windows=tuple(args.window))
    contents = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(contents, encoding="utf-8")
    else:
        print(contents)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
