"""Probe whether adjacent speech islands are recoverable without relaxing gates.

This is offline evidence only: a strong whole-span score can hide a speaker
change, and this script never writes accepted audio or changes pipeline policy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extractor.audio import load_mono  # noqa: E402
from extractor.pipeline import ExtractionPipeline, PipelineOptions  # noqa: E402
from extractor.speaker import DualSpeakerVerifier  # noqa: E402
from extractor.types import TimeSpan  # noqa: E402


def parse_pair(value: str) -> tuple[TimeSpan, TimeSpan]:
    try:
        left, right = value.split(",")
        a, b = (float(item) for item in left.split(":"))
        c, d = (float(item) for item in right.split(":"))
        if not 0 <= a < b <= c < d:
            raise ValueError("spans must be ordered and non-overlapping")
        return TimeSpan(a, b), TimeSpan(c, d)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def clips(path: Path) -> list[Path]:
    result = sorted(path.glob("*.wav"))
    if not result:
        raise FileNotFoundError(f"No reference clips in {path}")
    return result


def score(match, exclusion) -> dict:
    return {
        "accepted_by_whole_span_verifier": match.accepted,
        "tier": match.tier,
        "primary": round(match.primary.score, 5),
        "secondary": round(match.secondary.score, 5) if match.secondary else None,
        "paired_reference_median": round(match.paired_reference_median, 5),
        "primary_window_scores": [round(v, 5) for v in match.primary.window_scores],
        "secondary_window_scores": (
            [round(v, 5) for v in match.secondary.window_scores]
            if match.secondary else []
        ),
        "raw_rescue": bool(match.diagnostics.get("raw_rescue")),
        "excluded_role_rejected": bool(exclusion and exclusion.get("excluded_role_rejected")),
        "excluded_role": exclusion.get("excluded_role") if exclusion else None,
    }


def embedding_similarity(left, right, channel: str) -> float | None:
    a = getattr(left, channel)
    b = getattr(right, channel)
    if a is None or b is None or a.embedding is None or b.embedding is None:
        return None
    return round(float(a.embedding @ b.embedding), 5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("--pair", type=parse_pair, action="append")
    parser.add_argument("--cases", type=Path, help="Read named pairs from a frozen case file")
    parser.add_argument("--wavlm", action="store_true", help="Also inspect the local WavLM representation")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    named_pairs = []
    if args.cases:
        for case in json.loads(args.cases.read_text(encoding="utf-8"))["cases"]:
            a, b = case["left"], case["right"]
            left, right = parse_pair(f"{a[0]}:{a[1]},{b[0]}:{b[1]}")
            named_pairs.append((case["id"], left, right))
    named_pairs.extend((None, left, right) for left, right in (args.pair or []))
    if not named_pairs:
        parser.error("Provide --cases or at least one --pair")
    work = args.work_dir.resolve()
    pipeline = ExtractionPipeline(PipelineOptions())
    verifier = DualSpeakerVerifier(pipeline.device)
    try:
        profile = verifier.build_channel_profile(
            clips(work / "reference_voice_clips"),
            clips(work / "reference_original_voice_clips"),
            pipeline.options.speaker_threshold,
        )
        negative_root = work / "negative_reference_voice_clips"
        raw_negative_root = work / "negative_reference_original_voice_clips"
        groups = sorted(negative_root.glob("role_*"))
        exclusions = verifier.build_channel_exclusion_profiles(
            [clips(group) for group in groups],
            profile,
            raw_reference_groups=[clips(raw_negative_root / group.name) for group in groups],
        ) if groups else []
        stem = load_mono(work / "stems" / "target_vocals.wav", 16000)
        pipeline._raw_target_waveform = load_mono(work / "target_normalized.wav", 16000)
        threshold = max(pipeline.options.speaker_threshold, profile.primary.suggested_threshold)
        rows = []
        for case_id, left, right in named_pairs:
            whole = TimeSpan(left.start, right.end)
            matches = [
                pipeline._verify_speaker_span(verifier, stem, span, profile, threshold)
                for span in (left, right, whole)
            ]
            audits = [
                verifier.exclusion_audit(match, profile, exclusions) if exclusions else None
                for match in matches
            ]
            rows.append({
                "id": case_id,
                "left": [left.start, left.end],
                "right": [right.start, right.end],
                "gap_seconds": round(right.start - left.end, 5),
                "primary_pair_similarity": embedding_similarity(matches[0], matches[1], "primary"),
                "secondary_pair_similarity": embedding_similarity(matches[0], matches[1], "secondary"),
                "left_result": score(matches[0], audits[0]),
                "right_result": score(matches[1], audits[1]),
                "whole_result": score(matches[2], audits[2]),
            })
        if args.wavlm:
            tertiary, tertiary_profile = verifier._tertiary_pair(profile)
            spans = [
                span
                for _case_id, left, right in named_pairs
                for span in (left, right, TimeSpan(left.start, right.end))
            ]
            embeddings = tertiary._embeddings_from_waveforms(
                [pipeline._waveform_span(stem, span) for span in spans],
                batch_size=8,
            )
            for index, row in enumerate(rows):
                a, b, whole = embeddings[index * 3 : index * 3 + 3]
                row["wavlm"] = {
                    "pair_similarity": round(float(a @ b), 5),
                    "left_target": round(float(a @ tertiary_profile.centroid), 5),
                    "right_target": round(float(b @ tertiary_profile.centroid), 5),
                    "whole_target": round(float(whole @ tertiary_profile.centroid), 5),
                }
        report = {
            "warning": "Offline diagnostic only. Whole-span verification alone cannot rule out mixed speakers.",
            "threshold": threshold,
            "pairs": rows,
        }
        contents = json.dumps(report, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(contents, encoding="utf-8")
        else:
            print(contents)
        return 0
    finally:
        verifier.close()


if __name__ == "__main__":
    raise SystemExit(main())
