"""Replay small reviewed joins through frozen char/va ONNX local evidence.

Development regression only. Labels are read only after decisions; neither
this script nor its first-episode cases are a production calibration artifact.
It reuses cached UVR/raw files and never runs singing, UVR, STT or a full show.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import librosa
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extractor.nextgen.anime_embedding import AnimeSpeakerOnnx  # noqa: E402
from extractor.nextgen.domain_identity import (  # noqa: E402
    DomainReferenceBank, ModelViewScore, ReferenceAudio, classify, review_join,
)


def read_audio(path: Path, start: float | None = None,
               end: float | None = None) -> torch.Tensor:
    with sf.SoundFile(path) as stream:
        rate = stream.samplerate
        if start is not None:
            stream.seek(round(start * rate))
        count = -1 if end is None else round((end - (start or 0.0)) * rate)
        samples = stream.read(count, dtype="float32", always_2d=True).mean(axis=1)
    if rate != 16000:
        samples = librosa.resample(samples, orig_sr=rate, target_sr=16000)
    return torch.from_numpy(np.ascontiguousarray(samples))


def paired_clips(work: Path, stem_directory: str, raw_directory: str,
                 role: str) -> list[ReferenceAudio]:
    stem = work / stem_directory
    raw = work / raw_directory
    stem_files = {file.name: file for file in stem.glob("*.wav")}
    raw_files = {file.name: file for file in raw.glob("*.wav")}
    if not stem_files or stem_files.keys() != raw_files.keys():
        raise ValueError(f"Missing or mispaired reference files for {role}")
    return [ReferenceAudio(role, read_audio(stem_files[name]), read_audio(raw_files[name]), 16000)
            for name in sorted(stem_files)]


def build_bank(work: Path) -> DomainReferenceBank:
    references = paired_clips(work, "reference_voice_clips",
                              "reference_original_voice_clips", "target")
    negatives = work / "negative_reference_voice_clips"
    for group in sorted(negatives.glob("role_*")):
        if group.is_dir():
            references.extend(paired_clips(
                work, str(group.relative_to(work)),
                str(Path("negative_reference_original_voice_clips") / group.name),
                group.name,
            ))
    encoders = {variant: AnimeSpeakerOnnx(ROOT / "models", variant)
                for variant in ("char", "va")}
    return DomainReferenceBank(encoders, references)


def heldout_reference_stress(bank: DomainReferenceBank) -> dict:
    """Leave query audio out, including its whole exclusion role if non-target."""
    variants = ("char", "va")
    channels = ("stem", "raw")
    target_count = len(bank.vectors[("char", "stem", "target")])
    if target_count < 2:
        raise ValueError("Reference holdout requires at least two target clips")
    cases = [("target", index) for index in range(target_count)]
    cases.extend((role, index) for role in bank.roles
                 for index in range(len(bank.vectors[("char", "stem", role)])))
    rows = []
    for role, index in cases:
        model_scores = []
        for variant in variants:
            for channel in channels:
                vector = bank.vectors[(variant, channel, role)][index]
                target = bank.vectors[(variant, channel, "target")]
                if role == "target":
                    target = np.delete(target, index, axis=0)
                target_median = float(np.median(target @ vector))
                # A held-out exclusion role is entirely absent at inference.
                # Letting its remaining clips vote negative would leak the
                # answer and make this open-set check misleading.
                others = {
                    other: float((bank.vectors[(variant, channel, other)] @ vector).max())
                    for other in bank.roles if other != role
                }
                nearest = max(others, key=others.get) if others else None
                model_scores.append(ModelViewScore(
                    variant, channel, target_median, others.get(nearest), nearest,
                ))
        result = classify(tuple(model_scores))
        rows.append({"kind": "target" if role == "target" else "unseen_other",
                     "role": role, "index": index, "state": result.state})
    return {
        "warning": "Held-out reference clips only, not natural-dialogue speaker truth.",
        "target_supported": sum(row["kind"] == "target" and row["state"] == "target_supported"
                                for row in rows),
        "target_total": sum(row["kind"] == "target" for row in rows),
        "unseen_other_supported_as_target": sum(
            row["kind"] == "unseen_other" and row["state"] == "target_supported"
            for row in rows),
        "unseen_other_total": sum(row["kind"] == "unseen_other" for row in rows),
        "rows": rows,
    }


def view_margins(verdict) -> dict[str, float | None]:
    """Expose raw/stem disagreement without changing the join decision."""
    return {
        f"{row.variant}_{row.channel}": (
            round(row.margin, 6) if row.margin is not None else None
        )
        for row in verdict.scores
    }


def run(work: Path, evidence: Path, cases_path: Path) -> dict:
    bank = build_bank(work)
    reference_holdout = heldout_reference_stress(bank)
    raw_file = work / "target_normalized.wav"
    stem_file = work / "stems/target_vocals.wav"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))["cases"]
    whole = json.loads((evidence / "join_probe_corrected.json").read_text(encoding="utf-8"))
    stem_change = json.loads((evidence / "join_boundary_stem_probe.json").read_text(encoding="utf-8"))
    raw_change = json.loads((evidence / "join_boundary_raw_probe.json").read_text(encoding="utf-8"))
    whole_by_pair = {
        tuple(round(float(v), 5) for side in ("left", "right") for v in row[side]): row
        for row in whole["pairs"]
    }
    changes = {
        channel: {row["id"]: bool(row["join_change_detected"]) for row in report["cases"]}
        for channel, report in (("stem", stem_change), ("raw", raw_change))
    }
    results = []
    for case in cases:
        side = {}
        for name in ("left", "right"):
            start, end = case[name]
            side[name] = bank.score(
                stem=read_audio(stem_file, start, end),
                raw=read_audio(raw_file, start, end),
                sample_rate=16000,
            )
        key = tuple(round(float(v), 5) for name in ("left", "right") for v in case[name])
        old = whole_by_pair[key]
        decision = review_join(
            side["left"], side["right"],
            whole_verified=(bool(old["whole_result"]["accepted_by_whole_span_verifier"])
                            and not bool(old["whole_result"]["excluded_role_rejected"])),
            confirmed_change=changes["stem"][case["id"]] or changes["raw"][case["id"]],
            gap_allowed=0.0 <= float(old["gap_seconds"]) <= 0.85,
        )
        results.append({
            "id": case["id"], "expected": bool(case["same_target"]),
            "join": decision.allowed, "reason": decision.reason,
            "left": side["left"].state, "right": side["right"].state,
            "left_view_margins": view_margins(side["left"]),
            "right_view_margins": view_margins(side["right"]),
        })
    mixed = json.loads((evidence.parent / "20260925_104500_stt_policy_candidate"
                        / "local_identity_review.json").read_text(encoding="utf-8"))["pairs"][0]
    left = bank.score(stem=read_audio(stem_file, *mixed["left"]),
                      raw=read_audio(raw_file, *mixed["left"]), sample_rate=16000)
    right = bank.score(stem=read_audio(stem_file, *mixed["right"]),
                       raw=read_audio(raw_file, *mixed["right"]), sample_rate=16000)
    mixed_decision = review_join(
        left, right,
        whole_verified=bool(mixed["whole_result"]["accepted_by_whole_span_verifier"]),
        confirmed_change=False, gap_allowed=True,
    )
    return {
        "warning": "First-episode development examples only; no independent natural-dialogue validation.",
        "pairs_passed": sum(row["join"] == row["expected"] for row in results),
        "pairs_total": len(results),
        "pairs": results,
        "reference_holdout": reference_holdout,
        "known_mixed_525": {"join": mixed_decision.allowed, "reason": mixed_decision.reason,
                            "left": left.state, "right": right.state,
                            "left_view_margins": view_margins(left),
                            "right_view_margins": view_margins(right),
                            "boundary_note": "526.81 is an acoustic probe, not human-labelled precision"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--cases", type=Path, default=ROOT / "evaluation/join_cases.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.work, args.evidence, args.cases)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    holdout = report["reference_holdout"]
    return 0 if (report["pairs_passed"] == report["pairs_total"]
                 and not report["known_mixed_525"]["join"]
                 and holdout["target_supported"] == holdout["target_total"]
                 and not holdout["unseen_other_supported_as_target"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
