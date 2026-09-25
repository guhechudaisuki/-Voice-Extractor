"""Stress-test domain embedding witnesses on historical utterance edges.

The historical clips are development references, not complete human-labelled
micro-span truth. Results never authorize a production crop by themselves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

VARIANTS = ("char", "va")


def select_edge_windows(report: dict, *, window_seconds: float = 0.6) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    selected: list[dict] = []
    for row in report["windows"]:
        if row["window_seconds"] != window_seconds:
            continue
        if row["clean_speech_fraction"] < report["minimum_clean_speech_fraction"]:
            continue
        if row["label"] == "other":
            selected.append(row)
        else:
            grouped.setdefault(row["case_id"], []).append(row)
    for rows in grouped.values():
        ordered = sorted(rows, key=lambda row: (row["start"], row["end"]))
        selected.append(ordered[0])
        if ordered[-1] is not ordered[0]:
            selected.append(ordered[-1])
    return sorted(selected, key=lambda row: (row["start"], row["end"]))


def run(work_dir: Path, short_report: dict, *, variant: str) -> dict:
    import torch
    from anime_speaker_embedding.model import AnimeSpeakerEmbedding
    from probe_anime_embedding import MODELS, encode, read_audio, scores

    channel = short_report["channel"]
    if variant not in MODELS or channel not in ("stem", "raw"):
        raise ValueError("Unknown model variant or channel")
    selected = select_edge_windows(short_report)
    if not selected or not any(row["label"] == "other" for row in selected):
        raise ValueError("Short-span report lacks comparison windows")
    source = work_dir / (
        "stems/target_vocals.wav" if channel == "stem" else "target_normalized.wav"
    )
    reference_dir = work_dir / (
        "reference_voice_clips" if channel == "stem"
        else "reference_original_voice_clips"
    )
    negative_dir = work_dir / (
        "negative_reference_voice_clips" if channel == "stem"
        else "negative_reference_original_voice_clips"
    )
    references = sorted(reference_dir.glob("*.wav"))
    negative_groups = [
        sorted(group.glob("*.wav"))
        for group in sorted(negative_dir.glob("role_*"))
    ]
    negative_groups = [group for group in negative_groups if group]
    if len(references) < 2 or not negative_groups:
        raise ValueError("Need two target references and one other group")
    model = AnimeSpeakerEmbedding(variant=variant, ckpt_path=MODELS[variant]).eval()
    target_vectors = torch.stack([encode(model, read_audio(path)) for path in references])
    other_vectors = [
        torch.stack([encode(model, read_audio(path)) for path in group])
        for group in negative_groups
    ]
    rows = []
    for selected_row in selected:
        start, end = selected_row["start"], selected_row["end"]
        vector = encode(model, read_audio(source, start, end))
        result = scores(vector, target_vectors, other_vectors)
        other_max = max(result["negative_group_max"])
        rows.append({
            "case_id": selected_row["case_id"],
            "label": selected_row["label"],
            "start": start,
            "end": end,
            "target_median": result["target_median"],
            "nearest_other": other_max,
            "margin": round(result["target_median"] - other_max, 5),
            "target_wins": result["target_median"] > other_max,
        })
    return {
        "warning": (
            "Development-only historical output edges. They have not all "
            "been independently audited for subsecond speaker purity."
        ),
        "variant": variant,
        "channel": channel,
        "target_reference_count": len(references),
        "negative_group_count": len(negative_groups),
        "summary": {
            label: {
                "windows": sum(row["label"] == label for row in rows),
                "target_wins": sum(
                    row["label"] == label and row["target_wins"] for row in rows
                ),
            }
            for label in ("target", "other")
        },
        "windows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("short_report", type=Path)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        args.work_dir,
        json.loads(args.short_report.read_text(encoding="utf-8")),
        variant=args.variant,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ANIME_EDGE_PROBE={args.output} WINDOWS={len(report['windows'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
