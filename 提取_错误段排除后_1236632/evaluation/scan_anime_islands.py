"""Scan full silence-delimited speech islands with local anime embeddings.

Only proposes adjacent islands for later whole-utterance and boundary review.
No score from this scanner is an output permission or a probability of identity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from anime_speaker_embedding.model import AnimeSpeakerEmbedding

from probe_anime_embedding import MODELS, encode, read_audio, scores


def score_islands(
    work_dir: Path, stage_audit: dict, *, minimum_gap: float, maximum_gap: float,
) -> dict:
    if not 0 <= minimum_gap <= maximum_gap:
        raise ValueError("Invalid join gap range")
    islands = stage_audit.get("stages", {}).get("clean_speech_islands", {}).get("spans")
    if not isinstance(islands, list):
        raise ValueError("Stage audit lacks clean_speech_islands")
    source = work_dir / "stems/target_vocals.wav"
    references = sorted((work_dir / "reference_voice_clips").glob("*.wav"))
    groups = [
        sorted(group.glob("*.wav"))
        for group in sorted((work_dir / "negative_reference_voice_clips").glob("role_*"))
    ]
    groups = [group for group in groups if group]
    if not references or not groups:
        raise ValueError("Experimental open-set competition needs target and other references")
    ordered = sorted((float(start), float(end)) for start, end in islands)
    if any(end <= start for start, end in ordered):
        raise ValueError("Invalid clean speech island")
    candidates = []
    for index in range(len(ordered) - 1):
        left, right = ordered[index], ordered[index + 1]
        gap = right[0] - left[1]
        if minimum_gap <= gap <= maximum_gap:
            candidates.append((index, index + 1, round(gap, 5)))
    relevant_indexes = sorted({index for left, right, _ in candidates for index in (left, right)})
    items: dict[int, dict] = {
        index: {"index": index, "span": list(ordered[index])}
        for index in relevant_indexes
    }
    for variant in ("char", "va"):
        model = AnimeSpeakerEmbedding(variant=variant, ckpt_path=MODELS[variant]).eval()
        target_vectors = torch.stack([encode(model, read_audio(path)) for path in references])
        negative_groups = [
            torch.stack([encode(model, read_audio(path)) for path in group])
            for group in groups
        ]
        for completed, index in enumerate(relevant_indexes, start=1):
            start, end = ordered[index]
            if end - start < 0.20:
                items[index][variant] = {"too_short": True, "target_wins": False}
                continue
            vector = encode(model, read_audio(source, start, end))
            evidence = scores(vector, target_vectors, negative_groups)
            nearest = max(evidence["negative_group_max"])
            items[index][variant] = {
                "target_median": evidence["target_median"],
                "nearest_other": nearest,
                "target_wins": bool(evidence["target_median"] > nearest),
            }
            if completed % 20 == 0 or completed == len(relevant_indexes):
                print(f"{variant} {completed}/{len(relevant_indexes)}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pair_rows = []
    for left, right, gap in candidates:
        left_item, right_item = items[left], items[right]
        left_wins = any(left_item[name]["target_wins"] for name in ("char", "va"))
        right_wins = any(right_item[name]["target_wins"] for name in ("char", "va"))
        pair_rows.append({
            "left_index": left,
            "right_index": right,
            "left": left_item["span"],
            "right": right_item["span"],
            "gap_seconds": gap,
            "both_parts_have_domain_witness": bool(left_wins and right_wins),
        })
    return {
        "warning": "Development-only domain-model proposals. The model may favor the wrong unseen character; whole-span verification and boundary checks remain necessary.",
        "minimum_gap_seconds": minimum_gap,
        "maximum_gap_seconds": maximum_gap,
        "clean_island_count": len(ordered),
        "scored_island_count": len(relevant_indexes),
        "target_reference_count": len(references),
        "negative_group_count": len(groups),
        "pair_count": len(pair_rows),
        "witness_pair_count": sum(row["both_parts_have_domain_witness"] for row in pair_rows),
        "islands": [items[index] for index in relevant_indexes],
        "pairs": pair_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("stage_audit", type=Path)
    parser.add_argument("--minimum-gap", type=float, default=0.20)
    parser.add_argument("--maximum-gap", type=float, default=0.85)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = score_islands(
        args.work_dir, json.loads(args.stage_audit.read_text(encoding="utf-8")),
        minimum_gap=args.minimum_gap, maximum_gap=args.maximum_gap,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ISLAND_SCAN={args.output} WITNESS_PAIRS={report['witness_pair_count']}/{report['pair_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
