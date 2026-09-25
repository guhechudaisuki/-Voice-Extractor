"""Leave-query-out anime embedding competition on grouped references.

The other person's own group is excluded because it has only one clip and
comparing a query with itself would be leakage. This is an open-set test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_anime_embedding import MODELS, encode, read_audio  # noqa: E402
from probe_reference_joins import select_pairs  # noqa: E402
from anime_speaker_embedding.model import AnimeSpeakerEmbedding  # noqa: E402


def compare(
    query: Path,
    *, target_paths: list[Path], negative_groups: dict[Path, str],
    embeddings: dict[Path, torch.Tensor],
) -> dict:
    target_bank = [path for path in target_paths if path != query]
    if not target_bank:
        raise ValueError("A query cannot be its only target reference")
    target_scores = torch.tensor([float(embeddings[query] @ embeddings[path]) for path in target_bank])
    query_group = negative_groups.get(query)
    group_scores: dict[str, float] = {}
    for path, group in negative_groups.items():
        if group == query_group:
            continue
        score = float(embeddings[query] @ embeddings[path])
        group_scores[group] = max(group_scores.get(group, -1.0), score)
    nearest_other = max(group_scores.values()) if group_scores else None
    return {
        "target_median": round(float(target_scores.median()), 5),
        "target_max": round(float(target_scores.max()), 5),
        "nearest_other": round(nearest_other, 5) if nearest_other is not None else None,
        "target_wins": (
            bool(float(target_scores.median()) > nearest_other)
            if nearest_other is not None else None
        ),
    }


def run(work_dir: Path, *, variant: str, per_kind: int, seed: int) -> dict:
    target_paths = sorted((work_dir / "reference_voice_clips").glob("*.wav"))
    negative_groups = {
        path: path.parent.name for path in sorted(
            (work_dir / "negative_reference_voice_clips").glob("role_*/*.wav")
        )
    }
    pairs = select_pairs(target_paths, list(negative_groups), per_kind=per_kind, seed=seed)
    model = AnimeSpeakerEmbedding(variant=variant, ckpt_path=MODELS[variant]).eval()
    unique_paths = set(target_paths) | set(negative_groups)
    embeddings = {path: encode(model, read_audio(path)) for path in sorted(unique_paths)}
    rows = []
    for kind, left, right, expected_same in pairs:
        sides = []
        for path in (left, right):
            open_set = compare(
                path, target_paths=target_paths, negative_groups=negative_groups,
                embeddings=embeddings,
            )
            sides.append({
                "is_target_reference": path in target_paths,
                "open_set_competition": open_set,
            })
        rows.append({
            "kind": kind,
            "expected_same_group": expected_same,
            "left": sides[0],
            "right": sides[1],
        })
    evaluations = [
        (side["open_set_competition"]["target_wins"], side["is_target_reference"])
        for row in rows for side in (row["left"], row["right"])
        if side["open_set_competition"]["target_wins"] is not None
    ]
    summary = {
        "evaluated_sides": len(evaluations),
        "correct_sides": sum(prediction == truth for prediction, truth in evaluations),
        "false_target_sides": sum(prediction and not truth for prediction, truth in evaluations),
        "missed_target_sides": sum(not prediction and truth for prediction, truth in evaluations),
    }
    return {
        "warning": "Leave-query-out development diagnostic on provided reference clips, not natural dialogue. The queried negative group is absent from enrollment to avoid self-comparison leakage; no known-other result is available with one clip per group.",
        "variant": variant,
        "target_reference_count": len(target_paths),
        "negative_group_count": len(set(negative_groups.values())),
        "summary": summary,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("--variant", choices=tuple(MODELS), default="char")
    parser.add_argument("--per-kind", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.work_dir, variant=args.variant, per_kind=args.per_kind, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"REFERENCE_COMPETITION={args.output} SIDES={len(result['rows']) * 2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
