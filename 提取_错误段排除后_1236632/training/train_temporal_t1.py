"""Train and compare a frozen-feature temporal head on LibriSpeech T1 scenes.

All results are synthetic-scene development results. No checkpoint from this
script is eligible for production without real-dialogue and independent tests.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
from temporal_identity_model import TemporalIdentityHead, temporal_loss  # noqa: E402


def load_scene(path: Path) -> dict:
    scene = torch.load(path, map_location="cpu")
    query = scene["query"]
    references = scene["target_references"]
    negatives = scene["negative_references"]
    negative_mask = scene.get("negative_mask")
    labels = scene["labels"]
    if (
        query.ndim != 2 or references.ndim != 2
        or query.shape[1] != references.shape[1]
        or labels.shape != (query.shape[0], 4)
        or (negatives is not None and (
            negatives.ndim != 3 or negatives.shape[2] != query.shape[1]
            or negative_mask is None or negative_mask.shape != negatives.shape[:2]
        ))
    ):
        raise ValueError(f"Invalid feature scene: {path}")
    if torch.any((labels < -1) | (labels > 1)):
        raise ValueError(f"Invalid labels: {path}")
    return scene


def model_logits(model: TemporalIdentityHead, scene: dict, device: torch.device) -> torch.Tensor:
    query = scene["query"].float().unsqueeze(0).to(device)
    references = scene["target_references"].float().unsqueeze(0).to(device)
    negatives = scene["negative_references"]
    return model(
        query, references,
        negative_references=(negatives.float().unsqueeze(0).to(device)
                             if negatives is not None else None),
        negative_mask=(scene["negative_mask"].unsqueeze(0).to(device)
                       if negatives is not None else None),
    )[0]


def cosine_baselines(scene: dict) -> tuple[np.ndarray, np.ndarray]:
    query = F.normalize(scene["query"].float(), dim=-1)
    targets = F.normalize(scene["target_references"].float(), dim=-1)
    target_score = (query @ targets.T).max(dim=-1).values
    negatives = scene["negative_references"]
    if negatives is not None:
        vectors = F.normalize(negatives.float(), dim=-1).flatten(0, 1)
        scores = query @ vectors.T
        scores = scores.masked_fill(~scene["negative_mask"].flatten()[None, :], -1e4)
        other_score = scores.max(dim=-1).values
        margin = target_score - other_score
    else:
        margin = target_score
    return target_score.numpy(), margin.numpy()


def auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Rank AUC with tie correction, no calibration or chosen threshold."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    cumulative_negatives = 0
    wins = 0.0
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        group_positive = int(sorted_labels[start:end].sum())
        group_negative = end - start - group_positive
        wins += group_positive * (cumulative_negatives + 0.5 * group_negative)
        cumulative_negatives += group_negative
        start = end
    return float(wins / (positives * negatives))


def collect_scores(
    model: TemporalIdentityHead, feature_root: Path, paths: list[str], device: torch.device,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result: dict[str, list[np.ndarray]] = {name: [] for name in ("labels", "head", "cosine", "margin")}
    model.eval()
    with torch.inference_mode():
        for relative_path in paths:
            scene = load_scene(feature_root / relative_path)
            labels = scene["labels"][:, 0].numpy()
            mask = labels >= 0
            logits = model_logits(model, scene, device)[:, 0].float().cpu().numpy()
            cosine, margin = cosine_baselines(scene)
            for name, values in (
                ("labels", labels), ("head", logits),
                ("cosine", cosine), ("margin", margin),
            ):
                result[name].append(values[mask])
    packed = {name: np.concatenate(parts) for name, parts in result.items()}
    labels = packed.pop("labels")
    return {name: (labels, scores) for name, scores in packed.items()}


def operating_point(
    calibration: tuple[np.ndarray, np.ndarray],
    evaluation: tuple[np.ndarray, np.ndarray], *, negative_quantile: float = 0.99,
) -> dict:
    train_labels, train_scores = calibration
    labels, scores = evaluation
    train_negatives = train_scores[train_labels == 0]
    if not len(train_negatives):
        raise ValueError("Calibration requires negative frames")
    threshold = float(np.quantile(train_negatives, negative_quantile))
    predicted = scores > threshold
    positives = labels == 1
    negatives = labels == 0
    return {
        "auc": auc(labels, scores),
        "calibration_negative_quantile": negative_quantile,
        "threshold": threshold,
        "target_frame_recall": float(predicted[positives].mean()) if positives.any() else None,
        "other_or_silent_frame_false_positive_rate": (
            float(predicted[negatives].mean()) if negatives.any() else None
        ),
        "positive_frames": int(positives.sum()),
        "negative_frames": int(negatives.sum()),
    }


def evaluate_scene_operating_point(
    calibration: tuple[np.ndarray, np.ndarray],
    feature_root: Path,
    validation_paths: list[str],
    model: TemporalIdentityHead,
    device: torch.device,
    *, negative_quantile: float = 0.99,
) -> dict:
    """Count erroneous target presence in target-absent scenes separately."""
    train_labels, train_scores = calibration
    threshold = float(np.quantile(train_scores[train_labels == 0], negative_quantile))
    target_absent = []
    target_present = []
    model.eval()
    with torch.inference_mode():
        for relative_path in validation_paths:
            scene = load_scene(feature_root / relative_path)
            labels = scene["labels"][:, 0].numpy()
            probabilities = model_logits(model, scene, device)[:, 0].float().cpu().numpy()
            valid = labels >= 0
            target_frames = labels == 1
            accepted_fraction = float((probabilities[valid] > threshold).mean())
            record = {
                "file": relative_path,
                "target_labeled_frames": int(target_frames.sum()),
                "accepted_frame_fraction": accepted_fraction,
            }
            (target_present if target_frames.any() else target_absent).append(record)
    return {
        "threshold_from_train_negatives": threshold,
        "target_absent_scene_count": len(target_absent),
        "target_absent_scene_false_presence_count": sum(
            item["accepted_frame_fraction"] > 0 for item in target_absent
        ),
        "target_absent_scenes": target_absent,
        "target_present_scene_count": len(target_present),
        "target_present_scenes": target_present,
        "warning": "Any accepted frame is counted as false presence in target-absent synthetic scenes; this is a diagnostic, not a final audio export decision.",
    }


def train(
    feature_root: Path, output_dir: Path, *, epochs: int, seed: int, learning_rate: float,
) -> dict:
    if epochs < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Expected positive epochs and learning rate")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    manifest = json.loads((feature_root / "manifest.json").read_text(encoding="utf-8"))
    train_paths = [item["file"] for item in manifest["scenes"] if item["split"] == "train"]
    validation_paths = [item["file"] for item in manifest["scenes"] if item["split"] == "validation"]
    if not train_paths or not validation_paths:
        raise ValueError("Both speaker-disjoint splits are required")
    if set(manifest["train_speakers"]) & set(manifest["validation_speakers"]):
        raise ValueError("Speaker leakage between training and validation")
    feature_dim = load_scene(feature_root / train_paths[0])["query"].shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalIdentityHead(feature_dim=feature_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    weights = torch.tensor([1.0, 1.0, 0.25, 0.10], device=device)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_auc = -1.0
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        random.Random(seed + epoch).shuffle(train_paths)
        losses = []
        for relative_path in train_paths:
            scene = load_scene(feature_root / relative_path)
            logits = model_logits(model, scene, device).unsqueeze(0)
            labels = scene["labels"].float().unsqueeze(0).to(device)
            loss = temporal_loss(
                logits, labels, torch.tensor([labels.shape[1]], device=device), weights,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = collect_scores(model, feature_root, validation_paths, device)
        validation_auc = auc(*validation["head"])
        record = {
            "epoch": epoch,
            "train_loss": round(float(np.mean(losses)), 6),
            "validation_target_frame_auc": validation_auc,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation_auc is not None and validation_auc > best_auc:
            best_auc = validation_auc
            best_epoch = epoch
            torch.save({
                "state_dict": model.state_dict(),
                "feature_dim": feature_dim,
                "model_sha256": manifest["model_sha256"],
                "wavlm_hidden_layer": manifest["wavlm_hidden_layer"],
                "seed": seed,
                "epoch": epoch,
            }, output_dir / "best.pt")
    checkpoint = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    train_scores = collect_scores(model, feature_root, train_paths, device)
    validation_scores = collect_scores(model, feature_root, validation_paths, device)
    comparison = {
        name: operating_point(train_scores[name], validation_scores[name])
        for name in ("head", "cosine", "margin")
    }
    report = {
        "warning": "Synthetic English audiobook scenes and speaker-disjoint development split only; not evidence of Japanese anime extraction gain.",
        "feature_manifest": str((feature_root / "manifest.json").resolve()),
        "best_epoch": best_epoch,
        "best_validation_auc": best_auc,
        "scene_counts": {"train": len(train_paths), "validation": len(validation_paths)},
        "speaker_counts": {
            "train": len(manifest["train_speakers"]),
            "validation": len(manifest["validation_speakers"]),
        },
        "comparison": comparison,
        "scene_diagnostic": evaluate_scene_operating_point(
            train_scores["head"], feature_root, validation_paths, model, device,
        ),
        "history": history,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feature_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=240927)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    args = parser.parse_args()
    train(args.feature_root, args.output_dir, epochs=args.epochs,
          seed=args.seed, learning_rate=args.learning_rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
