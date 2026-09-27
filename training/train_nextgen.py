"""Explicit offline frozen-feature training entry; never run by the desktop app.

Input safetensors contain paired query/reference tokens and independent local,
purity and boundary labels. -1 labels are unknown. No unpickling, pseudo-label
absorption, network requests, media copying or automatic dataset downloads.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extractor.nextgen.artifacts import write_research_checkpoint  # noqa: E402
from extractor.nextgen.features import file_digest  # noqa: E402
from extractor.nextgen.identity_model import HEADS, ReferenceFeatures, TemporalIdentityModel, identity_loss  # noqa: E402
from training.nextgen_data import DataRecord, audit_records, read_manifest  # noqa: E402


def load_example(root: Path, record: DataRecord, device: str) -> dict:
    from safetensors.torch import load_file

    root = root.resolve(strict=True)
    path = (root / record.features_path).resolve(strict=True)
    if root not in path.parents or file_digest(path) != record.features_digest:
        raise ValueError("Feature file escaped its root or failed checksum")
    return load_file(str(path), device=device)


def reference(example: dict, prefix: str) -> ReferenceFeatures:
    return ReferenceFeatures(*(example[f"{prefix}.{key}"] for key in ("raw", "stem", "valid", "quality")))


def example_loss(model: TemporalIdentityModel, example: dict) -> torch.Tensor:
    groups = sorted({key.split(".")[1] for key in example if key.startswith("excluded.")})
    output = model(example["raw"], example["stem"], reference(example, "target"),
                   tuple(reference(example, f"excluded.{group}") for group in groups), example["allowed"])
    return identity_loss(output, example["labels.frames"], example["labels.purity"], example["labels.boundaries"])


def supervision_counts(examples) -> dict[str, dict[str, int]]:
    """Inventory only known training labels; unknown (-1) never counts as negative."""
    names = (*HEADS, "purity", "boundary_start", "boundary_end")
    counts = {name: {"positive": 0, "negative": 0} for name in names}
    for example in examples:
        frames = example["labels.frames"]
        if frames.ndim != 2 or frames.shape[1] != len(HEADS):
            raise ValueError("Frame labels do not match the model decision heads")
        labels = {name: frames[:, index] for index, name in enumerate(HEADS)}
        labels["purity"] = example["labels.purity"].reshape(-1)
        boundary = example["labels.boundaries"].reshape(-1)
        if boundary.numel() != 2:
            raise ValueError("Expected two boundary labels")
        labels["boundary_start"], labels["boundary_end"] = boundary[:1], boundary[1:]
        for name, values in labels.items():
            if not torch.isfinite(values).all() or not torch.all(
                (values == -1) | (values == 0) | (values == 1)
            ):
                raise ValueError(f"Invalid supervision label for {name}")
            counts[name]["positive"] += int((values == 1).sum())
            counts[name]["negative"] += int((values == 0).sum())
    return counts


def train(manifest: Path, root: Path, destination: Path, *, epochs: int,
          learning_rate: float = 1e-4, seed: int = 20260925,
          feature_dim: int = 1500, projection_dim: int = 128,
          hidden_dim: int = 96, device: str = "cpu") -> dict:
    if (type(epochs) is not int or epochs < 1 or not math.isfinite(learning_rate)
            or learning_rate <= 0 or destination.exists()):
        raise ValueError("Invalid training configuration or existing output directory")
    records = read_manifest(manifest)
    audit = audit_records(records, for_training=True)
    training = [row for row in records if row.split == "train"]
    development = [row for row in records if row.split == "development"]
    if not training or not development:
        raise ValueError("Independent training and development partitions are required")
    coverage = supervision_counts(load_example(root, row, "cpu") for row in training)
    # Check the complete dataset's split metadata, but do not load calibration
    # or sealed test features to select epochs or optimize parameters.
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = TemporalIdentityModel(feature_dim, projection_dim, hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_loss, best_weights, history = math.inf, None, []
    for epoch in range(epochs):
        model.train()
        rng.shuffle(training)
        training_loss = 0.0
        for row in training:
            optimizer.zero_grad(set_to_none=True)
            loss = example_loss(model, load_example(root, row, device))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            training_loss += float(loss.detach())
        model.eval()
        with torch.inference_mode():
            development_loss = sum(float(example_loss(model, load_example(root, row, device)))
                                   for row in development) / len(development)
        if not math.isfinite(development_loss):
            raise ValueError("Nonfinite development loss")
        if development_loss < best_loss:
            best_loss = development_loss
            best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history.append({"epoch": epoch + 1, "train_loss": training_loss / len(training),
                        "development_loss": development_loss})
        print(json.dumps(history[-1]), flush=True)
    model.load_state_dict(best_weights)
    card = write_research_checkpoint(
        destination, model, backbone_digest=audit["backbone_digest"],
        dataset_digest=audit["dataset_digest"], split_digest=audit["split_digest"],
        limitations="Frozen-feature research head; development-selected epoch. No claim of anime-domain "
                     "accuracy, independent-test success or deployment eligibility.",
        supervision=coverage,
    )
    report = {"schema": 1, "seed": seed, "audit": audit, "history": history,
              "training_supervision": coverage,
              "weights_digest": card.weights_digest, "status": "requires_calibration_and_independent_evaluation"}
    (destination / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train(args.manifest, args.feature_root, args.output, epochs=args.epochs, device=args.device)


if __name__ == "__main__":
    main()
