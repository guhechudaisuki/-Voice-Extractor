"""Versioned model/calibration artifacts; failed research weights stay research."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

from .decision_policy import Calibration, probability
from .features import file_digest
from .identity_model import HEADS, TemporalIdentityModel
from .timeline import require_digest


@dataclass(frozen=True)
class InferenceCalibration:
    identity: Calibration
    observable_min: float
    singing_max: float
    overlap_max: float
    boundary_min: float
    speech_min: float
    change_min: float

    def __post_init__(self) -> None:
        for value in (self.observable_min, self.singing_max, self.overlap_max,
                      self.boundary_min, self.speech_min, self.change_min):
            probability(value)


@dataclass(frozen=True)
class ModelCard:
    architecture: str
    weights_digest: str
    backbone_digest: str
    dataset_digest: str
    split_digest: str
    stage: str  # research or validated; neither claims universal accuracy
    reports: tuple[str, ...]
    limitations: str
    supervision: dict[str, dict[str, int]] = field(default_factory=dict)
    calibration_digest: str | None = None

    def __post_init__(self) -> None:
        if self.architecture not in TemporalIdentityModel.compatible_architectures:
            raise ValueError("Incompatible model architecture")
        for value in (self.weights_digest, self.backbone_digest, self.dataset_digest, self.split_digest):
            require_digest(value)
        if self.stage not in ("research", "validated") or not self.limitations:
            raise ValueError("Model card requires stage and limitations")
        if self.stage == "validated" and not self.reports:
            raise ValueError("Validated status requires independent evaluation artifacts")
        if self.calibration_digest is not None:
            require_digest(self.calibration_digest)
        if self.stage == "validated" and self.calibration_digest is None:
            raise ValueError("Validated model requires a fingerprinted calibration artifact")
        for report in self.reports:
            require_digest(report)
        required = (*HEADS, "purity", "boundary_start", "boundary_end")
        if any(name not in required or set(counts) != {"positive", "negative"}
               or any(type(value) is not int or value < 0 for value in counts.values())
               for name, counts in self.supervision.items()):
            raise ValueError("Invalid model-head supervision counts")
        if self.stage == "validated" and any(
            name not in self.supervision
            or self.supervision[name]["positive"] == 0
            or self.supervision[name]["negative"] == 0 for name in required
        ):
            raise ValueError("Validated model needs positive and negative supervision for every decision head")


def model_from_metadata(metadata: dict) -> TemporalIdentityModel:
    """Construct the recorded architecture, including reproducible v1 research weights."""
    architecture = metadata["card"]["architecture"]
    if architecture not in TemporalIdentityModel.compatible_architectures:
        raise ValueError("Incompatible model architecture")
    version = TemporalIdentityModel.compatible_architectures.index(architecture) + 1
    config = dict(metadata["config"])
    if "architecture_version" in config and config["architecture_version"] != version:
        raise ValueError("Model architecture and configuration disagree")
    config["architecture_version"] = version
    return TemporalIdentityModel(**config)


def load_bundle(directory: Path, *, device: str = "cpu", research: bool = False):
    """Safetensors avoids unpickling arbitrary checkpoint code."""
    from safetensors.torch import load_file

    directory = directory.resolve(strict=True)
    metadata = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    if metadata.get("schema") != 1 or tuple(metadata.get("heads", ())) != HEADS:
        raise ValueError("Unsupported checkpoint schema/heads")
    card = ModelCard(**{**metadata["card"], "reports": tuple(metadata["card"]["reports"])})
    weights = directory / "weights.safetensors"
    if file_digest(weights) != card.weights_digest:
        raise ValueError("Checkpoint checksum mismatch")
    if card.stage != "validated" and not research:
        raise ValueError("Research checkpoint is not eligible for production inference")
    raw = json.loads((directory / "calibration.json").read_text(encoding="utf-8"))
    if card.calibration_digest is not None and file_digest(directory / "calibration.json") != card.calibration_digest:
        raise ValueError("Calibration artifact checksum mismatch")
    calibration = InferenceCalibration(**{**raw, "identity": Calibration(**raw["identity"])})
    if calibration.identity.model_digest != card.weights_digest:
        raise ValueError("Calibration was not fitted for these weights")
    if calibration.identity.data_digest != card.dataset_digest:
        raise ValueError("Calibration dataset does not match the model card")
    model = model_from_metadata(metadata)
    model.load_state_dict(load_file(str(weights)), strict=True)
    return model.to(device).eval(), card, calibration


def write_research_checkpoint(directory: Path, model: TemporalIdentityModel, *,
                              backbone_digest: str, dataset_digest: str, split_digest: str,
                              limitations: str,
                              supervision: dict[str, dict[str, int]] | None = None) -> ModelCard:
    from safetensors.torch import save_file

    for value in (backbone_digest, dataset_digest, split_digest):
        require_digest(value)
    if not limitations:
        raise ValueError("Record training limitations")
    directory.mkdir(parents=True, exist_ok=False)
    weights = directory / "weights.safetensors"
    save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()},
              str(weights))
    card = ModelCard(model.architecture, file_digest(weights), backbone_digest,
                     dataset_digest, split_digest, "research", (), limitations,
                     supervision or {})
    (directory / "model.json").write_text(json.dumps({
        "schema": 1, "heads": HEADS, "config": model.config, "card": asdict(card),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    # Calibration is deliberately absent. Saving trained weights does not make
    # a valid operating point appear, nor authorize export through old gates.
    return card


def write_research_calibration(directory: Path, calibration: InferenceCalibration, *,
                               dataset_digest: str) -> Path:
    """Bind an audited calibration selection to exact research weights/data.

    This does not promote model status. Independent acoustic evaluation and a
    separately reviewed promotion are required before production may load it.
    """
    require_digest(dataset_digest)
    directory = Path(directory).resolve(strict=True)
    metadata = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    card = ModelCard(**{**metadata["card"], "reports": tuple(metadata["card"]["reports"])})
    if card.stage != "research":
        raise ValueError("Only an unpromoted research bundle may receive research calibration")
    if (file_digest(directory / "weights.safetensors") != card.weights_digest
            or calibration.identity.model_digest != card.weights_digest):
        raise ValueError("Calibration/model weights digest mismatch")
    if (dataset_digest != card.dataset_digest
            or calibration.identity.data_digest != dataset_digest):
        raise ValueError("Calibration dataset does not match the checkpoint dataset")
    destination = directory / "calibration.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(asdict(calibration), stream, ensure_ascii=False, indent=2)
    # A research card records the exact calibration artifact too. This still
    # cannot promote the model; validated status additionally requires audits.
    metadata["card"]["calibration_digest"] = file_digest(destination)
    temporary = directory / "model.json.tmp"
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(directory / "model.json")
    return destination
