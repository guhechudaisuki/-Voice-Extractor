"""Training authorization and split-leakage checks, independent of inference.

Source and person IDs include both query and references. A reference from a
held-out speaker/source cannot slip into training merely by being a reference.
Manifests are declarations to audit, not a way to infer legal permissions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from extractor.nextgen.timeline import require_digest


SPLITS = ("train", "development", "calibration", "test")


@dataclass(frozen=True)
class DataRecord:
    example_id: str
    split: str
    sources: tuple[str, ...]
    speakers: tuple[str, ...]
    content_digests: tuple[str, ...]
    features_path: str
    features_digest: str
    backbone_digest: str
    license_reference: str
    training_allowed: bool
    human_labels: bool
    synthetic: bool

    def __post_init__(self) -> None:
        if (not self.example_id or self.split not in SPLITS or not self.sources or not self.speakers
                or not self.content_digests or not self.license_reference):
            raise ValueError("Data record requires split, all query/reference identities, and provenance")
        if any(not value for value in (*self.sources, *self.speakers)):
            raise ValueError("Empty source/person identity")
        for digest in (*self.content_digests, self.features_digest, self.backbone_digest):
            require_digest(digest)
        if any(type(flag) is not bool for flag in (self.training_allowed, self.human_labels, self.synthetic)):
            raise ValueError("Data rights/label flags must be explicit booleans")
        path = Path(self.features_path)
        if path.is_absolute() or path.drive or path.root or ".." in path.parts or not path.name:
            raise ValueError("Feature paths must remain relative to the declared data root")


def audit_records(records: tuple[DataRecord, ...], *, for_training: bool = False) -> dict:
    if not records or len({row.example_id for row in records}) != len(records):
        raise ValueError("Empty dataset or duplicate examples")
    owners: dict[tuple[str, str], str] = {}
    if len({row.backbone_digest for row in records}) != 1:
        raise ValueError("Different encoder/preprocessing versions cannot share one training run")
    for row in records:
        # The development partition selects checkpoints and is part of the
        # training workflow, even though its gradients are not updated.
        if for_training and row.split in ("train", "development") and not row.training_allowed:
            raise ValueError(f"Training use has not been authorized: {row.example_id}")
        if not row.synthetic and not row.human_labels:
            raise ValueError(f"Natural-dialogue labels lack human verification: {row.example_id}")
        for kind, values in (("source", row.sources), ("speaker", row.speakers),
                             ("content", row.content_digests)):
            for value in values:
                previous = owners.setdefault((kind, value), row.split)
                if previous != row.split:
                    raise ValueError(f"Cross-split {kind} leakage: {value}")
    ordered = sorted(records, key=lambda row: row.example_id)
    serialized = json.dumps([asdict(row) for row in ordered], sort_keys=True, ensure_ascii=False)
    splits = [(row.example_id, row.split, row.sources, row.speakers) for row in ordered]
    return {
        "dataset_digest": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "split_digest": hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest(),
        "backbone_digest": records[0].backbone_digest,
        "counts": {split: sum(row.split == split for row in records) for split in SPLITS},
        "synthetic_examples": sum(row.synthetic for row in records),
        "natural_human_verified_examples": sum(not row.synthetic and row.human_labels for row in records),
    }


def read_manifest(path: Path) -> tuple[DataRecord, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != 1:
        raise ValueError("Unsupported training manifest")
    return tuple(DataRecord(**{**row, **{key: tuple(row[key]) for key in (
        "sources", "speakers", "content_digests",
    )}}) for row in raw["records"])
