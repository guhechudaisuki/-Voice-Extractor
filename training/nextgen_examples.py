"""Turn human-verified source intervals into frozen-feature training examples.

Unknown boundaries stay unknown (-1); a VAD gap is never silently labelled as
"no other speaker". This module does not infer identities from subtitles, STT,
filenames or model predictions. Rights/split metadata is audited separately.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch

from extractor.nextgen.features import FeatureSequence, file_digest, require_paired
from extractor.nextgen.identity_model import HEADS, ReferenceFeatures
from extractor.nextgen.inference import EncodedReference
from extractor.nextgen.ledger import Candidate
from extractor.nextgen.reference_bank import ReferenceBank
from extractor.nextgen.scene_adapter import PreparedScene
from extractor.nextgen.timeline import SampleSpan
from training.nextgen_data import DataRecord, audit_records


def _validate_label(value: float) -> None:
    if type(value) not in (int, float) or value not in (-1, 0, 1):
        raise ValueError("Training labels must be -1, 0 or 1")


@dataclass(frozen=True)
class AcousticAnnotation:
    output: SampleSpan
    target: tuple[SampleSpan, ...] = ()
    other: tuple[SampleSpan, ...] = ()
    singing: tuple[SampleSpan, ...] = ()
    overlap: tuple[SampleSpan, ...] = ()
    change: tuple[SampleSpan, ...] = ()
    unknown: tuple[SampleSpan, ...] = ()
    observable: tuple[SampleSpan, ...] = ()
    unobservable: tuple[SampleSpan, ...] = ()
    uncertain: tuple[SampleSpan, ...] = ()
    certain: tuple[SampleSpan, ...] = ()
    complete_timeline: bool = False
    purity: float = -1
    start_complete: float = -1
    end_complete: float = -1

    def __post_init__(self) -> None:
        if type(self.complete_timeline) is not bool:
            raise ValueError("Complete timeline must be an explicit boolean")
        for value in (self.purity, self.start_complete, self.end_complete):
            _validate_label(value)
        for spans in (self.target, self.other, self.singing, self.overlap,
                      self.change, self.unknown, self.observable,
                      self.unobservable, self.uncertain, self.certain):
            if (tuple(sorted(spans)) != spans or any(a.end > b.start for a, b in zip(spans, spans[1:]))
                    or any(not self.output.contains(span) for span in spans)):
                raise ValueError("Annotation spans must be ordered, separate and inside output")
        if any(left.intersection(right) for positive, negative in (
            (self.observable, self.unobservable), (self.uncertain, self.certain)
        ) for left in positive for right in negative):
            raise ValueError("opposite supervision intervals must not overlap")
        if not self.complete_timeline and self.purity != -1:
            raise ValueError("Partial labels cannot certify whole-span purity")
        if self.purity == 1 and (not self.target or self.other or self.singing or self.overlap
                                 or self.unknown):
            raise ValueError("Pure target label contradicts known contamination or missing target")
        if (self.purity == 0 and self.complete_timeline and self.target
                and not (self.other or self.singing or self.overlap or self.unknown)):
            raise ValueError("A target-only span cannot be labelled impure")
        for target in self.target:
            for other in self.other:
                shared = target.intersection(other)
                if shared is not None and not any(mask.contains(shared) for mask in self.overlap):
                    raise ValueError("Simultaneous target/other speech requires an overlap label")


def _event_label(cell: SampleSpan, events: tuple[SampleSpan, ...],
                 fully_annotated: bool) -> float:
    if any(event.contains(cell) for event in events):
        return 1.0
    if any(event.intersection(cell) for event in events):
        return -1.0
    return 0.0 if fully_annotated else -1.0


def _explicit_binary_label(cell: SampleSpan, positive: tuple[SampleSpan, ...],
                           negative: tuple[SampleSpan, ...]) -> float:
    """Only explicit acoustic labels may train observability/uncertainty."""
    if any(span.contains(cell) for span in positive):
        return 1.0
    if any(span.contains(cell) for span in negative):
        return 0.0
    return -1.0


def frame_labels(cells: tuple[SampleSpan, ...], annotation: AcousticAnnotation) -> torch.Tensor:
    if not cells or any(a.end > b.start for a, b in zip(cells, cells[1:])):
        raise ValueError("Expected ordered nonoverlapping feature cells")
    output = torch.full((len(cells), len(HEADS)), -1.0, dtype=torch.float32)
    index = {name: column for column, name in enumerate(HEADS)}
    for row, cell in enumerate(cells):
        if not annotation.output.contains(cell):
            continue
        output[row, index["observable"]] = _explicit_binary_label(
            cell, annotation.observable, annotation.unobservable,
        )
        output[row, index["uncertainty"]] = _explicit_binary_label(
            cell, annotation.uncertain, annotation.certain,
        )
        if any(cell.intersection(span) for span in annotation.unknown):
            continue
        complete = annotation.complete_timeline
        target = _event_label(cell, annotation.target, complete)
        other = _event_label(cell, annotation.other, complete)
        output[row, index["target"]] = target
        output[row, index["other"]] = other
        output[row, index["speech"]] = (
            1.0 if target == 1 or other == 1 else
            0.0 if complete and target == 0 and other == 0 else -1.0
        )
        for name in ("singing", "overlap", "change"):
            output[row, index[name]] = _event_label(cell, getattr(annotation, name), complete)
        # Explicit observability/uncertainty supervision is independent of the
        # person/activity interval; a known identity does not imply every
        # consonant frame carries a standalone voiceprint.
    return output


def build_feature_example(raw: FeatureSequence, stem: FeatureSequence,
                          annotation: AcousticAnnotation, target: ReferenceFeatures,
                          exclusions: tuple[ReferenceFeatures, ...] = ()) -> dict[str, torch.Tensor]:
    require_paired(raw, stem)
    if any(tensor.device.type != "cpu" for tensor in (raw.values, stem.values)):
        raise ValueError("Frozen feature examples must be serialized on CPU")
    if not any(cell.intersection(annotation.output) for cell in raw.cells):
        raise ValueError("The annotated output has no encoded acoustic frame")
    for role in (target, *exclusions):
        role.validate(raw.values.shape[1], torch.device("cpu"))
    allowed = torch.tensor([cell.intersection(annotation.output) is not None
                            for cell in raw.cells], dtype=torch.bool)
    indexes = torch.where(allowed)[0]
    if int(indexes[-1] - indexes[0] + 1) != len(indexes):
        raise ValueError("The output mask cannot skip an internal event")
    labels = frame_labels(raw.cells, annotation)
    if not (labels.ge(0).any() or annotation.purity >= 0
            or annotation.start_complete >= 0 or annotation.end_complete >= 0):
        raise ValueError("Example has no human-confirmed training label")
    tensors = {
        "raw": raw.values.detach().cpu().contiguous(),
        "stem": stem.values.detach().cpu().contiguous(),
        "allowed": allowed,
        "labels.frames": labels,
        "labels.purity": torch.tensor(float(annotation.purity)),
        "labels.boundaries": torch.tensor([annotation.start_complete, annotation.end_complete],
                                          dtype=torch.float32),
    }
    for prefix, reference in (("target", target),
                              *((f"excluded.{index:03d}", group)
                                for index, group in enumerate(exclusions))):
        for name in ("raw", "stem", "valid", "quality"):
            tensors[f"{prefix}.{name}"] = getattr(reference, name).detach().cpu().contiguous()
    return tensors


def build_prepared_example(scene: PreparedScene, candidate: Candidate,
                           bank: ReferenceBank, encoded: tuple[EncodedReference, ...],
                           encoder, annotation: AcousticAnnotation, *,
                           query_source_group: str,
                           reference_source_groups: dict[str, str]) -> dict[str, torch.Tensor]:
    """Connect a real prepared candidate and independent references to labels."""
    if candidate not in scene.candidates or annotation.output != candidate.output:
        raise ValueError("Training labels must name an actual prepared candidate interval")
    if (not query_source_group or set(reference_source_groups) !=
            {row.clip_digest for row in bank.entries}
            or any(not source for source in reference_source_groups.values())):
        raise ValueError("Every query/reference needs its original-recording group ID")
    if query_source_group in reference_source_groups.values():
        raise ValueError("A training query cannot enroll another crop of its source recording")
    if any(row.source_digest == candidate.source_sha256 for row in bank.entries):
        raise ValueError("A training query cannot enroll audio from its own source recording")
    mapping = {row.clip_digest: row for row in encoded}
    if len(mapping) != len(encoded) or set(mapping) != {row.clip_digest for row in bank.entries}:
        raise ValueError("Encoded reference set does not match its immutable bank")
    raw, stem = scene.audio.encode(candidate.context, encoder)
    if raw.backbone_digest != encoder.digest:
        raise ValueError("Prepared query used another feature encoder")
    roles = sorted({row.role for row in bank.entries})
    grouped = {}
    def token(member: EncodedReference, field: str) -> torch.Tensor:
        value = getattr(member, field)
        return value.values if isinstance(value, FeatureSequence) else value

    for role in roles:
        members = [mapping[row.clip_digest] for row in bank.entries if row.role == role]
        if any(member.raw.backbone_digest != encoder.digest
               or member.stem.backbone_digest != encoder.digest for member in members):
            raise ValueError("Reference and query backbone fingerprints differ")
        grouped[role] = ReferenceFeatures(*(
            torch.cat([token(member, field) for member in members])
            for field in ("raw", "stem", "valid", "quality")
        ))
    target = grouped.pop("target", None)
    if target is None:
        raise ValueError("No independent target reference remains")
    return build_feature_example(raw, stem, annotation, target,
                                 tuple(grouped[role] for role in sorted(grouped)))


def save_feature_example(tensors: dict[str, torch.Tensor], destination: Path) -> str:
    """Write one immutable safetensors example and return its SHA-256."""
    from safetensors.torch import save_file

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(destination))
    return file_digest(destination)


def data_record_for_prepared(example_id: str, split: str, scene: PreparedScene,
                             bank: ReferenceBank, *, query_source_group: str,
                             reference_source_groups: dict[str, str],
                             role_speaker_ids: dict[str, str],
                             query_speaker_ids: tuple[str, ...],
                             features_path: str, features_digest: str,
                             backbone_digest: str, license_reference: str,
                             training_allowed: bool, human_labels: bool,
                             synthetic: bool) -> DataRecord:
    """Bind a feature shard to all source/person groups and actual audio hashes."""
    roles = {row.role for row in bank.entries}
    if (not query_source_group or set(reference_source_groups) !=
            {row.clip_digest for row in bank.entries}
            or set(role_speaker_ids) != roles
            or not query_speaker_ids
            or any(not value for value in (*reference_source_groups.values(),
                                           *role_speaker_ids.values(),
                                           *query_speaker_ids))):
        raise ValueError("Record lacks a complete source/person provenance mapping")
    if query_source_group in reference_source_groups.values():
        raise ValueError("A reference crop shares the query's original recording")
    alignment = scene.audio.alignment_report
    content = {alignment.raw_digest, alignment.stem_digest}
    for row in bank.entries:
        content.update((row.raw_digest, row.stem_digest))
    return DataRecord(
        example_id, split,
        tuple(sorted({query_source_group, *reference_source_groups.values()})),
        tuple(sorted({*role_speaker_ids.values(), *query_speaker_ids})),
        tuple(sorted(content)),
        features_path, features_digest, backbone_digest, license_reference,
        training_allowed, human_labels, synthetic,
    )


def write_manifest(records: tuple[DataRecord, ...], destination: Path) -> dict:
    """Audit all splits before publishing a local training manifest."""
    audit = audit_records(records, for_training=True)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": 1, "records": [record.__dict__ for record in records],
               "audit": audit}
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit
