"""Build audited training shards from cached scenes and human annotations.

This command never prepares media, downloads a model, or invents labels.  It
only binds exact cached candidates to explicit annotations and independently
prepared target/exclusion references.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extractor.nextgen.features import WavLMSpeakerFeatures, file_digest  # noqa: E402
from extractor.nextgen.ledger import Candidate  # noqa: E402
from extractor.nextgen.prepare_media import load_prepared_scene  # noqa: E402
from extractor.nextgen.reference_preparation import (  # noqa: E402
    ReferenceMaterial,
    prepare_reference_bank,
)
from extractor.nextgen.scene_adapter import PreparedScene  # noqa: E402
from extractor.nextgen.timeline import SampleSpan  # noqa: E402
from training.nextgen_data import DataRecord, SPLITS, read_manifest  # noqa: E402
from training.nextgen_examples import (  # noqa: E402
    AcousticAnnotation,
    build_prepared_example,
    data_record_for_prepared,
    save_feature_example,
    write_manifest,
)


_EXAMPLE_FIELDS = {
    "id", "split", "prepared_scene", "candidate_key", "query_source_group",
    "query_speaker_ids", "license_reference", "use_allowed", "references",
    "annotation",
}
_REFERENCE_FIELDS = {
    "role", "prepared_scene", "source_group", "speaker_id",
    "license_reference", "use_allowed",
}
_SPAN_FIELDS = {"start", "end"}
_ANNOTATION_INTERVAL_FIELDS = {
    "target", "other", "singing", "overlap", "change", "unknown",
    "observable", "unobservable", "uncertain", "certain",
}
_ANNOTATION_FIELDS = {
    "output", *_ANNOTATION_INTERVAL_FIELDS, "complete_timeline", "purity",
    "start_complete", "end_complete",
}
_PORTABLE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_WINDOWS_RESERVED = {
    "con", "prn", "aux", "nul", *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


@dataclass(frozen=True)
class _ReferenceInput:
    role: str
    configured_path: str
    path: Path
    scene: PreparedScene
    source_group: str
    speaker_id: str
    license_reference: str


@dataclass(frozen=True)
class _ExampleInput:
    example_id: str
    split: str
    configured_path: str
    path: Path
    scene: PreparedScene
    candidate: Candidate
    candidate_key: str
    query_source_group: str
    query_speaker_ids: tuple[str, ...]
    license_reference: str
    references: tuple[_ReferenceInput, ...]
    annotation: AcousticAnnotation


def _object(value: Any, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return value


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_json_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON numeric value: {value}")


def _exact_fields(value: dict[str, Any], expected: set[str], label: str, *,
                  required: set[str] | None = None) -> None:
    required = expected if required is None else required
    missing = sorted(required - value.keys())
    unknown = sorted(value.keys() - expected)
    if missing:
        raise ValueError(f"{label} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{label} has unknown fields: {', '.join(unknown)}")


def _text(value: Any, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _identity(value: Any, label: str) -> str:
    text = _text(value, label)
    if text != text.strip() or any(ord(character) < 32 or ord(character) == 127
                                   for character in text):
        raise ValueError(f"{label} must not contain surrounding whitespace or control characters")
    return text


def _authorized(value: Any, label: str) -> None:
    if type(value) is not bool:
        raise ValueError(f"{label} must be an explicit boolean")
    if not value:
        raise ValueError(f"Use is not authorized for {label}")


def _scene_path(value: Any, config_directory: Path, label: str) -> tuple[str, Path]:
    configured = _text(value, label)
    path = Path(configured)
    if not path.is_absolute() and (path.drive or path.root):
        raise ValueError(f"{label} must be fully absolute or relative to the config")
    if not path.is_absolute():
        path = config_directory / path
    try:
        return configured, path.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"{label} does not exist: {configured}") from error


def _span(value: Any, label: str) -> SampleSpan:
    raw = _object(value, label)
    _exact_fields(raw, _SPAN_FIELDS, label)
    return SampleSpan(raw["start"], raw["end"])


def _annotation(value: Any, label: str) -> AcousticAnnotation:
    raw = _object(value, label)
    _exact_fields(raw, _ANNOTATION_FIELDS, label, required={"output"})
    values: dict[str, Any] = {"output": _span(raw["output"], f"{label}.output")}
    for field in _ANNOTATION_INTERVAL_FIELDS:
        if field not in raw:
            continue
        rows = raw[field]
        if type(rows) is not list:
            raise ValueError(f"{label}.{field} must be a JSON array")
        values[field] = tuple(_span(row, f"{label}.{field}[{index}]")
                              for index, row in enumerate(rows))
    for field in ("complete_timeline", "purity", "start_complete", "end_complete"):
        if field in raw:
            values[field] = raw[field]
    return AcousticAnnotation(**values)


def _example_id(value: Any) -> str:
    identifier = _identity(value, "example.id")
    if (_PORTABLE_ID.fullmatch(identifier) is None or identifier.endswith(".")
            or identifier.split(".", 1)[0].casefold() in _WINDOWS_RESERVED):
        raise ValueError("example.id must be a portable file-safe identifier")
    return identifier


def _load_inputs(config_path: Path) -> tuple[_ExampleInput, ...]:
    try:
        payload = json.loads(
            config_path.read_text(encoding="utf-8"),
            object_pairs_hook=_json_object,
            parse_constant=_invalid_json_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read dataset configuration: {config_path}") from error
    root = _object(payload, "dataset")
    _exact_fields(root, {"schema", "examples"}, "dataset")
    if type(root["schema"]) is not int or root["schema"] != 1:
        raise ValueError("Unsupported prepared-dataset schema")
    if type(root["examples"]) is not list or not root["examples"]:
        raise ValueError("dataset.examples must be a nonempty JSON array")

    cache: dict[Path, PreparedScene] = {}

    def scene(path: Path) -> PreparedScene:
        if path not in cache:
            cache[path] = load_prepared_scene(path)
        return cache[path]

    examples: list[_ExampleInput] = []
    identifiers: set[str] = set()
    folded_identifiers: set[str] = set()
    for index, value in enumerate(root["examples"]):
        label = f"dataset.examples[{index}]"
        raw = _object(value, label)
        _exact_fields(raw, _EXAMPLE_FIELDS, label)
        identifier = _example_id(raw["id"])
        if identifier in identifiers or identifier.casefold() in folded_identifiers:
            raise ValueError(f"Duplicate example id: {identifier}")
        identifiers.add(identifier)
        folded_identifiers.add(identifier.casefold())
        split = _identity(raw["split"], f"{label}.split")
        if split not in SPLITS:
            raise ValueError(f"Unknown dataset split: {split}")
        configured_path, query_path = _scene_path(
            raw["prepared_scene"], config_path.parent, f"{label}.prepared_scene",
        )
        query_scene = scene(query_path)
        candidate_key = _text(raw["candidate_key"], f"{label}.candidate_key")
        candidates = {candidate.key: candidate for candidate in query_scene.candidates}
        candidate = candidates.get(candidate_key)
        if candidate is None:
            raise ValueError(f"Exact candidate key not found for {identifier}: {candidate_key}")
        annotation = _annotation(raw["annotation"], f"{label}.annotation")
        if annotation.output != candidate.output:
            raise ValueError(f"Annotation output does not match candidate output: {identifier}")
        query_source_group = _identity(
            raw["query_source_group"], f"{label}.query_source_group",
        )
        speakers = raw["query_speaker_ids"]
        if type(speakers) is not list or not speakers:
            raise ValueError(f"{label}.query_speaker_ids must be a nonempty JSON array")
        query_speaker_ids = tuple(
            _identity(value, f"{label}.query_speaker_ids[{speaker_index}]")
            for speaker_index, value in enumerate(speakers)
        )
        if len(set(query_speaker_ids)) != len(query_speaker_ids):
            raise ValueError(f"{label}.query_speaker_ids contains duplicates")
        license_reference = _text(
            raw["license_reference"], f"{label}.license_reference",
        )
        _authorized(raw["use_allowed"], f"{identifier} query")

        references_raw = raw["references"]
        if type(references_raw) is not list or not references_raw:
            raise ValueError(f"{label}.references must contain a target reference")
        references: list[_ReferenceInput] = []
        speakers_by_role: dict[str, str] = {}
        roles_by_speaker: dict[str, str] = {}
        source_metadata: dict[tuple[str, str], tuple[str, str, str]] = {}
        for reference_index, reference_value in enumerate(references_raw):
            reference_label = f"{label}.references[{reference_index}]"
            reference = _object(reference_value, reference_label)
            _exact_fields(reference, _REFERENCE_FIELDS, reference_label)
            role = _identity(reference["role"], f"{reference_label}.role")
            reference_configured, reference_path = _scene_path(
                reference["prepared_scene"], config_path.parent,
                f"{reference_label}.prepared_scene",
            )
            reference_scene = scene(reference_path)
            source_group = _identity(
                reference["source_group"], f"{reference_label}.source_group",
            )
            speaker_id = _identity(
                reference["speaker_id"], f"{reference_label}.speaker_id",
            )
            reference_license = _text(
                reference["license_reference"],
                f"{reference_label}.license_reference",
            )
            _authorized(reference["use_allowed"], f"{identifier} reference {role}")
            previous_speaker = speakers_by_role.setdefault(role, speaker_id)
            if previous_speaker != speaker_id:
                raise ValueError(f"Reference role has conflicting speakers: {role}")
            previous_role = roles_by_speaker.setdefault(speaker_id, role)
            if previous_role != role:
                raise ValueError("One reference speaker cannot be both target and exclusion")
            source_key = (role, reference_scene.audio.source.source_sha256)
            metadata = (source_group, speaker_id, reference_license)
            previous_metadata = source_metadata.setdefault(source_key, metadata)
            if previous_metadata != metadata:
                raise ValueError(f"One reference source has conflicting metadata: {role}")
            if source_group == query_source_group:
                raise ValueError("A reference cannot share the query source group")
            if reference_scene.audio.source.source_sha256 == query_scene.audio.source.source_sha256:
                raise ValueError("A reference cannot reuse the query source recording")
            references.append(_ReferenceInput(
                role, reference_configured, reference_path, reference_scene,
                source_group, speaker_id, reference_license,
            ))
        if not any(reference.role == "target" for reference in references):
            raise ValueError(f"Example has no target reference: {identifier}")
        examples.append(_ExampleInput(
            identifier, split, configured_path, query_path, query_scene, candidate,
            candidate_key, query_source_group, query_speaker_ids, license_reference,
            tuple(references), annotation,
        ))
    return tuple(examples)


def _content_digests(scene: PreparedScene) -> tuple[str, str]:
    alignment = scene.audio.alignment_report
    return alignment.raw_digest, alignment.stem_digest


def _preflight_separation(examples: tuple[_ExampleInput, ...]) -> None:
    counts = {split: 0 for split in SPLITS}
    owners: dict[tuple[str, str], str] = {}
    for example in examples:
        counts[example.split] += 1
        sources = {example.query_source_group,
                   *(reference.source_group for reference in example.references)}
        speakers = {*example.query_speaker_ids,
                    *(reference.speaker_id for reference in example.references)}
        content = {*_content_digests(example.scene),
                   *(digest for reference in example.references
                     for digest in _content_digests(reference.scene))}
        for kind, values in (("source", sources), ("speaker", speakers),
                             ("content", content)):
            for value in values:
                previous = owners.setdefault((kind, value), example.split)
                if previous != example.split:
                    raise ValueError(f"Cross-split {kind} leakage: {value}")
    missing = [split for split, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"Every split is required; missing: {', '.join(missing)}")


def _bank_metadata(example: _ExampleInput, bank) -> tuple[dict[str, str], dict[str, str]]:
    by_source: dict[tuple[str, str], tuple[str, str]] = {}
    for reference in example.references:
        key = (reference.role, reference.scene.audio.source.source_sha256)
        value = reference.source_group, reference.speaker_id
        previous = by_source.setdefault(key, value)
        if previous != value:
            raise ValueError(f"Reference metadata changed for role {reference.role}")
    retained_sources = {(entry.role, entry.source_digest) for entry in bank.entries}
    missing = sorted(set(by_source) - retained_sources)
    if missing:
        raise ValueError("A configured reference produced no usable clean speech")
    source_groups = {
        entry.clip_digest: by_source[(entry.role, entry.source_digest)][0]
        for entry in bank.entries
    }
    role_speakers: dict[str, str] = {}
    for entry in bank.entries:
        speaker = by_source[(entry.role, entry.source_digest)][1]
        previous = role_speakers.setdefault(entry.role, speaker)
        if previous != speaker:
            raise ValueError(f"Reference role has conflicting speakers: {entry.role}")
    return source_groups, role_speakers


def _license_summary(example: _ExampleInput) -> str:
    return json.dumps({
        "query": example.license_reference,
        "references": [
            {"role": reference.role, "license_reference": reference.license_reference}
            for reference in example.references
        ],
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _provenance(example: _ExampleInput) -> dict[str, Any]:
    annotation = asdict(example.annotation)
    annotation_digest = hashlib.sha256(json.dumps(
        annotation, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return {
        "example_id": example.example_id,
        "candidate_key": example.candidate_key,
        "annotation": annotation,
        "annotation_digest": annotation_digest,
        "query": {
            "prepared_scene": example.configured_path,
            "source_group": example.query_source_group,
            "speaker_ids": list(example.query_speaker_ids),
            "license_reference": example.license_reference,
            "use_allowed": True,
            "content_digests": list(_content_digests(example.scene)),
        },
        "references": [
            {
                "role": reference.role,
                "prepared_scene": reference.configured_path,
                "source_group": reference.source_group,
                "speaker_id": reference.speaker_id,
                "license_reference": reference.license_reference,
                "use_allowed": True,
                "content_digests": list(_content_digests(reference.scene)),
            }
            for reference in example.references
        ],
    }


def _remove_staging(temporary: Path, parent: Path) -> None:
    if not temporary.exists():
        return
    resolved = temporary.resolve(strict=True)
    intended_parent = parent.resolve(strict=True)
    if resolved.parent != intended_parent or not resolved.name.startswith(".ngprep-"):
        raise RuntimeError(f"Refusing to remove unverified staging path: {resolved}")
    shutil.rmtree(resolved)


def build_dataset(config_path: str | Path, destination: str | Path, *, encoder) -> dict:
    """Build one immutable manifest plus per-example frozen feature shards."""
    config_path = Path(config_path)
    destination = Path(destination)
    if not config_path.is_absolute() and (config_path.drive or config_path.root):
        raise ValueError("Config path must be fully absolute or relative")
    if not destination.is_absolute() and (destination.drive or destination.root):
        raise ValueError("Destination path must be fully absolute or relative")
    config_path = config_path.resolve(strict=True)
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite dataset: {destination}")
    if not destination.name:
        raise ValueError("Dataset destination must name a new directory")
    config_digest = file_digest(config_path)
    examples = _load_inputs(config_path)
    if file_digest(config_path) != config_digest:
        raise ValueError("Dataset configuration changed while it was being read")
    _preflight_separation(examples)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=".ngprep-", dir=destination.parent,
    )).resolve(strict=True)
    if temporary.parent != destination.parent.resolve(strict=True):
        raise RuntimeError("Temporary dataset was not created beside its destination")
    try:
        records: list[DataRecord] = []
        for example in examples:
            bank, encoded = prepare_reference_bank(tuple(
                ReferenceMaterial(reference.role, reference.scene)
                for reference in example.references
            ), encoder)
            reference_source_groups, role_speaker_ids = _bank_metadata(example, bank)
            tensors = build_prepared_example(
                example.scene, example.candidate, bank, encoded, encoder,
                example.annotation, query_source_group=example.query_source_group,
                reference_source_groups=reference_source_groups,
            )
            relative = Path("features") / example.split / f"{example.example_id}.safetensors"
            features_digest = save_feature_example(tensors, temporary / relative)
            records.append(data_record_for_prepared(
                example.example_id, example.split, example.scene, bank,
                query_source_group=example.query_source_group,
                reference_source_groups=reference_source_groups,
                role_speaker_ids=role_speaker_ids,
                query_speaker_ids=example.query_speaker_ids,
                features_path=relative.as_posix(), features_digest=features_digest,
                backbone_digest=encoder.digest,
                license_reference=_license_summary(example), training_allowed=True,
                human_labels=True, synthetic=False,
            ))

        manifest_path = temporary / "manifest.json"
        audit = write_manifest(tuple(records), manifest_path)
        audit = {
            **audit,
            "authorized_assets": sum(1 + len(example.references) for example in examples),
            "human_annotation_examples": len(examples),
            "separation_checked": ["source", "speaker", "content"],
        }
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["audit"] = audit
        manifest["config_digest"] = config_digest
        manifest["provenance"] = [_provenance(example) for example in examples]
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        if file_digest(config_path) != config_digest:
            raise ValueError("Dataset configuration changed during the build")
        for example in examples:
            reloaded = load_prepared_scene(example.path)
            if (_content_digests(reloaded) != _content_digests(example.scene)
                    or example.candidate_key not in {row.key for row in reloaded.candidates}):
                raise ValueError(f"Query scene changed during the build: {example.example_id}")
            for reference in example.references:
                reloaded_reference = load_prepared_scene(reference.path)
                if _content_digests(reloaded_reference) != _content_digests(reference.scene):
                    raise ValueError(
                        f"Reference scene changed during the build: {example.example_id}",
                    )
        if read_manifest(manifest_path) != tuple(records):
            raise RuntimeError("Published manifest does not reproduce the audited records")
        if any(file_digest(temporary / record.features_path) != record.features_digest
               for record in records):
            raise RuntimeError("A feature shard changed before publication")
        temporary.rename(destination)
        return audit
    except BaseException:
        try:
            _remove_staging(temporary, destination.parent)
        except BaseException as cleanup_error:
            print(f"Staging cleanup failed and was left at {temporary}: {cleanup_error}",
                  file=sys.stderr)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--backbone", type=Path, required=True,
                        help="Complete local WavLM speaker checkpoint directory")
    parser.add_argument("--device", required=True,
                        help="Torch device for the local encoder, for example cpu or cuda:0")
    args = parser.parse_args()
    encoder = WavLMSpeakerFeatures(args.backbone, device=args.device)
    print(json.dumps(build_dataset(args.config, args.destination, encoder=encoder),
                     ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
