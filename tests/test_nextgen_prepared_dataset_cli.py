from __future__ import annotations

import copy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf
import torch

from extractor.nextgen.features import FeatureSequence, file_digest
from extractor.nextgen.prepared_audio import AlignmentReport
from extractor.nextgen.prepare_media import load_prepared_scene
from extractor.nextgen.scene_adapter import make_scene
from extractor.nextgen.timeline import SampleSpan, SourceTimeline
from training.build_nextgen_prepared import build_dataset
from training.nextgen_data import read_manifest


BACKBONE = "b" * 64
SPLITS = ("train", "development", "calibration", "test")


class TinyEncoder:
    digest = BACKBONE
    geometry = SimpleNamespace(convolution_support=4000)

    def encode(self, waveform, alignment, view_span):
        source_span = alignment.to_source(view_span)
        cells = tuple(
            SampleSpan(start, start + 320)
            for start in range(source_span.start + 2240, source_span.end - 2240, 320)
        )
        values = torch.ones((len(cells), 4), dtype=torch.float32)
        return FeatureSequence(values, cells, alignment.source.source_sha256, self.digest)


def write_scene(directory: Path, seed: int) -> str:
    directory.mkdir(parents=True)
    signal = np.random.default_rng(seed).standard_normal(64000).astype(np.float32) * 0.02
    raw_path = directory / "original_16000.wav"
    stem_path = directory / "stem_16000.wav"
    sf.write(str(raw_path), signal, 16000, subtype="PCM_24")
    sf.write(str(stem_path), signal, 16000, subtype="PCM_24")
    alignment = AlignmentReport(
        file_digest(raw_path), file_digest(stem_path), 0,
        (SampleSpan(5000, 14000), SampleSpan(35000, 45000)),
        (1.0, 1.0), 8,
    )
    source = SourceTimeline(alignment.raw_digest, 16000, len(signal))
    audio = SimpleNamespace(source=source)
    # Build the exact candidate geometry without invoking any detector/model.
    scene = make_scene(
        audio,
        raw_voice=(SampleSpan(16000, 32000),),
        stem_voice=(SampleSpan(16000, 32000),),
    )
    report = {
        "schema": 1,
        "singing_before_uvr": True,
        "source_sha256": source.source_sha256,
        "alignment": asdict(alignment),
        "silence": asdict(scene.silence),
        "raw_voice": [asdict(row) for row in scene.raw_voice],
        "stem_voice": [asdict(row) for row in scene.stem_voice],
        "singing_masks": [],
        "overlap_masks": [],
        "subtitle_hints": [],
        "candidate_count": len(scene.candidates),
        "candidates": [
            {
                "output": asdict(row.output),
                "origin": row.origin,
                "start_complete": row.start_complete,
                "end_complete": row.end_complete,
            }
            for row in scene.candidates
        ],
    }
    (directory / "preparation.json").write_text(json.dumps(report), encoding="utf-8")
    return load_prepared_scene(directory).candidates[0].key


def span(value: SampleSpan) -> dict[str, int]:
    return {"start": value.start, "end": value.end}


def configured_examples(root: Path) -> list[dict]:
    examples = []
    for index, split in enumerate(SPLITS):
        query = root / f"query-{split}"
        reference = root / f"reference-{split}"
        candidate_key = write_scene(query, index * 2 + 1)
        write_scene(reference, index * 2 + 2)
        candidate = load_prepared_scene(query).candidates[0]
        examples.append({
            "id": f"example-{split}",
            "split": split,
            "prepared_scene": query.name,
            "candidate_key": candidate_key,
            "query_source_group": f"source-{split}-query",
            "query_speaker_ids": [f"speaker-{split}"],
            "license_reference": "local human-labelled test fixture",
            "use_allowed": True,
            "references": [{
                "role": "target",
                "prepared_scene": reference.name,
                "source_group": f"source-{split}-reference",
                "speaker_id": f"speaker-{split}",
                "license_reference": "local reference fixture",
                "use_allowed": True,
            }],
            "annotation": {
                "output": span(candidate.output),
                "target": [span(candidate.output)],
                "observable": [span(candidate.output)],
                "certain": [span(candidate.output)],
                "complete_timeline": True,
                "purity": 1,
                "start_complete": 1,
                "end_complete": 1,
            },
        })
    return examples


class PreparedDatasetCliTests(unittest.TestCase):
    def test_builds_all_four_partitions_from_exact_cached_candidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            examples = configured_examples(root)
            config = root / "dataset.json"
            config.write_text(json.dumps({"schema": 1, "examples": examples}), encoding="utf-8")
            output = root / "dataset"

            audit = build_dataset(config, output, encoder=TinyEncoder())

            self.assertEqual(audit["counts"], {split: 1 for split in SPLITS})
            records = read_manifest(output / "manifest.json")
            self.assertEqual({row.split for row in records}, set(SPLITS))
            self.assertTrue(all((output / row.features_path).is_file() for row in records))
            self.assertTrue(all(row.human_labels and not row.synthetic for row in records))

    def test_candidate_key_must_match_even_when_the_span_matches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            examples = configured_examples(root)
            query = load_prepared_scene(root / examples[0]["prepared_scene"])
            examples[0]["candidate_key"] = query.candidates[0].span_key
            config = root / "dataset.json"
            config.write_text(json.dumps({"schema": 1, "examples": examples}), encoding="utf-8")
            output = root / "dataset"

            with self.assertRaisesRegex(ValueError, "Exact candidate key"):
                build_dataset(config, output, encoder=TinyEncoder())

            self.assertFalse(output.exists())

    def test_cross_split_source_speaker_and_content_leakage_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = configured_examples(root)
            train = baseline[0]
            cases = {}

            examples = copy.deepcopy(baseline)
            examples[-1]["query_source_group"] = train["query_source_group"]
            cases["source"] = examples

            examples = copy.deepcopy(baseline)
            examples[-1]["query_speaker_ids"] = train["query_speaker_ids"]
            cases["speaker"] = examples

            examples = copy.deepcopy(baseline)
            examples[-1]["prepared_scene"] = train["prepared_scene"]
            examples[-1]["candidate_key"] = train["candidate_key"]
            examples[-1]["annotation"] = train["annotation"]
            cases["content"] = examples

            for kind, examples in cases.items():
                with self.subTest(kind=kind):
                    config = root / f"dataset-{kind}.json"
                    config.write_text(json.dumps({"schema": 1, "examples": examples}),
                                      encoding="utf-8")
                    output = root / f"dataset-{kind}"
                    with self.assertRaisesRegex(ValueError, f"Cross-split {kind} leakage"):
                        build_dataset(config, output, encoder=TinyEncoder())
                    self.assertFalse(output.exists())

    def test_every_partition_and_reference_requires_explicit_authorization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = configured_examples(root)
            cases = {}

            examples = copy.deepcopy(baseline)
            examples[2]["use_allowed"] = False
            cases["calibration-query"] = examples

            examples = copy.deepcopy(baseline)
            examples[3]["references"][0]["use_allowed"] = False
            cases["test-reference"] = examples

            for name, examples in cases.items():
                with self.subTest(asset=name):
                    config = root / f"dataset-{name}.json"
                    config.write_text(json.dumps({"schema": 1, "examples": examples}),
                                      encoding="utf-8")
                    output = root / f"dataset-{name}"
                    with self.assertRaisesRegex(ValueError, "not authorized"):
                        build_dataset(config, output, encoder=TinyEncoder())
                    self.assertFalse(output.exists())

    def test_subtitle_and_model_pseudo_labels_are_not_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = configured_examples(root)
            for field, value in (("subtitle_text", "speaker says hello"),
                                 ("model_prediction", {"target": 0.99})):
                with self.subTest(field=field):
                    examples = copy.deepcopy(baseline)
                    examples[0]["annotation"][field] = value
                    config = root / f"dataset-{field}.json"
                    config.write_text(json.dumps({"schema": 1, "examples": examples}),
                                      encoding="utf-8")
                    output = root / f"dataset-{field}"
                    with self.assertRaisesRegex(ValueError, f"unknown fields: {field}"):
                        build_dataset(config, output, encoder=TinyEncoder())
                    self.assertFalse(output.exists())

    def test_optional_exclusion_metadata_is_preserved_and_audited(self):
        from safetensors.torch import load_file

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            examples = configured_examples(root)
            exclusion = root / "exclusion-train"
            write_scene(exclusion, 99)
            examples[0]["references"].append({
                "role": "background-speaker",
                "prepared_scene": exclusion.name,
                "source_group": "source-train-exclusion",
                "speaker_id": "speaker-train-exclusion",
                "license_reference": "local exclusion fixture",
                "use_allowed": True,
            })
            config = root / "dataset.json"
            config.write_text(json.dumps({"schema": 1, "examples": examples}), encoding="utf-8")
            output = root / "dataset"

            audit = build_dataset(config, output, encoder=TinyEncoder())

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            provenance = next(row for row in manifest["provenance"]
                              if row["example_id"] == "example-train")
            recorded = next(row for row in provenance["references"]
                            if row["role"] == "background-speaker")
            self.assertEqual(recorded["source_group"], "source-train-exclusion")
            self.assertEqual(recorded["speaker_id"], "speaker-train-exclusion")
            train = next(row for row in read_manifest(output / "manifest.json")
                         if row.example_id == "example-train")
            self.assertIn("source-train-exclusion", train.sources)
            self.assertIn("speaker-train-exclusion", train.speakers)
            self.assertIn("excluded.000.raw", load_file(output / train.features_path))
            self.assertEqual(audit["authorized_assets"], 9)

    def test_windows_ambiguous_paths_and_device_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = configured_examples(root)
            cases = (("device-id", "id", "NUL.json", "portable file-safe"),
                     ("drive-relative", "prepared_scene", "C:query", "fully absolute"),
                     ("root-only", "prepared_scene", "\\query", "fully absolute"))
            for name, field, value, message in cases:
                with self.subTest(case=name):
                    examples = copy.deepcopy(baseline)
                    examples[0][field] = value
                    config = root / f"dataset-{name}.json"
                    config.write_text(json.dumps({"schema": 1, "examples": examples}),
                                      encoding="utf-8")
                    output = root / f"dataset-{name}"
                    with self.assertRaisesRegex(ValueError, message):
                        build_dataset(config, output, encoder=TinyEncoder())
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
