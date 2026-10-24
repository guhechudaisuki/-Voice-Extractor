"""Measure P4 retention and prepare P6 listening queues from completed batches.

Example: --development 1 output/baseline output/enforce --output docs/report.json
Held-out machine completion never implies that the human listening gate passed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def artifact(path: Path) -> dict:
    return {"path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def completed_batch(directory: Path) -> tuple[dict, dict, dict, dict]:
    manifest_path = directory / "batch_manifest.json"
    provenance_path = directory / "evaluation_provenance.json"
    manifest, provenance = read(manifest_path), read(provenance_path)
    reports = list(directory.rglob("cluster_report.json"))
    if len(reports) != 1 or manifest.get("target_count") != 1:
        raise ValueError(f"Expected exactly one completed target: {directory}")
    cluster = read(reports[0])
    count = sum(bool(row["accepted"]) for row in manifest["sentences"])
    if not (count == manifest["accepted_count"] == provenance["accepted_count"]
            == cluster["final_output_count"]):
        raise ValueError(f"Inconsistent output counts: {directory}")
    if cluster["mode"] != "enforce" or provenance["options"]["cluster_identity_mode"] != "enforce":
        raise ValueError(f"Expected enforce: {directory}")
    if manifest.get("options") != provenance["options"]:
        raise ValueError(f"Manifest and provenance options differ: {directory}")
    for row in manifest["sentences"]:
        if row["accepted"] and not (directory / row["audio_file"]).is_file():
            raise ValueError(f"Missing exported audio: {row['audio_file']}")
    evidence = {"manifest": artifact(manifest_path), "provenance": artifact(provenance_path),
                "cluster_report": artifact(reports[0])}
    return manifest, provenance, cluster, evidence


def freeze_signature(provenance: dict, cluster: dict) -> dict:
    inputs = provenance["inputs_sha256"]
    return {"code": provenance["code_sha256_at_start"], "options": provenance["options"],
            "references": sorted(inputs["references"]),
            "negative_groups": sorted(sorted(group) for group in inputs["negative_groups"]),
            "models": [item["sha256"] for item in cluster["provenance"]["inputs"][1:3]],
            "encoding_version": cluster["provenance"]["encoding_version"]}


def compare_inputs(baseline: Path, current: Path, provenance: dict) -> dict:
    old = read(baseline / "evaluation_provenance.json")
    old_inputs, inputs = old["inputs_sha256"], provenance["inputs_sha256"]
    for key in ("subtitle", "references", "negative_groups", "excluded_spans"):
        if old_inputs.get(key) != inputs.get(key):
            raise ValueError(f"Baseline and current {key} differ")
    old_options = {k: v for k, v in old["options"].items() if not k.startswith("cluster_")}
    new_options = {k: v for k, v in provenance["options"].items() if not k.startswith("cluster_")}
    if old_options != new_options:
        raise ValueError("Non-cluster options differ from baseline")
    result = {"baseline_target_sha256": old_inputs["target"], "current_target_sha256": inputs["target"]}
    if old_inputs["target"] != inputs["target"]:
        # Derived work audio may be masked or regenerated. It cannot establish
        # that two different original inputs constitute the same experiment.
        raise ValueError("Baseline and current source files differ; rerun the same source")
    return {**result, "method": "identical_source_file"}


def review_queue(directory: Path, manifest: dict, provenance: dict, evidence: dict) -> dict:
    source = provenance["inputs_sha256"]["target"]
    clips = []
    for row in manifest["sentences"]:
        if not row["accepted"]:
            continue
        start, end = float(row["start"]), float(row["end"])
        clips.append({
            "source_id": f"{source}:{round(start * 16000)}:{round(end * 16000)}",
            "start": start, "end": end,
            "audio_file": str((directory / row["audio_file"]).resolve()),
            "text": row.get("text", ""),
            "human_review": {"wrong_person": None, "singing": None, "overlap": None,
                             "notes": None},
        })
    return {"schema_version": 1, "status": "pending_human_listening",
            "source_sha256": source, "evidence": evidence, "clips": clips,
            "warning": "Machine decisions are not human labels. Do not tune on held-out results."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", action="append", nargs=3, default=[],
                        metavar=("EPISODE", "BASELINE", "ENFORCE"))
    parser.add_argument("--held-out", action="append", nargs=2, default=[],
                        metavar=("EPISODE", "ENFORCE"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    development, held_out, signatures = [], [], []
    for episode, baseline, current in args.development:
        baseline_path, current_path = Path(baseline) / "batch_manifest.json", Path(current)
        old = read(baseline_path)
        manifest, provenance, cluster, evidence = completed_batch(current_path)
        comparison = compare_inputs(Path(baseline), current_path, provenance)
        original = old["accepted_count"]
        if original != sum(bool(row["accepted"]) for row in old["sentences"]):
            raise ValueError("Baseline count disagrees with its sentence records")
        if original <= 0:
            raise ValueError("A non-empty baseline is required")
        final = manifest["accepted_count"]
        signatures.append(freeze_signature(provenance, cluster))
        development.append({
            "episode": int(episode), "baseline_count": original, "final_count": final,
            "retention": final / original, "minimum_retention": 0.40,
            "retention_passed": final / original >= 0.40,
            "target_cluster_id": cluster["target_cluster_id"], "reason": cluster["reason"],
            "accepted_turn_count": cluster["accepted_turn_count"], "changes": cluster["changes"],
            "comparison_inputs": comparison,
            "baseline": artifact(baseline_path), "evidence": evidence,
        })
    for episode, current in args.held_out:
        if int(episode) not in {6, 9, 12}:
            raise ValueError("The frozen held-out set is episodes 6, 9 and 12")
        directory = Path(current)
        manifest, provenance, cluster, evidence = completed_batch(directory)
        signatures.append(freeze_signature(provenance, cluster))
        queue_path = directory / "cluster_listening_queue.json"
        # Human fields may already have been filled in; never overwrite a queue.
        if not queue_path.exists():
            queue_path.write_text(json.dumps(review_queue(directory, manifest, provenance, evidence),
                                            ensure_ascii=False, indent=2), encoding="utf-8")
        held_out.append({
            "episode": int(episode), "final_count": manifest["accepted_count"],
            "target_cluster_id": cluster["target_cluster_id"], "reason": cluster["reason"],
            "changes": cluster["changes"], "evidence": evidence,
            "listening_queue": artifact(queue_path), "human_acceptance": "pending",
            "nonempty_output": manifest["accepted_count"] > 0,
        })
    uniform = bool(signatures) and all(s == signatures[0] for s in signatures)
    p4_complete = sorted(r["episode"] for r in development) == [1, 2, 3]
    p4_passed = p4_complete and uniform and all(r["retention_passed"] for r in development)
    report = {
        "schema_version": 1,
        "scope": "development_retention_and_held_out_machine_execution_only",
        "p4_status": "passed" if p4_passed else "failed" if p4_complete else "incomplete",
        "p4_passed": p4_passed, "development": development,
        "frozen_code_options_and_references_equal": uniform,
        "held_out_machine_complete": sorted(r["episode"] for r in held_out) == [6, 9, 12],
        "held_out": held_out, "p6_acceptance": "pending_human_listening" if held_out else "not_run",
        "limitations": ["Retention is a clip count ratio, not recall or accuracy.",
                        "No human wrong-person, singing, or overlap claims are made.",
                        "An empty output cannot establish useful held-out accuracy."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"development", "held_out"}}))
    return 0 if p4_passed and uniform else 2


if __name__ == "__main__":
    raise SystemExit(main())
