"""Replay one file's canonical boundaries; history is only a test fixture.

The production bootstrap entry takes current audio/turns directly and never
reads a batch, annotation, another episode or this evaluation module.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.cluster_environment import configure_pipeline_cpu_threads
from extractor.nextgen.bootstrap_training import analyze_with_bootstrap
from extractor.nextgen.cluster_gate import HARD_VETOES
from extractor.nextgen.cluster_runtime import ClusterAnalyzer
from extractor.nextgen.cluster_reporting import summarize
from extractor.nextgen.embedding_cache import file_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--single-reference", action="store_true")
    parser.add_argument("--with-exclusions", action="store_true")
    args = parser.parse_args()
    fixture = json.loads(args.cluster_report.read_text(encoding="utf-8"))
    audit_path = args.cluster_report.parent / "stage_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    manifest = json.loads((args.cluster_report.parent.parent / "batch_manifest.json").read_text(encoding="utf-8"))
    blocked = list(manifest.get("options", {}).get("user_excluded_spans", []))
    for name in ("pre_uvr_singing_mask", "residual_singing_evidence", "singing_blocked_islands",
                 "overlap_evidence", "overlap_blocked_islands"):
        blocked.extend(audit["stages"].get(name, {}).get("spans", []))
    blocked.extend((s["start"], s["end"]) for s in manifest["sentences"]
        if s["target"] == fixture["meta"]["source_name"]
        and any(s.get("diagnostics", {}).get(key) for key in HARD_VETOES))
    references = sorted(fixture["provenance"]["reference_paths"])
    if args.single_reference:
        references = references[:1]
    negatives = fixture["provenance"]["exclusion_groups"] if args.with_exclusions else []
    environment = configure_pipeline_cpu_threads()
    analyzer = ClusterAnalyzer(fixture["meta"]["source_stem"], references, negatives,
                               progress=lambda message: print(message, flush=True))
    try:
        if analyzer.source_hash != fixture["provenance"]["source_sha256"]:
            raise ValueError("fixture audio fingerprint changed")
        spans = audit["stages"]["speaker_turns"]["spans"]
        baseline = analyzer.analyze(spans, require_dominant=False)[1]
        turns, result, local = analyze_with_bootstrap(analyzer, spans,
            output_dir=args.output.parent / args.output.stem, blocked_spans=blocked)
        report = summarize(result, turns, [])
        report.update(local_windows=local, provenance=analyzer.provenance(),
            test_inputs={"file_count": 1, "reference_count": len(references),
                "exclusion_group_count": len(negatives), "human_labels_used": False,
                "stage_audit_sha256": file_digest(audit_path), "environment": environment},
            comparison={"baseline_turns": sum(baseline.membership), "bootstrap_turns": sum(result.membership),
                "target_unchanged": baseline.target_cluster_id == result.target_cluster_id,
                "baseline_preserved": all(not old or new for old, new in zip(baseline.membership, result.membership)),
                "scope": "cluster_local_identity_before_STT_not_final_export_or_accuracy"})
        assert report["comparison"]["target_unchanged"]
        assert report["comparison"]["baseline_preserved"]
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report["comparison"], ensure_ascii=False), flush=True)
    finally:
        analyzer.close()


if __name__ == "__main__":
    main()
