"""Replay the production cluster analyzer on a saved episode's exact turns."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extractor.nextgen.cluster_runtime import ClusterAnalyzer
from extractor.nextgen.cluster_reporting import summarize, load_review
from evaluation.cluster_report import collect_audio
from evaluation.export_episode_embeddings import find_stem
from evaluation.cluster_environment import configure_pipeline_cpu_threads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--exclude-dir", type=Path, required=True)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--require-dominant", action="store_true")
    parser.add_argument("--check-cache-parity", action="store_true",
                        help="Recompute without disk cache and compare all embeddings and decisions")
    args = parser.parse_args()
    if args.no_cache and args.check_cache_parity:
        parser.error("--check-cache-parity requires an initial cached run")
    environment = configure_pipeline_cpu_threads()
    audit_path = next(args.batch.rglob("stage_audit.json"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    stem = find_stem(ROOT / "work", args.batch.name)
    analyzer = ClusterAnalyzer(stem, collect_audio([args.reference]),
                               [[p] for p in collect_audio([args.exclude_dir])],
                               cache_enabled=not args.no_cache,
                               progress=lambda message: print(message, flush=True))
    try:
        turns, result, local = analyzer.analyze(
            audit["stages"]["speaker_turns"]["spans"],
            require_dominant=args.require_dominant,
        )
        report = summarize(result, turns, load_review(args.review, source_sha256=analyzer.source_hash))
        report.update(local_windows=local, provenance=analyzer.provenance(),
                      meta={"batch_id": args.batch.name, "stage_audit": str(audit_path), "source_stem": str(stem)})
        report["provenance"]["runtime_environment"] = environment
        if args.check_cache_parity:
            analyzer.cache.enabled = False
            analyzer.profile, analyzer.negatives = analyzer._profiles()
            fresh_turns, fresh_result, fresh_local = analyzer.analyze(
                audit["stages"]["speaker_turns"]["spans"],
                require_dominant=args.require_dominant,
            )
            max_error = 0.0
            for old, fresh in zip(turns, fresh_turns):
                for name in ("primary", "secondary"):
                    a, b = getattr(old, name), getattr(fresh, name)
                    np.testing.assert_allclose(a, b, rtol=0, atol=1e-6,
                                               err_msg=f"turn {old.index} channel {name}")
                    max_error = max(max_error, float(np.max(np.abs(a - b))))
            assert len(turns) == len(fresh_turns), "turn count changed"
            assert result.membership == fresh_result.membership, "membership changed"
            assert result.cluster_of == fresh_result.cluster_of, "cluster assignment changed"
            assert result.target_cluster_id == fresh_result.target_cluster_id, "target changed"
            assert result.reason == fresh_result.reason, "decision reason changed"
            assert result.diagnostics == fresh_result.diagnostics, "accepted intervals changed"
            decisions = lambda windows: {k: [(r["span"], r["accepted"]) for r in rows]
                                         for k, rows in windows.items()}
            assert decisions(local) == decisions(fresh_local), "local decisions changed"
            report["cache_parity"] = {
                "passed": True, "turn_count": len(turns),
                "maximum_embedding_absolute_error": max_error,
                "membership_and_accepted_intervals_identical": True,
                "scope": "real_audio_cluster_decisions_before_STT",
            }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k not in {"clusters", "local_windows", "provenance", "review_check"}}, ensure_ascii=False))
        print(json.dumps({k:v for k,v in report["review_check"].items() if k != "details"}, ensure_ascii=False))
    finally:
        analyzer.close()


if __name__ == "__main__":
    main()
