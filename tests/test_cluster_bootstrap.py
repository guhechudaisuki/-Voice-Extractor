import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extractor.nextgen.bootstrap_bank import (BootstrapVector, independent_vectors,
    merge_profile, load, save, deduplicate)
from extractor.nextgen.bootstrap_training import analyze_with_bootstrap, collect_vectors
from extractor.nextgen.cluster_runtime import ClusterAnalyzer
from extractor.nextgen.identity_cluster import TargetProfile
from extractor.pipeline import ExtractionPipeline, PipelineOptions
from extractor.types import CandidateSentence, TimeSpan


def vector(degrees):
    angle = np.deg2rad(degrees)
    return np.array([np.cos(angle), np.sin(angle)])


def synthetic_analyzer():
    """Real clustering/local gates, deterministic speaker embeddings, no corpus."""
    analyzer = ClusterAnalyzer.__new__(ClusterAnalyzer)
    analyzer.profile = TargetProfile(vector(0), vector(0), .68, .64)
    analyzer.negatives = []
    analyzer.source_hash = "only-input-file"
    analyzer.progress = MagicMock()
    analyzer.bank_context = lambda: {"references": ["user-reference"], "models": ["p", "s"]}
    analyzer.bootstrap_provenance = None
    analyzer.close = MagicMock()
    def embed(start, end):
        # Whole-turn evidence identifies the target, but a local vocal change
        # fails the original reference and needs a different, independent turn.
        angle = (45 if end - start > 2 else 60) if start >= 8 else 30
        return vector(angle), vector(angle)
    analyzer.embed = embed
    return analyzer


SPANS = [(0, 3), (4, 7), (8, 11)]


class SingleFileBootstrapTests(unittest.TestCase):
    def test_one_file_no_history_or_exclusion_references_actually_changes_decision(self):
        analyzer = synthetic_analyzer()
        _, baseline, _ = analyzer.analyze(SPANS, require_dominant=False)
        self.assertEqual(baseline.membership, (True, True, False))
        original = analyzer.profile
        with tempfile.TemporaryDirectory() as directory:
            _, result, local = analyze_with_bootstrap(analyzer, SPANS, output_dir=directory)
            self.assertEqual(result.membership, (True, True, True))
            self.assertTrue(all(row["reference"] == "bootstrap" for row in local["2"]))
            self.assertEqual(result.diagnostics["bootstrap_supports"]["2"],
                             (("only-input-file", 0., 3.),))
            report = json.loads((Path(directory) / "bootstrap_report.json").read_text())
            self.assertEqual(report["status"], "converged_no_new_turns")
            self.assertEqual([r["new_accepted_turns"] for r in report["rounds"]], [1, 0])
            self.assertEqual(report["scope"], "current_file_only")
            self.assertEqual(report["added_turns"], 1)
            meta, bank = load(Path(directory) / report["bank"])
            self.assertEqual(meta["context"]["identity"], analyzer.bank_context())
            self.assertTrue(all(row.source_sha256 == "only-input-file" for row in bank))
        self.assertIs(analyzer.profile, original)
        # A subsequent disabled run has no bank carried over from this run.
        self.assertEqual(analyzer.analyze(SPANS, require_dominant=False)[1].membership, baseline.membership)

    def test_current_turn_and_overlaps_cannot_certify_themselves(self):
        analyzer = synthetic_analyzer()
        turns, candidates, _ = analyzer.analyze(SPANS, temporal=False, require_dominant=False)
        for start, end in [(8, 11), (7.5, 9)]:
            bank = [BootstrapVector(analyzer.source_hash, start, end, end-start, vector(60), vector(60))]
            result, _ = analyzer.verify_temporal(turns, candidates, bank=bank)
            self.assertFalse(result.membership[2])
        independent = [BootstrapVector(analyzer.source_hash, 0, 3, 3, vector(30), vector(30))]
        result, _ = analyzer.verify_temporal(turns, candidates, bank=independent)
        self.assertTrue(result.membership[2])

    def test_descendant_evidence_cannot_feed_back_to_its_ancestor(self):
        bank = [BootstrapVector("file", 10, 12, 2, vector(30), vector(30),
                                supports=(("file", 1., 3.), ("file", 5., 7.)))]
        self.assertEqual(independent_vectors(bank, "file", 2, 4), [])
        self.assertEqual(independent_vectors(bank, "file", 6, 8), [])
        self.assertEqual(independent_vectors(bank, "file", 8, 10), bank)

    def test_no_independent_seed_keeps_baseline_and_saves_no_fake_bank(self):
        analyzer = synthetic_analyzer()
        baseline = analyzer.analyze(SPANS, require_dominant=False)[1]
        with tempfile.TemporaryDirectory() as directory:
            _, result, _ = analyze_with_bootstrap(analyzer, SPANS, output_dir=directory,
                                                  blocked_spans=[(0, 7)])
            self.assertEqual(result.membership, baseline.membership)
            self.assertEqual(analyzer.bootstrap_provenance["status"], "no_eligible_evidence")
            self.assertEqual(analyzer.bootstrap_provenance["rounds"], [])
            self.assertEqual(list(Path(directory).glob("*.npz")), [])

    def test_no_target_and_below_seed_score_do_not_expand(self):
        for angle in [0, 12]:
            analyzer = synthetic_analyzer()
            if angle:
                analyzer.profile = replace(analyzer.profile,
                    primary_centroid=vector(-angle), secondary_centroid=vector(-angle))
                spans = SPANS
            else:
                spans = []
            with tempfile.TemporaryDirectory() as directory:
                analyze_with_bootstrap(analyzer, spans, output_dir=directory)
                self.assertEqual(analyzer.bootstrap_provenance["vector_count"], 0)
                self.assertFalse(analyzer.bootstrap_provenance["qualification"]["qualified"])

    def test_exclusion_margin_in_either_channel_blocks_collection(self):
        analyzer = synthetic_analyzer()
        turns, result, _ = analyzer.analyze(SPANS, require_dominant=False)
        for channel in (0, 1):
            negative = [vector(180), vector(180)]
            negative[channel] = vector(30)
            analyzer.negatives = [("other", *negative)]
            rows, details = collect_vectors(analyzer, turns, result)
            self.assertFalse(rows)
            self.assertFalse(details["qualified"])

    def test_max_rounds_is_bounded_without_lowering_thresholds(self):
        analyzer = synthetic_analyzer()
        with tempfile.TemporaryDirectory() as directory:
            analyze_with_bootstrap(analyzer, SPANS, output_dir=directory, max_rounds=1)
            report = analyzer.bootstrap_provenance
            self.assertEqual(report["status"], "max_rounds_reached")
            self.assertEqual(len(report["rounds"]), 1)
            self.assertEqual(report["thresholds"]["primary_floor"], .68)
            for value in (0, 11, 1.5, True):
                with self.assertRaises(ValueError):
                    analyze_with_bootstrap(analyzer, SPANS, output_dir=directory, max_rounds=value)

    def test_channels_cannot_mix_original_and_expanded_profiles(self):
        analyzer = synthetic_analyzer()
        turns, candidates, _ = analyzer.analyze(SPANS, temporal=False, require_dominant=False)
        candidates = replace(candidates, clusters=tuple(
            replace(c, secondary_centroid=vector(-10)) for c in candidates.clusters))
        # Original passes only secondary, expanded passes only primary.
        analyzer.embed = lambda a, b: (vector(60), vector(-40))
        bank = [BootstrapVector("only-input-file", 0, 3, 3, vector(30), vector(30))]
        result, _ = analyzer.verify_temporal(turns, candidates, bank=bank)
        self.assertFalse(result.membership[2])

    def test_windows_cannot_mix_original_and_expanded_profiles(self):
        analyzer = synthetic_analyzer()
        turns, candidates, _ = analyzer.analyze(SPANS, temporal=False, require_dominant=False)
        candidates = replace(candidates, clusters=tuple(replace(c,
            primary_centroid=vector(10), secondary_centroid=vector(10), primary_median=.6)
            for c in candidates.clusters))
        analyzer.embed = lambda a, b: (vector(-40 if a < 8.5 else 60),) * 2
        bank = [BootstrapVector("only-input-file", 0, 3, 3, vector(30), vector(30))]
        result, _ = analyzer.verify_temporal(turns, candidates, bank=bank)
        # 整轮不得因扩充参考而放行（无混合证书）；只有原始参考通过的窗口
        # 可以作为干净子区间保留。
        spans = result.diagnostics["turn_clean_spans"]["2"]
        self.assertNotIn([8.0, 11.0], spans)
        self.assertTrue(all(end <= 9.8 + 1e-9 for _, end in spans))
        self.assertNotIn("2", result.diagnostics.get("bootstrap_supports", {}))

    def test_profile_merge_is_duration_weighted_and_keeps_original_floors(self):
        profile = synthetic_analyzer().profile
        bank = [BootstrapVector("s", 0, 1, 1, vector(0), vector(0)),
                BootstrapVector("s", 2, 5, 3, vector(60), vector(60))]
        merged = merge_profile(profile, bank)
        center = vector(0) + (vector(0) + 3 * vector(60)) / np.linalg.norm(vector(0) + 3 * vector(60))
        np.testing.assert_allclose(merged.primary_centroid, center / np.linalg.norm(center), atol=1e-7)
        self.assertEqual(merged.primary_reference_floor, profile.primary_reference_floor)
        self.assertIs(merge_profile(profile, []), profile)
        with self.assertRaises(ValueError):
            merge_profile(profile, [replace(bank[0], primary=np.ones(3))])

    def test_bank_roundtrip_preserves_dependencies_and_rejects_truncated_arrays(self):
        row = BootstrapVector("s", 0, 2, 2, vector(30), vector(30), .9, (("s", 5., 7.),))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.npz"
            save(path, [row], source_ids=["s"], context={"identity": {}})
            self.assertEqual(load(path)[1][0].supports, row.supports)
            with np.load(path, allow_pickle=False) as data:
                fields = {k: data[k] for k in data.files}
            fields["secondary"] = fields["secondary"][:0]
            np.savez_compressed(path, **fields)
            with self.assertRaises(ValueError):
                load(path)

    def test_malformed_vectors_and_evidence_are_rejected(self):
        row = BootstrapVector("s", 0, 2, 2, vector(30), vector(30))
        for invalid in (replace(row, primary=np.zeros(2)), replace(row, primary=np.ones((1, 2))),
                        replace(row, duration=float("nan")), replace(row, confidence=float("inf")),
                        replace(row, supports=(("s", float("nan"), 3),)),
                        replace(row, supports=(("s", 3, 2),))):
            with self.assertRaises(ValueError):
                deduplicate([invalid])
        with self.assertRaises(ValueError):
            deduplicate([row, replace(row, primary=np.ones(3))])

    def test_no_cross_file_state_is_reused(self):
        analyzer = synthetic_analyzer()
        with tempfile.TemporaryDirectory() as directory:
            analyze_with_bootstrap(analyzer, SPANS, output_dir=Path(directory) / "first")
            analyzer.source_hash = "second-file"
            # A completed first file cannot seed a second file with no clean evidence.
            _, result, _ = analyze_with_bootstrap(analyzer, SPANS,
                output_dir=Path(directory) / "second", blocked_spans=[(0, 7)])
            self.assertEqual(result.membership, (True, True, False))
            self.assertEqual(analyzer.bootstrap_provenance["vector_count"], 0)


class BootstrapIntegrationTests(unittest.TestCase):
    def test_pipeline_uses_bootstrap_result_in_final_gate_and_writes_artifacts(self):
        analyzer = synthetic_analyzer()
        analyzer.provenance = lambda: {"bootstrap": analyzer.bootstrap_provenance}
        pipeline = ExtractionPipeline(PipelineOptions(cluster_bootstrap=True, cluster_identity_mode="enforce"))
        accepted = [CandidateSentence(0, 3, "", accepted=True), CandidateSentence(4, 7, "", accepted=True)]
        candidate = CandidateSentence(8, 11, "", reject_reason="声纹匹配不足")
        rejected = [candidate]
        with tempfile.TemporaryDirectory() as directory:
            with patch("extractor.nextgen.cluster_runtime.ClusterAnalyzer", return_value=analyzer), \
                    patch.object(pipeline, "_job_paths", return_value={"output": Path(directory)}):
                pipeline._apply_cluster_identity(Path("input"), [Path("reference")], [], MagicMock(), None,
                    [TimeSpan(a, b) for a, b in SPANS], accepted, rejected, Path("input"), "test", MagicMock())
            self.assertIn(candidate, accepted)
            self.assertTrue(candidate.accepted)
            self.assertEqual(pipeline._cluster_report["changes"]["promoted"], 1)
            self.assertTrue((Path(directory) / "bootstrap" / "bootstrap_report.json").is_file())
            analyzer.close.assert_called_once()

    def test_pipeline_passes_user_detector_and_hard_veto_spans_to_training(self):
        from extractor.nextgen.cluster_gate import HARD_VETOES
        for veto in HARD_VETOES:
            pipeline = ExtractionPipeline(PipelineOptions(cluster_bootstrap=True,
                cluster_identity_mode="enforce", user_excluded_spans=((1, 2),)))
            pipeline._raw_blocked_spans = (TimeSpan(4, 5), TimeSpan(6, 7))
            pipeline._stage_audit = {"stages": {
                "pre_uvr_singing_mask": {"spans": [[12, 13]]},
                "overlap_evidence": {"spans": [[14, 15]]}}}
            rejected = [CandidateSentence(9, 10, "", diagnostics={veto: True})]
            analyzer = synthetic_analyzer()
            analyzer.provenance = lambda: {}
            values = analyzer.analyze(SPANS, require_dominant=False)
            with patch("extractor.nextgen.cluster_runtime.ClusterAnalyzer", return_value=analyzer), \
                    patch("extractor.nextgen.bootstrap_training.analyze_with_bootstrap", return_value=values) as train:
                pipeline._apply_cluster_identity(Path("input"), [], [], MagicMock(), None,
                    [], [], rejected, Path("input"), "test", MagicMock())
                self.assertEqual({tuple(span) for span in train.call_args.kwargs["blocked_spans"]},
                                 {(1, 2), (4, 5), (6, 7), (9, 10), (12, 13), (14, 15)})

    def test_disabled_bootstrap_never_calls_training(self):
        analyzer = synthetic_analyzer()
        analyzer.provenance = lambda: {}
        pipeline = ExtractionPipeline(PipelineOptions(cluster_bootstrap=False))
        with patch("extractor.nextgen.cluster_runtime.ClusterAnalyzer", return_value=analyzer), \
                patch("extractor.nextgen.bootstrap_training.analyze_with_bootstrap") as train:
            pipeline._apply_cluster_identity(Path("input"), [], [], MagicMock(), None,
                [], [], [], Path("input"), "test", MagicMock())
            train.assert_not_called()

    def test_worker_forwards_ui_bootstrap_and_enforce_options(self):
        import desktop_worker
        batch = MagicMock(results=[], accepted=[], rejected=[], batch_id="test")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            path.write_text(json.dumps({"output_root": directory,
                "references": ["reference.wav"], "targets": ["input.wav"],
                "negative_groups": [["other.wav"]], "subtitles": {"input.wav": "input.srt"},
                "options": {"cluster_bootstrap": True, "cluster_bootstrap_max_rounds": 2,
                            "silence_min_seconds": .25, "silence_max_seconds": 1.05}}))
            with patch.object(sys, "argv", ["desktop_worker", str(path)]), \
                    patch.dict("os.environ"), patch.object(desktop_worker, "emit"), \
                    patch("extractor.pipeline.ExtractionPipeline") as factory:
                factory.return_value.run_many.return_value = batch
                self.assertEqual(desktop_worker.main(), 0)
                options = factory.call_args.kwargs["options"]
                self.assertTrue(options.cluster_bootstrap)
                self.assertEqual(options.cluster_identity_mode, "enforce")
                self.assertEqual(options.cluster_bootstrap_max_rounds, 2)
                self.assertEqual(options.silence_min_seconds, .25)
                self.assertEqual(options.silence_max_seconds, 1.05)
                forwarded = factory.return_value.run_many.call_args.kwargs
                self.assertEqual(forwarded["negative_references"], [[Path("other.wav")]])
                self.assertEqual(forwarded["subtitles"], {"input.wav": "input.srt"})

    def test_conflicting_or_invalid_options_are_rejected(self):
        with self.assertRaises(ValueError):
            PipelineOptions(cluster_bootstrap=True, cluster_identity_mode="off")
        with self.assertRaises(ValueError):
            PipelineOptions(cluster_bootstrap=True, export_all_sentences=True)
        for value in (0, 11, True, 1.5):
            with self.assertRaises(ValueError):
                PipelineOptions(cluster_bootstrap_max_rounds=value)


if __name__ == "__main__":
    unittest.main()
