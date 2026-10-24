import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extractor.nextgen.embedding_cache import EmbeddingCache, cache_key, file_digest
from extractor.nextgen.cluster_gate import apply_cluster_gate, candidate_membership
from extractor.nextgen.cluster_runtime import window_spans
from extractor.nextgen.bootstrap_bank import BootstrapVector, deduplicate, save, load
from extractor.nextgen.identity_cluster import AssignResult, TurnEvidence
from extractor.types import CandidateSentence
from extractor.pipeline import PipelineOptions, ExtractionPipeline


class ClusterRuntimeTests(unittest.TestCase):
    def test_bootstrap_bank_deduplicates_with_provenance(self):
        a = np.array([1., 0., 0.]); b = np.array([.99, .1, 0.])
        rows = [BootstrapVector("dev-a", 0, 2, 2, a, a, .8),
                BootstrapVector("dev-b", 2, 6, 4, b, b, .7)]
        selected = deduplicate(rows)
        self.assertEqual(len(selected), 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.npz"
            meta = save(path, rows, source_ids=["dev-a", "dev-b"])
            loaded_meta, loaded = load(path)
            self.assertEqual(meta["count"], loaded_meta["count"])
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded_meta["purpose"], "development_target_evidence_only")

    def test_audit_failure_preserves_extraction_but_enforce_fails_closed(self):
        candidate = CandidateSentence(1, 4, "", accepted=True)
        for mode in ("off", "audit", "enforce"):
            pipeline = ExtractionPipeline(PipelineOptions(cluster_identity_mode=mode))
            accepted, rejected = [candidate], []
            def run():
                pipeline._apply_cluster_identity(Path("stem.wav"), [], [], MagicMock(), None,
                    [], accepted, rejected, Path("source.wav"), "job", lambda *args: None)
            with patch("extractor.nextgen.cluster_runtime.ClusterAnalyzer", side_effect=OSError("cache denied")) as factory:
                if mode == "enforce":
                    with self.assertRaises(OSError): run()
                else:
                    run()
                    if mode == "audit": self.assertEqual(pipeline._cluster_report["reason"], "audit_failed")
                    else: factory.assert_not_called()
            self.assertEqual(accepted, [candidate])
            self.assertEqual(rejected, [])

    def test_cache_is_exact_and_content_bound(self):
        self.assertNotEqual(cache_key("a", 1, 2, "m"), cache_key("b", 1, 2, "m"))
        self.assertNotEqual(cache_key("a", 1, 2, "m"), cache_key("a", 1, 2, "n"))
        self.assertNotEqual(cache_key("a", 1, 2, "m"), cache_key("a", 1.001, 2, "m"))
        with tempfile.TemporaryDirectory() as directory:
            cache = EmbeddingCache(Path(directory))
            vectors = (np.array([0.6, 0.8]), np.array([1., 0., 0.]))
            first = cache.get_or_compute("abcdef", lambda: vectors)
            second = cache.get_or_compute("abcdef", lambda: self.fail("cache missed"))
            for a, b in zip(first, second):
                np.testing.assert_array_equal(a, b)
            self.assertEqual((cache.hits, cache.misses), (1, 1))
            file = Path(directory) / "source"
            file.write_bytes(b"old")
            old = file_digest(file)
            file.write_bytes(b"new")
            self.assertNotEqual(old, file_digest(file))

    def test_cache_enabled_disabled_parity_and_fingerprint_invalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vectors = (np.array([0.6, 0.8]), np.array([1., 0., 0.]))
            key = cache_key("source-v1", 1, 2, "model-v1", 16000)
            enabled = EmbeddingCache(root / "enabled", enabled=True)
            disabled = EmbeddingCache(root / "disabled", enabled=False)
            cached = enabled.get_or_compute(key, lambda: vectors)
            uncached = disabled.get_or_compute(key, lambda: vectors)
            for actual, expected in zip(cached, uncached):
                np.testing.assert_array_equal(actual, expected)
            self.assertNotEqual(
                cache_key("source-v1", 1, 2, "model-v1", 16000),
                cache_key("source-v2", 1, 2, "model-v1", 16000),
            )
            self.assertNotEqual(
                cache_key("source-v1", 1, 2, "model-v1", 16000),
                cache_key("source-v1", 1, 2, "model-v2", 16000),
            )
            self.assertNotEqual(
                cache_key("source-v1", 1, 2, "model-v1", 16000),
                cache_key("source-v1", 1, 2, "model-v1", 8000),
            )

    def test_analyzer_invalidates_content_changes_and_keeps_role_order_stable(self):
        import torch
        from extractor.nextgen.cluster_runtime import ClusterAnalyzer
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stem, primary_model, secondary_model, reference, role_a, role_b = [
                root / name for name in ("stem.wav", "p.pt", "s.pt", "ref.wav", "a.wav", "b.wav")]
            for path in (stem, primary_model, secondary_model, reference, role_a, role_b):
                path.write_bytes(path.name.encode())
            encoders = (MagicMock(), MagicMock())
            for encoder in encoders:
                encoder.embedding_from_file.return_value = torch.tensor([0.6, 0.8])
                encoder._embedding_from_waveform.return_value = torch.tensor([0.6, 0.8])
            with patch.multiple("extractor.config", CACHE_ROOT=root / "cache",
                                SV_MODEL=primary_model, CAMPLUS_MODEL=secondary_model):
                def analyze(groups):
                    return ClusterAnalyzer(stem, [reference], groups, device="cpu", encoders=encoders)
                first = analyze([[role_a], [role_b]])
                reordered = analyze([[role_b], [role_a]])
                self.assertEqual([r[0] for r in first.negatives], [r[0] for r in reordered.negatives])
                self.assertEqual(reordered.cache.misses, 0)
                reference.write_bytes(b"same-path-new-reference")
                changed_reference = analyze([[role_a], [role_b]])
                self.assertEqual(changed_reference.cache.misses, 1)
                primary_model.write_bytes(b"same-path-new-model")
                changed_model = analyze([[role_a], [role_b]])
                self.assertNotEqual(first.model_hash, changed_model.model_hash)
                self.assertEqual(changed_model.cache.misses, 3)
                stem.write_bytes(b"same-path-new-audio")
                changed_audio = analyze([[role_a], [role_b]])
                self.assertNotEqual(changed_model.source_hash, changed_audio.source_hash)
                for analyzer in (changed_model, changed_audio):
                    analyzer.waveform = np.zeros(16000)
                    analyzer.embed(0, 1)
                self.assertEqual(encoders[0]._embedding_from_waveform.call_count, 2)

    def evidence(self):
        turns = [TurnEvidence(i, a, b, b-a, np.ones(2), np.ones(2))
                 for i, (a,b) in enumerate([(0,2),(2,4),(5,7)])]
        result = AssignResult((), 0, (True, False, True), (0,1,0), "target_cluster_found")
        return turns, result

    def test_merge_cannot_cross_other_speaker_or_gap(self):
        turns, result = self.evidence()
        self.assertFalse(candidate_membership(CandidateSentence(1,3,""), turns, result)[0])
        self.assertFalse(candidate_membership(CandidateSentence(4,6,""), turns, result)[0])
        self.assertTrue(candidate_membership(CandidateSentence(.3,1.7,""), turns, result)[0])

    def test_audit_preserves_output_and_enforce_is_bidirectional(self):
        turns, result = self.evidence()
        original = CandidateSentence(2,4,"",accepted=True)
        recover = CandidateSentence(5,7,"",reject_reason="声纹匹配不足")
        accepted, rejected = [original], [recover]
        apply_cluster_gate(accepted,rejected,turns,result,mode="audit")
        self.assertEqual(accepted,[original]); self.assertEqual(rejected,[recover])
        self.assertTrue(original.accepted)
        apply_cluster_gate(accepted,rejected,turns,result,mode="enforce")
        self.assertEqual(accepted,[recover]); self.assertEqual(rejected,[original])
        self.assertTrue(recover.accepted); self.assertFalse(original.accepted)

    def test_hard_veto_cannot_be_promoted(self):
        turns, result = self.evidence()
        candidate = CandidateSentence(5,7,"",reject_reason="声纹匹配不足",
                                      diagnostics={"excluded_role_rejected": True})
        accepted, rejected = [], [candidate]
        apply_cluster_gate(accepted,rejected,turns,result,mode="enforce")
        self.assertFalse(accepted)

    def test_explicit_user_exclusion_cannot_be_promoted(self):
        turns, result = self.evidence()
        candidate = CandidateSentence(5,7,"",reject_reason="声纹匹配不足")
        accepted, rejected = [], [candidate]
        apply_cluster_gate(accepted,rejected,turns,result,mode="enforce",blocked_spans=[(6,6.5)])
        self.assertFalse(accepted)

    def test_crop_does_not_duplicate_retained_child(self):
        turns, result = self.evidence()
        from dataclasses import replace
        result = replace(result, diagnostics={"accepted_spans": [[0,2]]})
        parent = CandidateSentence(0,4,"",accepted=True)
        child = CandidateSentence(0,2,"",accepted=True)
        accepted, rejected = [parent,child], []
        apply_cluster_gate(accepted,rejected,turns,result,mode="enforce")
        self.assertEqual(accepted,[child])

    def test_windows_cover_exact_tail(self):
        spans = window_spans(1,8.13)
        self.assertAlmostEqual(spans[0][0],1)
        self.assertAlmostEqual(spans[-1][1],8.13)

    def test_invalid_mode_fails(self):
        with self.assertRaises(ValueError):
            PipelineOptions(cluster_identity_mode="enfore")


if __name__ == "__main__":
    unittest.main()
