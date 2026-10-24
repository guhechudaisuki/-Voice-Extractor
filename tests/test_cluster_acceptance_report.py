import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.cluster_acceptance_report import completed_batch, review_queue, compare_inputs


class ClusterAcceptanceReportTests(unittest.TestCase):
    def batch(self, root):
        (root / "episode").mkdir()
        (root / "episode" / "clip.wav").write_bytes(b"audio fixture")
        manifest = {"target_count": 1, "accepted_count": 1, "sentences": [
            {"accepted": True, "start": 1, "end": 3, "audio_file": "episode/clip.wav"}]}
        provenance = {"accepted_count": 1, "options": {"cluster_identity_mode": "enforce"},
                      "inputs_sha256": {"target": "a" * 64}}
        manifest["options"] = provenance["options"]
        cluster = {"mode": "enforce", "final_output_count": 1}
        for path, data in ((root / "batch_manifest.json", manifest),
                           (root / "evaluation_provenance.json", provenance),
                           (root / "episode" / "cluster_report.json", cluster)):
            path.write_text(json.dumps(data), encoding="utf-8")
        return manifest, provenance

    def test_refuses_inconsistent_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self.batch(root)
            manifest["accepted_count"] = 2
            (root / "batch_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Inconsistent"):
                completed_batch(root)

    def test_refuses_missing_exported_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.batch(root)
            (root / "episode" / "clip.wav").unlink()
            with self.assertRaisesRegex(ValueError, "Missing exported audio"):
                completed_batch(root)

    def test_queue_never_assigns_machine_labels_as_human_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.batch(root)
            manifest, provenance, _, evidence = completed_batch(root)
            queue = review_queue(root, manifest, provenance, evidence)
            self.assertEqual(queue["status"], "pending_human_listening")
            self.assertEqual(queue["clips"][0]["source_id"], "a" * 64 + ":16000:48000")
            self.assertTrue(all(value is None for value in queue["clips"][0]["human_review"].values()))

    def test_different_sources_rejected_even_when_cached_audio_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, new = root / "output" / "before", root / "output" / "after"
            old.mkdir(parents=True)
            new.mkdir(parents=True)
            old_provenance = {"inputs_sha256": {"target": "a" * 64}, "options": {"speaker_threshold": .68}}
            new_provenance = {"inputs_sha256": {"target": "b" * 64}, "options": {"speaker_threshold": .68}}
            (old / "evaluation_provenance.json").write_text(json.dumps(old_provenance), encoding="utf-8")
            for batch in (old, new):
                work = root / "work" / (batch.name + "_001")
                work.mkdir(parents=True)
                (work / "target_normalized.wav").write_bytes(b"same PCM and header")
            with self.assertRaisesRegex(ValueError, "source files differ"):
                compare_inputs(old, new, new_provenance)

    def test_same_source_does_not_hide_other_parameter_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provenance = {"inputs_sha256": {"target": "a" * 64}, "options": {"speaker_threshold": .68}}
            (root / "evaluation_provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
            self.assertEqual(compare_inputs(root, root, provenance)["method"], "identical_source_file")
            changed = {**provenance, "options": {"speaker_threshold": .5}}
            with self.assertRaisesRegex(ValueError, "Non-cluster options differ"):
                compare_inputs(root, root, changed)


if __name__ == "__main__":
    unittest.main()
