import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from launcher import webui_server as server
from launcher.review_store import review_path


class ClusterWebUITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / 'output'
        self.root.mkdir()
        self.batch = self.root / "batch"
        self.batch.mkdir()
        self.docs = self.root / "docs"
        self.docs.mkdir()
        self.patches = [patch.object(server,"OUTPUT_DIR",self.root), patch.object(server,"ROOT_DOCS",self.docs)]
        for item in self.patches: item.start()
        self.addCleanup(self.temporary.cleanup)
        for item in self.patches: self.addCleanup(item.stop)

    def report(self, target="episode"):
        directory=self.batch/target
        directory.mkdir(exist_ok=True)
        (directory/"cluster_report.json").write_text(json.dumps({
            "target_cluster_id":0,"clusters":[{"cluster_id":0,"is_target":True,"member_count":3,"members":[{"index":0,"span":[1,3]}]}],
            "review_check":{"verdict":"no_target_cluster","whole_wrong_withheld_no_target":10},
        }),encoding="utf-8")

    def test_legacy_batch_degrades_without_report(self):
        self.assertFalse(server.get_batch_clusters("batch")["available"])

    def test_override_roundtrip_and_review_save_preserve_each_other(self):
        self.report()
        current=server.get_batch_clusters("batch")
        server.save_cluster_override("batch",server.ClusterOverridePayload(
            cluster_id=0,target_dir="episode",report_sha256=current["report_sha256"]))
        server.save_batch_review("batch",server.BatchReviewPayload(batch_id="batch",reviews={}))
        self.assertEqual(server.get_batch_clusters("batch")["override"]["cluster_id"],0)
        self.assertIn("cluster_override",json.loads(review_path(self.batch).read_text(encoding="utf-8")))

    def test_invalid_or_stale_override_rejected(self):
        self.report()
        current=server.get_batch_clusters("batch")
        with self.assertRaises(server.HTTPException):
            server.save_cluster_override("batch",server.ClusterOverridePayload(
                cluster_id=9,target_dir="episode",report_sha256=current["report_sha256"]))
        with self.assertRaises(server.HTTPException):
            server.save_cluster_override("batch",server.ClusterOverridePayload(
                cluster_id=0,target_dir="episode",report_sha256="stale"))

    def test_multiple_targets_have_independent_reports(self):
        self.report("one"); self.report("two")
        report=server.get_batch_clusters("batch","two")
        self.assertEqual(report["target_dir"],"two")
        self.assertEqual(report["targets"],["one","two"])

    def test_foreign_historical_report_not_selected(self):
        (self.docs/"cluster_batch.json").write_text(json.dumps({"meta":{"batch_id":"other"},"clusters":[]}))
        self.assertFalse(server.get_batch_clusters("batch")["available"])

    def test_cluster_removed_audio_is_reviewable_without_changing_manifest(self):
        directory = self.batch / "episode" / "cluster_outliers"
        directory.mkdir(parents=True)
        clip = directory / "0001.wav"
        clip.write_bytes(b"audio fixture")
        clip.with_suffix(".txt").write_text("1.000–3.000\ncluster veto", encoding="utf-8")
        manifest = {"targets": [{"target": "source.wav", "output_dir": "episode"}], "sentences": [
            {"target": "source.wav", "start": 1, "end": 3, "accepted": False, "audio_file": "",
             "diagnostics": {"cluster_removed": True}}]}
        path = self.batch / "batch_manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        details = server.get_batch_details("batch")
        self.assertEqual(details["total_segments"], 1)
        self.assertFalse(details["segments"][0]["accepted"])
        self.assertEqual(Path(server.stream_audio("batch", 1).path), clip)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), manifest)
        clip.with_suffix(".txt").write_text("4.000–7.000\nwrong mapping", encoding="utf-8")
        self.assertEqual(server.get_batch_details("batch")["total_segments"], 0)


if __name__=="__main__": unittest.main()
