import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from launcher import webui_server as server


class WebUISubtitleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.target = self.root / "一段音频.wav"
        self.reference = self.root / "reference.wav"
        self.subtitle = self.root / "对应字幕.srt"
        for path in (self.target, self.reference, self.root / "desktop_worker.py"):
            path.touch()
        self.subtitle.write_text("1\n00:00:00,000 --> 00:00:02,000\n测试对白\n", encoding="utf-8")
        self.payload = dict(targets=[str(self.target)], references=[str(self.reference)],
            negative_groups=[[str(self.root / "other.wav")]],
            options={"cluster_bootstrap": True, "cluster_identity_mode": "enforce",
                     "silence_min_seconds": .25, "silence_max_seconds": 1.05})
        for obj, name, value in (
            (server, "BASE_DIR", self.root), (server, "OUTPUT_DIR", self.root / "output"),
            (server, "RUN", server._Run())):
            context = patch.object(obj, name, value)
            context.start()
            self.addCleanup(context.stop)
        process_module = MagicMock(PIPE=server.subprocess.PIPE, STDOUT=server.subprocess.STDOUT,
            CREATE_NEW_PROCESS_GROUP=getattr(server.subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        for context in (patch.object(server, "subprocess", process_module),
                        patch.object(server, "threading", MagicMock()),
                        patch.object(server, "load_settings", return_value={})):
            context.start()
            self.addCleanup(context.stop)
        self.spawn = process_module.Popen

    def test_valid_binding_is_normalized_and_forwarded_with_other_features(self):
        response = server.start_extract(server.ExtractPayload(**self.payload,
            subtitles={str(self.target): str(self.subtitle)}))
        saved = json.loads((self.root / "work/webui_requests" / response["request"]).read_text(encoding="utf-8"))
        self.assertEqual(saved["subtitles"], {str(self.target.resolve()): str(self.subtitle.resolve())})
        self.assertEqual(saved["negative_groups"], self.payload["negative_groups"])
        self.assertEqual(saved["options"], self.payload["options"])
        self.spawn.assert_called_once()

    def test_subtitles_are_optional(self):
        response = server.start_extract(server.ExtractPayload(**self.payload))
        saved = json.loads((self.root / "work/webui_requests" / response["request"]).read_text(encoding="utf-8"))
        self.assertEqual(saved["subtitles"], {})
        self.spawn.assert_called_once()

    def test_missing_invalid_and_foreign_bindings_fail_before_worker_starts(self):
        invalid = self.root / "not-subtitles.txt"
        invalid.write_text("text")
        empty = self.root / "empty.srt"
        empty.write_text("")
        cases = [
            {str(self.target): str(self.root / "missing.srt")},
            {str(self.root / "different.wav"): str(self.subtitle)},
            {str(self.target): str(invalid)},
            {str(self.target): str(empty)},
        ]
        for bindings in cases:
            with self.subTest(bindings=bindings), self.assertRaises(server.HTTPException) as caught:
                server.start_extract(server.ExtractPayload(**self.payload, subtitles=bindings))
            self.assertEqual(caught.exception.status_code, 400)
            self.assertIn("字幕绑定无效", caught.exception.detail)
        self.spawn.assert_not_called()
        self.assertFalse((self.root / "work/webui_requests").exists())

    def test_partial_binding_in_a_batch_keeps_unbound_files_optional(self):
        second = self.root / "另一个文件.mkv"
        second.touch()
        self.payload["targets"].append(str(second))
        response = server.start_extract(server.ExtractPayload(**self.payload,
            subtitles={str(second): str(self.subtitle)}))
        saved = json.loads((self.root / "work/webui_requests" / response["request"]).read_text(encoding="utf-8"))
        self.assertEqual(saved["subtitles"], {str(second.resolve()): str(self.subtitle.resolve())})


if __name__ == "__main__":
    unittest.main()
