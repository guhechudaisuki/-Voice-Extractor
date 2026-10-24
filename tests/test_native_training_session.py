import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from launcher.training_session import TrainingSession


class NativeTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.session = TrainingSession(self.base)
        self.addCleanup(self.session.close)
        self.raw = self.base / 'sample.zip'
        self.raw.write_bytes(b'fixture')

    def test_native_picker_rejects_non_zip_file(self):
        outer = self.base / 'audio.wav'
        outer.write_bytes(b'fixture')
        with self.assertRaisesRegex(ValueError, 'ZIP'):
            self.session.start([outer])
        self.assertEqual(self.session.snapshot()['state'], 'idle')

    def test_failed_worker_launch_cleans_raw_copy_and_allows_clear(self):
        with patch('launcher.training_session.subprocess.Popen', side_effect=OSError('launch failed')):
            self.session.start([self.raw])
            self.session.thread.join(timeout=5)
        self.assertEqual(self.session.snapshot()['state'], 'failed')
        self.assertFalse(self.session.directory.exists())
        self.assertTrue(self.raw.exists())
        self.session.discard()
        self.assertEqual(self.session.snapshot()['state'], 'idle')

    def test_cancel_stops_real_child_and_preserves_original_material(self):
        actual_popen = subprocess.Popen
        def start_sleeping_child(*args, **kwargs):
            return actual_popen([sys.executable, '-c', 'import time; time.sleep(60)'], **kwargs)
        with patch('launcher.training_session.subprocess.Popen', side_effect=start_sleeping_child):
            self.session.start([self.raw])
            deadline = time.monotonic() + 5
            while self.session.process is None and time.monotonic() < deadline:
                time.sleep(.01)
            process = self.session.process
            self.assertIsNotNone(process)
            directory = self.session.directory
            self.session.discard()
        self.assertIsNotNone(process.poll())
        self.assertFalse(directory.exists())
        self.assertEqual(self.raw.read_bytes(), b'fixture')

    def complete(self):
        self.session.directory = self.session.workspace.job('complete')
        self.session.state = 'complete'
        (self.session.directory / 'training_report.json').write_text(json.dumps({'status': 'updated_in_place'}))

    def test_current_model_survives_native_session_close(self):
        self.complete()
        current = self.base / 'classifier_v2.pt'
        current.write_bytes(b'updated model')
        self.session.close()
        self.assertEqual(current.read_bytes(), b'updated model')
        self.assertFalse(self.session.workspace.root.exists())

    def test_completed_job_requires_temporary_cleanup_before_next_training(self):
        self.complete()
        with self.assertRaisesRegex(ValueError, '清理'):
            self.session.start([self.raw])
        self.assertEqual(self.session.snapshot()['state'], 'complete')


if __name__ == '__main__':
    unittest.main()
