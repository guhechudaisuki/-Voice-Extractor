import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from launcher.review_store import read_review, review_path, write_review


class ReviewStoreTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.batch = self.root / 'output/episode-a'
        self.batch.mkdir(parents=True)

    def test_flat_files_keep_batches_separate(self):
        first = write_review(self.batch, {'reviews': {'1': {'label': 'correct'}}})
        second = write_review(self.batch.with_name('episode-b'), {'reviews': {'1': {'label': 'wrong'}}})
        self.assertEqual(first, self.root / 'annotations/episode-a.json')
        self.assertEqual(second, self.root / 'annotations/episode-b.json')
        self.assertTrue(all(path.is_file() for path in first.parent.iterdir()))
        self.assertEqual(read_review(self.batch)['reviews']['1']['label'], 'correct')

    def test_legacy_read_and_next_save_preserves_override_and_uses_new_path(self):
        legacy = self.batch / 'review_v2.json'
        data = {'reviews': {}, 'cluster_override': {'episode': {'cluster_id': 1}}}
        legacy.write_text(json.dumps(data), encoding='utf-8')
        self.assertEqual(read_review(self.batch), data)
        updated = {**read_review(self.batch), 'reviewer': 'new'}
        write_review(self.batch, updated)
        self.assertEqual(read_review(self.batch), updated)
        self.assertEqual(json.loads(legacy.read_text(encoding='utf-8')), data)

    def test_corrupt_current_does_not_fall_back_or_get_overwritten_by_api(self):
        from launcher import webui_server as server
        path = write_review(self.batch, {'reviews': {}})
        path.write_text('broken', encoding='utf-8')
        (self.batch / 'review_v2.json').write_text('{}', encoding='utf-8')
        with self.assertRaises(ValueError):
            read_review(self.batch)
        with patch.object(server, 'OUTPUT_DIR', self.batch.parent), self.assertRaises(server.HTTPException):
            server.save_batch_review(self.batch.name, server.BatchReviewPayload(batch_id=self.batch.name, reviews={}))
        self.assertEqual(path.read_text(encoding='utf-8'), 'broken')

    def test_failed_save_keeps_original_and_removes_temporary_file(self):
        path = write_review(self.batch, {'reviews': {}})
        before = path.read_bytes()
        with patch.object(Path, 'replace', side_effect=PermissionError('busy')), self.assertRaises(PermissionError):
            write_review(self.batch, {'reviews': {'1': {'label': 'wrong'}}})
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(path.parent.iterdir()), [path])

    def test_desktop_reports_save_error_and_retains_existing_data(self):
        from launcher.desktop_review_tab import DesktopReviewTab
        from types import SimpleNamespace
        from unittest.mock import Mock
        path = write_review(self.batch, {'reviews': {}})
        path.write_text('broken', encoding='utf-8')
        tab = SimpleNamespace(current_batch_path=self.batch, reviews={}, current_status_var=Mock(), root=None)
        with patch('tkinter.messagebox.showerror') as error:
            self.assertFalse(DesktopReviewTab.save_review(tab, False))
        error.assert_called_once()
        self.assertEqual(path.read_text(encoding='utf-8'), 'broken')


if __name__ == '__main__':
    unittest.main()
