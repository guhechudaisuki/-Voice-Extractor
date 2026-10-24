import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
import wave
import zipfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from launcher.training_material import export_package, import_packages, validated_rows
from launcher.training_workspace import Workspace, clean_stale
from extractor.training_context import save_training_context
from launcher.review_store import review_path, write_review


class MaterialTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'output'
        self.root.mkdir()
        self.batch = self.root / 'batch'
        self.batch.mkdir()
        self.manifest = {'sentences': []}
        self.reviews = {}
        for ordinal in range(1, 5):
            name = f'{ordinal}.wav'
            with wave.open(str(self.batch / name), 'wb') as stream:
                stream.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                stream.writeframes(bytes([ordinal, 0]) * 16000)
            self.manifest['sentences'].append({'audio_file': name, 'accepted': ordinal != 4,
                'diagnostics': {'cluster_removed': ordinal == 4}, 'start': ordinal * 2, 'end': ordinal * 2 + 1})
            self.reviews[str(ordinal)] = {'ordinal': ordinal, 'filename': name,
                                          'label': 'correct' if ordinal < 3 else 'wrong'}

        save_training_context(self.batch, self.batch / '1.wav', [self.batch / '1.wav'])

    def package(self, name='material.zip', reviews=None):
        path = self.root / name
        export_package(self.batch, self.manifest, self.reviews if reviews is None else reviews, path)
        return path

    def test_requires_all_reviewable_rows_including_removed_audio(self):
        reviews = copy.deepcopy(self.reviews)
        del reviews['4']
        with self.assertRaisesRegex(ValueError, '未打标'):
            self.package(reviews=reviews)
        self.assertFalse((self.root / 'material.zip').exists())

    def test_rejects_stale_file_invalid_ordinal_label_and_partial_range(self):
        changes = ({'filename': 'stale.wav'}, {'ordinal': 99}, {'label': 'unknown'},
                   {'label': 'partial', 'partial_start': 0, 'partial_end': 5},
                   {'label': 'partial', 'partial_start': float('nan'), 'partial_end': .5})
        for change in changes:
            reviews = copy.deepcopy(self.reviews)
            reviews['1'].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validated_rows(self.batch, self.manifest, reviews)

    def test_unavailable_audio_does_not_block_and_skip_is_retained(self):
        self.manifest['sentences'].append({'accepted': True, 'audio_file': 'missing.wav'})
        self.reviews['4']['label'] = 'skip'
        rows = import_packages([self.package()], self.root / 'imported')
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[-1]['review']['label'], 'skip')

    def test_portable_merge_deduplicates_and_rejects_conflicts(self):
        first = self.package()
        second = self.package('second.zip')
        rows = import_packages([first, second], self.root / 'merged')
        self.assertEqual(len(rows), 4)
        self.reviews['1']['label'] = 'wrong'
        conflicting = self.package('conflict.zip')
        with self.assertRaisesRegex(ValueError, '冲突'):
            import_packages([first, conflicting], self.root / 'conflicts')

    def test_rejects_corrupt_duplicate_package_and_path_checksum(self):
        original = self.package()
        with zipfile.ZipFile(original) as archive:
            content = {name: archive.read(name) for name in archive.namelist()}
        for name, mutate in (
            ('corrupt.zip', lambda data: data.update({'audio/000001.wav': b'broken'})),
            ('escape.zip', lambda data: data.update({'material.json': data['material.json'].replace(
                json.loads(data['material.json'])['items'][0]['sha256'].encode(), (('../' * 20) + 'abcd').encode())})),
        ):
            changed = dict(content)
            mutate(changed)
            destination = self.root / name
            with zipfile.ZipFile(destination, 'w') as archive:
                for member, value in changed.items():
                    archive.writestr(member, value)
            with self.subTest(name=name), self.assertRaises(ValueError):
                import_packages([original, destination], self.root / name.removesuffix('.zip'))

    def test_import_reads_training_zip_directly_and_rejects_plain_audio_zip(self):
        raw = self.package()
        outer = self.root / 'audio.zip'
        with zipfile.ZipFile(outer, 'w') as archive:
            archive.writestr('sample.wav', b'plain audio without labels')
        with self.assertRaisesRegex(ValueError, '不含训练标注'):
            import_packages([outer], self.root / 'import')
        self.assertEqual(len(import_packages([raw], self.root / 'valid_import')), 4)
        with self.assertRaisesRegex(ValueError, 'ZIP'):
            import_packages([raw.with_suffix('.txt')], self.root / 'import')

    def test_optional_material_requires_complete_labels_but_normal_zip_remains_available(self):
        from fastapi import HTTPException
        from launcher import webui_server as server
        (self.batch / 'batch_manifest.json').write_text(json.dumps(self.manifest), encoding='utf-8')
        data = copy.deepcopy(self.reviews)
        data.pop('4')
        (self.batch / 'review_v2.json').write_text(json.dumps({'reviews': data}), encoding='utf-8')
        payload = server.VerifiedExportPayload(include_material=True)
        with patch.object(server, 'OUTPUT_DIR', self.root), self.assertRaises(HTTPException) as caught:
            server.export_verified_segments('batch', payload)
        self.assertEqual(caught.exception.status_code, 400)
        with patch.object(server, 'OUTPUT_DIR', self.root):
            result = server.export_verified_segments('batch')
        with zipfile.ZipFile(result['zip_path']) as archive:
            self.assertNotIn('material.json', archive.namelist())
            self.assertEqual(len(archive.namelist()), 2)
        write_review(self.batch, {'reviews': self.reviews})
        with patch.object(server, 'OUTPUT_DIR', self.root):
            result = server.export_verified_segments('batch', payload)
        with zipfile.ZipFile(result['zip_path']) as archive:
            self.assertIn('material.json', archive.namelist())
            self.assertFalse(any(name.endswith('.zip') for name in archive.namelist()))
        self.assertEqual(Path(result['zip_path']), review_path(self.batch).with_suffix('.zip'))
        rows = import_packages([Path(result['zip_path'])], self.root / 'raw_import')
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(path.is_file() for path in review_path(self.batch).parent.iterdir()))
        # An incomplete retry must not destroy the previous complete ZIP.
        before = Path(result['zip_path']).read_bytes()
        write_review(self.batch, {'reviews': data})
        with patch.object(server, 'OUTPUT_DIR', self.root), self.assertRaises(HTTPException):
            server.export_verified_segments('batch', payload)
        self.assertEqual(Path(result['zip_path']).read_bytes(), before)

    def test_download_cleanup_keeps_saved_annotation_zip(self):
        from fastapi.testclient import TestClient
        from launcher import webui_server as server
        (self.batch / 'batch_manifest.json').write_text(json.dumps(self.manifest), encoding='utf-8')
        write_review(self.batch, {'reviews': self.reviews})
        client = TestClient(server.app)
        with patch.object(server, 'OUTPUT_DIR', self.root):
            response = client.post('/api/export_verified/batch', json={'include_material': True})
        self.assertEqual(response.status_code, 200)
        saved = review_path(self.batch).with_suffix('.zip')
        expected = saved.read_bytes()
        downloaded = client.get(response.json()['download_url'])
        self.assertEqual(downloaded.content, expected)
        self.assertEqual(saved.read_bytes(), expected)
        self.assertEqual(client.get(response.json()['download_url']).status_code, 404)

    def test_mixed_targets_keep_reference_conditioned_labels_without_names(self):
        first = self.package()
        self.reviews['1']['speaker_tag'] = 'unneeded name'
        for item in self.reviews.values():
            item['label'] = 'wrong' if item['label'] == 'correct' else 'correct'
        save_training_context(self.batch, self.batch / '3.wav', [self.batch / '3.wav'])
        second = self.package('other-episode.zip')
        rows = import_packages([first, second, first], self.root / 'mixed')
        self.assertEqual(len(rows), 8)
        same_audio = [row for row in rows if row['filename'] == '1.wav']
        self.assertEqual({row['review']['label'] for row in same_audio}, {'correct', 'wrong'})
        self.assertEqual(len({row['reference_key'] for row in same_audio}), 2)
        with zipfile.ZipFile(second) as archive:
            material = json.loads(archive.read('material.json'))
            self.assertNotIn('identity', material)
            self.assertTrue(material['contexts'])
            self.assertNotIn('speaker_tag', material['items'][0]['review'])
        # The reference files travel inside the material; no original paths are needed.
        self.assertTrue(all(Path(path).is_file() for row in rows for path in row['reference_paths']))

    def test_missing_reference_context_rejects_raw_export(self):
        (self.batch / '_training/context.json').unlink()
        with self.assertRaisesRegex(ValueError, '参考音频'):
            self.package()
        self.assertFalse((self.root / 'material.zip').exists())

    def test_reference_snapshot_and_raw_survive_original_reference_removal(self):
        original = self.root / 'original_reference.wav'
        original.write_bytes((self.batch / '1.wav').read_bytes())
        save_training_context(self.batch, self.batch / '1.wav', [original])
        original.unlink()
        raw = self.package()
        rows = import_packages([raw], self.root / 'portable')
        self.assertEqual(Path(rows[0]['reference_paths'][0]).read_bytes(), (self.batch / '1.wav').read_bytes())
        with zipfile.ZipFile(raw) as archive:
            self.assertNotIn(str(self.root), archive.read('material.json').decode('utf-8'))

    def test_old_raw_without_reference_audio_has_clear_migration_error(self):
        raw = self.root / 'old.zip'
        with zipfile.ZipFile(raw, 'w') as archive:
            archive.writestr('material.json', json.dumps({'kind': 'voice_extractor_training_material', 'schema': 1}))
        with self.assertRaisesRegex(ValueError, '重新导出'):
            import_packages([raw], self.root / 'old')

    def test_backend_does_not_accept_or_persist_person_names(self):
        from launcher import webui_server as server
        review = server.ItemReview(ordinal=1, filename='1.wav', label='correct', speaker_tag='old name')
        self.assertNotIn('speaker_tag', review.model_dump())
        with patch.object(server, 'OUTPUT_DIR', self.root):
            server.save_batch_review('batch', server.BatchReviewPayload(batch_id='batch', reviews={'1': review}))
            persisted = json.loads(review_path(self.batch).read_text(encoding='utf-8'))
            self.assertNotIn('speaker_tag', persisted['reviews']['1'])



class WorkspaceTests(unittest.TestCase):
    def test_disk_workspace_is_cleaned_without_removing_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = Workspace(root)
            job = workspace.job('job')
            (job / 'candidate.pt').write_bytes(b'model')
            exported = root / 'exported.pt'
            exported.write_bytes((job / 'candidate.pt').read_bytes())
            self.assertEqual(clean_stale(root), 0)
            workspace.close()
            self.assertFalse(job.exists())
            self.assertEqual(exported.read_bytes(), b'model')

    def test_stale_sweep_preserves_active_and_unrecognized_folders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            active = Workspace(root)
            stale = root / 'voice_extractor_training_stale'
            stale.mkdir()
            (stale / 'owner.json').write_text(json.dumps({'application': 'voice-extractor', 'pid': 99999999, 'created': 0}))
            unknown = root / 'voice_extractor_training_unknown'
            unknown.mkdir()
            self.assertEqual(clean_stale(root), 1)
            self.assertTrue(active.root.exists())
            self.assertTrue(unknown.exists())
            active.close()


class TrainingLifecycleTests(unittest.TestCase):
    def setUp(self):
        from launcher import webui_server as server
        self.server = server
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Workspace(Path(self.temporary.name))
        self.addCleanup(self.workspace.close)
        for name, value in (('TRAINING_WORKSPACE', self.workspace), ('TRAINING_JOBS', {}),
                            ('TRAINING_PROCESSES', {}), ('DOWNLOADS', {}), ('_closing', threading.Event())):
            mocked = patch.object(server, name, value)
            mocked.start()
            self.addCleanup(mocked.stop)

    def test_failed_process_start_leaves_no_job_or_raw_copy(self):
        from fastapi import HTTPException, UploadFile
        upload = UploadFile(filename='sample.zip', file=io.BytesIO(b'sample'))
        with patch.object(self.server.subprocess, 'Popen', side_effect=OSError('cannot start')), self.assertRaises(HTTPException):
            self.server.start_training([upload])
        self.assertEqual(self.server.TRAINING_JOBS, {})
        self.assertEqual(self.server.TRAINING_PROCESSES, {})
        self.assertEqual([path.name for path in self.workspace.root.iterdir()], ['owner.json'])

    def complete_job(self):
        directory = self.workspace.job('job')
        (directory / 'training_report.json').write_text('{}')
        self.server.TRAINING_JOBS['job'] = {'state': 'complete', 'temporary_path': str(directory)}
        return directory

    def test_clearing_training_preserves_the_current_model(self):
        directory = self.complete_job()
        current = Path(self.temporary.name) / 'classifier_v2.pt'
        current.write_bytes(b'updated current model')
        self.server.discard_training('job')
        self.assertFalse(directory.exists())
        self.assertEqual(current.read_bytes(), b'updated current model')

    def test_model_export_endpoint_is_removed(self):
        from fastapi.testclient import TestClient
        self.assertEqual(TestClient(self.server.app).post('/api/training/export/job').status_code, 404)

    def test_new_training_requires_clearing_previous_temporary_model(self):
        from fastapi import HTTPException, UploadFile
        directory = self.complete_job()
        upload = UploadFile(filename='sample.zip', file=io.BytesIO(b'sample'))
        with self.assertRaises(HTTPException) as caught:
            self.server.start_training([upload])
        self.assertEqual(caught.exception.status_code, 409)
        self.assertTrue((directory / 'training_report.json').exists())

    def test_close_only_acknowledges_after_files_are_removed(self):
        self.complete_job()
        with patch.object(self.server.threading, 'Thread'):
            result = self.server.close_application()
        self.assertEqual(result['status'], 'cleaned')
        self.assertFalse(self.workspace.root.exists())

    def test_failed_cleanup_keeps_service_open_and_allows_retry(self):
        from fastapi import HTTPException
        self.complete_job()
        with patch.object(self.workspace, 'close', side_effect=PermissionError('locked')), \
                patch.object(self.server.threading, 'Thread') as stop, self.assertRaises(HTTPException):
            self.server.close_application()
        stop.assert_not_called()
        self.assertFalse(self.server._closing.is_set())
        self.assertTrue(self.workspace.root.exists())

    def test_large_upload_rejected_before_starting_training(self):
        from fastapi.testclient import TestClient
        from launcher import training_upload
        client = TestClient(self.server.app)
        with patch.object(training_upload, 'MAX_UPLOAD_BYTES', 10):
            response = client.post('/api/training/start', files={'files': ('data.zip', b'01234567890')})
        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.server.TRAINING_JOBS, {})

    def test_streamed_oversized_upload_closes_partial_spool(self):
        import asyncio
        from launcher import training_upload
        head = b'--raw\r\nContent-Disposition: form-data; name="files"; filename="a.zip"\r\n\r\nfirst'
        chunks = [head, b'over-limit-data\r\n--raw--\r\n']
        messages, spools = [], []
        async def receive():
            return {'type': 'http.request', 'body': chunks.pop(0), 'more_body': bool(chunks)}
        async def send(message):
            messages.append(message)
        def spool(*args, **kwargs):
            result = tempfile.SpooledTemporaryFile(*args, **kwargs)
            spools.append(result)
            return result
        scope = {'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1', 'method': 'POST',
                 'scheme': 'http', 'path': '/api/training/start', 'query_string': b'',
                 'headers': [(b'content-type', b'multipart/form-data; boundary=raw')],
                 'client': ('127.0.0.1', 1), 'server': ('127.0.0.1', 7860)}
        with patch.object(training_upload, 'MAX_UPLOAD_BYTES', len(head) + 8), \
                patch('starlette.formparsers.SpooledTemporaryFile', side_effect=spool):
            asyncio.run(self.server.app(scope, receive, send))
        self.assertEqual(messages[0]['status'], 400)
        self.assertEqual(len(spools), 1)
        self.assertTrue(spools[0].closed)
        self.assertEqual(self.server.TRAINING_JOBS, {})


if __name__ == '__main__':
    unittest.main()
