"""Continuation and durable replacement of the classifier used by extraction."""
from contextlib import ExitStack
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extractor.model_store import active_classifier_path, file_sha256, reinforcement_lock, replace_model
from extractor.pipeline import ExtractionPipeline, PipelineOptions
from launcher.model_reinforcement import train, source_groups
from launcher.reinforcement_history import read_history, stage_checkpoint


class ModelFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.current = self.root / 'models/anime_t3/classifier_v2.pt'
        self.current.parent.mkdir(parents=True)
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(47)
            self.classifier = torch.nn.Sequential(torch.nn.Linear(3, 256), torch.nn.ReLU(), torch.nn.Linear(256, 1))
        self.payload = {'input_dim': 3, 'state_dict': self.classifier.state_dict(),
                        'feature_version': 'wavlm-tdnn-unpooled-v1', 'threshold': 2.5,
                        'reject_threshold': -4.0, 'calibration_status': 'passed',
                        'reference_sha256': ['bound-reference'], 'existing_metadata': 'keep-me'}
        torch.save(self.payload, self.current)
        self.original = self.current.read_bytes()


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.current = self.root / 'pytorch_model.bin'
        self.raw = self.root / 'material.zip'
        self.raw.write_bytes(b'ZIP fixture; contents supplied by mocked importer')
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        torch.manual_seed(47)
        self.head = torch.nn.Linear(4, 3)
        self.state = {'feature_extractor.'+key: value.clone() for key, value in self.head.state_dict().items()}
        self.state['frozen_backbone'] = torch.tensor([17.])
        torch.save(self.state, self.current)
        self.original = self.current.read_bytes()
        self.feature_values = torch.tensor([[1., .2, 0., 0., -.5, .3], [.8, .1, 0., 0., -.4, .2],
            [-1., -.2, 0., 0., .5, -.3], [-.8, -.1, 0., 0., .4, -.2]])
        files = []
        for i in range(4):
            path = self.root / f'{i}.wav'
            path.write_bytes(bytes([i+1]))
            files.append(path)
        self.items = [{'path': str(path), 'sha256': file_sha256(path),
            'review': {'label': 'correct' if i < 2 else 'wrong'}, 'source_groups': ['source-a'],
            'source_sha256': None, 'reference_key': 'reference-a', 'reference_paths': [str(files[0])]}
            for i, path in enumerate(files)]
        # One audio may legitimately carry the opposite label for another target.
        self.items += [{**item, 'review': {'label': 'wrong' if item['review']['label'] == 'correct' else 'correct'},
            'reference_key': 'reference-b', 'reference_paths': [str(files[2])], 'source_groups': ['source-b']}
            for item in self.items[:]]
        def load_encoder(*args, **kwargs):
            encoder = torch.nn.Module()
            encoder.feature_extractor = torch.nn.Linear(4, 3)
            saved = torch.load(self.current, weights_only=True)
            encoder.feature_extractor.load_state_dict({key: saved['feature_extractor.'+key] for key in ('weight', 'bias')})
            return encoder
        for target, kwargs in [
            ('launcher.model_reinforcement.import_packages', {'return_value': self.items}),
            ('torch.cuda.is_available', {'return_value': False}),
            ('transformers.WavLMForXVector.from_pretrained', {'side_effect': load_encoder}),
            ('transformers.Wav2Vec2FeatureExtractor.from_pretrained', {'return_value': object()}),
            ('extractor.audio.load_mono', {'side_effect': lambda path, rate: torch.full((8000,), int(path.stem))}),
            ('extractor.local_identity_classifier.pooled_features', {'side_effect': lambda encoder, processor, device, wave: self.feature_values[int(wave[0])]})]:
            self.context.enter_context(patch(target, **kwargs))
        self.context.enter_context(patch('launcher.model_reinforcement.EPOCHS', 10))

    def continue_training(self, name='training', progress=lambda message: None, paths=None):
        destination = self.root / name
        destination.mkdir()
        return train(paths or [self.raw], destination, progress, model_path=self.current)

    def test_mixed_targets_continue_existing_weights_then_duplicate_zip_is_noop(self):
        first_sha = file_sha256(self.current)
        feature = torch.cat((self.feature_values[:, :2], self.feature_values[:, -2:]), dim=1)
        with torch.no_grad():
            embeddings = torch.nn.functional.normalize(self.head(feature), dim=-1)
            scores = torch.cat((embeddings @ embeddings[0], embeddings @ embeddings[2]))
            expected_loss = float(torch.nn.functional.binary_cross_entropy_with_logits(
                (scores - .8) * 20, torch.tensor([1., 1., 0., 0., 0., 0., 1., 1.])))
        report = self.continue_training()
        self.assertAlmostEqual(report['initial_loss'], expected_loss, places=5)
        self.assertLess(report['metrics']['training']['loss'], report['initial_loss'])
        self.assertEqual(report['model_path'], str(self.current.resolve()))
        self.assertEqual(report['base_model_sha256'], first_sha)
        self.assertEqual(report['reference_sets'], 2)
        self.assertEqual(report['source_count'], 2)
        self.assertNotIn('identity', report)
        self.assertEqual(self.current.with_suffix('.bin.previous').read_bytes(), self.original)
        updated = torch.load(self.current, weights_only=True)
        self.assertFalse(torch.equal(updated['feature_extractor.weight'], self.state['feature_extractor.weight']))
        self.assertTrue(torch.equal(updated['frozen_backbone'], self.state['frozen_backbone']))
        second_sha = file_sha256(self.current)
        stamp = self.current.stat().st_mtime_ns
        renamed = self.root / 'renamed.zip'
        renamed.write_bytes(self.raw.read_bytes())
        with patch('launcher.model_reinforcement.import_packages') as importer:
            second = self.continue_training('second', paths=[renamed, self.raw])
        importer.assert_not_called()
        self.assertEqual(second['base_model_sha256'], second_sha)
        self.assertEqual(second['status'], 'already_applied')
        self.assertEqual(second['duplicate_packages'], 2)
        self.assertEqual(second['new_packages'], 0)
        self.assertEqual(file_sha256(self.current), second_sha)
        self.assertEqual(self.current.stat().st_mtime_ns, stamp)
        self.assertEqual(self.current.with_suffix('.bin.previous').read_bytes(), self.original)
        self.assertEqual(read_history(self.current)['material_sha256'], [file_sha256(self.raw)])

    def test_changed_zip_hash_is_new_even_with_same_samples(self):
        self.continue_training()
        self.raw.write_bytes(b'repacked ZIP with same samples')
        report = self.continue_training('repacked')
        self.assertEqual(report['status'], 'updated_in_place')
        self.assertEqual(report['new_packages'], 1)
        self.assertEqual(len(read_history(self.current)['material_sha256']), 2)

    def test_mixed_old_new_and_same_request_duplicates_import_only_new_zip(self):
        self.continue_training()
        added = self.root / 'new.zip'
        added.write_bytes(b'new episode ZIP')
        with patch('launcher.model_reinforcement.import_packages', return_value=self.items[:1]) as importer:
            report = self.continue_training('increment', paths=[self.raw, added, added])
        self.assertEqual(importer.call_args.args[0], [added])
        self.assertEqual(report['duplicate_packages'], 2)
        self.assertEqual(report['training_clips'], 1)
        self.assertEqual(report['new_packages'], 1)
        self.assertEqual(len(read_history(self.current)['material_sha256']), 2)

    def test_failure_before_publication_keeps_existing_model(self):
        def fail_last_epoch(message):
            if message == '训练 10/10':
                raise RuntimeError('training interrupted')
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            self.continue_training(progress=fail_last_epoch)
        self.assertEqual(self.current.read_bytes(), self.original)
        self.assertEqual(read_history(self.current)['material_sha256'], [])
        self.assertEqual(self.continue_training('retry')['status'], 'updated_in_place')

    def test_failed_publication_does_not_remember_zip_and_retry_trains(self):
        with patch('launcher.model_reinforcement.replace_model', side_effect=PermissionError('busy')):
            with self.assertRaises(PermissionError):
                self.continue_training()
        self.assertEqual(read_history(self.current)['material_sha256'], [])
        self.assertEqual(self.continue_training('retry')['new_packages'], 1)

    def test_hash_record_survives_checkpoint_copy_without_sidecar(self):
        self.continue_training()
        copied = self.root / 'copied/pytorch_model.bin'
        copied.parent.mkdir()
        copied.write_bytes(self.current.read_bytes())
        self.current = copied
        self.assertEqual(self.continue_training('after_move')['status'], 'already_applied')

    def test_safetensors_preserves_metadata_and_history(self):
        from safetensors import safe_open
        from safetensors.torch import save_file, load_file
        original = self.root / 'model.safetensors'
        staged = self.root / 'staged.safetensors'
        save_file(self.state, str(original), metadata={'format': 'pt', 'keep': 'value'})
        history = {'schema': 1, 'material_sha256': [file_sha256(self.raw)]}
        stage_checkpoint(original, staged, self.head, history)
        self.assertEqual(read_history(staged), history)
        self.assertTrue(torch.equal(load_file(str(staged))['frozen_backbone'], self.state['frozen_backbone']))
        with safe_open(str(staged), framework='pt') as checkpoint:
            self.assertEqual(checkpoint.metadata()['keep'], 'value')

    def test_corrupt_hash_record_prevents_retraining(self):
        from collections import OrderedDict
        from launcher.reinforcement_history import HISTORY_KEY
        state = OrderedDict(self.state)
        state._metadata = {'': {HISTORY_KEY: {'schema': 1, 'material_sha256': ['bad']}}}
        torch.save(state, self.current)
        before = self.current.read_bytes()
        with self.assertRaisesRegex(ValueError, '哈希记录无效'):
            self.continue_training()
        self.assertEqual(self.current.read_bytes(), before)

    def test_temporary_cleanup_failure_does_not_report_saved_model_as_failed(self):
        actual_unlink = Path.unlink
        def busy_stage(path, *args, **kwargs):
            if path.name == 'pytorch_model.bin':
                raise PermissionError('temporary file busy')
            return actual_unlink(path, *args, **kwargs)
        with patch.object(Path, 'unlink', busy_stage):
            report = self.continue_training()
        self.assertEqual(report['status'], 'updated_in_place')
        self.assertNotEqual(self.current.read_bytes(), self.original)


class ModelStoreTests(ModelFixture):
    def test_duplicate_queries_link_source_groups_before_holdout(self):
        groups = source_groups([
            {'sha256': 'one', 'source_groups': ['episode-a']},
            {'sha256': 'two', 'source_groups': ['episode-b']},
            {'sha256': 'one', 'source_groups': ['episode-b']},
            {'sha256': 'three', 'source_groups': ['episode-c']},
        ])
        self.assertEqual(len(set(groups[:3])), 1)
        self.assertNotEqual(groups[0], groups[3])

    def test_active_path_matches_production_v2_then_legacy_fallback(self):
        self.assertEqual(active_classifier_path(self.root), self.current)
        self.current.unlink()
        self.assertEqual(active_classifier_path(self.root), self.current.with_name('classifier.pt'))

    def test_concurrent_writer_rejected_and_lock_released(self):
        with reinforcement_lock(self.current):
            with self.assertRaisesRegex(ValueError, '另一个窗口'):
                with reinforcement_lock(self.current):
                    self.fail('A second writer acquired the model')
        with reinforcement_lock(self.current):
            pass

    def test_external_change_prevents_overwrite(self):
        expected = file_sha256(self.current)
        staged = self.root / 'staged.pt'
        staged.write_bytes(b'new trained model')
        self.current.write_bytes(b'external update')
        with self.assertRaisesRegex(ValueError, '发生变化'):
            replace_model(staged, self.current, expected)
        self.assertEqual(self.current.read_bytes(), b'external update')

    def test_failed_atomic_replace_preserves_current_and_removes_temporary_files(self):
        staged = self.root / 'staged.pt'
        staged.write_bytes(b'new trained model')
        actual_replace = Path.replace
        def fail_publication(source, destination):
            if destination == self.current:
                raise PermissionError('model busy')
            return actual_replace(source, destination)
        with patch.object(Path, 'replace', fail_publication), self.assertRaises(PermissionError):
            replace_model(staged, self.current, file_sha256(self.current))
        self.assertEqual(self.current.read_bytes(), self.original)
        self.assertEqual(self.current.with_suffix('.pt.previous').read_bytes(), self.original)
        self.assertFalse(list(self.current.parent.glob('.reinforced_*')))


class ExtractionReloadTests(ModelFixture):
    def setUp(self):
        super().setUp()
        self.context.enter_context(patch('extractor.pipeline.ASSET_ROOT', self.root))
        self.context.enter_context(patch('extractor.pipeline.WAVLM_SV_MODEL', self.root))
        self.context.enter_context(patch('transformers.WavLMForXVector.from_pretrained', return_value=torch.nn.Identity()))
        self.context.enter_context(patch('transformers.Wav2Vec2FeatureExtractor.from_pretrained', return_value=object()))
        self.pipeline = ExtractionPipeline(PipelineOptions(), device='cpu')
        self.pipeline._reference_sha256 = ['bound-reference']

    def test_next_job_loads_updated_weights_and_current_job_keeps_consistent_weights(self):
        first = self.pipeline._ensure_tail_scorer()
        self.assertIsNotNone(first)
        stamp = self.current.stat().st_mtime_ns
        state = {key: value.clone() for key, value in self.payload['state_dict'].items()}
        state['2.bias'] += 3
        torch.save({**self.payload, 'state_dict': state, 'calibration_status': 'reinforced'}, self.current)
        os.utime(self.current, ns=(stamp + 1_000_000, stamp + 1_000_000))
        self.assertIs(self.pipeline._ensure_tail_scorer(), first)
        self.pipeline._invalidate_changed_tail_scorer()
        second = self.pipeline._ensure_tail_scorer()
        self.assertIsNot(second, first)
        sample = torch.tensor([[1., 0., -1.]])
        with torch.no_grad():
            self.assertAlmostEqual(float(second[2](sample) - first[2](sample)), 3., places=5)
        self.assertEqual(self.pipeline._local_identity_model_sha256, file_sha256(self.current))

    def test_reinforcement_does_not_bypass_reference_binding(self):
        torch.save({**self.payload, 'calibration_status': 'reinforced'}, self.current)
        self.pipeline._reference_sha256 = ['other-person']
        self.assertIsNone(self.pipeline._ensure_tail_scorer())
        self.assertIsNone(self.pipeline._local_identity_reject_threshold)


if __name__ == '__main__':
    unittest.main()
