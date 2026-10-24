"""Exercise real local training with synthetic audio; no quality claim."""
from __future__ import annotations

import io
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import time
import wave
import zipfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastapi.testclient import TestClient
from launcher import webui_server as server
from launcher.training_material import digest, export_package
from extractor.training_context import save_training_context


def main():
    start = time.monotonic()
    production = ROOT / 'model/speaker/wavlm-base-plus-sv/pytorch_model.bin'
    before = digest(production) if production.is_file() else None
    production_classifier = ROOT / 'models/anime_t3/classifier_v2.pt'
    classifier_before = (digest(production_classifier)
                         if production_classifier.is_file() else None)
    if production_classifier.is_file():
        isolated_dir = Path(tempfile.mkdtemp(prefix='voice_classifier_isolated_'))
        isolated_classifier = isolated_dir / 'classifier_v2.pt'
        shutil.copyfile(production_classifier, isolated_classifier)
        os.environ['VOICE_EXTRACT_CLASSIFIER_V2_MODEL'] = str(isolated_classifier)
    with tempfile.TemporaryDirectory(prefix='voice_workflow_smoke_') as temporary:
        root = Path(temporary) / 'output'
        root.mkdir()
        assets = root / 'isolated_assets'
        current = assets / 'model/speaker/wavlm-base-plus-sv/pytorch_model.bin'
        shutil.copytree(production.parent, current.parent, ignore=shutil.ignore_patterns('.cache', '*.previous', '*.lock'))
        batch = root / 'synthetic'
        batch.mkdir()
        rows, reviews = [], {}
        for ordinal, frequency in enumerate((170, 210, 330, 410), 1):
            name = f'{ordinal}.wav'
            with wave.open(str(batch / name), 'wb') as stream:
                stream.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                stream.writeframes(b''.join(struct.pack('<h', int(8000 * math.sin(2 * math.pi * frequency * sample / 16000))) for sample in range(16000)))
            rows.append({'accepted': True, 'audio_file': name, 'start': ordinal * 2, 'end': ordinal * 2 + 1})
            reviews[str(ordinal)] = {'ordinal': ordinal, 'filename': name, 'label': 'correct' if ordinal <= 2 else 'wrong'}
        (batch / 'batch_manifest.json').write_text(json.dumps({'sentences': rows}), encoding='utf-8')
        (batch / 'review_v2.json').write_text(json.dumps({'reviews': reviews}), encoding='utf-8')
        save_training_context(batch, batch / '1.wav', [batch / '1.wav'])
        second_batch = root / 'another_episode'
        shutil.copytree(batch, second_batch)
        save_training_context(second_batch, second_batch / '3.wav', [second_batch / '3.wav'])
        second_reviews = {key: {**item, 'label': 'wrong' if item['label'] == 'correct' else 'correct'} for key, item in reviews.items()}
        second_raw = root / 'second.zip'
        export_package(second_batch, {'sentences': rows}, second_reviews, second_raw)
        client = TestClient(server.app)
        with patch.object(server, 'OUTPUT_DIR', root):
            export = client.post('/api/export_verified/synthetic', json={'include_material': True})
            assert export.status_code == 200, export.text
            downloaded = client.get(export.json()['download_url'])
        with zipfile.ZipFile(io.BytesIO(downloaded.content)) as archive:
            assert 'material.json' in archive.namelist()
            assert len([name for name in archive.namelist() if name.startswith('audio/')]) == 4
        material = downloaded.content
        rejected = client.post('/api/training/start', files=[('files', ('wrong.wav', material))])
        assert rejected.status_code == 400, rejected.text
        # The trainer saves in place. Never run synthetic examples on installed weights.
        with patch.dict(os.environ, {'VOICE_EXTRACT_ASSET_ROOT': str(assets),
                                     'VOICE_EXTRACT_WAVLM_MODEL': str(current.parent)}):
            response = client.post('/api/training/start', files=[('files', ('first.zip', material)), ('files', ('second.zip', second_raw.read_bytes()))])
        assert response.status_code == 200, response.text
        job_id = response.json()['job_id']
        previous = ''
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            status = client.get(f'/api/training/status/{job_id}').json()
            if status['message'] != previous:
                print(status['message'], flush=True)
                previous = status['message']
            if status['state'] != 'running':
                break
            time.sleep(.3)
        assert status['state'] == 'complete', status
        assert status['report']['unique_clips'] == 8
        assert status['report']['source_count'] == 2
        assert status['report']['reference_sets'] == 2
        job_dir = Path(status['temporary_path'])
        assert ROOT not in job_dir.parents
        assert status['report']['status'] == 'updated_in_place'
        assert status['report']['model_path'] == str(current.resolve())
        assert status['report']['base_model_sha256'] == before
        assert digest(current) != before
        assert digest(current.with_suffix('.bin.previous')) == before
        updated_sha = digest(current)
        import torch
        from extractor.audio import load_mono
        from extractor.speaker import WavLMSpeakerVerifier
        updated = torch.load(current, map_location='cpu', weights_only=True)
        original_projection = torch.load(current.with_suffix('.bin.previous'), map_location='cpu', weights_only=True)['feature_extractor.weight']
        assert not torch.equal(updated['feature_extractor.weight'], original_projection)
        with patch('extractor.speaker.WAVLM_SV_MODEL', current.parent):
            verifier = WavLMSpeakerVerifier(device='cpu')
        assert torch.equal(verifier.model.feature_extractor.weight, updated['feature_extractor.weight'])
        embedding = verifier._embedding_from_waveform(load_mono(batch / '1.wav', 16000))
        assert torch.isfinite(embedding).all()
        verifier.close()
        assert not list(job_dir.glob('*.pt'))
        assert client.post(f'/api/training/export/{job_id}').status_code == 404
        assert client.delete(f'/api/training/{job_id}').status_code == 200
        assert not job_dir.exists()
        assert digest(current) == updated_sha
        stamp = current.stat().st_mtime_ns
        with patch.dict(os.environ, {'VOICE_EXTRACT_WAVLM_MODEL': str(current.parent)}):
            repeated = client.post('/api/training/start', files=[('files', ('renamed.zip', material))])
        assert repeated.status_code == 200, repeated.text
        repeated_id = repeated.json()['job_id']
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            repeated_status = client.get(f'/api/training/status/{repeated_id}').json()
            if repeated_status['state'] != 'running':
                break
            time.sleep(.2)
        assert repeated_status['state'] == 'complete', repeated_status
        assert repeated_status['report']['status'] == 'already_applied'
        assert '模型未改动' in repeated_status['message']
        assert digest(current) == updated_sha
        assert current.stat().st_mtime_ns == stamp
        assert digest(current.with_suffix('.bin.previous')) == before
        assert client.delete(f'/api/training/{repeated_id}').status_code == 200
        assert (digest(production) if production.is_file() else None) == before
        if classifier_before is not None:
            assert digest(production_classifier) == classifier_before
        report = {'passed': True, 'synthetic_audio_only': True, 'seconds': round(time.monotonic() - start, 2),
                  'checks': ['direct_zip_import', 'non_zip_rejected', 'duplicate_renamed_zip_skipped_after_restart', 'duplicate_weights_backup_mtime_unchanged', 'real_wavlm_training', 'mixed_episode_and_reference_material', 'same_audio_different_targets',
                             'current_weights_continued', 'automatic_in_place_save', 'previous_model_backup',
                             'updated_model_loaded_and_scored_by_extraction',
                             'cleanup_preserves_updated_model', 'production_model_unchanged', 'production_classifier_untouched_by_smoke']}
        output = ROOT / 'work/cluster_p0p2_logs/training_workflow_smoke.json'
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
