"""Exercise native Tk controls with synthetic audio and real local training."""
import io
import json
import math
import os
import shutil
from pathlib import Path
import struct
import sys
import tempfile
import time
import wave
import zipfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'launcher'))
os.environ['VOICE_EXTRACT_INSTALL_ROOT'] = str(ROOT)

import tkinter as tk
from PIL import ImageGrab
import voice_extractor_desktop as desktop
from desktop_review_tab import DesktopReviewTab
from extractor.model_store import file_sha256
from extractor.training_context import save_training_context
from launcher.review_store import review_path, read_review


def main():
    output = ROOT / 'work/cluster_p0p2_logs'
    output.mkdir(parents=True, exist_ok=True)
    root = tk.Tk()
    errors = []
    root.report_callback_exception = lambda *args: errors.append(str(args[1]))
    app = None
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='voice_native_smoke_') as temporary, \
            patch.object(DesktopReviewTab, 'play_current'), \
            patch('tkinter.messagebox.showinfo'), patch('tkinter.messagebox.showerror') as dialog_error:
        directory = Path(temporary)
        production = ROOT / 'model/speaker/wavlm-base-plus-sv/pytorch_model.bin'
        production_sha = file_sha256(production)
        production_classifier = ROOT / 'models/anime_t3/classifier_v2.pt'
        production_classifier_sha = (file_sha256(production_classifier)
                                     if production_classifier.is_file() else None)
        assets = directory / 'isolated_assets'
        model_path = assets / 'model/speaker/wavlm-base-plus-sv/pytorch_model.bin'
        shutil.copytree(production.parent, model_path.parent, ignore=shutil.ignore_patterns('.cache', '*.previous', '*.lock'))
        isolated_classifier = None
        if production_classifier.is_file():
            isolated_classifier = assets / 'models/anime_t3/classifier_v2.pt'
            isolated_classifier.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(production_classifier, isolated_classifier)
        try:
            app = desktop.VoiceExtractorDesktop(root)
            root.geometry('1240x820+40+30')
            root.update()
            assert root.title() == 'Voice Extractor Desktop'
            app.setup_page.pack_forget()
            app.main_page.pack(fill='both', expand=True)
            assert len(app.main_notebook.tabs()) == 4
            batch = directory / 'output' / 'batch'
            audio = batch / 'source' / 'audio'
            audio.mkdir(parents=True)
            rows = []
            for ordinal, frequency in enumerate((170, 210, 330, 410), 1):
                name = f'{ordinal}.wav'
                with wave.open(str(audio / name), 'wb') as stream:
                    stream.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                    stream.writeframes(b''.join(struct.pack('<h', int(8000 * math.sin(2 * math.pi * frequency * n / 16000))) for n in range(16000)))
                rows.append({'accepted': ordinal <= 2, 'audio_file': f'source/audio/{name}', 'start': 0, 'end': 1,
                             'diagnostics': {'cluster_removed': ordinal > 2}})
            (batch / 'batch_manifest.json').write_text(json.dumps({'sentences': rows}), encoding='utf-8')
            save_training_context(batch / 'source', audio / '1.wav', [audio / '1.wav'])
            review = app.review_tab
            review.current_batch_path = batch
            review.batch_combo.set('synthetic-smoke')
            review.load_current_batch()
            review.auto_next_var.set(False)
            assert len(review.segments) == 4
            assert review.material_check.instate(['disabled'])
            for index in range(4):
                review.select_segment(index)
                (review.btn_correct if index < 2 else review.btn_wrong).invoke()
            assert not review.material_check.instate(['disabled'])
            review.material_check.invoke()
            assert not hasattr(review, 'identity_var')
            assert not hasattr(review, 'role_combo')
            assert review_path(batch) == directory / 'annotations/batch.json'
            assert len(read_review(batch)['reviews']) == 4
            review.load_current_batch()
            assert len(review.reviews) == 4
            if not review.include_material_var.get():
                review.material_check.invoke()
            assert review.include_material_var.get()
            archive_path = directory / 'annotations/batch.zip'
            app.main_notebook.select(review.tab_frame)
            root.update()
            review.review_form.canvas.yview_moveto(1)
            root.update()
            assert review.export_button.winfo_rooty() + review.export_button.winfo_height() <= root.winfo_rooty() + root.winfo_height()
            snapshot(root, output / 'native_desktop_review.png')
            with patch('tkinter.filedialog.asksaveasfilename', return_value=str(archive_path)):
                review.export_button.invoke()
            wait(root, lambda: not review.export_busy, 30)
            assert all(path.is_file() for path in archive_path.parent.iterdir())
            with zipfile.ZipFile(archive_path) as archive:
                assert 'material.json' in archive.namelist()
                assert len([name for name in archive.namelist() if name.startswith('audio/')]) == 4
                assert any(name.startswith('references/') for name in archive.namelist())
                assert not any(name.endswith('.zip') for name in archive.namelist())
            raw = archive_path
            train = app.training_tab
            environment = {'VOICE_EXTRACT_ASSET_ROOT': str(assets),
                'VOICE_EXTRACT_WAVLM_MODEL': str(model_path.parent)}
            if isolated_classifier is not None:
                environment['VOICE_EXTRACT_CLASSIFIER_V2_MODEL'] = str(isolated_classifier)
            train.environment_provider = lambda: environment
            app.main_notebook.select(train.tab_frame)
            with patch('tkinter.filedialog.askopenfilenames', return_value=(str(raw), str(raw))):
                train.choose_button.invoke()
            train.start_button.invoke()
            wait(root, lambda: train.session.snapshot()['state'] in {'complete', 'failed'}, 120)
            result = train.session.snapshot()
            assert result['state'] == 'complete', result
            assert result['report']['unique_clips'] == 4
            assert not hasattr(train, 'export_button')
            assert result['report']['base_model_sha256'] == production_sha
            assert result['report']['model_path'] == str(model_path.resolve())
            assert file_sha256(model_path) != production_sha
            assert file_sha256(model_path.with_suffix('.bin.previous')) == production_sha
            assert result['report']['new_packages'] == 1
            assert result['report']['duplicate_packages'] == 1
            wait(root, lambda: str(model_path.resolve()) in train.report.get('1.0', 'end'), 5)
            train.form.canvas.yview_moveto(.3)
            snapshot(root, output / 'native_desktop_training.png')
            updated_sha = file_sha256(model_path)
            stamp = model_path.stat().st_mtime_ns
            train.session.discard()
            train.start()
            wait(root, lambda: train.session.snapshot()['state'] in {'complete', 'failed'}, 120)
            repeated = train.session.snapshot()
            assert repeated['state'] == 'complete', repeated
            assert repeated['report']['status'] == 'already_applied'
            assert repeated['report']['duplicate_packages'] == 2
            assert file_sha256(model_path) == updated_sha
            assert model_path.stat().st_mtime_ns == stamp
            wait(root, lambda: '模型未改动' in train.report.get('1.0', 'end'), 5)
            temp_root = train.session.workspace.root
            app._on_close()
            root.mainloop()
            assert not temp_root.exists()
            assert model_path.is_file()
            assert file_sha256(production) == production_sha
            if production_classifier_sha is not None:
                assert file_sha256(production_classifier) == production_classifier_sha
            assert errors == [], errors
            assert not dialog_error.called, dialog_error.call_args_list
            report = {'passed': True, 'ui': 'native_tk_desktop', 'browser_required': False,
                      'synthetic_audio_only': True, 'seconds': round(time.monotonic() - started, 2),
                      'checks': ['native_controls', 'removed_clips_reviewable', 'complete_labels_gate',
                                 'flat_annotation_files', 'saved_labels_reload', 'zip_contains_labels_audio_references', 'native_zip_picker', 'real_wavlm_training',
                                 'duplicate_zip_skipped_in_new_worker', 'duplicate_no_weight_or_mtime_change',
                                 'cross_machine_dedup', 'reference_audio_bundled', 'no_manual_person_fields', 'current_weights_continued', 'automatic_in_place_save',
                                 'previous_model_backup', 'cleanup_preserves_updated_model', 'production_untouched_by_smoke', 'production_classifier_untouched_by_smoke']}
            (output / 'native_desktop_workflow_smoke.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps(report), flush=True)
        finally:
            if app:
                app.training_tab.session.close()
            try:
                root.destroy()
            except tk.TclError:
                pass


def wait(root, predicate, seconds):
    deadline = time.monotonic() + seconds
    while not predicate():
        root.update()
        if time.monotonic() > deadline:
            raise TimeoutError('Native UI operation timed out')
        time.sleep(.03)
    root.update()


def snapshot(root, destination):
    root.lift()
    root.attributes('-topmost', True)
    root.update()
    time.sleep(.3)
    root.update()
    x, y = root.winfo_rootx(), root.winfo_rooty()
    ImageGrab.grab(bbox=(x, y, x + root.winfo_width(), y + root.winfo_height())).save(destination)
    root.attributes('-topmost', False)


if __name__ == '__main__':
    main()
