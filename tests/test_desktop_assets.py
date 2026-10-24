import hashlib
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from launcher import desktop_bootstrap as bootstrap
from launcher import portable_resources as portable


class DesktopAssetTests(unittest.TestCase):
    def test_downloaded_component_is_returned_after_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.json'
            source.write_text('{"downloaded": true}', encoding='utf-8')
            spec = bootstrap.ComponentSpec('fixture', 'Fixture', 'model/fixture', 'direct', 1,
                (bootstrap.DirectFile('config.json', source.as_uri(),
                    hashlib.sha256(source.read_bytes()).hexdigest()),))
            layout = bootstrap.InstallLayout(root / 'install')
            with patch.object(bootstrap, 'COMPONENTS', (spec,)), \
                    patch.object(bootstrap, '_legacy_component_paths',
                        side_effect=lambda item, install, _gpt: [install.resource / item.relative_path]):
                assets = bootstrap.ensure_assets(Path('unused'), layout, root, None, lambda *_: None)
            self.assertEqual(assets, {'fixture': layout.resource / spec.relative_path})
            self.assertEqual((assets['fixture'] / 'config.json').read_bytes(), source.read_bytes())

    def test_layout_keeps_dependencies_in_resource(self):
        layout = bootstrap.InstallLayout(Path('portable').resolve())
        for path in (layout.model, layout.models, layout.packages, layout.tools, layout.python):
            self.assertTrue(path.is_relative_to(layout.resource))

    def test_frozen_scan_ignores_external_models_and_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = bootstrap.InstallLayout(Path(directory))
            spec = bootstrap.COMPONENTS[0]
            with patch.object(bootstrap.sys, 'frozen', True, create=True), \
                    patch.object(bootstrap, '_candidate_roots', side_effect=AssertionError('external scan')):
                self.assertEqual(bootstrap._legacy_component_paths(spec, layout, Path('other-program')),
                                 [layout.resource / spec.relative_path])

    def test_archive_cannot_escape_resource(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'bad.zip'
            with zipfile.ZipFile(archive, 'w') as package:
                package.writestr('valid.txt', 'valid')
                package.writestr('../escape.txt', 'escape')
            with self.assertRaisesRegex(RuntimeError, '非法路径'):
                portable.extract_checked(archive, root / 'resource')
            self.assertFalse((root / 'escape.txt').exists())
            self.assertFalse((root / 'resource/valid.txt').exists())

    def test_portable_download_is_persistent_and_hash_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'program.zip'
            files = ('desktop_worker.py', 'extractor/pipeline.py', 'launcher/model_reinforcement.py',
                     'tools/ffmpeg.exe', 'tools/ffprobe.exe', 'dependencies/uvr5/vr.py',
                     'dependencies/eres2net/ERes2NetV2.py')
            with zipfile.ZipFile(archive, 'w') as package:
                for file in files:
                    package.writestr(file, 'fixture')
            install = root / 'install'
            with patch.object(portable.sys, 'frozen', True, create=True), \
                    patch.object(portable, 'RESOURCE_URL', archive.as_uri()), \
                    patch.object(portable, 'RESOURCE_SHA256', portable.sha256(archive)):
                result = portable.ensure_program_files(install, lambda *_: None)
                self.assertEqual(result, install / 'resource')
                self.assertTrue(portable.program_ready(result))
                with patch.object(portable, 'download', side_effect=AssertionError('redownload')):
                    self.assertEqual(portable.ensure_program_files(install, lambda *_: None), result)
                (result / files[0]).unlink()
                self.assertFalse(portable.program_ready(result))

    def test_portable_root_is_executable_directory(self):
        with patch.object(bootstrap.sys, 'frozen', True, create=True), \
                patch.object(bootstrap.sys, 'executable', str(Path('portable/VoiceExtractor.exe').resolve())), \
                patch.object(bootstrap, 'load_global_settings', side_effect=AssertionError('global settings')):
            self.assertEqual(bootstrap.suggested_install_root(), Path(bootstrap.sys.executable).parent)


if __name__ == '__main__':
    unittest.main()
