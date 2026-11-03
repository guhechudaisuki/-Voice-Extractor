"""Package runtime files and pin the desktop resource download checksum."""
from __future__ import annotations

import argparse
import ast
import hashlib
import os
from pathlib import Path
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
URL = 'https://github.com/guhechudaisuki/-Voice-Extractor/releases/download/v3.0/VoiceExtractor_resources.zip'
EXCLUDED_DIRECTORIES = {'__pycache__', 'bin', 'obj', 'tests', 'test', 'evaluation', 'docs', 'examples'}
RUNTIME_SUFFIXES = {'.py', '.json', '.yaml', '.yml', '.txt', '.csv', '.html', '.css', '.js',
                    '.svg', '.png', '.ico', '.typed', '.dll', '.pyd', '.so'}
MODEL_FILES = ('models/anime_t3/classifier_v2.pt',
               'models/anime-speaker-char/anime_speaker_char_ecapa.onnx',
               'models/anime-speaker-va/anime_speaker_va_ecapa.onnx')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def runtime_file(relative: Path) -> bool:
    if any(part.startswith('.') or part.lower() in EXCLUDED_DIRECTORIES for part in relative.parts):
        return False
    if relative.name.lower().startswith(('test_', 'conftest.', 'readme')):
        return False
    return (relative.suffix.lower() in RUNTIME_SUFFIXES
            or relative.name.upper().startswith(('LICENSE', 'COPYING', 'NOTICE'))
            or any(part.endswith('.dist-info') for part in relative.parts))


def validate_model(path: Path) -> None:
    if path.suffix == '.onnx':
        if path.stat().st_size == 0:
            raise ValueError(f'Empty ONNX model: {path}')
        return
    if path.name == 'classifier_v2.pt':
        manifest = ast.parse((ROOT / 'launcher/desktop_bootstrap.py').read_text(encoding='utf-8'))
        asset = next(node for node in ast.walk(manifest)
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                     and node.func.id == 'DirectFile' and node.args
                     and isinstance(node.args[0], ast.Constant) and node.args[0].value == path.name)
        if sha256(path) != ast.literal_eval(asset.args[2]):
            raise ValueError('The trained classifier does not match the desktop manifest')
        return
    raise ValueError(f'Unsupported runtime model: {path.name}')


def collect_files(gpt_root: Path) -> dict[str, Path]:
    entries: dict[str, Path] = {}

    def add(file: Path, name: str, source: Path) -> None:
        if file.is_symlink() or not file.resolve().is_relative_to(source.resolve()):
            raise ValueError(f'Runtime file escapes its source directory: {name}')
        if not file.is_file():
            raise FileNotFoundError(file)
        entries[name] = file

    tracked = subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard', '--', 'extractor', 'launcher', 'vendor'], cwd=ROOT,
    ).decode('utf-8').split('\0')
    for name in filter(None, tracked):
        if (ROOT / name).is_file() and runtime_file(Path(name)):
            add(ROOT / name, name, ROOT)

    for source, target in ((gpt_root / 'tools/uvr5', 'dependencies/uvr5'),
                           (gpt_root / 'GPT_SoVITS/eres2net', 'dependencies/eres2net')):
        if not source.is_dir():
            raise FileNotFoundError(source)
        for file in sorted(source.rglob('*')):
            relative = file.relative_to(source)
            if file.is_file() and runtime_file(relative) and 'uvr5_weights' not in relative.parts:
                add(file, (Path(target) / relative).as_posix(), source)
    add(gpt_root / 'LICENSE', 'dependencies/LICENSE.GPT-SoVITS', gpt_root)
    for name in ('ffmpeg.exe', 'ffprobe.exe'):
        add(gpt_root / 'runtime' / name, f'tools/{name}', gpt_root)
    for name in ('desktop_worker.py', 'extract_cli.py', 'app.py', 'README.md', 'LICENSE',
                 'requirements-extra.txt'):
        add(ROOT / name, name, ROOT)
    for name in MODEL_FILES:
        file = ROOT / name
        add(file, name, ROOT)
        validate_model(file)
    required = ('extractor/pipeline.py', 'launcher/model_reinforcement.py',
                'launcher/release_manifest.py', 'dependencies/uvr5/vr.py',
                'dependencies/eres2net/ERes2NetV2.py', 'dependencies/LICENSE.GPT-SoVITS')
    for name in required:
        if name not in entries:
            raise FileNotFoundError(f'Missing runtime source: {name}')
    return entries


def build(gpt_root: Path, destination: Path) -> str:
    entries = collect_files(gpt_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.part')
    try:
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as package:
            for name, file in sorted(entries.items()):
                if name == 'launcher/release_manifest.py':
                    package.writestr(name, f'RESOURCE_URL = {URL!r}\nRESOURCE_SHA256 = ""\n'
                                     f'REQUIRED_MODEL_FILES = {MODEL_FILES!r}\n')
                elif name.startswith('vendor/') and (
                        file.suffix in {'.py', '.typed'}
                        or any(part.endswith('.dist-info') for part in Path(name).parts)):
                    # Wheel RECORD checksums describe the original LF text files.
                    content = file.read_bytes()
                    content.decode('utf-8')
                    package.writestr(name, content.replace(b'\r\n', b'\n'))
                else:
                    package.write(file, name)
        digest = sha256(temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    (ROOT / 'launcher/release_manifest.py').write_text(
        '"""Generated by scripts/package_resources.py before building the EXE."""\n'
        f'RESOURCE_URL = {URL!r}\nRESOURCE_SHA256 = {digest!r}\n'
        f'REQUIRED_MODEL_FILES = {MODEL_FILES!r}\n', encoding='utf-8')
    return digest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpt-root', type=Path,
                        default=Path(os.environ.get('VOICE_EXTRACT_GPT_ROOT', ROOT.parent / 'GPT-SoVITS')))
    parser.add_argument('--output', type=Path, default=ROOT / 'dist/VoiceExtractor_resources.zip')
    args = parser.parse_args()
    digest = build(args.gpt_root.resolve(), args.output.resolve())
    print(f'{args.output}: {args.output.stat().st_size} bytes; SHA256 {digest}')
