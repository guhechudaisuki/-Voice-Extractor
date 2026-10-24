"""Download and contain the portable desktop's program files."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import urllib.request
import zipfile

from launcher.release_manifest import RESOURCE_SHA256, RESOURCE_URL


def portable_environment(root: Path) -> dict[str, str]:
    resource = root / 'resource'
    cache = resource / 'cache'
    temporary = resource / 'temp'
    temporary.mkdir(parents=True, exist_ok=True)
    return {
        'VOICE_EXTRACT_INSTALL_ROOT': str(root),
        'VOICE_EXTRACT_SOURCE_ROOT': str(resource),
        'VOICE_EXTRACT_DATA_ROOT': str(root),
        'VOICE_EXTRACT_ASSET_ROOT': str(resource),
        'VOICE_EXTRACT_MODEL_ROOT': str(resource / 'model'),
        'VOICE_EXTRACT_CACHE_ROOT': str(cache),
        'VOICE_EXTRACT_WORK_ROOT': str(root / 'work'),
        'VOICE_EXTRACT_OUTPUT_ROOT': str(root / 'output'),
        'VOICE_EXTRACT_FFMPEG': str(resource / 'tools/ffmpeg.exe'),
        'VOICE_EXTRACT_FFPROBE': str(resource / 'tools/ffprobe.exe'),
        'HF_HOME': str(cache / 'huggingface'),
        'HF_HUB_CACHE': str(cache / 'huggingface' / 'hub'),
        'TORCH_HOME': str(cache / 'torch'),
        'MODELSCOPE_CACHE': str(cache / 'modelscope'),
        'XDG_CACHE_HOME': str(cache),
        'PIP_CACHE_DIR': str(cache / 'pip'),
        'PIP_DISABLE_PIP_VERSION_CHECK': '1',
        'PYTHONNOUSERSITE': '1',
        'PYTHONIOENCODING': 'utf-8',
        'TEMP': str(temporary),
        'TMP': str(temporary),
    }


def initialize_portable_process(root: Path) -> None:
    os.environ.update(portable_environment(root))
    tempfile.tempdir = os.environ['TEMP']


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path, reporter, expected_hash: str = '') -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + '.part')
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'VoiceExtractor/3.0'})
        with urllib.request.urlopen(request, timeout=90) as response, partial.open('wb') as output:
            size = int(response.headers.get('Content-Length') or 0)
            received = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
                received += len(block)
                reporter(received / size if size else 0,
                         f'正在下载运行文件：{received / 1024**2:.1f} MB')
            if size and received != size:
                raise RuntimeError('运行文件下载不完整，请重新启动程序重试')
        if expected_hash and sha256(partial) != expected_hash:
            raise RuntimeError('运行文件 SHA-256 校验失败')
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def extract_checked(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as package:
        members = package.infolist()
        if sum(item.file_size for item in members) > 2 * 1024**3:
            raise RuntimeError('运行文件解压大小超过限制')
        paths = []
        for item in members:
            name = PurePosixPath(item.filename.replace('\\', '/'))
            target = (root / str(name)).resolve()
            if (name.is_absolute() or '..' in name.parts or ':' in str(name)
                    or not target.is_relative_to(root)
                    or (item.external_attr >> 16) & 0o170000 == 0o120000):
                raise RuntimeError('运行文件压缩包包含非法路径')
            paths.append((item, target))
        for item, target in paths:
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with package.open(item) as source, target.open('wb') as output:
                shutil.copyfileobj(source, output)


def program_ready(resource: Path) -> bool:
    try:
        marker = json.loads((resource / 'program_manifest.json').read_text(encoding='utf-8'))
        return (marker.get('sha256') == RESOURCE_SHA256 and bool(RESOURCE_SHA256)
                and all((resource / file).is_file() for file in marker['files']))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def ensure_program_files(root: Path, reporter) -> Path:
    if not getattr(sys, 'frozen', False):
        return Path(__file__).resolve().parents[1]
    resource = root / 'resource'
    if program_ready(resource):
        return resource
    if not RESOURCE_SHA256:
        raise RuntimeError('EXE 缺少运行文件校验清单，请下载完整发布版本')
    resource.mkdir(parents=True, exist_ok=True)
    archive = resource / 'program.zip'
    download(RESOURCE_URL, archive, reporter, RESOURCE_SHA256)
    try:
        with tempfile.TemporaryDirectory(prefix='program_', dir=resource) as directory:
            staged = Path(directory)
            extract_checked(archive, staged)
            required = ('desktop_worker.py', 'extractor/pipeline.py', 'launcher/model_reinforcement.py',
                        'tools/ffmpeg.exe', 'tools/ffprobe.exe', 'dependencies/uvr5/vr.py',
                        'dependencies/eres2net/ERes2NetV2.py')
            if not all((staged / name).is_file() for name in required):
                raise RuntimeError('运行文件不完整，请重新下载发布版本')
            files = []
            for path in staged.rglob('*'):
                if path.is_file():
                    relative = path.relative_to(staged)
                    target = resource / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(path, target)
                    files.append(relative.as_posix())
            marker = resource / 'program_manifest.json'
            temporary = marker.with_suffix('.tmp')
            temporary.write_text(json.dumps({'sha256': RESOURCE_SHA256, 'files': files}), encoding='utf-8')
            temporary.replace(marker)
    finally:
        archive.unlink(missing_ok=True)
    readme = resource / 'README.md'
    if readme.is_file():
        shutil.copy2(readme, root / 'README.md')
    return resource


def install_embedded_python(root: Path, reporter) -> Path:
    """Use the embeddable distribution; no registry or PATH installation."""
    import subprocess

    resource = root / 'resource'
    destination = resource / 'python'
    python = destination / 'python.exe'
    archive = resource / 'python-3.9.13-embed-amd64.zip'
    marker = destination / '.python-ready'
    if not python.is_file() or not marker.is_file():
        download('https://www.python.org/ftp/python/3.9.13/python-3.9.13-embed-amd64.zip',
                 archive, reporter, '938a1f3b80d580320836260612084d74ce094a261e36f9ff3ac7b9463df5f5e4')
        try:
            extract_checked(archive, destination)
            marker.touch()
        finally:
            archive.unlink(missing_ok=True)
    (destination / 'python39._pth').write_text(
        'python39.zip\n.\nLib/site-packages\n../python_packages\n..\n../vendor\nimport site\n', encoding='utf-8')
    environment = {**os.environ, **portable_environment(root)}
    packages = resource / 'python_packages'
    packages.mkdir(parents=True, exist_ok=True)

    def run(arguments):
        process = subprocess.Popen([str(python), *arguments], cwd=resource, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8',
            errors='replace', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        for line in process.stdout:
            if line.strip():
                reporter(0.8, line.strip()[-500:])
        if process.wait():
            raise RuntimeError('便携 Python 依赖安装失败；重新点击安装可重试')

    pip_script = resource / 'get-pip.py'
    probe = subprocess.run([str(python), '-m', 'pip', '--version'], env=environment,
        capture_output=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), timeout=30)
    if probe.returncode:
        download('https://bootstrap.pypa.io/pip/3.9/get-pip.py', pip_script, reporter)
        try:
            run([str(pip_script), '--no-warn-script-location'])
        finally:
            pip_script.unlink(missing_ok=True)
    reporter(0.8, '正在安装便携 PyTorch 和 torchaudio（首次约需下载 2.5 GB）')
    run(['-m', 'pip', 'install', '--no-input', '--no-warn-script-location',
         'torch==2.0.0+cu118', 'torchaudio==2.0.1+cu118',
         '--index-url', 'https://download.pytorch.org/whl/cu118'])
    return python
