"""Complete and verify the selected installation inside the NSIS wizard."""
from __future__ import annotations

import argparse
import json
import locale
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback


def install(root: Path, runtime: str, models: str, device: str) -> None:
    resource = root / 'resource'
    sys.path.insert(0, str(resource))
    from launcher import desktop_bootstrap as bootstrap
    from launcher import portable_resources as portable

    layout = bootstrap.InstallLayout(root)
    layout.prepare()
    environment = {**os.environ, **portable.portable_environment(root)}
    environment['PIP_PROGRESS_BAR'] = 'off'
    os.environ.update(environment)
    log_path = layout.logs / 'installer.log'
    with log_path.open('a', encoding='utf-8', buffering=1) as log:
        def report(_progress: float, message: str) -> None:
            print(message, flush=True)
            log.write(message + '\n')

        def run(arguments: list[str]) -> None:
            process = subprocess.Popen(arguments, cwd=resource, env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding='utf-8', errors='replace',
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            for line in process.stdout:
                if line.strip():
                    report(0, line.rstrip())
            if process.wait():
                raise RuntimeError(f'安装命令失败，退出代码 {process.returncode}；日志：{log_path}')

        # A failed repair must never leave a previously successful marker in use.
        layout.settings_path.write_text(json.dumps({'complete': False}), encoding='utf-8')
        try:
            if not portable.program_ready(resource):
                raise RuntimeError('运行资源不完整，请重新运行安装包。')
            python = layout.python
            if runtime == 'install':
                report(0.1, '[1/4] 安装独立 Python 与所选计算环境')
                if not python.is_file():
                    shutil.copytree(Path(sys.executable).parent, python.parent, dirs_exist_ok=True)
                (python.parent / 'python39._pth').write_text(
                    'python39.zip\n.\nLib/site-packages\n../python_packages\n..\n../vendor\nimport site\n',
                    encoding='utf-8')
                (python.parent / '.python-ready').touch()
                pip = subprocess.run([str(python), '-m', 'pip', '--version'],
                    capture_output=True, env=environment, timeout=30,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                if pip.returncode:
                    pip_script = resource / 'setup' / 'get-pip.py'
                    portable.download('https://bootstrap.pypa.io/pip/3.9/get-pip.py', pip_script, report)
                    run([str(python), str(pip_script), '--no-warn-script-location'])
                constraints = resource / 'runtime-constraints.txt'
                constraints.write_text(
                    f'torch==2.0.0+{device}\ntorchaudio==2.0.1+{device}\n'
                    'numpy<2\nscipy==1.11.4\nrotary-embedding-torch==0.6.4\nsetuptools<71\n', encoding='utf-8')
                run([str(python), '-m', 'pip', 'install', '--no-input', '--no-warn-script-location',
                     f'torch==2.0.0+{device}', f'torchaudio==2.0.1+{device}',
                     '--index-url', f'https://download.pytorch.org/whl/{device}'])
                packages = [value for value in bootstrap.MODULE_PACKAGES.values() if value]
                run([str(python), '-m', 'pip', 'install', '--no-input', '--no-warn-script-location',
                     '--constraint', str(constraints), 'setuptools<71', 'matplotlib<3.10', *packages])
            elif not python.is_file():
                raise RuntimeError('未找到已安装的运行环境；请返回勾选“Python 与运行依赖”。')

            report(0.4, '[2/4] 验证依赖和计算环境')
            state = bootstrap.probe_modules(python, [layout.packages, resource / 'vendor'])
            missing = [name for name, ready in state.items() if not ready]
            if missing:
                raise RuntimeError('缺少或无法加载依赖：' + '、'.join(missing))
            runtime_state = bootstrap.probe_python_runtime(python)
            if device == 'cu118' and not runtime_state.get('cuda_available'):
                raise RuntimeError('CUDA 无法使用，请检查 NVIDIA 驱动，或重新安装并选择 CPU。')
            if device == 'cpu' and runtime_state.get('cuda_build'):
                raise RuntimeError('现有环境与所选 CPU 模式不一致，请勾选运行依赖以切换环境。')

            report(0.6, '[3/4] 安装并检查全部提取模型')
            assets = {}
            for spec in bootstrap.COMPONENTS:
                destination = resource / spec.relative_path
                ready = bootstrap._component_ready(spec, destination)
                # User-reinforced weights are intentionally different from release hashes.
                if spec.key in {'wavlm_model', 'classifier_v2'}:
                    ready = ready or (bootstrap._component_quick_ready(spec, destination)
                                      and bootstrap._valid_file(destination / spec.check_file))
                if not ready:
                    if models == 'reuse':
                        raise RuntimeError(f'缺少 {spec.label}；请勾选“补齐全部提取模型”。')
                    report(0.6, f'正在安装：{spec.label}')
                    if spec.kind == 'direct':
                        for item in spec.files:
                            bootstrap._download_file(item, destination / item.filename,
                                                     report, 0.6, 0.2, spec.label)
                    elif spec.kind == 'modelscope':
                        bootstrap._download_modelscope(python, spec, destination, layout,
                                                       resource, report, 0.6)
                    else:
                        bootstrap._download_whisper(python, spec, destination, layout,
                                                     resource, report, 0.6)
                    if not bootstrap._component_ready(spec, destination):
                        raise RuntimeError(f'{spec.label} 安装后仍不完整。')
                assets[spec.key] = destination

            report(0.9, '[4/4] 检查后台程序与媒体工具')
            ffmpeg, ffprobe, uvr_code, sv_code = bootstrap.ensure_local_tools(layout, resource, None, report)
            context = bootstrap.BootstrapContext(resource, layout, python, assets,
                                                uvr_code, sv_code, ffmpeg, ffprobe, [layout.packages])
            environment.update(context.backend_environment())
            run([str(ffmpeg), '-version'])
            run([str(ffprobe), '-version'])
            run([str(python), '-c', 'import extractor.pipeline; import launcher.model_reinforcement; '
                 'import launcher.webui_server; from extractor.config import ensure_local_assets; '
                 'ensure_local_assets(); print("Backend imports and installed assets OK")'])
            settings = {
                'version': bootstrap.APP_VERSION, 'cache_schema': bootstrap.CACHE_SCHEMA_VERSION,
                'complete': True, 'install_root': str(root), 'python': str(python),
                'assets': {key: str(path) for key, path in assets.items()}, 'gpt_root': '',
                'ffmpeg': str(ffmpeg), 'ffprobe': str(ffprobe),
                'uvr_code': str(uvr_code), 'sv_code': str(sv_code), 'device': device,
            }
            temporary = layout.settings_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding='utf-8')
            temporary.replace(layout.settings_path)
            report(1.0, '安装完成，运行环境、模型和后台程序检查通过。')
        except Exception:
            log.write(traceback.format_exc())
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--runtime', choices=('install', 'reuse'), required=True)
    parser.add_argument('--models', choices=('install', 'reuse'), required=True)
    parser.add_argument('--device', choices=('cpu', 'cu118'), required=True)
    args = parser.parse_args()
    # nsExec reads output using the Windows code page; persistent logs use UTF-8.
    if os.name == 'nt':
        sys.stdout.reconfigure(encoding=locale.getpreferredencoding(False), errors='replace')
    try:
        install(args.root.resolve(), args.runtime, args.models, args.device)
    except Exception as error:
        print(f'安装失败：{error}', flush=True)
        sys.exit(1)
