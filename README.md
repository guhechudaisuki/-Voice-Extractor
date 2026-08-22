# 参考音色句子提取器

从一个或多个参考音频中建立目标说话人声纹，在一个或多个目标音频中提取该人物的完整讲话回合，并输出纯人声 WAV、逐句 TXT、SRT 和批次清单。

本仓库只包含提取程序本体和部署脚本。模型、CUDA/PyTorch、GPT-SoVITS runtime、Whisper 缓存及用户音频均为外部资源，不提交到 Git。

## 功能

- 多段参考音频和多目标音频批量处理。
- UVR5 人声分离，去除纯音乐和无人声空段。
- ERes2NetV2 + CAM++ 双模型声纹匹配。
- 先检测重叠说话和歌声，再切分说话人，最后执行 STT。
- 支持中文、日文、英文等 Whisper 文本；中文可用本地 Paraformer 精修。
- 对目标人声的自然响度做安全增益：只提升过低音量，不压低正常音量，并限制峰值。
- 最终生成一个包含所有目标文件结果的批次 ZIP。

## 目录

```text
app.py                         Gradio 图形界面
extract_cli.py                 命令行入口
extractor/                     提取、分离、声纹、过滤和 STT 实现
vendor/                        轻量 Python 源码依赖
scripts/check_assets.ps1       检查外部资源是否齐全
scripts/download_assets.ps1    下载公开模型/资源的命令模板
scripts/build_launcher.ps1    构建 Windows 启动器 EXE
launcher/                      启动器源码
启动提取工具.bat                直接启动脚本
```

以下目录不会进入仓库：`models/`、`work/`、`output/`、`runtime/`、`omnvoice/`、本地音频和 Python 缓存。

## 外部依赖清单

| 依赖 | 用途 | 放置位置或要求 |
| --- | --- | --- |
| GPT-SoVITS v2Pro runtime | Python、FFmpeg、UVR5、FunASR、ERes2Net 代码 | 工作区根目录 `GPT-SoVITS-v2pro-20250604/` |
| CUDA + PyTorch | GPU 推理 | 使用本机已有安装；不要在仓库内重复打包 |
| UVR5 `HP2_all_vocals.pth` | 人声分离 | GPT-SoVITS `tools/uvr5/uvr5_weights/` |
| ERes2NetV2 checkpoint | 主声纹模型 | GPT-SoVITS `GPT_SoVITS/pretrained_models/sv/` |
| Paraformer、FSMN-VAD、CT-Punc | 中文 VAD/STT | GPT-SoVITS `tools/asr/models/` |
| Whisper large-v3-turbo | 日文/英文等 STT | `omnvoice/hf_cache/` 下的 Hugging Face cache |
| CAM++ VoxCeleb checkpoint | 第二声纹模型 | `models/campplus_voxceleb/campplus_voxceleb.bin` |
| overlap ONNX model | 多人同时说话检测 | `models/overlap/model.onnx` |
| PANNs Cnn10 + labels | 歌声检测 | `models/panns/` |

模型许可证和来源以各上游项目为准。部署时请遵守 GPT-SoVITS、FunASR、Hugging Face、ModelScope 及各模型自身许可证。

## 快速部署

要求 Windows 10/11、Python 3.9+（推荐使用 GPT-SoVITS 自带 runtime）。CUDA 和 PyTorch 已经存在时，不要再次安装 CUDA/PyTorch。

1. 克隆本仓库：

   ```powershell
   git clone https://github.com/guhechudaisuki/- 提取
   cd 提取
   ```

2. 将 GPT-SoVITS v2Pro 放在仓库同级目录，或设置自定义 runtime：

   ```powershell
   $env:VOICE_EXTRACT_PYTHON = 'D:\AI\GPT-SoVITS-v2pro-20250604\runtime\python.exe'
   ```

3. 下载/准备模型后检查资源：

   ```powershell
   .\scripts\check_assets.ps1 -WorkspaceRoot (Resolve-Path '..')
   ```

4. 启动：

   ```powershell
   .\启动提取工具.bat
   # 或
   ..\GPT-SoVITS-v2pro-20250604\runtime\python.exe app.py
   ```

   浏览器访问 `http://127.0.0.1:7865`。

## 下载命令模板

脚本不会下载 CUDA/PyTorch，也不会覆盖已有文件。先查看命令：

```powershell
.\scripts\download_assets.ps1 -WorkspaceRoot (Resolve-Path '..') -WhatIf
```

确认来源后执行：

```powershell
.\scripts\download_assets.ps1 -WorkspaceRoot (Resolve-Path '..')
```

脚本使用 `git`、`huggingface-cli`/`hf` 和 `modelscope`（若已安装）。网络受限时，可手动下载后放到上表路径，再运行 `check_assets.ps1`。

## 命令行提取

```powershell
..\GPT-SoVITS-v2pro-20250604\runtime\python.exe extract_cli.py `
  -r ref_1.wav -r ref_2.wav `
  -t target_1.wav -t target_2.wav
```

输出位于 `output/`：每个目标文件的 WAV/TXT、单文件清单、批次 `batch_manifest.json/csv`、合并 SRT 和一个总 ZIP。

## 构建 EXE 启动器

EXE 是轻量启动器，负责定位外部 Python/runtime 并启动 `app.py`；模型和 CUDA 仍然外置，这是可维护且不会产生几十 GB 单文件的部署方式。

```powershell
.\scripts\build_launcher.ps1
```

生成：`dist/VoiceExtractor.exe`。把 `dist/VoiceExtractor.exe` 放在本仓库根目录附近运行，或设置 `VOICE_EXTRACT_PYTHON` 指向 GPT-SoVITS runtime。首次启动前仍需完成模型清单中的下载。

仓库也提供一个预构建的轻量启动器：[dist/VoiceExtractor.exe](dist/VoiceExtractor.exe)。它不包含模型、CUDA 或 PyTorch，运行时仍需准备外部资源。

## 日志与故障排查

- 服务日志：`output/app.log`
- 资源检查：`scripts/check_assets.ps1`
- 端口冲突：设置 `$env:VOICE_EXTRACT_PORT = '7866'` 后重新启动。
- 4 GB 显存建议一次只处理一个批次。

## 许可

提取程序源码的许可证为 MIT，见 [LICENSE](LICENSE)。外部模型、GPT-SoVITS 和 vendor 依赖继续遵守其原始许可证。
