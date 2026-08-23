# 参考音色句子提取器

根据一组参考音频，从一组目标音频中提取指定说话人的完整讲话句子，并输出可用于语音数据整理的音频和文本。

## 功能

- 参考音频支持多选。
- 目标音频支持多选。
- 可添加多个排除角色，每个角色支持多段参考音频。
- 匹配指定说话人的连续讲话回合。
- 自动排除多人同时说话、歌声、纯音乐和无人声空段。
- 在语音识别之后输出对应文本，支持中文、日文和英文等语言。
- 对低响度语音进行安全增益，避免导出的训练音频过小。
- 所有目标文件统一生成一个批次压缩包。

## 处理流程

```text
参考音频
    -> 参考音色建立
目标音频
    -> 人声分离
    -> 讲话检测
    -> 多人/歌声过滤
    -> 说话人切分
    -> ERes2Net / CAM++ 指定音色匹配
    -> WavLM 歧义窗口复核
    -> WeSpeaker 剩余歧义裁决
    -> 可选排除角色复核
    -> 完整句子修正
    -> 语音识别
    -> 音频、文本和清单
```

## 输出内容

每个目标文件包含：

- `audio/`：匹配到的单人讲话 WAV 文件。
- `text/`：与 WAV 同名的文本文件。
- `manifest.json`：句子时间、文本、匹配分数和处理结果。
- `manifest.csv`：便于表格处理的清单。
- `transcript.srt`：带时间轴的文本。

批次目录还包含：

- `batch_manifest.json`
- `batch_manifest.csv`
- `batch_transcript.srt`
- 一个包含全部目标文件结果的 ZIP 压缩包。

## 使用

1. 在“参考音频”中选择同一个人的一段或多段讲话。
2. 在“待提取音频”中选择一个或多个目标音频。
3. 如果某些主要配角色与目标人物音色接近，可点击“添加一个排除角色”，并为每个角色上传一段或多段同一人的讲话。
4. 保持多人和歌声过滤开启；程序会自动完成多模型分工复核和完整句检查。
5. 点击开始，等待批次完成。

仓库内提供两个入口：

- `Start Voice Extractor.bat`：图形界面入口。
- `extract_cli.py`：批处理入口。

预构建启动器： [dist/VoiceExtractor.exe](dist/VoiceExtractor.exe)

首次运行 `VoiceExtractor.exe` 时会自动检查本地 Python、音频工具和模型资源：

- 已存在的资源会直接复用，不会重复下载。
- 缺失的 UVR、声纹、多人检测、歌声检测、FunASR 和 Whisper 资源会自动下载到对应目录。
- CUDA、PyTorch 和本地 Python runtime 不会由启动器下载；它们需要先随运行环境准备好。
- 启动器完成检查后才会打开中文网页界面。

仅检查和补齐资源、不启动界面：

```powershell
.\dist\VoiceExtractor.exe --check-only
```

## 资源清单

以下资源需要单独准备，资源文件不包含在源码提交中：

| 文件 | 用途 |
| --- | --- |
| `HP2_all_vocals.pth` | 人声分离 |
| `pretrained_eres2netv2w24s4ep4.ckpt` | 主声纹匹配 |
| `campplus_voxceleb.bin` | 二次声纹匹配 |
| `wavlm-base-plus-sv/` | 双模型歧义窗口与完整句复核（约 386 MiB） |
| `wespeaker-resnet34-lm/onnx/model.onnx` | 剩余声纹歧义的确定性第四模型裁决（约 25.3 MiB） |
| `model.onnx` | 多人同时说话检测 |
| `Cnn10_mAP=0.380.pth` | 歌声检测 |
| `class_labels_indices.csv` | 歌声检测标签 |
| Paraformer 模型文件 | 中文文本识别 |
| FSMN-VAD 模型文件 | 中文讲话检测 |
| CT-Punc 模型文件 | 中文标点恢复 |
| Whisper large-v3-turbo 模型文件 | 日文、英文等文本识别 |

资源下载模板：

```powershell
.\scripts\download_assets.ps1 -WhatIf
```

确认下载目标后执行：

```powershell
.\scripts\download_assets.ps1
```

资源检查：

```powershell
.\scripts\check_assets.ps1
```

## 源码结构

```text
app.py                         图形界面
extract_cli.py                 批处理入口
extractor/                     核心提取流程
launcher/                      启动器源码
scripts/                       资源检查、下载和构建脚本
vendor/                        随源码提供的轻量组件
```

## 许可

本项目源码使用 MIT License，见 [LICENSE](LICENSE)。外部资源按其各自许可证使用。
