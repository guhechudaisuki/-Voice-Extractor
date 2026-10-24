# Voice Extractor

根据一组参考音频，从多个目标音频或视频中提取指定说话人的完整讲话，并输出对应音频与 STT 文本。界面为中文原生 Windows 桌面程序，不启动浏览器，也不占用网页端口。

Windows 桌面安装程序可从 [GitHub Releases](https://github.com/guhechudaisuki/-Voice-Extractor/releases/latest) 下载。

## 功能

- 参考音频可多选。
- 待提取音频或视频可多选。
- 每个待提取文件可绑定可选的 ASS/SSA/SRT/VTT 字幕，仅使用时间轴辅助切分与补全。
- 可添加任意数量的排除人物，每个人物可提供多段参考音频，也可以完全不添加。
- 自动过滤有人唱歌、纯音乐、无人声、多说话人重叠和非目标人物片段。
- 支持中文、日文、英文等语言的 STT 文本。
- 可输出目标人物结果，或输出清理后的全部单句供人工筛选。
- 视频输入可选择同时导出对应视频片段。
- 低响度语音会进行安全增益，便于整理训练数据。
- 多个目标文件共用一个批次目录和一个 ZIP 压缩包。
- 每次任务可单独选择结果保存位置，默认是安装目录下的 `output`。

## 桌面版使用

1. 从 [v3.0 Release](https://github.com/guhechudaisuki/-Voice-Extractor/releases/tag/v3.0) 下载 `VoiceExtractor.exe`，放入准备长期使用的文件夹。
2. 双击 EXE，程序自动下载运行文件、便携 Python、PyTorch、其他依赖和全部模型。无需选择安装目录，也无需手动下载或移动模型。
3. 文件固定保存在 **EXE 同级的 `resource`** 中。首次安装需要联网和足够的磁盘空间；下载失败时点击“重试安装”，完整模型会保留。
4. 安装完成自动进入主界面。以后启动直接使用已下载的文件。添加参考音频、待提取文件和可选排除人物，开始提取。

```text
VoiceExtractor/
├── VoiceExtractor.exe
├── README.md                    首次安装后自动生成的使用说明
├── resource/                    自动下载，用户无需整理
│   ├── desktop_worker.py        后台程序
│   ├── extractor/、launcher/、vendor/、dependencies/
│   ├── python/                  便携 Python，不修改系统 PATH/注册表
│   ├── python_packages/         PyTorch 和其他运行依赖
│   ├── model/                   自动下载的声纹和语音识别模型
│   ├── models/                  自动下载的分类器、歌声与重叠检测模型
│   ├── tools/                   FFmpeg、FFprobe
│   ├── cache/                   包下载与模型缓存
│   └── temp/                    临时文件
├── output/                      默认提取结果
├── work/                        任务文件
├── logs/                        程序日志
└── voice_extractor_installation.json
```

需要移动程序时，退出后移动整个文件夹，保留 `resource`。程序不依赖其他软件的运行环境和模型目录，不把安装文件存入 AppData。输入素材仍从用户选择的位置读取。

首次安装可能下载数 GB 文件；PyTorch 使用自带 CUDA 运行库，无需另装 CUDA Toolkit。使用 GPU 加速需有兼容的 NVIDIA 显卡驱动。

## 处理流程

```text
参考音频
    -> 多参考音色建模

目标音频或视频
    -> 检测并移除有人演唱区间（纯器乐不按歌声处理）
    -> UVR 人声分离与环境音清理
    -> 讲话/静音边界检测
    -> 静音范围合并与硬切分
    -> 多人重叠和说话人切换检查
    -> 多声纹模型目标人物复核
    -> 可选排除人物降分与否决
    -> 完整句边界修正
    -> 响度整理
    -> STT
    -> 音频、文本、清单与批次 ZIP
```

STT 是最后一步，不参与决定说话人身份。

### 可选字幕

在待提取列表中选中一个文件，点击“绑定字幕”；“移除字幕”可取消选中文件的绑定。不提供字幕时保持原流程。

字幕先与人声检测结果估计时间偏移，再在字幕边界附近寻找实际停顿；找不到可靠停顿时不会直接按时间戳切声音。字幕标出的漏检区域会局部重做 VAD，并继续经过歌声、多人重叠和声纹检查。同一条字幕内被拆开的片段只有在新增部分、说话人连续性与整句都通过检查后才补全，不跨越静音上限或已确认的换人区间。

中文字幕、翻译字幕均可作为时间提示，但不会提供人物身份，也不会替换、提示或纠正 STT；日语原声仍转写成日文。歌词和屏幕文字样式不用于对话补全。时间轴无法可靠对齐时回退到原流程，并在日志中说明。

`manifest.json` 的 `subtitle_assistance` 记录时间偏移、实际停顿细分、局部补检和补句结果。字幕原文仅保留在清单的参考信息中，不写入训练用 TXT。

## 输出内容

每个目标文件包含：

- `audio/`：通过检查的 WAV 单句。
- `text/`：与 WAV 同名的 STT 文本。
- `video/`：启用视频片段输出时生成。
- `manifest.json` 和 `manifest.csv`：时间、文本、分数与处理信息。
- `transcript.srt`：带时间轴的文本。

批次目录还包含统一的批次清单、批次字幕和一个包含所有目标文件结果的 ZIP。

## 界面选项

- `声纹阈值`：默认值是当前推荐值；调高更保守，调低可能增加召回，也可能混入相近声音。
- `静音下限`：短于该值的停顿按连续讲话处理。
- `静音上限`：长于该值的停顿直接切成两个句段。
- 位于上下限之间的停顿：仅当前后声纹一致时合并，并保留原停顿。
- `输出全部单句`：完成歌声、纯音乐、无人声和多人重叠清理后输出全部静音分隔单句及 STT，供人工筛选。
- `同时输出视频片段`：目标为视频时导出对应画面片段。
- `结果保存位置`：可为当前任务另选目录；留空会自动恢复为安装目录下的 `output`。

## 命令行

源码方式也可以使用批处理入口：

```powershell
python extract_cli.py `
  --reference ref-a.wav --reference ref-b.wav `
  --target episode-01.mp4 --target episode-02.wav `
  --threshold 0.68 --silence-min 0.20 --silence-max 0.85 `
  --video-segments
```

人工筛选模式增加 `--all-sentences`。

指定字幕绑定可增加 `--subtitle episode-01.mp4 episode-01.sc.ass`，多文件逐个重复此参数；不会按输入顺序猜测字幕属于哪一集。

## 构建桌面 EXE

```powershell
.\scripts\build_launcher.ps1
```

生成文件：`dist/VoiceExtractor.exe` 和 `dist/VoiceExtractor_resources.zip`。将两者上传到 v3.0 Release；EXE 首次启动会下载并验证运行文件 ZIP，解压到自己的 `resource`。构建结果为无控制台窗口的单文件 EXE。

## 自动安装的模型

模型自动下载到 `resource/model` 和 `resource/models`。WavLM 使用 v3.0 Release 中与本地版本一致的权重，人物分类器使用仓库中的 `classifier_v2.pt`。其他模型继续从原模型站点下载。用户无需手动放置文件；训练素材、标注和本地输出不随发布包分发。

| 资源 | 用途 |
| --- | --- |
| UVR5 HP2 | 人声分离 |
| ERes2NetV2 | 主声纹匹配 |
| CAM++ | 第二声纹匹配 |
| WavLM SV | 歧义窗口复核 |
| WeSpeaker ONNX | 剩余歧义裁决 |
| 重叠检测模型 | 多人同时说话检测 |
| PANNs | 有人唱歌检测 |
| Paraformer、FSMN-VAD、CT-Punc | 中文识别、讲话检测与标点 |
| Whisper large-v3-turbo | 日文、英文等语言识别 |

手动资源脚本：

```powershell
.\scripts\download_assets.ps1 -WhatIf
.\scripts\download_assets.ps1
.\scripts\check_assets.ps1
```

## 源码结构

```text
desktop_worker.py               桌面后台任务进程
extract_cli.py                  命令行批处理入口
extractor/                      核心提取流程
launcher/                       原生桌面与安装检测代码
scripts/                        资源和构建脚本
vendor/                         随源码提供的轻量组件
```

## 许可

项目源码使用 MIT License。外部资源按各自许可证使用。
