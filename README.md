# Voice Extractor

根据一组参考音频，从多个目标音频或视频中提取指定说话人的完整讲话，并输出对应音频与 STT 文本。界面为中文原生 Windows 桌面程序，不启动浏览器，也不占用网页端口。

Windows 桌面安装程序可从 [GitHub Releases](https://github.com/guhechudaisuki/-Voice-Extractor/releases/latest) 下载。

## 功能

- 参考音频可多选。
- 待提取音频或视频可多选。
- 可添加任意数量的排除人物，每个人物可提供多段参考音频，也可以完全不添加。
- 自动过滤有人唱歌、纯音乐、无人声、多说话人重叠和非目标人物片段。
- 支持中文、日文、英文等语言的 STT 文本。
- 可输出目标人物结果，或输出清理后的全部单句供人工筛选。
- 视频输入可选择同时导出对应视频片段。
- 低响度语音会进行安全增益，便于整理训练数据。
- 多个目标文件共用一个批次目录和一个 ZIP 压缩包。
- 每次任务可单独选择结果保存位置，默认是安装目录下的 `output`。

## 桌面版使用

1. 启动 `VoiceExtractor.exe`。
2. 首次启动时只需选择安装目录，不需要填写或选择 `python.exe`。
3. 程序会完整检查环境变量、`PATH`、Windows `py.exe`、注册表、Conda/venv、常见 Python 安装目录和 GPT-SoVITS runtime，并实际验证 PyTorch、torchaudio 与 CUDA。
4. 同一次扫描还会检查全部模型、Hugging Face/ModelScope 常见缓存、FFmpeg、UVR 和 ERes2Net 后端。
5. 如果完全没有 Python，可在结果弹窗中确认自动安装官方 Python 3.9.13；不要求用户填写路径。
6. 如果 CUDA/PyTorch 未就绪，弹窗可直接打开 PyTorch 或 CUDA 官网，也可以停止安装。本工具不会擅自下载或替换 CUDA/PyTorch。
7. 第一次完整检查若全部满足，会自动保存永久依赖缓存并直接进入主界面，不再询问。
8. 只有发现缺失依赖时才询问是否安装。已存在的资源只读复用，不移动、不修改、不重复下载。
9. 后续启动直接读取永久缓存并进入主界面，不再导入 PyTorch 或重新扫描模型；只有缓存对应文件被删除或移动时才重新完整检查。
10. 进入中文桌面界面后，添加参考音频、待提取文件和可选排除人物，确认保存位置后开始提取。

安装目录结构：

```text
VoiceExtractor.exe
python/                 找不到 Python 时自动安装的工具运行环境
models/                 缺失时下载的模型
python_packages/        本工具独立补充的 Python 包
output/                 默认结果目录
work/                   处理缓存与任务配置
logs/                   桌面程序日志
tools/                  本工具独立使用的媒体工具
```

自动安装的 Python 下载量约 28 MB、安装后约占 120 MB。CUDA 和 PyTorch 不会由本工具下载或升级。

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

## 构建桌面 EXE

```powershell
.\scripts\build_launcher.ps1
```

生成文件：`dist/VoiceExtractor.exe`。构建结果是单文件、无控制台窗口的原生桌面程序。

## 资源清单

模型权重不提交到源码仓库。桌面版会先扫描本地资源，只下载缺失项；全部缺失时预计约需 3.3 GB。

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
