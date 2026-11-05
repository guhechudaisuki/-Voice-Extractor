# Voice Extractor

根据参考音频，从音频或视频中提取指定说话人的讲话，整理为带文本的语音数据集。提供 Windows 桌面界面、浏览器工作台和命令行入口。

## 功能

- 多参考音频建模与自适应说话人聚类，处理同一人物的不同声线；可添加排除人物参考。
- 批量处理音频和视频，检查歌声、多人重叠、说话人切换与句段边界。
- 可绑定 ASS、SSA、SRT、VTT 字幕，用时间轴辅助切分；语音识别独立生成文本。
- 筛选结果可逐条试听、局部裁切和盲听；需要人工筛选所有人物的讲话时，可开启“输出全部单句”。
- 角色簇探索与参考种子导出，便于补充参考音频或提取其他人物。
- 声学质量分级、重复内容检查和训练集导出，可生成 GPT-SoVITS `train.list`。
- 使用人工标注的训练 ZIP 强化模型；成功保存的权重供后续任务使用。

聚类和自动筛选可能受到配乐、混响、重叠讲话及相近声线影响，训练集导出前建议试听复核。

## 安装与启动

从 [GitHub Releases](https://github.com/guhechudaisuki/-Voice-Extractor/releases) 下载 `VoiceExtractor.exe`，放入准备长期使用的文件夹后启动。首次安装需要联网，会下载运行资源、Python、依赖与模型，通常需要数 GB 磁盘空间。安装器按发布配置校验文件；下载中断后可重试。

运行资源保存在 EXE 同级的 `resource` 中。移动安装目录时，退出程序并保留整个文件夹。使用 NVIDIA GPU 加速需要兼容的显卡驱动。

```text
VoiceExtractor/
├── VoiceExtractor.exe
├── resource/        程序、运行环境、媒体工具和模型
├── output/          默认提取结果
├── work/            任务中间文件
└── logs/            运行日志
```

源码入口使用已配置依赖的 Python 环境：

```powershell
python launcher/app_entry.py --ui desktop
python launcher/app_entry.py --ui webui
```

桌面模式使用原生窗口；浏览器模式启动本机服务并打开工作台。源码当前功能与已发布安装包可能不同，请以对应 Release 的说明为准。

## 提取与复核

1. 添加目标人物的参考音频，尽量选择少配乐、无重叠、人物明确的片段。
2. 添加待处理文件，按需绑定字幕和排除人物参考，选择结果保存位置。
3. 选择导出方式。默认将声学和字幕两路已选时间轴合并、去重后导出；开启“分流检验”时追加声纹与边界检验。“输出全部单句”另外导出最终目标结果之外的全部干净单人单句，单独保存供人工复核，剩余句子不按目标声纹删减。各模式均使用分离干声，保留歌声与重叠过滤。
4. 在复核工作台逐条试听导出结果；局部混入他人声音的片段可标记区间。
5. 按质量等级导出数据集，需要时勾选 GPT-SoVITS 清单或附带训练原料。

复核快捷键：

| 按键 | 操作 |
| --- | --- |
| `1` | 目标人物 |
| `2` | 非目标人物 |
| `3` | 局部混入其他人物 |
| `4` | 歌声 |
| `5` | 多人重叠 |
| `6` | 跳过 |
| `Space` | 播放或停止 |

输入框内不会触发标注快捷键。角色探索可试听聚类结果，并导出参考 WAV 用于后续提取。

质量等级包含 `Gold`、`Silver` 和 `Review`，用于筛选声学条件，不代替人物身份判断。导出时可选择最低等级；裁切后的音频会重新检查质量。

训练音频始终来自分离后的干声，不再提供原轨高频回填。旧客户端提交该选项也不会混入原轨。训练清单优先使用人工修订和最终转写；局部裁切只使用人工填写的局部文本或完整覆盖的可靠 ASR 时间戳，不根据时长比例猜词，也不把翻译字幕当作训练文本。文本未确定的干声音频仍保留，暂不写入 `train.list`、`slicer_opt.list` 和 `metadata.jsonl`。

### 字幕与文本

字幕只提供时间信息，不决定人物身份，也不代替语音识别。翻译字幕可辅助原声切分，输出文本仍以识别语言为准。时间轴无法可靠对齐时，程序回退到音频切分。

### 输出文件

每个目标文件对应音频、文本与处理清单：

- `audio/`：WAV 句段。
- `text/`：与音频同名的转写文本。
- `remaining_audio/`、`remaining_text/`、`remaining_manifest.json`：开启“输出全部单句”后另外保存的剩余单句，和最终目标范围不重叠；人工确认后才可进入训练导出。
- `video/`：启用视频片段输出时生成。
- `manifest.json`、`manifest.csv`：时间、文本、分数与处理信息。
- `transcript.srt`：带时间轴的文本。

批次目录包含汇总清单和 ZIP。复核导出的数据集可附带 `train.list`；附带训练原料时，标注和音频另存为训练 ZIP。

## 命令行

```powershell
python extract_cli.py `
  --reference reference-a.wav --reference reference-b.wav `
  --target input-01.mp4 --target input-02.wav `
  --threshold 0.68 --silence-min 0.20 --silence-max 0.85 `
  --video-segments
```

增加 `--all-sentences` 可追加输出最终目标结果之外的全部干净单句供人工筛选；`--branch-validation` 开启合流追加检验，默认关闭。字幕按目标文件绑定，例如 `--subtitle input-01.mp4 input-01.ass`；多个目标可重复指定。排除人物使用 `--exclude-role reference-other.wav`，同一参数后的参考属于同一人物。

## 模型与资源

安装器在 `resource/model` 和 `resource/models` 下管理模型。主要资源包括 UVR5 人声分离、ERes2NetV2 与 CAM++ 声纹匹配、WavLM 与 WeSpeaker 复核、歌声及重叠检测，以及中文和多语言语音识别模型。

`models/anime_t3/classifier_v2.pt` 保存全局参考条件分类器的校准参数，身份评分使用当前任务参考音频与 WavLM 投影。安装器从 [模型资源发布页](https://github.com/guhechudaisuki/-Voice-Extractor/releases/tag/model-assets) 下载经过多说话人配对训练的 WavLM 权重及角色、声优声纹 ONNX 模型，并校验 SHA-256。WavLM 仅更新说话人投影层，保留预训练骨干；同一模型服务于不同目标参考，无须为每个角色建立独立模型。

模型强化成功后保留本地权重。训练素材、标注记录及本机配置不进入发布包。模型校验值更新时应同时更新安装配置与发布清单。

全局 WavLM 强化同时检查留出来源上的整体指标和逐条参考配对决策：旧的正确正例不能变成漏判，旧的正确负例不能变成误收。新的正确结果不能抵消另一条旧结果的退化。该检查针对验证样本，不代表对未标注角色和视频的准确率保证；所有角色仍共用一套全局权重、回放池和备份回滚。

普通提取默认只在内存中复用声纹特征；`VOICE_EXTRACT_PERSIST_CACHE=1` 可显式启用研发回放缓存。UVR 逐块拼接并及时清理临时分块；高采样率输出在同卷文件系统上通过硬链接保留干声母带，跨卷时才复制，不再生成额外整集原轨母带。

资源脚本：

脚本默认从仓库同级的 `GPT-SoVITS` 目录查找运行组件；其他位置可通过 `-GptRoot` 参数或 `VOICE_EXTRACT_GPT_ROOT` 环境变量指定。

```powershell
.\scripts\download_assets.ps1 -WhatIf
.\scripts\download_assets.ps1
.\scripts\check_assets.ps1
```

## 构建与发布

```powershell
.\scripts\build_launcher.ps1
```

构建脚本生成 `dist/VoiceExtractor.exe` 和 `dist/VoiceExtractor_resources.zip`，并更新资源包校验值。EXE 与资源包应成对发布到安装器配置的 Release，避免下载地址和 SHA-256 不一致。外部运行组件由构建脚本指定的依赖目录提供。

```text
desktop_worker.py    桌面后台任务
extract_cli.py       命令行入口
extractor/           提取、聚类与声学处理
launcher/            界面、复核、强化与安装管理
scripts/             资源检查、打包与构建
vendor/              随源码分发的轻量依赖
```

## 许可

项目源码使用 [MIT License](LICENSE)。第三方代码、预训练模型及输入素材按各自许可使用。
