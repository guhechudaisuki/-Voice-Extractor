# 冻结记录：VoiceExtractor 2.02（高召回簇级判决 + Tier A/B 复核工作台）

> 本文件所在提交即冻结点（tag `v2.02`）。只记录事实与复现命令；数字均为**离线回放口径**，不构成泛化或人工听音结论。按 `docs/improvement_plan_20260929.md` 附录 D 基线冻结协议与 §6.4 发布检查单执行；**未 push、未发布 Release**（等待所有者明确要求）。

## 冻结内容（相对 2.01 基线 `a287fe1`）

- **Stage 1**：`extractor/nextgen/identity_cluster.py` —— 多原型 Top-2 参考打分、目标簇联盟（取代 `ambiguous_target_cluster` 整集弃权）、中位数离群基准 + Cohort AS-Norm 竞争守卫、双通道排除余量、类目标簇不入冒认者队列。
- **Stage 2**：`cluster_runtime.py` + `cluster_gate.py` —— 候选句级滑窗核验（不再父轮次连坐）、长句换气平滑、连续干净子区间裁切、窗口级 Cohort 守卫、救援标记严审、混合句裁切、分类器冲突仲裁池、Tier A/B 分流。
- **Stage 3**：`ClusterAnalyzer` 第三通道（强化后 WavLM，独立缓存 `cache/wavlm_embeddings`）、第四通道 WeSpeaker 仲裁（救援标记 Tier A 否决票）、Stem+Raw 双轨边缘复核（只救不杀）；`launcher/model_reinforcement.py` 加性角度边际（AAM）+ L2-SP 防遗忘。
- **Stage 4**：Desktop/WebUI Tier A·B·撤下三档筛选、一键信任主输出、partial 保留段试听与裁切导出、GPT-SoVITS `train.list`、困难负例自动入强化包、`_borderline/` 边缘池导出、`write_clip` 15ms 淡入淡出。
- **计划补齐项**：ASS `Actor` 字段解析与 `overlapping_speech_spans()` 多人重叠拦截、零静音能量谷底切分（`split_at_subtitle_pauses` 升级）、STT 分片幻觉过滤 + 声纹/字幕双证据精裁正文 + 字幕正字。

## 验收证据（复现命令）

```
runtime/python.exe -B evaluation/verify_high_recall_cluster.py --json <out>.json
runtime/python.exe -B -m unittest discover -s tests        # 518 项全部通过
```

回放（6 集，`output/20261008_002053_batch_3866bb` + 双嵌入缓存，热缓存约 25–60 秒）：

| 门禁 | 结果 |
|---|---|
| 精度：Ep1–3 已知错人拦截 | **97.56%**（41 个负例泄漏 1 处：Ep3 `[550.64,555.74]` 高分同声线，四模型不可分，走人工复核） |
| 召回：Tier A 主输出 | 23/28/20/28/21/33 段 vs enforce 基线 7/5/9/0/15/6，**平均 2.86 倍**；Ep4 从 0 → 28 |
| 覆盖：Tier A ∪ Tier B | 人工确认正确句 Ep1 **95%** / Ep2 **92.5%** / Ep3 **100%** |
| 单元测试 | **518 项通过**（含 `test_high_recall_cluster.py` 18 项、`test_plan_completion.py` 12 项） |
| Smoke | cluster/subtitle/bootstrap/desktop_switch/training 浏览器 smoke、`tier_ui_smoke.cjs`（新增）、原生桌面全流程（真实 WavLM 训练）、训练工作流（真实 WavLM 训练）、desktop_subtitle 均通过 |

证据归档：`docs/high_recall_stage1_4_20261023.json`、`docs/high_recall_replay_final_20261023.json`、`docs/plan_completion_20261024.json`、`docs/plan_completion_replay_20261024.json`（回放数字与本冻结树复算一致）。

## 模型与参考指纹

`work/freeze_model_hashes.json`（本地，因 `work/` 不入库）：目标参考 6 文件、排除组 5 文件、classifier_v2 / ERes2NetV2 / CAM++ / WavLM-SV / WeSpeaker 五模型 sha256。B0 附录 E 的"待补齐"在本冻结点补齐。

## 已知弱点与未完成（继续工作入口）

1. **Stage 5–8 未完成**（`计划方案/实施方案.md` §4）：零泄漏最后一处泄漏、Tier A 内部父子重叠消解、Tier B NMS 瘦身（当前每集 109–167 段）、Tier B 补 STT 台词、`auto_negatives.npz` 负例原型库。**Stage 5–6 首轮实验的代码快照存于 `work/stage56_wip/`**（本地）；实验结论：逐窗软兜底收紧（-0.045）与裁切父比率门在实测中净伤召回/覆盖，已回退；有效组件是 WeSpeaker 仲裁（拦截 95.12%→97.56%）与 Raw 双轨救回（+2 轮/6 集）。
2. `training_desktop_smoke.py` 断言 Edge 套壳时代 `edge_app` 模式，已被原生桌面流程取代，未删除（规则：不经批准不删）。
3. 第 3 集 `[500.31,501.80]` 已由 WeSpeaker 仲裁拦截；`[550.64,555.74]` 建议人工听音后标 `wrong` 入强化闭环。
4. `audio.py` 15ms 淡入淡出改变 `encoding_version`：换机首次回放会整体重算嵌入（约 10 分钟），之后热缓存。
5. 本提交同时包含此前各轮（簇自举、桌面原生端、训练闭环）的未提交工作——冻结点即当前完整工作版本。

## 回滚

`git checkout v2.01`（tag 不存在则 `a287fe1`）即回 2.01 基线；模型 `.previous` 备份机制不变。


---

# 定档补记：v2.03（Stage 9(3)、11(2)、12 尾部项补齐 —— Stage 1–12 计划 100% 完整）

本节追加于 v2.02 定档之后。三个尾部项全部落地，**计划书 Stage 1–12 至此无遗留**。

## 本轮新增

1. **Stage 9(3)：classifier_v2.pt 参考重绑定（`launcher/model_reinforcement.py: rebind_classifier_references`）**
   模型强化成功后自动执行：用本次标注的正/负片段对冻结分类头做**阈值再校准**（veto 阈值压到全部正例窗口分之下、rescue 阈值抬到全部负例整句均分之上，0.1 安全边距），并把 `reference_sha256` 重绑到本次参考集——"一换参考音频 0.5s 局部分类器就静默罢工"的历史顽疾就此关闭。权重不动、原子写入、`.previous` 备份；多参考集会话与正负分布重叠时保守跳过。
2. **Stage 11(2)：partial 保留段台词修正输入框**
   Desktop（`askstring` 预填 `_partial_clip_text` 自动切片结果）与 WebUI（`#partialText` 输入框）均可编辑 `review.partial_text`；导出端（`d497067` 已落地的 `_partial_clip_text`）按 显式填写 > 字幕子片段 > 比例切片 优先级取词，`train.list` 音字 100% 对齐。`clean_reviews` 白名单与 `ItemReview` 模型已放行该字段。
3. **Stage 12：run_many 批内跨集高纯度共享池（`cluster_cross_episode_pool` 选项，默认开）**
   每集簇级判决完成后，收集四模型一致满分核心句（p≥0.82 / s≥0.80 / WavLM≥0.93 / ≥2.0s / 非边缘 / 不与排除区间重叠）入批内池（`deduplicate` 去重 + 24 条上限按 confidence 降序）；后续集数的 `verify_temporal` 在原始剖面失败时尝试跨集合并剖面（`merge_profile` 一半权重仍锚在用户原始参考、参考下限不变、全部窗口 strict 通过才放行），命中窗口标记 `reference="cross_episode"` 并计入 `cross_episode_rescued_turns`。单文件自举的同源豁免语义不受影响。

## 验证（v2.03 定档口径）

- 离线回放（`docs/high_recall_stage5_11_20261024.json`）：**拦截 41/41 = 100%、Tier A 重叠对 0（6 集）、覆盖 95/92.5/100% 一分不掉**，四门禁全过。
- 单元测试 **532 项全过**（新增重绑定×2、跨集池×2、partial_text 管道×2）。
- Smoke：训练工作流 + 原生桌面全流程（真实 WavLM 训练 + 真实重绑定 + 原型库，**新增生产 classifier_v2.pt 哈希不变断言**）；tier/training/cluster 三个浏览器 smoke 通过（QA 18784）。
- 说明：真实整集端到端推理与人工听音按所有者要求留待 UI 实跑；4→6 个本地提交未 push。

## 定档

tag `v2.03` = 本文件所在提交。回滚链：`v2.03 → v2.02 → v2.01(a287fe1)`。Stage 1–12 全部完成，无计划内遗留。
