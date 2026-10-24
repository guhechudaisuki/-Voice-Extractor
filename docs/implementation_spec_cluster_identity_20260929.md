# Voice Extractor 可实施方案：簇级身份判决

日期：2026-09-29
基线：main `a287fe1`（2.01）
性质：**可实施的工程方案，不由我执行**。前一份 `improvement_suggestions_20260929.md` 是方向层，本文把它落到文件、函数、数据结构、阈值来源、测试与验收。凡未验证的一律标「待复核」。

---

## 0. 一页结论

### 已经存在的东西，比你以为的多

`extractor/pipeline.py:3280` `_promote_global_target_cluster` **已经在做半件这件事**：它把已接受的强轮次平均成两个质心（ERes2Net 主通道 + CAM++ 副通道），再去把被拒的轮次救回来。它证明了两件事：质心方案在你的数据上能跑通；双模型一致性可以作为独立证据。

但它有四个缺陷，正好对应第三集的错误：

| 缺陷 | 代码证据 | 后果 |
|---|---|---|
| 单向：只加回，从不撤下 | 遍历对象是 `rejected`（3335 行） | 已接受但错人的段无人质疑 |
| 它自己就是救援路径 | 带 `global_target_cluster` 标记 | 救回的段没被复核过 |
| 只处理一种拒绝理由 | 3336 行 `!= "声纹匹配不足"` 直接留 | 其他理由被拒的段永不进入这个视图 |
| 没有"簇"概念 | 单段对质心打分（3352–3353） | 相似女声无法表达"她自成一群" |

所以方案不是「新增一个聚类模块」，而是**把这件事从"救援"改成"判决"，并把方向反过来**：先聚类、再定目标簇，接受集合由目标簇**双向**决定——簇内的进、簇外的出、已接受但不在簇内的**也出**。

### 真正稀缺的东西

`_build_adaptive_speaker_profile`（2842 行）已经把「集内锚点」做出来了。它和 `_promote_global_target_cluster` 的关系是：前者造域匹配声纹，后者拿它去救。这两者合起来，缺的只有**否定能力**。

一个具体的数字：建自适应 profile 的门槛是 `len(anchors) >= 3`（2866 行），而第三集有 254 个说话轮次。**254 个轮次里只用 3 个当锚点。** 聚类要做的就是把 254 个全部用上。

### 改动规模（真实估计）

| 文件 | 性质 | 行数 |
|---|---|---|
| `extractor/nextgen/identity_cluster.py`（新） | 纯函数，无 IO、无模型 | 约 260 |
| `extractor/nextgen/embedding_cache.py`（新） | 磁盘缓存 | 约 120 |
| `extractor/pipeline.py` | 插入 + 少量改写 | 约 150 改 / 40 删 |
| `extractor/config.py` | 加一个 CACHE_ROOT | 约 8 |
| `evaluation/cluster_report.py`（新） | 诊断报告 | 约 130 |
| `evaluation/export_episode_embeddings.py`（新） | 导出嵌入做实验 | 约 110 |
| `tests/test_identity_cluster.py`（新） | 合成数据单测 | 约 180 |

合计约 1000 行，其中 460 行是新增的独立文件（不影响现有流程）。**核心算法 260 行，其余是缓存、报告、测试。**

---

## 1. 为什么"簇"能同时解决错人和边界两个问题

这不是类比，是同一个机制。

**错人**：现在的判定是「这段像不像目标」。若一个不在排除集里的相似女声，单段分数略高于阈值，她就被输出——因为没有机制能说"她是一个独立的人"。改成簇级后，她会和她的其他轮次聚成一个簇；只要这个簇不是目标簇，她的每一段都被同一句话拒掉。判定粒度从「一次一个片段」变成「一次一个人」。

**边界**：`_merge_short_silence_same_speaker`（463 行）之所以危险，是因为它把两段拼起来之后必须**重新做一次身份判定**（2785 行 `_verify_speaker_span`）。重新判定就是重新掷一次骰子。而合并后的段又触发 `post_target_silence_merge`（2801 行），这个标记在 `_install_acoustic_identity_certificate`（1464 行）里**直接换发一张"已验证"证书**——不需要任何额外证据。这就是"合并段自己给自己发证"的机制性来源。

簇级判决下这件事消失：合并前的两段**属于哪个簇，是已经知道的事实**，不需要重新判决。同簇则合并（正常整理），异簇则不合并（本来就是硬边界）。

一句话：**合并从"高风险操作"降级为"同簇内的整理"。** 28.6% 的合并段来自这条路径（第三集 28 段里 8 段带 `post_target_silence_merge` 或 `final_same_speaker_*`）。

---

## 2. 新模块：`extractor/nextgen/identity_cluster.py`

纯函数，不碰 IO、不碰模型、不碰 global 状态。输入是嵌入向量和元数据，输出是簇划分与判决。**这样它可以被 100% 单测覆盖，不需要跑整集。**

### 2.1 数据结构

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np


# 两条嵌入通道。名字与 DomainReferenceBank 的 CHANNELS 保持一致，
# 避免又造一套术语。
Channel = Literal["primary", "secondary"]


@dataclass(frozen=True)
class TurnEvidence:
    """一个说话轮次的全部身份证据。索引即轮次序号，全局稳定。"""

    index: int
    start: float
    end: float
    speech_seconds: float
    primary: np.ndarray          # 单位向量，ERes2NetV2 整段嵌入
    secondary: np.ndarray        # 单位向量，CAM++ 整段嵌入
    # 该轮次与用户提供的排除角色的最大相似度，两者取高。用来阻止
    # "目标簇把排除角色吸进来"这种合并错误。没有排除角色时为 None。
    exclusion_max: float | None
    exclusion_role: str | None
    # 该轮次是否被现有的单段判决接受。聚类覆盖它，但保留原值用于诊断。
    single_turn_accepted: bool


@dataclass(frozen=True)
class Cluster:
    """一个簇 = 程序认为的一个说话人。"""

    cluster_id: int
    members: tuple[int, ...]            # TurnEvidence.index 的升序元组
    primary_centroid: np.ndarray
    secondary_centroid: np.ndarray
    # 簇内每个成员对簇质心的余弦分数，用于算"簇内一致性"
    primary_cohesion: float             # 最低成员的分数（最差成员）
    primary_median: float
    total_speech_seconds: float
    # 与用户参考集的匹配结果，在 assign_target 里填
    target_score: float = -1.0
    nearest_other_score: float = -1.0
    target_margin: float = -1.0


@dataclass(frozen=True)
class AssignResult:
    """整集的身份判决结果。"""

    clusters: tuple[Cluster, ...]
    target_cluster_id: int | None
    # 每个轮次的最终结论。True 才允许进入下游。
    membership: tuple[bool, ...]
    # 每个轮次属于哪个簇，-1 表示没有并入任何簇（噪声）
    cluster_of: tuple[int, ...]
    reason: str
    # 完整诊断，直接进 stage_audit
    diagnostics: dict = field(default_factory=dict)
```

设计约束：**`TurnEvidence` 用 `index` 而不是对象引用。** 现有代码到处用 `id(candidate)` 做键（3297、3298、3339、5193 行），跨函数传递时很脆。新模块只用整数索引，调用方负责维持 `index → CandidateSentence` 的映射。

### 2.2 常量与来源

每个数字都要有来源，不能拍脑袋。

```python
# 单链合并阈值（主通道，ERes2NetV2）。
# 来源：speaker.py:271 用 floor=0.50 做参考集内点过滤，
# speaker.py:471 用 floor=0.35 给 CAM++。参考集的"同人下限"是
# 0.50 量级。聚类是放宽的候选生成，取比它低的一档。
LINK_PRIMARY = 0.46
# 单链合并阈值（副通道，CAM++）。CAM++ 的类内余弦普遍高于 ERes2Net，
# 但它的 floor 被设成 0.35，说明它的尺度不同。保守起见不再抬高。
LINK_SECONDARY = 0.42

# 双通道必须同时成立。这是防"相似女声被并进目标簇"的主闸门。
# 来源：domain_identity.py:78-99 的 classify() 已经确立了
# "两个独立模型一致才算证据"的原则，此处沿用。

# 簇的最小规模。一集的女性龙套可能有 1-2 句，把它们当"人"没有意义，
# 它们更可能是嵌入噪声。低于此规模的簇不参与目标簇竞争。
MIN_CLUSTER_MEMBERS = 3
MIN_CLUSTER_SPEECH_SECONDS = 6.0

# 单链传递的刹车：一个簇内最差成员与质心的相似度不得低于此值。
# 单链的已知缺陷是"链式漂移"——A~B、B~C 但 A 与 C 差很远，
# 结果 A 和 C 被并到一起。这条闸门在合并后检查，超了就拆开。
CLUSTER_COHESION_FLOOR = 0.30

# 目标簇判定的余量下限。
# 来源：第三集第 11 段的排除余量 0.341 仍然错人，说明 0.34 量级
# 的"排他证据"不足以支撑判决。目标判定需要比这更强的分离度。
TARGET_MARGIN_FLOOR = 0.06
```

关于 `TARGET_MARGIN_FLOOR` 的说明：这是一个**保守起点**，其真实取值必须由第 6 节的实验决定，不能直接写死上线。文档给的是流程和默认值，不是结论。

### 2.3 核心算法：双通道约束的单链聚类

单链（single-linkage）在这里是对的选择，原因很具体：**同一角色在不同情绪、不同录音条件下，相邻轮次的相似度是连续衰减的**。用 k-means 或谱聚类需要预先给定簇数，而一集里有几个说话人你并不知道；用 DBSCAN 需要密度参数，而龙套的密度极不均匀。单链只需要一个"同人的最低相似度"，且天然处理不均匀簇大小。

它的缺陷是链式漂移，用 `CLUSTER_COHESION_FLOOR` 事后拆解。

```python
def cluster_turns(
    turns: Sequence[TurnEvidence],
    *,
    link_primary: float = LINK_PRIMARY,
    link_secondary: float = LINK_SECONDARY,
    min_members: int = MIN_CLUSTER_MEMBERS,
    min_speech_seconds: float = MIN_CLUSTER_SPEECH_SECONDS,
    cohesion_floor: float = CLUSTER_COHESION_FLOOR,
) -> tuple[Cluster, ...]:
    """把一集的说话轮次聚成"人"。纯函数，无副作用。

    只做合并与拆解，不做目标判定——目标判定在 assign_target()。
    分离这两件事是为了让"聚类好不好"和"目标选对没选对"能分别评估。
    """
```

实现步骤，逐条：

**第一步，构造邻接。** 对每一对轮次 `(i, j)`，计算两个通道的余弦（向量已是单位向量，点积即可）。条件为 `primary >= link_primary and secondary >= link_secondary`。

复杂度控制：254 个轮次的两两比较是 32k 次 512 维点积，在 CPU 上约 30 毫秒。**不要做任何近似或抽样**，直接算全矩阵。矩阵化写法是 `primary_matrix @ primary_matrix.T`，不要写 Python 双层循环。

**第二步，排除闸门。** 若 `i` 的 `exclusion_role` 与 `j` 的 `exclusion_max` 都指向同一角色，且两者的 `exclusion_max` 都 `>= 0.55`，则禁止合并。这条专门防「目标簇通过链式路径把某个明确被排除的角色吸进来」。

理由：第三集第 11 段有 0.341 的排除余量却仍错人，说明排除角色本身也有"像目标"的段（同一个 VA 配不同角色，或者录音条件相似）。如果目标簇有一条链穿过这些段，单链会把她整个吸进来。

**第三步，并查集合并。** 标准 union-find，路径压缩。不要用 networkx——多一个依赖不值得。

**第四步，拆解链式漂移。** 对每个合并后的簇：
- 算质心（**注意：用修剪均值，不用普通均值**，见 2.4）
- 算每个成员对质心的分数
- 若最低分 `< cohesion_floor`，把最低分的成员踢出去，重新算质心，重复直到全部达标
- 被踢出的成员**不再回到候选池**（不重新尝试并入其他簇）。这是保守选择：宁可漏，不要错。

**第五步，过滤小簇。** 成员数 `< min_members` 或总语音时长 `< min_speech_seconds` 的簇标记为"噪声"，它们的成员不参与目标竞争，`cluster_of = -1`。这些段仍然可能被输出，但需要单独的证据路径（见 4.3）。

**第六步，算簇统计量。** 返回 `Cluster` 元组，按 `total_speech_seconds` 降序排列（最大的簇是主角的先验，也是报告里最好读的顺序）。

### 2.4 质心用修剪均值

```python
def _trimmed_centroid(
    vectors: np.ndarray,           # [n, d]，单位向量
    trim_ratio: float = 0.10,
) -> np.ndarray:
    """算质心前先丢掉偏离中位数最远的那一档成员。

    理由：一个簇里混进 1-2 个"像目标但其实是别人"的段时，普通均值会把
    质心拖向他们。而质心一旦被拖走，后续所有成员的分都会下降，形成
    自证循环——这正是 speaker.py:296 注释提到的 centroid filtering 问题，
    那里用 leave-one-out 解决，这里用修剪均值解决同一个问题。
    """
```

实现：先算普通均值并归一化，算成员对它的分数，`torch.quantile` 取 `trim_ratio` 分位，丢掉低于该分位的成员，用剩下的重算并归一化。`trim_ratio=0.10` 对应"254 个轮次里丢掉 25 个最差的"，与 `_build_adaptive_speaker_profile` 里 `quantile(gains, 0.10)` 的用法（2916 行）同一个量级，属于沿用既有惯例而非新引入的自由度。

### 2.5 目标簇判定

```python
@dataclass(frozen=True)
class TargetProfile:
    """目标人物的声纹参考，可以来自用户文件，也可以来自自举库。"""

    primary_centroid: np.ndarray        # 单位向量
    secondary_centroid: np.ndarray
    # 参考集自身的一致性下限。用于把"绝对分数"转成"相对分离度"，
    # 避免换一集录音条件变化后阈值失配。
    primary_reference_floor: float
    secondary_reference_floor: float
    source: Literal["user_reference", "bootstrap", "merged"] = "user_reference"


def assign_target(
    clusters: Sequence[Cluster],
    profile: TargetProfile,
    exclusion_centroids: Sequence[tuple[str, np.ndarray]],
    *,
    margin_floor: float = TARGET_MARGIN_FLOOR,
    require_dominant: bool = True,
) -> AssignResult:
    """在所有簇里挑出目标簇，并给出每个轮次的最终归属。"""
```

判定流程：

**第一，给每个簇打分。** 用簇质心而不是单个成员：

```
target_score(c)     = max(cos(c.primary_centroid,   p.primary_centroid),
                          cos(c.secondary_centroid, p.secondary_centroid))
nearest_other(c)    = max over 排除角色 o of cos(c.primary_centroid, o.centroid)
target_margin(c)    = target_score(c) - nearest_other(c)
```

注意 `target_score` 取两通道的**最大值**而不是平均值。理由：CAM++ 在跨语言/情绪场景下更稳，ERes2NetV2 在干净场景下更准（从 `_classify_match` 的分层阈值可以看出两者的尺度不同）。取最大值等价于"任一模型认为是目标"，这个选择**偏松**，因此必须由下面的双重条件兜住。

**第二，准入条件。** 一个簇成为目标簇，必须同时满足：

- `target_score >= profile.primary_reference_floor`——分数不能低于参考集自身的下限，否则说明这一集根本不像参考
- `target_margin >= margin_floor`——与最强排除角色的分离度足够
- 规模达标（已在聚类的第五步过滤）
- 若 `require_dominant`：该簇的 `total_speech_seconds` 必须排在所有达标簇的第一位。**这不是硬性要求，是一个强烈建议保留的交叉验证**：目标是主角，她在一集里的语音时长应当最长。如果判定出的目标簇不是最大的簇，这本身就是一个可疑信号，应该记入诊断。

**第三，多簇并列。** 若两个簇同时达标（例如久美子和她的姐姐，或者同一 VA 的两个角色），**不要挑一个**。返回 `target_cluster_id = None`，`reason = "ambiguous_target_cluster"`，并让这一集走保守档（见 5.3）。这是"错人 0 段"目标的直接推论：**不确定时不输出，比输出一个需要人复查的集合更有价值。**

**第四，没有簇达标。** 返回 `target_cluster_id = None`，`reason = "no_qualifying_cluster"`。这一集大概率没有目标角色出场（或出场的集数不对），走保守档。

**第五，输出 `membership`。** 目标簇的成员为 `True`，其余为 `False`。**这一步是双向的**——已经在 `accepted_turns` 里的段，若不在目标簇内，`membership` 为 `False`，必须被撤下。这是与现有 `_promote_global_target_cluster` 的根本区别。

### 2.6 簇内一致性防错

一个必须处理的边界情况：目标簇里混进了别人。表现是某些成员对簇质心的分数明显偏低。三个处理层级：

- `Cluster.primary_cohesion`（最低成员）低于 `CLUSTER_COHESION_FLOOR` 时，拆解已在聚类阶段做了
- 拆解剩下的簇里，若仍有成员的分数低于 `target_score - 0.18`，标记为 `cluster_outlier` 写入诊断，**并且不输出**。这个 0.18 来自 `SpeakerVerifier.verify_waveform` 的 `p20 >= threshold - 0.18`（380 行），是既有的"允许的个体差异"量级
- 被排除的成员记录成独立条目进 `cluster_report`，可以人工确认是拆错了还是确实混进了别人

### 2.7 与现有代码的接口契约

新模块**不导入任何 `extractor.` 里的东西**（除了类型）。理由：`pipeline.py` 6400 行、循环导入风险高，且新模块要能被 `evaluation/` 下的实验脚本单独调用，不能拖着整个 pipeline 起来。

数据转换在调用方做：

```python
# pipeline.py 里的适配函数
def _build_turn_evidence(
    self,
    scored_turns: list[tuple[TimeSpan, CandidateSentence, SpeakerMatchDecision | None]],
    exclusion_profiles: Sequence[ExclusionSpeakerProfile],
) -> list[TurnEvidence]:
    """从现有 scored_turns 提取聚类输入。缺失嵌入的轮次仍进入列表，
    但标记为零向量，由 clustering 跳过——不静默丢弃，否则索引会错位。"""
```

**关键：索引必须稳定。** `TurnEvidence.index` 一旦分配，在整集生命周期内不变。`scored_turns` 会在流程中被 `extend`（5282、5643 行），新增项追加在尾部，索引天然稳定。但 `_recover_target_segments` 会**替换**父轮次（5264 行 `accepted_turns.append`），这时父轮次的索引会失效。处理方式：被替换的父轮次保留索引但标记 `superseded=True`，其嵌入不参与聚类，替换出的子段获得新索引。

---

## 3. 接入点：改哪几行

所有行号对应 `extractor/pipeline.py` @ `a287fe1`。

### 3.1 新增选项（`PipelineOptions`，55 行）

```python
    # 簇级身份判决。off 时行为与 2.01 完全一致。
    cluster_identity_mode: Literal["off", "audit", "enforce"] = "audit"
    # 目标簇判定的余量下限，来源见 identity_cluster.TARGET_MARGIN_FLOOR。
    cluster_target_margin: float = 0.06
    # 允许在目标簇外保留的例外路径（见 4.3），默认只有旁白。
    cluster_keep_narration: bool = True
```

`__post_init__` 里加校验：`cluster_identity_mode` 必须是三选一，否则 `ValueError`。理由：这个选项决定整集输出，拼错一个字母导致静默走默认档是必须避免的。

**`audit` 是默认值的理由**：先让它在全部现有集数上跑一遍、产出报告、跟你现有的试听结论对齐，再考虑切成 `enforce`。直接 `enforce` 上线会让第一集从 55 段掉到可能十几段，而你看不出掉的是对的还是错的。

### 3.2 插入位置：在 `_repair_final_sentence_boundaries` 之前

这是最关键的一个决定。

现在的顺序（`run()`，按行号）：

```
5107  scored_turns 构建（254 个说话轮次）      ← 嵌入在这里产生
5164  recovery_turns（疑似混合回合）
5198  _recover_target_segments（局部补切）
5633  _build_contrastive_edge_candidates
5653  _promote_multimodel_target_subclusters
5670  _merge_verified_target_turns（短静音合并）
5692  _build_adaptive_speaker_profile
5877  _repair_final_sentence_boundaries         ← 边界修复
5900  _extend_target_clip_over_tail_sliver
5916  _classifier_target_rescue
5952  restore_subtitle_sentences
```

**簇级判决必须插在 5877 之前，也就是紧接在 5692 的 adaptive profile 之后。**

理由有三条，每条都是从现有代码的约束推出来的：

**其一，必须在所有救援路径之后。** 救援路径（`_classifier_target_rescue` 5916、字幕补全 5952）是错误的主要来源，第三集定位器 6 段错 5 段。如果聚类在它们之前跑，它们救回来的段会绕过判决。插在 5877 之前意味着：`_recover_target_segments`、`_promote_multimodel_target_subclusters`、`_build_adaptive_speaker_profile` 的产物**已经都在 `accepted_turns` 里**，全都在聚类视野内，可以被撤下。

**其二，必须在 `_repair_final_sentence_boundaries` 之前。** 边界修复会**裁切**已接受的段（`boundary_cropped`）。裁切改变 `start/end`，而嵌入是按 span 算的——裁切后的段嵌入失效，必须重算。让聚类在裁切之前跑，嵌入是有效的。裁切后的时间段仍然属于同一个簇（裁切只动边界，不动身份），所以判决结果依然适用。

**其三，必须在 `_merge_verified_target_turns`（5670）之后。** 短静音合并正是在这里发生的，而合并前必须知道两段是否同簇。这个顺序天然成立。

插入后的代码形态：

```python
                # 5877 之前插入
                cluster_verdict = self._apply_cluster_identity(
                    scored_turns, accepted_turns, rejected,
                    profile, exclusion_profiles,
                    progress=lambda _v, m: progress(0.80, m),
                )
                if cluster_verdict is not None:
                    progress(0.80, cluster_verdict)
```

### 3.3 新增方法 `_apply_cluster_identity`

```python
    def _apply_cluster_identity(
        self,
        scored_turns: list[
            tuple[TimeSpan, CandidateSentence, SpeakerMatchDecision | None]
        ],
        accepted_turns: list[CandidateSentence],
        rejected: list[CandidateSentence],
        profile: SpeakerMatchProfile,
        exclusion_profiles: Sequence[ExclusionSpeakerProfile],
        progress: ProgressCallback,
    ) -> str | None:
        """整集簇级身份判决。mode=off 时立即返回 None，不做任何事。

        返回一句人类可读的进度摘要，写入 stage_audit。
        """
```

内部步骤：

**（1）早退。** `if self.options.cluster_identity_mode == "off": return None`

**（2）构造 `TurnEvidence`。** 遍历 `scored_turns`，从 `match.primary.embedding` / `match.secondary.embedding` 取嵌入。这两个字段已经存在（`speaker.py:53`、`65`），是 `SpeakerDecision.embedding`，无需重算。

**嵌入缺失的轮次**（`match is None`，例如因为 5140–5144 行被提前拒绝的段）**不能跳过**，要记为零向量 + `usable=False`。原因：第三集漏斗显示 254 个轮次里只有 48 个通过了 STT 前身份，206 个在 5136–5157 行被拒。**如果聚类只看到已通过的段，它看到的是一个已经被单段判决筛选过的子集——错人段会因为在子集内而显得一致。** 聚类必须看到全部 254 个，这正是它比现有机制强的根本原因。

同时收集 `exclusion_max`：对每个轮次，算它和每个排除角色 profile 的最大相似度。`ExclusionSpeakerProfile` 已存在（`speaker.py:136`）。

**（3）`cluster_turns()` → `assign_target()`。** 纯函数调用。

**（4）按 `mode` 处理。**

`audit` 模式：**不改动 `accepted_turns`**，只把完整结果写进 `self._stage_audit["stages"]["cluster_identity"]` 和每个候选的 `diagnostics`。目的是让你拿到报告后能直接对比：程序认为的目标簇有几个成员、你试听认为对的有几段、差在哪。

`enforce` 模式：执行撤下。遍历 `accepted_turns`，`membership[index]` 为 `False` 的移入 `rejected`，设 `reject_reason = f"不属于目标簇（簇{cid}，目标簇为{tid}）"`，并按 4.1 处理。同时**允许提升**：`membership` 为 `True` 但当前在 `rejected` 里的段，移入 `accepted_turns`。

**（5）写审计。** 无论哪个模式都调用 `self._trace_spans("cluster_target_members", ...)`，把目标簇成员的时间段写进 `stage_audit`。

### 3.4 改写 `_promote_global_target_cluster`（3280）

**在 `cluster_identity_mode != "off"` 时，这个函数必须停用。**

理由：它的职责和新的簇级判决完全重叠，但方向相反（它只加回）。同时启用会出现"聚类撤下 → 全局簇又加回"的振荡，而且没有任何机制能保证最终结果稳定。

具体做法：在 `run()` 里 5881 行附近的调用处加条件。**不要删函数**——`off` 模式还需要它，而且删了就没有回退路径。

```python
                if self.options.cluster_identity_mode == "off":
                    global_recovered = self._promote_global_target_cluster(...)
                else:
                    global_recovered = 0
```

### 3.5 `run_many`（6252）

批量处理时 `self._stage_audit = None`（6246 行）。簇级报告需要一个独立的位置：**不要塞进 `stage_audit`，写到 `output_dir/cluster_report.json`。** 理由见第 6 节——报告是给人看的，结构和 `stage_audit` 的 `{count, spans}` 契约不同，混在一起会让 `audit_stage_reachability.py` 的 `records(stage)` 解析失败。

---

## 4. 判决的边界行为

### 4.1 撤下已接受的段

`enforce` 模式下被撤下的段，处理方式和现有被拒段**必须不同**：它们要保留音频。

理由：这些段是"程序曾经认为是目标"，是最有价值的人工复核材料。现有 `keep_rejected` 只对被拒段生效，且只写音频不写文本（4432–4437 行）。对簇级撤下的段，建议单独写 `cluster_outliers/` 目录，含音频 + 一个说明属于哪个簇的 txt。这是可选项，但**在调参阶段强烈建议开**——你需要听到"程序为什么认为她不是久美子"。

### 4.2 边界裁切与簇归属的交互

`_repair_final_sentence_boundaries`（5877）在聚类之后跑，它会裁切段的边界。裁切后必须检查一件事：**裁切是否跨越了簇边界**。

具体场景：一个段在 5877 之前被判为目标簇成员，但它内部其实混了别人的一句短话。5877 的裁切逻辑会尝试把别人的部分切掉（`boundary_cropped`）。这个行为是**正确的**，不需要干预。

但反过来的情况需要干预：裁切后剩下的部分**太短**（`< min_output_seconds = 1.20`），这段应该被丢弃，而不是作为一个 0.6 秒的碎片输出。现有的 `min_output_seconds` 门控在哪里生效待复核——如果它只在 STT 之后生效，聚类阶段就要补一道。

### 4.3 例外路径：什么可以不进目标簇

严格"只输出目标簇"会丢掉一类正当内容：**旁白**。第三集那段旁白（1 段，0 错）时长短、音色与角色对话不同，很可能不并入任何簇。

处理方式：给 `AssignResult` 加一个例外判定，需要**同时**满足：

- 该轮次不在任何簇内（`cluster_of == -1`），即聚类把它当噪声
- 该轮次的 `TargetProfile` 分数 `>= profile.primary_reference_floor`
- 该轮次的 `exclusion_max`（若存在）`< 0.40`

三条同时成立才保留。这个条件很严，预期只会保留极少数段。**不要为它单独放宽阈值**——旁白的训练价值有限，不值得为它引入风险。

---

## 5. 缓存与自举

### 5.1 跨运行嵌入缓存

现状：只有内存里的 `_tail_scorer_cache`（161 行），没有任何跨运行缓存。改一行要看 23 分钟结果。

新增 `extractor/nextgen/embedding_cache.py`：

```python
def cache_key(source_sha256: str, start: float, end: float,
              model_sha256: str) -> str:
    """稳定键。采样索引量化到 10 ms，避免浮点噪声导致缓存不命中。"""
    return f"{source_sha256[:16]}:{round(start*100)}:{round(end*100)}:{model_sha256[:12]}"
```

这个键的设计**直接沿用 `evaluation/build_review_queue.py` 已有的 `source_id` 方案**（`f"{source_hash}:{round(start*sample_rate)}:{round(end*sample_rate)}"`）。理由：那个系统已经在用，试听标签和嵌入能通过同一个键关联起来——你今天试听标出来的"第 11 段错人"，明天能直接对上它的嵌入。**这是自举和验收都依赖的连接点。**

存储：`CACHE_ROOT / "embeddings" / f"{source_sha256[:16]}.npz`，一个源文件（一集）一个 npz，键是上述 cache_key，值是 float32 向量。放在 `config.py` 新增的 `CACHE_ROOT`（默认 `ASSET_ROOT / "cache"`）。

**注意**：`cache_key` 里必须含模型 sha256。你换一次参考音频，`classifier_v2.pt` 就整体关闭（1863–1872 行的门控），同样的逻辑——嵌入和模型绑定，缓存也必须绑。

### 5.2 目标声纹自举

2026-09-30 按用户的通用性要求修订：只要求目标人物参考音频（至少一段）与当前待提取文件；不要求多集、历史批次、人工标签、排除角色或预建库。批量处理时，各文件独立自举，不跨文件继承库。

生产入口为 `extractor/nextgen/bootstrap_training.py::analyze_with_bootstrap`，在 pipeline 最终簇门控前执行。原始用户参考始终决定聚类身份锚点和目标簇；自举只参与候选轮次的局部身份核验，不重新选择目标人物，也不假设目标是主角或最大簇。

1. 用原始参考完成基础簇分析与整轮时序核验。目标簇对原参考的双通道分数均须至少 0.75，存在排除角色时双通道余量均须至少 0.15，才可收集种子。
2. 仅收集已通过全部局部窗口、属于目标簇、非离群且不与用户排除、歌声、重叠或硬否决区间重叠的轮次。禁止直接收集全部簇成员。
3. 用上一轮可靠片段扩充参考，再核验原先未通过的候选。每个待判断轮次都剔除自身、重叠证据及依赖它得到的后续证据。两通道、全部窗口必须在同一参考配置下通过，不能逐窗口或逐模型拼凑通过结果。
4. 新通过且满足入库条件的轮次可加入下一轮。没有新增通过轮次即停止，默认最多 3 轮，可配置 1–10 轮；没有可靠种子时保留基础簇判决并报告未扩充。

库内按说话时长加权，双通道相似度同时至少 0.95 时去重，最多 500 条。原始参考始终占融合权重的一半，局部下限、排除余量、离群规则不会随轮次放宽。复用当前模型和嵌入，不反复运行 UVR/VAD/STT。

输出目录内保存 `bootstrap/bootstrap_report.json` 与逐轮 NPZ，记录模型、参考、源文件、编码指纹和证据依赖。这些是机器伪标签，不能作为人工准确率证明；当前不自动导入历史库。WebUI/桌面勾选“自动扩充参考并应用”时使用 enforce；关闭开关保持原来的 audit 行为。

### 5.3 保守档

`target_cluster_id is None`（无簇达标或目标簇歧义）时，该集走保守档：**不输出任何段**，只写 `cluster_report.json` 说明原因。

不输出代表证据不足，并不代表质量通过。通用用户可能只有一个文件，不能以“还有其他集”解释召回不足；需要在报告中明确记录失败原因，并将这一限制纳入后续效果验证。

---

## 6. 验证与测试

### 6.1 单元测试（不需要跑整集）

`tests/test_identity_cluster.py`，全部用合成单位向量，不加载任何模型。

| 用例 | 构造 | 期望 |
|---|---|---|
| 两簇分离 | 构造 40 个向量：A 簇 25 个围绕基向量 a，B 簇 15 个围绕 b，`cos(a,b)=0.35` | 恰好 2 个簇，成员数正确 |
| 相似女声不并入 | A 簇 25 个，C 簇 8 个，`cos(a,c)=0.55`（高于 A 簇内部的最差对） | A 与 C 分开，且 C 不是目标簇 |
| 双通道不一致不合并 | 主通道 `cos=0.60` 但副通道 `cos=0.20` | 不合并 |
| 链式漂移被拆解 | A-B `0.50`、B-C `0.50`、A-C `0.15` | 拆成两组，不出现 A 和 C 同簇 |
| 排除角色不被吸收 | 目标簇 25 个，其中 3 个带 `exclusion_role="X"` 且 `exclusion_max=0.80` | 那 3 个被挡在合并外 |
| 小簇被过滤 | 3 个孤立向量 + 1 个大簇 | 孤立向量 `cluster_of == -1` |
| 目标歧义 | 两个簇都达标且规模相当 | `target_cluster_id is None`，`reason == "ambiguous_target_cluster"` |
| 无簇达标 | 所有簇对目标的分数都低于参考下限 | `target_cluster_id is None`，走保守档 |
| 修剪均值抗污染 | 25 个好成员 + 3 个"像目标但不是"的成员 | 质心与纯 25 个的质心夹角 `< 8°` |

最后一条是关键用例。**如果修剪均值不起作用，整簇判决会被少数污染段拖偏，方案就失去意义。**

### 6.2 影子模式对照（不需要你试听）

这是整个方案里**最重要的一步验证**，而且它不消耗你的时间。

在第三集 `20260928_214330_batch_5bd1f3` 上跑 `cluster_identity_mode="audit"`，然后：

1. 你已有的试听结论在 `evaluation/episode3_user_review_20260928_5bd1f3.json`
2. 每个错误段在 manifest 里有精确的 `start/end`
3. 用 `evaluation/build_review_queue.py` 的 `source_id` 键把试听标签和嵌入对上
4. **检查：那 10 段错人的段，落在目标簇内还是簇外？**

这个检查的结论决定后面怎么走：

| 结果 | 含义 | 行动 |
|---|---|---|
| 10 段全部在簇外 | 聚类解决了错人问题 | 可以进 `enforce`，直接看第 3 步验收 |
| 5–9 段在簇外 | 部分解决 | 看落在簇内的那些有什么共同点（大概率是`_recover_target_segments` 的子段，边界错导致嵌入错） |
| 0–4 段在簇外 | 方案对这个失效模式无效 | **停下，重新想**。不要继续加码 |

**这一步必须先做。** 它花的是机器时间（一次 audit 跑集），不是你的时间，而且它能在写任何 pipeline 接入代码之前就把方案证伪。

命令形态（需要你或执行者写）：

```
python -m evaluation.cluster_report \
    --batch output/20260928_214330_batch_5bd1f3 \
    --review evaluation/episode3_user_review_20260928_5bd1f3.json \
    --output docs/cluster_vs_review_ep3.json
```

`cluster_report.py` 要输出一张表：簇 id、成员数、总时长、target_score、target_margin、以及**你标错的那几段在哪个簇**。

### 6.3 验收（只看一次）

`enforce` 模式在未跑过的集数上验证。通过线：**错人 0 段、歌声 0 段、重叠 0 段。**

关键约束：**验收集只看一次，看完不许用来调参。** 第三集已经用过一次了（61% 那次），它现在是训练集的一部分，不能再当验收。

因此验收需要**新的、没跑过的集**。你有 13 集 + OVA，前 3 集已被用于调参，**第 4 集往后是干净的**。建议验收集：第 6、9、12 集各一集，覆盖不同的场景密度。

---

## 7. 顺序与依赖

**不要一次性做完。** 按下面的顺序，每步有独立的停止点：

| 阶段 | 内容 | 停止条件 |
|---|---|---|
| **P0** | 写 `identity_cluster.py` + `test_identity_cluster.py`，跑通 9 个用例 | 全部通过才能进 P1 |
| **P1** | 写 `export_episode_embeddings.py`，从第三集导出 254 个轮次的嵌入 | 确认嵌入维度和数量符合预期 |
| **P2** | 写 `cluster_report.py`，在**不接 pipeline** 的情况下用 P1 的导出手动跑聚类，对照 6.2 的表 | 若 10 段错人只有 ≤4 段在簇外 → **停止，重新设计** |
| **P3** | 接 pipeline，`mode="audit"`，在第三集跑一次，确认 audit 结果与 P2 的手动结果一致 | 不一致说明接入有 bug |
| **P4** | `mode="enforce"`，在第一、二、三集上跑，**只看输出段数变化**，不听 | 段数不应低于原输出的 40%，否则目标簇判定过严 |
| **P5** | 加缓存（`embedding_cache.py`），验证同样的输入产出同样的输出 | 缓存必须不改变结果 |
| **P5.5** | 按 7.5 节加 UI：簇摘要、按簇筛选、簇视图与覆盖；确认旧批次能正常降级 | 四条验收标准逐条过 |
| **P6** | 验收：第 6、9、12 集 |

**P2 是真正的决策点。** 前三阶段全部是新增独立文件，**不动 pipeline 一行**。如果 P2 证伪，损失是 460 行新增代码和机器时间，你的试听时间是零。

---

## 7.5 簇级判决的 UI（提前写进去）

这一节是后补的：簇级判决会**改变界面上要展示的东西**，而界面属于交付物的一部分，不能等算法落地了再回头补。下面这些是 `launcher/webui_static/` 和桌面端都需要的改动。

### 7.5.1 为什么簇级判决必须配 UI

**它会让输出的段数变少。**第 4 节说过：`enforce` 模式下，不在目标簇内的段会被撤下。第一集现在是 55 段，切换后可能只剩二十几段。如果界面上没有任何解释，你看到的就是"东西变少了"，而这和"东西更准了"在视觉上没有区别。

**它是唯一能解释"为什么这段被撤下"的机制。**现在被拒的段只有一个 `reject_reason` 字符串（237 段都是"声纹匹配不足"，等于没说）。簇级判决能给出真正有用的信息：这段属于第 7 簇，目标簇是第 2 簇。

**它需要你确认目标簇选对了。**第 6.2 节的影子模式检验只做一次，之后每跑新集都需要一个低成本的方式让你看一眼"程序选的目标簇是不是久美子"。在界面上放一个按钮跳过 30 段音频，比读 JSON 快得多。

### 7.5.2 新增的展示项

`cluster_report.json` 已经由 6.2 节的 `evaluation/cluster_report.py` 产出，UI 只需读它并渲染。要展示的是**簇列表**，不是片段列表：

| 列 | 来源 | 用途 |
|---|---|---|
| 簇编号 | `Cluster.cluster_id` | 引用与讨论 |
| 成员数 / 总时长 | `len(members)` / `total_speech_seconds` | 判断谁是主角 |
| 目标分数 | `Cluster.target_score` | 与参考集的绝对匹配度 |
| 排除余量 | `Cluster.target_margin` | 与最强排除角色的分离度 |
| 状态 | `membership` 是否为目标簇 | 一眼看出选了谁 |

**关键交互：目标簇要能手动改。** 如果程序选错了簇，你点另一个簇就能覆盖，并把这次覆盖写进 `review_v2.json` 的 `cluster_override` 字段。这比回终端重新跑一遍便宜得多，而且覆盖记录本身是宝贵的标签——它直接告诉你簇级判定在哪一集失效了。

`webui_server.py` 需要新增一个接口：

```python
@app.get("/api/batches/{batch_id}/clusters")
def get_batch_clusters(batch_id: str):
    """读取 cluster_report.json；文件不存在时返回 available=false。"""
```

片段列表也要多一个筛选维度：**按簇过滤**。在第 6.2 节的验证阶段，你最需要的能力是"把目标簇的段全听一遍"和"把某个相似女声簇的段全听一遍"，这两个都是按簇筛选。

### 7.5.3 与现有 UI 的关系

`webui_static/` 的试听工作台已经有按片段裁决的完整能力（六个裁决按钮、区间标记、盲听模式、撤销、快捷键），这些**不需要改**。要加的是：

- 批次元信息区增加一行簇摘要："3 个簇 · 目标簇 #2（41 段）· 另有 2 簇与目标相似度 >0.6"
- 片段列表增加"按簇筛选"（接在现有的 全部/未标/已标 后面，不替换）
- 新增一个"簇视图"标签页，渲染 7.5.2 那张表，可点选覆盖

桌面端（`launcher/` 下的 tkinter 界面）同理，但优先级低——簇视图主要在验证阶段用，而验证用网页版更快。**桌面端可以只加簇摘要那一行**，不做完整簇视图。

### 7.5.4 这一节的验收

UI 部分能否算完成，标准很具体：

1. 打开一个批次，能在不看 JSON 的情况下说出"程序认为的目标簇是哪个、它有多少段"
2. 能按簇筛选片段并连续试听
3. 覆盖目标簇后，覆盖记录出现在 `review_v2.json` 里
4. `cluster_report.json` 不存在时（`cluster_identity_mode="off"` 或旧批次），界面正常降级，不报错

第 4 条容易被忽略但必须做：**你的 13 集历史批次都没有这个文件**，UI 必须能同时处理新旧两种批次。

---

## 8. 明确不做的

- **不训练任何模型。** 全部用现有嵌入。`classifier_v2.pt` 的门控问题（换参考就静默关闭）独立处理，不混进这次改动。
- **不碰 `extractor/nextgen/` 的 T2 时序权重。** 已放弃。
- **不在同一次改动里拆 `run()`。** 拆分需要重放等价性保证，那是另一件事，且会掩盖这次改动的真实效果。
- **不删 `_promote_global_target_cluster` 和任何救援路径。** 只加开关。回退路径必须始终存在。
- **不引入新依赖。** 并查集自己写 20 行，不用 networkx / sklearn / scipy。你现有代码里 numpy 和 torch 都在，够用。
- **不改静音规则、不改 STT、不改导出格式。** 你列出的硬约束逐条不动。

---

## 9. 待复核清单

写这份方案时我读到的东西，与我没有验证的假设，分开列清：

**已读代码确认的**：`_promote_global_target_cluster` 的四个缺陷（3280–3381）；`_build_adaptive_speaker_profile` 的 `>= 3` 门槛（2866）；`post_target_silence_merge` 换发证书（1432、1464）；`speaker.py` 的 profile 构建与阈值推导（266–296、338–396）；`local_identity_classifier.py` 的窗口逻辑（12–61）；`build_review_queue.py` 的 `source_id` 方案；`anime_embedding.py` 的 192 维契约；`config.py` 的路径常量。

**需要执行者复核的**：

1. `SpeakerDecision.embedding` 的实际维度。文档里按 512 维估算，**未核实**。不同模型维度不同（ERes2NetV2 常见 192，CAM++ 常见 192 或 512），实现前必须打印确认。
2. ~~`min_output_seconds` 的执行点~~ **已复核**：该门控在 pipeline 里有 25 处，其中 6159 行 `if turn.duration < self.options.min_output_seconds` 是最终回合门控，位于簇级判决（5877 前）**之后**。结论：4.2 提到的"裁切后过短"问题**已经被既有门控覆盖**，不需要在聚类阶段补一道。但这意味着 6159 会在聚类之后丢弃短段——**审计报告与最终输出的段数会不一致**，报告里必须标明这一点，否则对照时会产生误解。
3. `ExclusionSpeakerProfile` 的字段名。已确认是 `label` + `primary`（`speaker.py:136–138`），文档 3.3 里写的"每个排除角色 profile 的最大相似度"实现时要用 `profile.primary.centroid`。
4. 第 2801 行路径产生第 8 段（带 `post_target_silence_merge` 但无 `final_same_speaker_*`）是否成立。这是我此前的**假设**，需要重放确认。
5. 第三集 28 段按路径的归属表（0/3/5/1 那张表）是**我手工读 manifest 推断的**，需要用脚本重算。P2 阶段的第一步就是把它脚本化。
6. `silence_split_seconds` 与 `silence_max_seconds` 在 `__post_init__` 里的互相覆盖（92–99 行）意味着桌面 UI 传 `silence_max_seconds: 0.9` 时 `silence_split_seconds` 也被改成 0.9。这个行为是否正确**待你确认**——它影响所有静音判定，但不在这次改动范围内。
7. 4.3 的旁白例外条件里那条 `exclusion_max < 0.40` 是我按"排除角色相似度量级"估的，**无实验支撑**，实现时先按诊断值输出、不参与判决，跑几集看分布再定。

**关于 6.2 的诚实说明**：我预测那 10 段错人的段会大部分落在目标簇外，因为「不在排除集里的相似女声会自成一群」在机制上成立。**但这是预测，不是结论。** P2 就是要检验它。如果 P2 结果与预测不符，上面整套阈值推导都需要重做——那份 `TARGET_MARGIN_FLOOR = 0.06` 是我按既有代码量级给的一个起点，**没有任何实验支撑**。

---

*本文件是可实施方案，不含执行。文中所有行号对应 `a287fe1`；标注「待复核」的项在实现前必须由执行者验证。*
