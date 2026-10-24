# 缓存对照的 CPU 运行环境差异

完整缓存开关对照最初失败：第 2 个轮次（10.15–12.85 秒）的主通道最大绝对差异为 0.000104181，副通道为 0.0000525303，超过既定的 0.000001 容差。同一环境连续两次无缓存重算的差异为 0。

最小复现位于 `work/cluster_p0p2_logs/diagnose_cache_parity.py`。CUDA TF32 和 benchmark 开关没有复现缓存值；交替把 CPU 线程数设为 10、4、10、4，结果稳定如下：

| CPU 线程数 | 主通道最大差异 | 副通道最大差异 |
|---|---:|---:|
| 10（离线进程原默认值） | 0.000104181 | 0.0000525303 |
| 4（正式 pipeline 的 FunASR 设置） | 0 | 0 |

正式流程在 VAD 阶段构造 FunASR `AutoModel`，其默认 `ncpu=4` 会调用 `torch.set_num_threads(4)`。离线工具未经过 VAD，因此原先以不同线程设置计算 CPU 特征。这是运行环境不一致，不能通过放宽容差掩盖。

修复在 `evaluation/cluster_environment.py`：离线验证、嵌入导出和聚类报告入口统一采用正式流程的 4 线程设置，并把它写入证据。没有改动 `extractor/` 下的冻结代码，也没有更改身份阈值或删除缓存。

修复后重新执行完整 254 轮的缓存对照，主副通道向量最大差异为 **0**，目标选择、簇分配、成员、允许区间和局部判决均一致。原失败日志保留在 `work/cluster_p0p2_logs/33_cache_parity_original_failure.log`；线程交替对照保留在 `33_cache_parity_threads_repeated.log`。完整回归命令就是 `evaluation/verify_cluster_plan.py --check-cache-parity`（其余参数见配套运行脚本）。

该结论限定于本次相同代码、模型、设备与 4 线程运行环境，以及 STT 前的簇分析。它不声称任意线程/硬件设置下逐位一致，也不代替完整 STT 导出对照或人工试听。
