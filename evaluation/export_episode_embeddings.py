"""把一集的说话轮次导出成声纹嵌入。

这是簇级判决的取证步骤：嵌入在正式流程里只在内存中存在，跑完就没了。本脚本
从 UVR 人声轨上重新算一遍，输入是已完成的批次产物，不重跑 pipeline、不改任何
输出。

用法:

    python -m evaluation.export_episode_embeddings \\
        --batch output/20260928_214330_batch_5bd1f3 \\
        --reference "京吹原片音频/kumiko" \\
        --output work/embeddings/5bd1f3.npz

输出 npz 里的键:

    index / start / end / speech_seconds / usable / single_turn_accepted
        每个轮次的元数据，一维数组，行序与嵌入矩阵一致
    primary / secondary
        [n, d] 单位向量矩阵，两个声纹模型
    exclusion_max / exclusion_role_id
        每个轮次与排除角色的最大相似度与角色序号；没有排除角色时全为 nan / -1
    meta
        JSON 字符串，记录源文件、模型路径与 sha256、参数
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def find_stem(work_root: Path, batch_id: str) -> Path:
    """定位批次对应的人声轨。work 目录名是 <batch_id>_<序号>。"""
    candidates = sorted(work_root.glob(f"{batch_id}_*/stems/target_vocals.wav"))
    if not candidates:
        raise SystemExit(
            f"找不到 {batch_id} 的人声轨。\n"
            f"预期位置：{work_root}/{batch_id}_001/stems/target_vocals.wav\n"
            "如果 work 目录已被清理，需要重跑该集才能拿到该集的人声轨。"
        )
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, default=ROOT / "work")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reference", type=Path, action="append", required=True)
    parser.add_argument("--exclude-role", type=Path, action="append", nargs="+", default=[])
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    from evaluation.cluster_report import collect_audio
    from extractor.nextgen.cluster_runtime import ClusterAnalyzer
    from extractor.nextgen.embedding_cache import file_digest
    from evaluation.cluster_environment import configure_pipeline_cpu_threads
    environment = configure_pipeline_cpu_threads()
    batch = args.batch.resolve()
    audits = list(batch.rglob("stage_audit.json"))
    if len(audits) != 1:
        parser.error("Specify a batch with exactly one target audit")
    audit_path = audits[0]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    spans = audit["stages"]["speaker_turns"]["spans"]
    stem = find_stem(args.work_root, batch.name)
    analyzer = ClusterAnalyzer(stem, collect_audio(args.reference),
                               [collect_audio(group) for group in args.exclude_role],
                               device=args.device, cache_enabled=not args.no_cache,
                               progress=lambda m: print(m, flush=True))
    try:
        turns = analyzer.evidence(spans)
        if not any(t.usable for t in turns):
            parser.error("No usable speaker turns")
        pre = {tuple(x) for x in audit["stages"].get("identity_accepted_before_stt", {}).get("spans", [])}
        final = {tuple(x) for x in audit["stages"].get("final_accepted", {}).get("spans", [])}
        roles = sorted({t.exclusion_role for t in turns if t.exclusion_role})
        meta = {"schema": 2, "batch_id": batch.name, "source_stem": str(stem.resolve()),
                "source_sha256": analyzer.source_hash, "stage_audit": str(audit_path),
                "stage_audit_sha256": file_digest(audit_path), "provenance": analyzer.provenance(),
                "source_duration_seconds": audit.get("source_duration_seconds"),
                "primary_dim": len(turns[0].primary), "secondary_dim": len(turns[0].secondary),
                "exclusion_roles": roles}
        meta["provenance"]["runtime_environment"] = environment
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output,
            index=np.array([t.index for t in turns], dtype=np.int32),
            start=np.array([t.start for t in turns]), end=np.array([t.end for t in turns]),
            speech_seconds=np.array([t.speech_seconds for t in turns]),
            usable=np.array([t.usable for t in turns]),
            single_turn_accepted=np.array([(t.start,t.end) in pre for t in turns]),
            in_final_output=np.array([(t.start,t.end) in final for t in turns]),
            primary=np.stack([t.primary for t in turns]), secondary=np.stack([t.secondary for t in turns]),
            exclusion_max=np.array([t.exclusion_max if t.exclusion_max is not None else np.nan for t in turns]),
            exclusion_role_id=np.array([roles.index(t.exclusion_role) if t.exclusion_role else -1 for t in turns]),
            anchor_role=np.array([t.anchor_role or "" for t in turns]),
            meta=np.array(json.dumps(meta, ensure_ascii=False)))
        print(json.dumps({"output": str(args.output), "turns": len(turns),
                          "usable": sum(t.usable for t in turns),
                          "dimensions": [meta["primary_dim"], meta["secondary_dim"]]}, ensure_ascii=False))
    finally:
        analyzer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
