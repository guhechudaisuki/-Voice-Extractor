"""在已导出的嵌入上跑整集聚类，并对照人工试听结论。

这是簇级判决方案的决策点：它不接 pipeline、不改任何输出，只回答一个问题——
**那些被人工判定为"错人"的段，落在目标簇内还是簇外？**

    有目标且 10 段全部被拒 → 通过已标注整句错误检查，仍需边界检查与 P3 audit
    5–9 段在簇外       → 部分解决，看落在簇内的有什么共同点
    0–4 段在簇外       → 方案对这个失效模式无效，停下重新设计

用法:

    python -m evaluation.cluster_report \\
        --embeddings work/embeddings/5bd1f3.npz \\
        --reference "京吹原片音频/kumiko" \\
        --exclude-role "京吹原片音频/其他人物/绿" \\
        --review evaluation/episode3_user_review_20260928_5bd1f3.json \\
        --output docs/cluster_vs_review_ep3.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extractor.nextgen.identity_cluster import (  # noqa: E402
    TargetProfile,
    TurnEvidence,
    assign_target,
    cluster_turns,
)

AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma"}


def collect_audio(paths: list[Path]) -> list[Path]:
    """路径可以是文件也可以是目录，目录里递归收音频。"""
    found: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
            found.append(path)
        elif path.is_dir():
            found.extend(
                sorted(
                    child for child in path.rglob("*")
                    if child.is_file() and child.suffix.lower() in AUDIO_SUFFIXES
                )
            )
    return sorted(set(found))


def build_profile(
    references: list[Path], negatives: list[list[Path]], device: str,
) -> tuple[TargetProfile, list[tuple[str, np.ndarray]]]:
    """用真实模型算参考集质心。参考集很小，不需要缓存。"""
    import torch

    from extractor.speaker import CAMPlusVerifier, SpeakerVerifier

    verifier = SpeakerVerifier(device=device)
    secondary = CAMPlusVerifier(device=device)

    def centroid(paths: list[Path], encoder) -> np.ndarray:
        vectors = torch.stack([encoder.embedding_from_file(path) for path in paths])
        return (
            torch.nn.functional.normalize(vectors.mean(dim=0), dim=0)
            .cpu().numpy().astype(np.float64)
        )

    primary_centroid = centroid(references, verifier)
    secondary_centroid = centroid(references, secondary)

    # 参考集自身的一致性下限：用参考成员对质心的最低分做基准，这样换一集录音
    # 条件变化时"绝对分数"的门槛会跟着参考集一起移动。
    scores = [
        float(
            verifier.embedding_from_file(path).cpu().numpy() @ primary_centroid
        )
        for path in references
    ]
    floor = float(np.quantile(scores, 0.10)) if len(scores) > 1 else float(scores[0])
    primary_floor = min(0.68, max(0.52, floor - 0.22))
    secondary_floor = min(0.64, max(0.48, floor - 0.26))

    profile = TargetProfile(
        primary_centroid=primary_centroid,
        secondary_centroid=secondary_centroid,
        primary_reference_floor=primary_floor,
        secondary_reference_floor=secondary_floor,
    )

    exclusion_centroids = []
    for ordinal, group in enumerate(negatives, start=1):
        if not group:
            continue
        exclusion_centroids.append((f"排除{ordinal}", centroid(group, verifier)))

    verifier.close()
    secondary.close()
    return profile, exclusion_centroids


from extractor.nextgen.cluster_reporting import load_review, review_clip, summarize, _verdict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--reference", type=Path, action="append", default=[])
    parser.add_argument(
        "--exclude-role", type=Path, action="append", default=[],
        help="排除角色目录，可重复；每次重复算一个角色",
    )
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-dominant-check", action="store_true")
    args = parser.parse_args()

    from extractor.nextgen.cluster_runtime import ClusterAnalyzer
    from evaluation.cluster_environment import configure_pipeline_cpu_threads
    environment = configure_pipeline_cpu_threads()
    with np.load(args.embeddings, allow_pickle=False) as payload:
        meta = json.loads(str(payload["meta"]))
        spans = list(zip(payload["start"].tolist(), payload["end"].tolist()))
    # Legacy NPZ vectors lack a full encoding/input fingerprint: use only their
    # audited span list and obtain vectors via the shared content-addressed cache.
    audit = json.loads(Path(meta["stage_audit"]).read_text(encoding="utf-8"))
    if not np.allclose(spans, audit["stages"]["speaker_turns"]["spans"], rtol=0, atol=1e-5):
        raise ValueError("嵌入轮次与当前 audit 不一致，请重新导出")
    references = collect_audio(args.reference)
    negatives = [collect_audio([path]) for path in args.exclude_role]
    analyzer = ClusterAnalyzer(Path(meta["source_stem"]), references, negatives,
                               device=args.device, progress=lambda m: print(m, flush=True))
    try:
        turns, result, local = analyzer.analyze(spans, require_dominant=not args.no_dominant_check)
        report = summarize(result, turns, load_review(args.review, source_sha256=analyzer.source_hash))
        report.update(meta=meta, local_windows=local, provenance=analyzer.provenance())
        report["provenance"].update(command=sys.argv, legacy_vectors_reused=False,
            runtime_environment=environment,
            span_input_sha256=file_digest(args.embeddings),
            review_sha256=file_digest(args.review) if args.review else None)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"reason": result.reason, "accepted_turns": sum(result.membership),
                         "review_check": {k:v for k,v in report["review_check"].items() if k != "details"}}, ensure_ascii=False))
        return 0
    finally:
        analyzer.close()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
