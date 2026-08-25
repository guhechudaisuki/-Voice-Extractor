from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extractor.pipeline import ExtractionPipeline, PipelineOptions


def main() -> int:
    parser = argparse.ArgumentParser(description="按参考音色提取多个目标文件中的完整讲话句子")
    parser.add_argument("--reference", "-r", action="append", required=True, help="参考音频，可重复")
    parser.add_argument("--target", "-t", action="append", required=True, help="待提取音频，可重复")
    parser.add_argument(
        "--exclude-role",
        action="append",
        nargs="+",
        default=[],
        metavar="AUDIO",
        help="可选排除角色音频组；同一参数后的文件属于同一人物，可重复使用该参数",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.68,
        help="声纹阈值，默认 0.68",
    )
    parser.add_argument(
        "--silence-min",
        type=float,
        default=0.20,
        help="静音合并下限（秒），默认 0.20",
    )
    parser.add_argument(
        "--silence-max",
        type=float,
        default=0.85,
        help="静音切分上限（秒），默认 0.85",
    )
    parser.add_argument(
        "--all-sentences",
        action="store_true",
        help="输出去除歌声/多人后的全部静音单句，跳过目标声纹筛选",
    )
    parser.add_argument(
        "--video-segments",
        action="store_true",
        help="目标包含视频时，同时导出对应的视频片段",
    )
    parser.add_argument("--no-overlap", action="store_true", help="关闭多人重叠检测")
    parser.add_argument("--no-singing", action="store_true", help="关闭唱歌检测")
    args = parser.parse_args()

    def progress(value: float, message: str) -> None:
        print(f"[{value * 100:5.1f}%] {message}", flush=True)

    batch = ExtractionPipeline(
        PipelineOptions(
            speaker_threshold=args.threshold,
            silence_min_seconds=args.silence_min,
            silence_split_seconds=args.silence_max,
            silence_max_seconds=args.silence_max,
            use_overlap_detector=not args.no_overlap,
            use_singing_detector=not args.no_singing,
            export_all_sentences=args.all_sentences,
            export_video_clips=args.video_segments,
        )
    ).run_many(
        [Path(item) for item in args.reference],
        [Path(item) for item in args.target],
        negative_references=[
            [Path(item) for item in group] for group in args.exclude_role
        ],
        progress=progress,
    )
    accepted_count = sum(len(result.accepted) for result in batch.results)
    rejected_count = sum(len(result.rejected) for result in batch.results)
    print(f"批次目录: {batch.output_dir}")
    print(f"批次压缩包: {batch.archive_path}")
    print(f"批次清单: {batch.manifest_path}")
    print(f"批次文本: {batch.transcript_path}")
    print(f"目标文件: {len(batch.results)} 个; 保留: {accepted_count} 句; 舍弃: {rejected_count} 句")
    for target, result in zip(args.target, batch.results):
        print(f"  {Path(target).name}: 保留 {len(result.accepted)} 句; 舍弃 {len(result.rejected)} 句")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
