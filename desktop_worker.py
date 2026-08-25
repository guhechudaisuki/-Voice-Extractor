from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def emit(event_type: str, **payload: object) -> None:
    print(
        json.dumps({"type": event_type, **payload}, ensure_ascii=False),
        flush=True,
    )


def main() -> int:
    if len(sys.argv) != 2:
        emit("error", message="缺少桌面任务配置文件")
        return 2

    request_path = Path(sys.argv[1]).resolve()
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        output_root = Path(request["output_root"]).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        os.environ["VOICE_EXTRACT_OUTPUT_ROOT"] = str(output_root)

        from extractor.pipeline import ExtractionPipeline, PipelineOptions

        references = [Path(value) for value in request.get("references", [])]
        targets = [Path(value) for value in request.get("targets", [])]
        negative_groups = [
            [Path(value) for value in group]
            for group in request.get("negative_groups", [])
            if group
        ]
        options_data = request.get("options", {})
        options = PipelineOptions(
            speaker_threshold=float(options_data.get("speaker_threshold", 0.68)),
            silence_min_seconds=float(options_data.get("silence_min_seconds", 0.20)),
            silence_split_seconds=float(options_data.get("silence_max_seconds", 0.85)),
            silence_max_seconds=float(options_data.get("silence_max_seconds", 0.85)),
            use_overlap_detector=bool(options_data.get("use_overlap_detector", True)),
            use_singing_detector=bool(options_data.get("use_singing_detector", True)),
            export_all_sentences=bool(options_data.get("export_all_sentences", False)),
            export_video_clips=bool(options_data.get("export_video_clips", False)),
        )

        def progress(value: float, message: str) -> None:
            emit(
                "progress",
                value=max(0.0, min(1.0, float(value))),
                message=str(message),
            )

        batch = ExtractionPipeline(options=options).run_many(
            references,
            targets,
            negative_references=negative_groups,
            progress=progress,
        )
        rows: list[dict[str, object]] = []
        for target, result in zip(targets, batch.results):
            for sentence in result.accepted:
                audio_path = (
                    batch.output_dir / sentence.audio_file
                    if sentence.audio_file
                    else None
                )
                video_path = (
                    batch.output_dir / sentence.video_file
                    if sentence.video_file
                    else None
                )
                rows.append(
                    {
                        "target": target.name,
                        "text": sentence.text or sentence.whisper_text,
                        "language": sentence.language,
                        "start": round(sentence.start, 3),
                        "end": round(sentence.end, 3),
                        "duration": round(sentence.duration, 3),
                        "speaker_score": round(sentence.speaker_score, 4),
                        "audio_file": str(audio_path) if audio_path else "",
                        "video_file": str(video_path) if video_path else "",
                    }
                )

        emit(
            "complete",
            batch_id=batch.batch_id,
            output_dir=str(batch.output_dir),
            archive_path=str(batch.archive_path),
            manifest_path=str(batch.manifest_path),
            transcript_path=str(batch.transcript_path),
            accepted_count=len(batch.accepted),
            rejected_count=len(batch.rejected),
            rows=rows,
        )
        return 0
    except Exception as exc:
        emit(
            "error",
            message=str(exc),
            traceback=traceback.format_exc(),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
