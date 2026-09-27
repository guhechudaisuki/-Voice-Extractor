"""Export a small, explicitly unverified audio+STT preview from repair proposals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extractor.audio import write_clip  # noqa: E402
from extractor.pipeline import ExtractionPipeline  # noqa: E402
from extractor.transcription import WhisperSegmenter  # noqa: E402
from extractor.types import TimeSpan  # noqa: E402


def select_spans(report: dict, starts: list[float] | None = None) -> list[tuple[int, dict]]:
    if "rows" in report:
        selected = [
            (row["index"], child)
            for row in report["rows"] for child in row["children"]
            if child["state"] == "identity_supported_unverified_boundary"
        ]
    elif "sentences" in report:
        selected = [
            (index, {"span": [row["start"], row["end"]]})
            for index, row in enumerate(report["sentences"], 1)
            if row["accepted"]
        ]
    else:
        raise ValueError("Expected repair proposals or an accepted-span replay")
    if starts is not None:
        selected = [
            (index, child) for index, child in selected
            if any(abs(float(child["span"][0]) - start) < 0.05
                   for start in starts)
        ]
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("proposals", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--starts", type=float, nargs="+",
                        help="Only export child spans starting at these source times")
    args = parser.parse_args()
    report = json.loads(args.proposals.read_text(encoding="utf-8"))
    selected = select_spans(report, args.starts)
    replay_source = "sentences" in report
    if not selected:
        parser.error("No identity-supported child spans to preview")
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    audio_dir = destination / "audio"
    text_dir = destination / "text"
    audio_dir.mkdir()
    text_dir.mkdir()
    stem = args.work / "stems" / "target_vocals.wav"
    spans = [TimeSpan(*map(float, child["span"])) for _, child in selected]
    transcribed = WhisperSegmenter("cuda").transcribe_spans(
        stem, spans,
        progress=lambda value, message: print(f"[{value * 100:5.1f}%] {message}", flush=True),
    )
    text_by_index: dict[int, str] = {}
    for fragment in transcribed:
        index = int(fragment.diagnostics.get("transcription_span_index", -1))
        text_by_index[index] = ExtractionPipeline._join_transcript_text(
            text_by_index.get(index, ""), fragment.whisper_text
        )
    manifest = {"status": "UNVERIFIED_PREVIEW", "source": str(args.proposals),
                "warning": "Needs human listening. Do not use as training audio or replace the stable version.",
                "clips": []}
    for ordinal, (source_index, child) in enumerate(selected, 1):
        start, end = map(float, child["span"])
        origin_label = "replay" if replay_source else "from"
        basename = f"{ordinal:04d}_{origin_label}_{source_index:04d}_{start:.2f}-{end:.2f}"
        audio_file = audio_dir / f"{basename}.wav"
        text_file = text_dir / f"{basename}.txt"
        write_clip(stem, audio_file, start, end, sample_rate=16000,
                   normalize_level=True)
        transcription = text_by_index.get(ordinal - 1, "").strip()
        text_file.write_text(transcription, encoding="utf-8")
        manifest["clips"].append({
            ("replay_sentence_index" if replay_source else "parent_index"): source_index,
            "span": [start, end],
            "audio_file": str(audio_file.relative_to(destination)),
            "text_file": str(text_file.relative_to(destination)),
            "stt": transcription,
        })
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    archive = destination.with_suffix(".zip")
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as zipped:
        for file in sorted(destination.rglob("*")):
            if file.is_file():
                zipped.write(file, file.relative_to(destination))
    print(f"PREVIEW={destination}")
    print(f"ZIP={archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
