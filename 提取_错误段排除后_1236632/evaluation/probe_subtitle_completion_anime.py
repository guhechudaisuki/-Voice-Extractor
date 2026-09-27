"""Score subtitle-timed sentence candidates with local anime-domain models.

This is a diagnostic only. It reports target-vs-open-set competition and never
changes production acceptance or writes audio.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extractor.audio import load_mono
from extractor.subtitles import SubtitleGuide
from extractor.types import TimeSpan
from evaluation.probe_anime_embedding import MODELS, encode, read_audio, scores


def overlap(left: TimeSpan, right: TimeSpan) -> float:
    return max(0.0, min(left.end, right.end) - max(left.start, right.start))


def groups_from_stage(stage: dict, guide: SubtitleGuide) -> list[tuple[int, list[TimeSpan]]]:
    speech = [TimeSpan(*span) for span in stage["stages"]["clean_speech_islands"]["spans"]]
    return [(cue.index, parts) for cue, parts in guide.groups(speech, 0.85)]


def run(work_dir: Path, stage_path: Path, subtitle_path: Path) -> dict:
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    guide = SubtitleGuide.load(subtitle_path)
    guide.calibrate([TimeSpan(*span) for span in stage["stages"]["initial_vad"]["spans"]])
    if not guide.aligned:
        raise ValueError("Subtitle timeline did not align with frozen speech")
    groups = groups_from_stage(stage, guide)
    accepted = [TimeSpan(*span) for span in stage["stages"].get("identity_accepted_before_stt", {}).get("spans", [])]
    candidates = []
    for cue_index, parts in groups:
        span = TimeSpan(parts[0].start, parts[-1].end)
        cores = [item for item in accepted if overlap(span, item) > 0.02]
        if cores and all(item.start >= span.start - 0.001 and item.end <= span.end + 0.001 for item in cores):
            continue
        candidates.append({"cue_index": cue_index, "parts": parts, "span": span})

    source = work_dir / "stems/target_vocals.wav"
    refs = sorted((work_dir / "reference_voice_clips").glob("*.wav"))
    groups_paths = [sorted(group.glob("*.wav")) for group in sorted((work_dir / "negative_reference_voice_clips").glob("role_*"))]
    groups_paths = [group for group in groups_paths if group]
    if not refs:
        raise ValueError("No target reference clips")
    models = {}
    target_vectors = {}
    negative_vectors = {}
    try:
        for variant in ("char", "va"):
            model = __import__("anime_speaker_embedding.model", fromlist=["AnimeSpeakerEmbedding"]).AnimeSpeakerEmbedding(
                variant=variant, ckpt_path=MODELS[variant]
            ).eval()
            models[variant] = model
            target_vectors[variant] = torch.stack([encode(model, read_audio(path)) for path in refs])
            negative_vectors[variant] = [
                torch.stack([encode(model, read_audio(path)) for path in group])
                for group in groups_paths
            ]
        rows = []
        for number, candidate in enumerate(candidates, start=1):
            part_rows = []
            for part in candidate["parts"]:
                part_result = {}
                for variant in ("char", "va"):
                    vector = encode(models[variant], read_audio(source, part.start, part.end))
                    part_result[variant] = scores(vector, target_vectors[variant], negative_vectors[variant])
                part_rows.append({"span": [part.start, part.end], "models": part_result})
            whole_result = {}
            for variant in ("char", "va"):
                vector = encode(models[variant], read_audio(source, candidate["span"].start, candidate["span"].end))
                whole_result[variant] = scores(vector, target_vectors[variant], negative_vectors[variant])
            part_witness = [
                all(bool(row["models"][variant]["target_median"] > max(row["models"][variant]["negative_group_max"], default=-1.0)) for variant in ("char", "va"))
                for row in part_rows
            ]
            rows.append({
                "cue_index": candidate["cue_index"],
                "span": [candidate["span"].start, candidate["span"].end],
                "part_count": len(part_rows),
                "parts": part_rows,
                "whole": whole_result,
                "all_parts_win_both_models": bool(part_witness and all(part_witness)),
                "part_witness": part_witness,
            })
            if number % 10 == 0 or number == len(candidates):
                print(f"subtitle anime probe {number}/{len(candidates)}", flush=True)
        return {
            "warning": "Diagnostic only. Anime-domain scores are not calibrated probabilities and do not authorize export.",
            "candidate_count": len(rows),
            "all_parts_win_both_models_count": sum(row["all_parts_win_both_models"] for row in rows),
            "rows": rows,
        }
    finally:
        for model in models.values():
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("stage_audit", type=Path)
    parser.add_argument("subtitle", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = run(args.work_dir, args.stage_audit, args.subtitle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SUBTITLE_ANIME_PROBE={args.output} CANDIDATES={report['candidate_count']} WITNESS={report['all_parts_win_both_models_count']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
