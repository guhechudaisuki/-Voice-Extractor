"""Replay completion on a frozen pre-STT identity result, without exporting audio.

The saved stage interval set is authoritative for pre-STT acceptance. Rejected
manifest rows are retained, including their structural vetoes. No review labels
are passed to the completion function.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extractor.audio import load_mono
from extractor.pipeline import ExtractionPipeline, PipelineOptions
from extractor.speaker import DualSpeakerVerifier
from extractor.subtitle_assistance import restore_subtitle_sentences
from extractor.subtitles import SubtitleGuide
from extractor.types import CandidateSentence, TimeSpan
from evaluation.probe_adjacent_pairs import clips, score


def key(start: float, end: float) -> tuple[int, int]:
    return round(start * 16000), round(end * 16000)


def parse_window(value: str) -> TimeSpan:
    try:
        start, end = map(float, value.split(":"))
        if not 0 <= start < end < float("inf"):
            raise ValueError("expected finite 0 <= start < end")
        return TimeSpan(start, end)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use START:END in seconds") from error


def pre_stt_candidates(manifest: dict, stage: dict) -> tuple[list, list]:
    names = {field.name for field in fields(CandidateSentence)}
    records = [CandidateSentence(**{k: v for k, v in row.items() if k in names})
               for row in manifest["sentences"]]
    accepted = []
    for start, end in stage["stages"]["identity_accepted_before_stt"]["spans"]:
        matches = [row for row in records if key(row.start, row.end) == key(start, end)]
        if not matches:
            raise ValueError(f"No manifest evidence for pre-STT span {start}:{end}")
        record = next((row for row in matches if row.accepted), matches[0])
        record.accepted = False  # still a proposal before STT and export
        record.reject_reason = ""
        accepted.append(record)
    selected = {id(row) for row in accepted}
    rejected = [row for row in records if id(row) not in selected and not row.accepted]
    return accepted, rejected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("subtitle", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--window", action="append", type=parse_window, default=[],
                        help="Only replay cues intersecting START:END; repeat for local regressions")
    args = parser.parse_args()
    stage = json.loads((args.run_dir / "stage_audit.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    accepted, rejected = pre_stt_candidates(manifest, stage)
    original = {key(row.start, row.end) for row in accepted}
    pipeline = ExtractionPipeline(PipelineOptions(**manifest["options"]))
    work = args.work_dir
    pipeline._raw_target_waveform = load_mono(work / "target_normalized.wav", 16000)
    stem_path = work / "stems/target_vocals.wav"
    waveform = load_mono(stem_path, 16000)
    clean = [TimeSpan(*span) for span in stage["stages"]["clean_speech_islands"]["spans"]]
    blocked = [TimeSpan(*span) for name in ("singing_blocked_islands", "overlap_blocked_islands")
               for span in stage["stages"].get(name, {}).get("spans", [])]
    pipeline._raw_blocked_spans = tuple(blocked)
    guide = SubtitleGuide.load(args.subtitle)
    guide.calibrate([TimeSpan(*span) for span in stage["stages"]["initial_vad"]["spans"]])
    if not guide.aligned:
        raise ValueError("Frozen subtitle no longer aligns")
    if args.window:
        # Calibrate on the full frozen timeline first. Limit cue proposals,
        # never crop the waveform, speech islands, or existing veto evidence.
        guide.cues = [cue for cue in guide.cues
                      if (span := guide.cue_span(cue)) is not None and any(
                          min(span.end, window.end) > max(span.start, window.start)
                          for window in args.window)]
    evidence = []
    verify_original = pipeline._verify_speaker_span

    def verify(verifier, samples, span, profile, threshold):
        match = verify_original(verifier, samples, span, profile, threshold)
        evidence.append({"span": [span.start, span.end], **score(match, None)})
        return match

    pipeline._verify_speaker_span = verify
    started = time.monotonic()
    verifier = DualSpeakerVerifier(pipeline.device)
    try:
        profile = verifier.build_channel_profile(
            clips(work / "reference_voice_clips"),
            clips(work / "reference_original_voice_clips"),
            pipeline.options.speaker_threshold,
        )
        groups = sorted((work / "negative_reference_voice_clips").glob("role_*"))
        exclusions = verifier.build_channel_exclusion_profiles(
            [clips(group) for group in groups], profile,
            raw_reference_groups=[clips(work / "negative_reference_original_voice_clips" / g.name)
                                  for g in groups],
        ) if groups else []
        restored = restore_subtitle_sentences(
            pipeline, guide, accepted, rejected, clean, blocked, verifier, profile,
            stem_path, waveform, exclusions,
            max(pipeline.options.speaker_threshold, profile.primary.suggested_threshold),
            progress=lambda value, message: print(message, flush=True),
        )
    finally:
        verifier.close()
    final = {key(row.start, row.end) for row in accepted}
    report = {
        "warning": "Frozen-stage replay only; new candidates still require identity and acoustic completeness review. No audio is accepted or exported by this report.",
        "requested_windows": [[span.start, span.end] for span in args.window],
        "code_sha256": hashlib.sha256((ROOT / "extractor/subtitle_assistance.py").read_bytes()).hexdigest(),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "restored": restored,
        "original_span_count": len(original),
        "final_span_count": len(final),
        "added_spans": [[a / 16000, b / 16000] for a, b in sorted(final - original)],
        "removed_spans": [[a / 16000, b / 16000] for a, b in sorted(original - final)],
        "result_counts": dict(Counter(r["result"] for r in guide.report["completion_proposals"])),
        "subtitle_report": guide.report,
        "identity_probes": evidence,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"COMPLETION_REPLAY={args.output} RESTORED={restored}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
