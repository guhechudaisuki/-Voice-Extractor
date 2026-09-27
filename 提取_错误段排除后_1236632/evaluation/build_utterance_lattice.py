"""Generate non-exporting complete-utterance proposals from a frozen run."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extractor.candidate_lattice import build_utterance_lattice
from extractor.boundary_proposals import (
    propose_cross_channel_boundaries, subdivide_islands,
)
from extractor.subtitles import SubtitleGuide
from extractor.types import TimeSpan


def build_report(
    stage: dict,
    provenance: dict,
    *,
    max_gap_seconds: float,
    max_utterance_seconds: float,
    subtitle_guide: SubtitleGuide | None = None,
    stem_boundary_report: dict | None = None,
    raw_boundary_report: dict | None = None,
) -> dict:
    if (stem_boundary_report is None) != (raw_boundary_report is None):
        raise ValueError("Both boundary channels are required together")
    stages = stage["stages"]
    islands = [TimeSpan(*span) for span in stages["clean_speech_islands"]["spans"]]
    blocked = [
        TimeSpan(*span)
        for name in (
            "pre_uvr_singing_mask",
            "residual_singing_evidence",
            "singing_blocked_islands",
            "overlap_evidence",
            "overlap_blocked_islands",
        )
        for span in stages.get(name, {}).get("spans", [])
    ]
    source_hash = provenance["inputs_sha256"]["target"]
    cues: list[tuple[int, TimeSpan]] = []
    if subtitle_guide is not None:
        subtitle_guide.calibrate(
            [TimeSpan(*span) for span in stages["initial_vad"]["spans"]]
        )
        cues = [
            (cue.index, aligned)
            for cue in subtitle_guide.cues
            if (aligned := subtitle_guide.cue_span(cue)) is not None
        ]
    proposals = build_utterance_lattice(
        source_hash,
        islands,
        max_gap_seconds=max_gap_seconds,
        max_utterance_seconds=max_utterance_seconds,
        blocked=blocked,
        subtitle_cues=cues,
    )
    boundary_rows: list[dict] = []
    boundary_partition_variants = 0
    if stem_boundary_report is not None and raw_boundary_report is not None:
        cuts = propose_cross_channel_boundaries(
            islands,
            (
                item for case in stem_boundary_report["cases"]
                for item in case["boundaries"]
            ),
            (
                item for case in raw_boundary_report["cases"]
                for item in case["boundaries"]
            ),
            include_unmatched_stem=True,
        )
        boundary_rows = [cut.to_dict() for cut in cuts]
        seen = {item.source_id for item in proposals}
        # Two channels agreeing within tolerance do not locate the change at
        # the exact midpoint. Keep the center partition plus each observed
        # endpoint as a separate proposal. Never put all three cuts in one
        # partition: doing so invents 50 ms phonetic fragments.
        partitions = [cuts]
        for index, cut in enumerate(cuts):
            if cut.raw_time is None:
                continue
            for edge in (cut.stem_time, cut.raw_time):
                if round(edge * 16000) != round(cut.time * 16000):
                    partitions.append([
                        replace(row, time=edge) if position == index else row
                        for position, row in enumerate(cuts)
                    ])
        boundary_partition_variants = len(partitions)
        for partition in partitions:
            refined = build_utterance_lattice(
                source_hash,
                subdivide_islands(islands, partition),
                max_gap_seconds=max_gap_seconds,
                max_utterance_seconds=max_utterance_seconds,
                blocked=blocked,
                subtitle_cues=cues,
            )
            for item in refined:
                if item.source_id not in seen:
                    proposals.append(item)
                    seen.add(item.source_id)
    return {
        "schema_version": 1,
        "warning": (
            "Non-destructive candidate geometry only. No identity, sentence "
            "completeness, singing or overlap purity is inferred from a proposal."
        ),
        "source_sha256": source_hash,
        "analysis_sample_rate": 16000,
        "atomic_island_count": len(islands),
        "proposal_count": len(proposals),
        "internal_boundary_proposal_count": len(boundary_rows),
        "boundary_partition_variants": boundary_partition_variants,
        "internal_boundary_proposals": boundary_rows,
        "max_gap_seconds": max_gap_seconds,
        "max_utterance_seconds": max_utterance_seconds,
        "subtitle_timing": (
            subtitle_guide.report if subtitle_guide is not None else None
        ),
        "proposals": [item.to_dict() for item in proposals],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage_audit", type=Path)
    parser.add_argument("evaluation_provenance", type=Path)
    parser.add_argument("--max-gap-seconds", type=float, default=0.85)
    parser.add_argument("--max-utterance-seconds", type=float, default=45.0)
    parser.add_argument("--subtitle", type=Path)
    parser.add_argument("--stem-boundaries", type=Path)
    parser.add_argument("--raw-boundaries", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        json.loads(args.stage_audit.read_text(encoding="utf-8")),
        json.loads(args.evaluation_provenance.read_text(encoding="utf-8")),
        max_gap_seconds=args.max_gap_seconds,
        max_utterance_seconds=args.max_utterance_seconds,
        subtitle_guide=(
            SubtitleGuide.load(args.subtitle) if args.subtitle else None
        ),
        stem_boundary_report=(
            json.loads(args.stem_boundaries.read_text(encoding="utf-8"))
            if args.stem_boundaries else None
        ),
        raw_boundary_report=(
            json.loads(args.raw_boundaries.read_text(encoding="utf-8"))
            if args.raw_boundaries else None
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"UTTERANCE_LATTICE={args.output} PROPOSALS={report['proposal_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
