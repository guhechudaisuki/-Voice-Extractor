"""Export frozen adjacent-join proposals for human review, never training ZIPs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

import soundfile as sf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("probe", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Review output already exists; choose a new directory")
    probe = json.loads(args.probe.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    proposals = [row for row in probe["pairs"] if row.get("whole_still_possible")]
    stem = args.work / "stems" / "target_vocals.wav"
    args.output.mkdir(parents=True)
    rows = []
    with sf.SoundFile(stem) as source:
        for index, proposal in enumerate(proposals, 1):
            start = float(proposal["left"][0])
            end = float(proposal["right"][1])
            conflicts = [
                {
                    "span": [float(turn["start"]), float(turn["end"])],
                    "reason": turn["reject_reason"],
                }
                for turn in manifest["sentences"]
                if not turn["accepted"]
                and turn.get("diagnostics", {}).get("excluded_role_rejected")
                and min(end, float(turn["end"])) > max(start, float(turn["start"]))
            ]
            label = (
                "conflict" if conflicts
                else "local_review" if proposal.get("local_purity_unresolved")
                else "unreviewed"
            )
            name = f"R{index:02d}_{start:.2f}-{end:.2f}_{label}.wav"
            source.seek(round(start * source.samplerate))
            audio = source.read(
                round((end - start) * source.samplerate),
                dtype="float32", always_2d=True,
            ).mean(axis=1)
            sf.write(args.output / name, audio, source.samplerate, subtype="PCM_16")
            rows.append({
                "audio": name,
                "source_span": [start, end],
                "old_accepted_side": (
                    "both" if proposal["left_old_accepted"] and proposal["right_old_accepted"]
                    else "left" if proposal["left_old_accepted"]
                    else "right" if proposal["right_old_accepted"] else "neither"
                ),
                "contradictory_exclusion_parts": conflicts,
                "short_window_other_probes": proposal.get("local_other_windows", []),
                "short_window_unresolved_probes": proposal.get("local_unresolved_windows", []),
                "review_status": "not_reviewed",
            })
    report = {
        "warning": "REVIEW ONLY: these are unverified acoustic proposals, not confirmed target clips or STT/training outputs. A conflict-labeled clip contains an independently excluded subspan.",
        "source_audio": str(stem.resolve()),
        "candidates": rows,
    }
    (args.output / "review_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    archive = args.output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for file in sorted(args.output.iterdir()):
            bundle.write(file, file.name)
    print(json.dumps({"clips": len(rows), "conflicts": sum(bool(row["contradictory_exclusion_parts"]) for row in rows),
                      "archive": str(archive.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
