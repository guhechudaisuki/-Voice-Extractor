"""Run a repeatable local batch from a saved desktop request.

The request supplies reference groups and options. A target and optional
subtitle may be substituted for a held-out or regression source. Model weights,
audio inputs and generated reports remain in the local workspace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extractor.pipeline import ExtractionPipeline, PipelineOptions  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--subtitle", type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    references = [Path(value) for value in request["references"]]
    negatives = [[Path(value) for value in group] for group in request.get("negative_groups", [])]
    target = args.target.resolve()
    subtitle = args.subtitle.resolve() if args.subtitle else None
    all_inputs = [*references, *(item for group in negatives for item in group), target]
    if subtitle:
        all_inputs.append(subtitle)
    missing = [str(item) for item in all_inputs if not item.is_file()]
    if missing:
        parser.error("Missing input files: " + ", ".join(missing))
    options = PipelineOptions(**request["options"])
    code_hashes = {str(path.relative_to(ROOT)): sha256(path)
                   for path in sorted((ROOT / "extractor").glob("*.py"))}
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    inputs = {
        "target": sha256(target),
        "subtitle": sha256(subtitle) if subtitle else None,
        "references": [sha256(path) for path in references],
        "negative_groups": [[sha256(path) for path in group] for group in negatives],
    }
    started = time.monotonic()
    last_percent = -1
    last_message_time = 0.0

    def progress(value: float, message: str) -> None:
        nonlocal last_percent, last_message_time
        percent = int(value * 100)
        now = time.monotonic()
        if percent > last_percent or percent == 100 or now - last_message_time >= 5.0:
            print(f"[{value * 100:5.1f}%] {message}", flush=True)
            last_percent = percent
            last_message_time = now

    result = ExtractionPipeline(options).run_many(
        references,
        [target],
        negative_references=negatives,
        subtitles={target: subtitle} if subtitle else None,
        progress=progress,
    )
    provenance = {
        "commit": commit,
        "pipeline_sha256": sha256(ROOT / "extractor" / "pipeline.py"),
        "code_sha256_at_start": code_hashes,
        "request_sha256": sha256(args.request),
        "inputs_sha256": inputs,
        "options": request["options"],
        "accepted_count": len(result.accepted),
        "rejected_count": len(result.rejected),
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    report_path = result.output_dir / "evaluation_provenance.json"
    report_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"BATCH_MANIFEST={result.manifest_path}", flush=True)
    print(f"PROVENANCE={report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
