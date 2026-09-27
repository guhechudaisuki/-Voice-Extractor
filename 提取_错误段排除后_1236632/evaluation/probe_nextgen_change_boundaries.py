"""Probe independent local speaker-change hints on a prepared short scene.

Diagnostic only: a change hint is neither a target label nor an export cut.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extractor.nextgen.prepare_media import load_prepared_scene  # noqa: E402
from extractor.speaker import DualSpeakerVerifier, LocalSpeakerTurnSplitter  # noqa: E402
from extractor.types import TimeSpan  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--context", type=float, default=0.60)
    parser.add_argument("--hop", type=float, default=0.10)
    args = parser.parse_args()
    scene = load_prepared_scene(args.scene)
    spans = [TimeSpan(span.start / 16000, span.end / 16000)
             for span in scene.stem_voice]
    verifier = DualSpeakerVerifier(args.device)
    try:
        splitter = LocalSpeakerTurnSplitter(
            verifier.primary,
            secondary_factory=verifier._boundary_secondary_factory,
        )
        boundaries = splitter.detect_speaker_boundaries(
            args.scene / "stem_16000.wav", spans,
            context_seconds=args.context, scan_hop_seconds=args.hop,
            primary_candidate_threshold=0.78,
            minimum_separation_seconds=0.30,
        )
    finally:
        verifier.close()
    print(json.dumps({"scene": str(args.scene),
                      "boundaries": [row.to_dict() for row in boundaries]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
