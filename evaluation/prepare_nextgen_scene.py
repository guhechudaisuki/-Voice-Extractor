"""Short-scene preprocessing smoke tool; it never exports target-speaker audio."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extractor.nextgen.prepare_media import prepare_bounded_media  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Short source audio or video")
    parser.add_argument("destination", type=Path, help="New work directory")
    parser.add_argument("--subtitle", type=Path, help="Matching local subtitle file")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    result = prepare_bounded_media(
        args.source, args.destination, subtitle=args.subtitle, device=args.device,
        progress=lambda stage, value, message:
        print(f"[{stage} {value:.0%}] {message}", flush=True),
    )
    print(f"Prepared {len(result.scene.candidates)} unverified candidates; "
          f"report: {result.report_path}", flush=True)


if __name__ == "__main__":
    main()
