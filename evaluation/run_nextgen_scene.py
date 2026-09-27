"""Run one cached, bounded scene through the independent identity engine.

Research checkpoints are review-only: this command never emits training WAVs
or transcripts. It deliberately does not replace the desktop/production path.
Prepare each scene with prepare_nextgen_scene.py first (default maximum 90 s).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extractor.nextgen.artifacts import load_bundle  # noqa: E402
from extractor.nextgen.features import WavLMSpeakerFeatures  # noqa: E402
from extractor.nextgen.identity_model import HEADS  # noqa: E402
from extractor.nextgen.inference import IdentitySession  # noqa: E402
from extractor.nextgen.prepare_media import load_prepared_scene  # noqa: E402
from extractor.nextgen.reference_preparation import (ReferenceMaterial,  # noqa: E402
                                                     prepare_reference_bank)
from extractor.nextgen.runtime import run_scene  # noqa: E402


def run_cached_scene(scene_directory: Path, bundle: Path,
                     target_references: tuple[Path, ...],
                     exclusions: tuple[tuple[str, Path], ...], *,
                     research: bool = False, device: str = "cuda") -> dict:
    """Review exact preprocessed audio with fingerprint-matched weights.

    This bridge returns an auditable report only. Actual release/export remains
    gated on independent model validation and product-path integration.
    """
    if not target_references:
        raise ValueError("At least one target reference is required")
    if any(not role or role == "target" for role, _ in exclusions):
        raise ValueError("Exclusion roles must be named and distinct from target")
    scene = load_prepared_scene(scene_directory)
    if scene.audio.source.total_samples > 90 * 16000:
        raise ValueError("Research scene must be no longer than 90 seconds")
    encoder = WavLMSpeakerFeatures(ROOT / "model/speaker/wavlm-base-plus-sv", device=device)
    model, card, calibration = load_bundle(bundle, device=device, research=research)
    if card.stage != "validated" and not research:
        raise ValueError("Only a validated model can run outside explicit research mode")
    if encoder.digest != card.backbone_digest:
        raise ValueError("Cached scene encoder and model backbone do not match")
    materials = tuple(ReferenceMaterial("target", load_prepared_scene(path))
                      for path in target_references)
    materials += tuple(ReferenceMaterial(role, load_prepared_scene(path))
                       for role, path in exclusions)
    bank, encoded = prepare_reference_bank(materials, encoder)
    session = IdentitySession(model, card, calibration, bank, encoded, research=research)
    result = run_scene(scene, session, encoder)
    return {
        "schema": 1,
        "status": "research_review_only" if research else "validated_acoustic_review_no_export",
        "model_digest": card.weights_digest,
        "reference_digest": bank.digest,
        "candidate_count": len(scene.candidates),
        "review_count": len(result.reviews),
        "selected_count": len(result.selected),
        "cancelled": result.cancelled,
        "selected": [asdict(row) for row in result.selected],
        "heads": HEADS,
        "reviews": [{"candidate": asdict(row.candidate),
                     "prediction": asdict(row.prediction),
                     "head_scores": row.head_scores,
                     "boundary_scores": row.boundary_scores,
                     "state": row.assessment.state.value,
                     "reasons": row.assessment.reasons,
                     "risk_spans": [asdict(span) for span in row.assessment.risk_spans]}
                    for row in result.reviews],
        "note": "Uncalibrated sigmoid head scores and acoustic decisions are not human-verified truth or a complete audio delivery.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path, help="Previously prepared bounded query scene")
    parser.add_argument("bundle", type=Path, help="Versioned model/calibration directory")
    parser.add_argument("--target-reference", type=Path, action="append", required=True,
                        help="Previously prepared, clean target reference scene; may repeat")
    parser.add_argument("--exclude", action="append", default=[], metavar="ROLE=SCENE",
                        help="Optional named exclusion-role reference scene; may repeat")
    parser.add_argument("--output", type=Path, required=True, help="New JSON report path")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--research", action="store_true",
                        help="Explicitly permit a research bundle for a review-only report")
    args = parser.parse_args()
    exclusions = []
    for item in args.exclude:
        if "=" not in item:
            parser.error("--exclude requires ROLE=SCENE")
        role, path = item.split("=", 1)
        if not role or not path:
            parser.error("--exclude requires ROLE=SCENE")
        exclusions.append((role, Path(path)))
    report = run_cached_scene(args.scene, args.bundle, tuple(args.target_reference),
                              tuple(exclusions), research=args.research,
                              device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(f"{report['status']}: {report['selected_count']} model-selected intervals; {args.output}")


if __name__ == "__main__":
    main()
