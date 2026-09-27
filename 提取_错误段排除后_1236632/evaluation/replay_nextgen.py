"""Replay serialized scoped predictions without loading any model or audio.

This is a policy/code diagnostic, not a speaker-recognition accuracy test. The
input artifact must declare research provenance. It cannot create audio output.
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

from extractor.nextgen.boundary_decoder import select_verified
from extractor.nextgen.decision_policy import Calibration, FramePrediction, SpanPrediction, assess
from extractor.nextgen.ledger import Candidate, EvidenceKind, LocalEvidence
from extractor.nextgen.timeline import SampleSpan


def replay(payload: dict) -> dict:
    if payload.get("schema") != 1 or payload.get("purpose") != "research_policy_replay":
        raise ValueError("Expected an explicitly research-only prediction artifact")
    calibration = Calibration(**payload["calibration"])
    candidates, assessments = [], []
    for row in payload["rows"]:
        raw = row["candidate"]
        candidate = Candidate(**{**raw, "output": SampleSpan(**raw["output"]),
                                 "context": SampleSpan(**raw["context"]),
                                 "speech": tuple(SampleSpan(**part) for part in raw["speech"]),
                                 "parents": tuple(raw.get("parents", ()))})
        raw_prediction = row["prediction"]
        prediction = SpanPrediction(**{**raw_prediction, "frames": tuple(
            FramePrediction(**{**frame, "span": SampleSpan(**frame["span"])})
            for frame in raw_prediction["frames"]
        )})
        evidence = tuple(LocalEvidence(**{**item, "span": SampleSpan(**item["span"]),
                                          "kind": EvidenceKind(item["kind"])}) for item in row.get("evidence", ()))
        candidates.append(candidate)
        assessments.append(assess(candidate, prediction, calibration,
                                  reference_digest=payload["reference_digest"], local_evidence=evidence))
    selected = select_verified(tuple(candidates), tuple(assessments))
    return {"schema": 1, "purpose": "policy_replay_not_acoustic_validation",
            "assessments": [asdict(row) for row in assessments],
            "selected": [asdict(row) for row in selected]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = replay(json.loads(args.input.read_text(encoding="utf-8")))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
