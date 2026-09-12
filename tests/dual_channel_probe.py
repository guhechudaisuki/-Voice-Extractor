"""Small offline probe for the archived first-episode boundary fixture.

This is intentionally not a full pipeline run: it compares the two speaker
front ends on the already separated and time-aligned fixture.  It is useful
when changing channel fusion because it finishes in a few minutes instead of
re-running UVR/VAD/STT.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from extractor.audio import load_mono
from extractor.pipeline import ExtractionPipeline
from extractor.speaker import DualSpeakerVerifier
from extractor.transcription import FunASRTools


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    ROOT
    / "versions"
    / "VoiceExtractor_boundary_candidate"
    / "work"
    / "20260825_153507_batch_ba8782_001"
)
REGRESSION = (
    ROOT
    / "versions"
    / "VoiceExtractor_boundary_candidate"
    / "work"
    / "candidate_boundary_regression.json"
)


def main() -> None:
    regression = json.loads(REGRESSION.read_text(encoding="utf-8"))
    stem = load_mono(FIXTURE / "stems" / "target_vocals.wav", 16000)
    raw = load_mono(FIXTURE / "target_normalized.wav", 16000)
    stem_refs = sorted((FIXTURE / "reference_stems").glob("*.wav"))
    raw_sources = sorted((FIXTURE / "references_normalized").glob("*.wav"))
    vad = FunASRTools("cuda").vad_many(stem_refs)

    with tempfile.TemporaryDirectory(prefix="voice-extractor-dual-probe-") as temp:
        temp_root = Path(temp)
        raw_map = {
            raw_path: list(vad.get(stem_path, []))
            for raw_path, stem_path in zip(raw_sources, stem_refs)
        }
        raw_refs = ExtractionPipeline()._make_reference_clips(
            raw_sources,
            raw_map,
            {"reference_clips": temp_root},
            output_dir=temp_root,
            prefix="reference",
        )
        stem_clips = sorted((FIXTURE / "reference_voice_clips").glob("*.wav"))
        verifier = DualSpeakerVerifier("cuda")
        try:
            profile = verifier.build_channel_profile(stem_clips, raw_refs, 0.68)

            def clip(waveform, start: float, end: float):
                return waveform[round(start * 16000) : round(end * 16000)]

            cases = [
                *(('positive', index, start, end) for index, (start, end) in enumerate(regression["after"], 1)),
                ('negative', 1, 448.90, 450.45),
                ('negative', 2, 525.31, 526.81),
                ('negative', 3, 638.14, 643.54),
                ('negative', 4, 340.81, 342.62),
                ('negative', 5, 975.78, 980.58),
                ('negative', 6, 1056.34, 1059.36),
            ]
            if os.environ.get("DUAL_PROBE_FOCUS"):
                cases = [
                    case
                    for case in cases
                    if (case[0], case[1])
                    in {
                        ("positive", 4),
                        ("positive", 6),
                        ("positive", 11),
                        ("negative", 5),
                        ("negative", 6),
                    }
                ]
            rescued_positive: list[int] = []
            rescued_negative: list[int] = []
            for label, index, start, end in cases:
                decision = verifier.verify_dual_channel_waveform(
                    clip(stem, start, end),
                    clip(raw, start, end),
                    profile,
                    0.68,
                    duration=end - start,
                    clean_gate=True,
                    audit_raw_on_uvr_accept=True,
                )
                print(
                    label,
                    index,
                    f"{start:.2f}-{end:.2f}",
                    decision.accepted,
                    decision.tier,
                    decision.diagnostics.get("uvr_eres_score"),
                    decision.diagnostics.get("uvr_camplus_score"),
                    decision.diagnostics.get("raw_eres_score"),
                    decision.diagnostics.get("raw_camplus_score"),
                    decision.diagnostics.get("raw_rescue_reason"),
                    "uvr_ref",
                    decision.diagnostics.get("uvr_eres_reference_max"),
                    decision.diagnostics.get("uvr_camplus_reference_max"),
                    decision.diagnostics.get("uvr_paired_reference_median"),
                    "raw_ref",
                    decision.diagnostics.get("raw_eres_reference_max"),
                    decision.diagnostics.get("raw_camplus_reference_max"),
                    decision.diagnostics.get("raw_paired_reference_median"),
                    "p20",
                    decision.diagnostics.get("uvr_eres_window_p20"),
                    decision.diagnostics.get("uvr_camplus_window_p20"),
                    decision.diagnostics.get("raw_eres_window_p20"),
                    decision.diagnostics.get("raw_camplus_window_p20"),
                    "cross",
                    decision.diagnostics.get("raw_rescue_cross_primary_pair"),
                    decision.diagnostics.get("raw_rescue_cross_secondary_pair"),
                    "diag",
                    decision.diagnostics.get("raw_rescue_diagonal_reference_count"),
                    decision.diagnostics.get("raw_rescue_diagonal_reference_pair"),
                    decision.diagnostics.get("raw_rescue_diagonal_evidence"),
                    decision.diagnostics.get("raw_rescue_diagonal_primary_gain"),
                )
                if decision.diagnostics.get("raw_rescue"):
                    (rescued_positive if label == "positive" else rescued_negative).append(index)
                if os.environ.get("DUAL_PROBE_SCORES") and decision.raw_primary is not None and decision.raw_secondary is not None:
                    print(
                        "  per_ref_raw_eres",
                        [round(float(v), 4) for v in profile.raw_primary.embeddings @ decision.raw_primary.embedding],
                    )
            if not os.environ.get("DUAL_PROBE_FOCUS"):
                expected_positive = {4, 6}
                if set(rescued_positive) != expected_positive:
                    raise AssertionError(
                        f"unexpected diagonal rescues: {rescued_positive}; expected {sorted(expected_positive)}"
                    )
                if rescued_negative:
                    raise AssertionError(
                        f"known negative spans rescued: {rescued_negative}"
                    )
                    print(
                        "  per_ref_uvr_cam",
                        [round(float(v), 4) for v in verifier.secondary_profile.embeddings @ decision.secondary.embedding],
                    )
        finally:
            verifier.close()


if __name__ == "__main__":
    main()
