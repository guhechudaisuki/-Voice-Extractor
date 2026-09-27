"""Focused real-model checks, not a full extraction or a human accuracy label."""

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extractor.audio import load_mono
from extractor.pipeline import ExtractionPipeline
from extractor.speaker import DualSpeakerVerifier, LocalSpeakerTurnSplitter
from extractor.transcription import FunASRTools, WhisperSegmenter
from extractor.types import TimeSpan


def main():
    fixture = ROOT / "versions/VoiceExtractor_boundary_candidate/work/20260825_153507_batch_ba8782_001"
    data = json.loads((ROOT / "work/subtitle_timing_probe.json").read_text(encoding="utf-8"))
    pipeline = ExtractionPipeline()
    stem_path = fixture / "stems/target_vocals.wav"
    stem, raw = load_mono(stem_path, 16000), load_mono(fixture / "target_normalized.wav", 16000)
    stem_refs = sorted((fixture / "reference_stems").glob("*.wav"))
    raw_sources = sorted((fixture / "references_normalized").glob("*.wav"))
    vad = FunASRTools("cuda").vad_many(stem_refs)
    with TemporaryDirectory(prefix="subtitle_identity_", dir=ROOT / "work") as temp:
        raw_refs = pipeline._make_reference_clips(
            raw_sources, {p: vad.get(s, []) for p, s in zip(raw_sources, stem_refs)},
            {"reference_clips": Path(temp)}, output_dir=Path(temp), prefix="reference")
        verifier = DualSpeakerVerifier("cuda", status=lambda m: print(m, flush=True))
        try:
            profile = verifier.build_channel_profile(sorted((fixture / "reference_voice_clips").glob("*.wav")), raw_refs, .68)
            groups = [sorted(p.glob("*.wav")) for p in sorted((fixture / "negative_reference_voice_clips").glob("role_*"))]
            exclusions = verifier.build_channel_exclusion_profiles(groups, profile) if groups else []
            pipeline._raw_target_waveform = raw
            splitter = LocalSpeakerTurnSplitter(verifier.primary, secondary=verifier.secondary)
            cases = [("subtitle_self_introduction", *g["spans"][0]) for g in data["groups"] if 338 < g["subtitle"][0] < 339]
            cases.extend([("known_next_speaker", 340.8825, 342.2625), ("known_mixed_turn", 975.78, 980.58)])
            results = []
            for label, start, end in cases:
                span = TimeSpan(start, end)
                match = pipeline._verify_speaker_span(verifier, stem, span, profile, max(.68, profile.primary.suggested_threshold))
                exclusion = verifier.exclusion_audit(match, profile, exclusions)
                boundaries = splitter.detect_multiscale_speaker_boundaries(stem_path, [span])
                row = {"label": label, "span": [start, end], "identity_accepted": match.accepted,
                       "tier": match.tier, "exclusion": exclusion,
                       "boundaries": [b.to_dict() for b in boundaries]}
                results.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
            (ROOT / "work/subtitle_identity_probe.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        finally:
            verifier.close()
        self_introduction = [TimeSpan(row["span"][0], row["span"][1]) for row in results
                             if row["label"] == "subtitle_self_introduction" and row["identity_accepted"]]
        text = WhisperSegmenter("cuda").transcribe_spans(stem_path, self_introduction,
            progress=lambda _v, m: print(m, flush=True))
        (ROOT / "work/subtitle_stt_probe.json").write_text(
            json.dumps([c.to_dict() for c in text], ensure_ascii=False, indent=2), encoding="utf-8")
        print("STT", [(c.language, c.whisper_text) for c in text], flush=True)


if __name__ == "__main__":
    main()
