"""Check one bounded real raw/UVR pair without running a full episode.

Timing and feature geometry only. This script cannot produce a speaker verdict.
Temporary 16 kHz WAVs live under the chosen work directory and are removed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extractor.config import FFMPEG
from extractor.nextgen.acoustic_gaps import find_confirmed_gaps
from extractor.nextgen.features import WavLMSpeakerFeatures
from extractor.nextgen.ledger import Candidate
from extractor.nextgen.prepared_audio import PairedAudio, estimate_stem_delay
from extractor.nextgen.timeline import SampleSpan


def crop(source: Path, target: Path, start: float, length: float) -> None:
    subprocess.run([
        str(FFMPEG), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", str(source), "-ss", str(start), "-t", str(length),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s24le", str(target),
    ], check=True, timeout=120)


def smoke(raw: Path, stem: Path, *, start: float, length: float,
          anchors: tuple[tuple[float, float], ...], work: Path, encoder: Path,
          inspect_vad: bool = False) -> dict:
    if not 0 <= start or length <= 0 or len(anchors) < 2:
        raise ValueError("Choose a bounded range and two speech anchors")
    work = work.resolve(strict=True)
    with TemporaryDirectory(prefix="nextgen_pair_", dir=work) as temporary:
        directory = Path(temporary)
        prepared_raw, prepared_stem = directory / "raw.wav", directory / "stem.wav"
        crop(raw, prepared_raw, start, length)
        crop(stem, prepared_stem, start, length)
        spans = tuple(SampleSpan(round(left * 16000), round(right * 16000))
                      for left, right in anchors)
        report = estimate_stem_delay(prepared_raw, prepared_stem, spans)
        paired = PairedAudio(prepared_raw, prepared_stem, report)
        context = SampleSpan(spans[0].start, spans[-1].end)
        speaker_encoder = WavLMSpeakerFeatures(encoder)
        raw_features, stem_features = paired.encode(context, speaker_encoder)
        result = {
            "purpose": "bounded_alignment_and_feature_geometry_only",
            "crop_start_seconds": start,
            "crop_length_seconds": length,
            "measured_delay_samples_at_16khz": report.delay_samples,
            "anchor_correlations": report.correlations,
            "feature_shape": list(raw_features.values.shape),
            "paired_cells_equal": raw_features.cells == stem_features.cells,
            "first_source_cell": [raw_features.cells[0].start, raw_features.cells[0].end],
            "last_source_cell": [raw_features.cells[-1].start, raw_features.cells[-1].end],
        }
        if inspect_vad:
            from extractor.transcription import FunASRTools

            detected = FunASRTools(device="cpu").vad_many([prepared_raw, prepared_stem])

            def convert(path: Path) -> tuple[SampleSpan, ...]:
                return tuple(SampleSpan(round(row.start * 16000), round(row.end * 16000))
                             for row in detected[path] if row.end > row.start)

            raw_voice, stem_voice = convert(prepared_raw), convert(prepared_stem)
            output = SampleSpan(0, paired.source.total_samples)
            parent = Candidate(paired.source.source_sha256, 16000, output, output,
                               (output,), "bounded_vad_diagnostic")
            gaps = find_confirmed_gaps(parent, raw_voice, stem_voice, paired,
                                       minimum_gap_samples=800)
            gap_levels = []
            for left, right in zip(raw_voice, raw_voice[1:]):
                if right.start <= left.end or left.end < 1600 or right.start + 1600 > output.end:
                    continue
                before = SampleSpan(left.end - 1600, left.end)
                gap = SampleSpan(left.end, right.start)
                after = SampleSpan(right.start, right.start + 1600)
                levels = []
                for interval in (before, gap, after):
                    pair = paired.read_pair(interval)
                    levels.append([float(np.sqrt(np.mean(np.square(channel.astype(np.float64)))))
                                   for channel in pair])
                gap_levels.append({"gap": [gap.start, gap.end], "raw_stem_rms_before_gap_after": levels})
            result.update({
                "purpose": "bounded_alignment_feature_and_vad_gap_diagnostic_only",
                "raw_voice": [[row.start, row.end] for row in raw_voice],
                "stem_voice": [[row.start, row.end] for row in stem_voice],
                "confirmed_gaps": [[row.span.start, row.span.end] for row in gaps],
                "raw_vad_gap_levels": gap_levels,
            })
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--stem", required=True, type=Path)
    parser.add_argument("--start", required=True, type=float)
    parser.add_argument("--length", required=True, type=float)
    parser.add_argument("--anchor", nargs=2, type=float, action="append", required=True,
                        metavar=("START", "END"), help="seconds relative to the bounded crop")
    parser.add_argument("--work", type=Path, default=ROOT / "work")
    parser.add_argument("--encoder", type=Path, default=ROOT / "model/speaker/wavlm-base-plus-sv")
    parser.add_argument("--inspect-vad", action="store_true",
                        help="Run the existing local VAD on this short crop")
    args = parser.parse_args()
    print(json.dumps(smoke(args.raw, args.stem, start=args.start, length=args.length,
                           anchors=tuple(map(tuple, args.anchor)), work=args.work,
                           encoder=args.encoder, inspect_vad=args.inspect_vad), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
