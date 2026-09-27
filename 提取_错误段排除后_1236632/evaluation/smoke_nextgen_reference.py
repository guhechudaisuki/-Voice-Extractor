"""Check one existing synchronized reference pair without identity inference."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extractor.audio import normalize_audio  # noqa: E402
from extractor.nextgen.features import WavLMSpeakerFeatures  # noqa: E402
from extractor.nextgen.prepared_audio import PairedAudio  # noqa: E402
from extractor.nextgen.prepare_media import _alignment_anchors, _samples, _verify_alignment  # noqa: E402
from extractor.nextgen.reference_preparation import (ReferenceMaterial, prepare_reference_bank)  # noqa: E402
from extractor.nextgen.scene_adapter import make_scene  # noqa: E402
from extractor.transcription import FunASRTools  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("stem", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    raw = normalize_audio(args.raw.resolve(strict=True), destination / "raw.wav",
                          sample_rate=16000, stereo=False)
    stem = normalize_audio(args.stem.resolve(strict=True), destination / "stem.wav",
                           sample_rate=16000, stereo=False)
    detected = FunASRTools("cuda").vad_many([raw, stem])
    total = sf.info(raw).frames
    raw_voice, stem_voice = _samples(detected[raw], total), _samples(detected[stem], total)
    anchors = _alignment_anchors(raw_voice, stem_voice, (), total)
    report = _verify_alignment(raw, stem, anchors)
    scene = make_scene(PairedAudio(raw, stem, report), raw_voice=raw_voice,
                       stem_voice=stem_voice)
    encoder = WavLMSpeakerFeatures(ROOT / "model/speaker/wavlm-base-plus-sv",
                                   device="cuda" if torch.cuda.is_available() else "cpu")
    bank, encoded = prepare_reference_bank((ReferenceMaterial("target", scene),), encoder)
    print(f"Verified delay={report.delay_samples} samples; "
          f"reference clips={len(bank.entries)}; "
          f"valid TDNN tokens={sum(int(item.valid.sum()) for item in encoded)}")


if __name__ == "__main__":
    main()
