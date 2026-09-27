"""T0 shape/alignment smoke test with real audio and untrained temporal head.

No model score from this script is an identity decision.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extractor.audio import load_mono  # noqa: E402
from extractor.speaker import WavLMSpeakerVerifier  # noqa: E402
from probe_temporal_features import encode, pool  # noqa: E402
from temporal_identity_model import OUTPUTS, TemporalIdentityHead  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("--start", type=float, default=303.38)
    parser.add_argument("--end", type=float, default=304.885)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.start < args.end:
        parser.error("Invalid source interval")
    work = args.work_dir.resolve()
    reference_paths = sorted((work / "reference_voice_clips").glob("*.wav"))
    negative_groups = sorted((work / "negative_reference_voice_clips").glob("role_*"))
    if not reference_paths:
        parser.error("No reference clips")
    stem = load_mono(work / "stems/target_vocals.wav", 16000)
    raw = load_mono(work / "target_normalized.wav", 16000)
    start, end = round(args.start * 16000), round(args.end * 16000)
    verifier = WavLMSpeakerVerifier()
    try:
        layer = 9
        stem_frames = encode(verifier, stem[start:end])[layer]
        raw_frames = encode(verifier, raw[start:end])[layer]
        if stem_frames.shape != raw_frames.shape:
            raise ValueError("Raw and UVR frame counts differ; cannot fuse unaligned sequences")
        target_vectors = [
            pool(encode(verifier, load_mono(path, 16000))[layer])
            for path in reference_paths
        ]
        negative_group_vectors = [
            [pool(encode(verifier, load_mono(path, 16000))[layer])
             for path in sorted(group.glob("*.wav"))]
            for group in negative_groups
        ]
        negative_group_vectors = [group for group in negative_group_vectors if group]
        negative_tensor = None
        negative_mask = None
        if negative_group_vectors:
            maximum_references = max(map(len, negative_group_vectors))
            negative_tensor = torch.zeros(
                1, len(negative_group_vectors), maximum_references, stem_frames.shape[-1],
            )
            negative_mask = torch.zeros(
                1, len(negative_group_vectors), maximum_references, dtype=torch.bool,
            )
            for group_index, group in enumerate(negative_group_vectors):
                negative_tensor[0, group_index, :len(group)] = torch.stack(group)
                negative_mask[0, group_index, :len(group)] = True
        torch.manual_seed(0)
        model = TemporalIdentityHead(feature_dim=stem_frames.shape[-1]).eval()
        with torch.inference_mode():
            logits = model(
                stem_frames.unsqueeze(0),
                torch.stack(target_vectors).unsqueeze(0),
                raw_frames=raw_frames.unsqueeze(0),
                negative_references=negative_tensor,
                negative_mask=negative_mask,
            )
        if not torch.isfinite(logits).all():
            raise ValueError("Untrained model produced non-finite logits")
        report = {
            "warning": "Untrained T0 interface check only. Logits are intentionally not exported.",
            "interval": [args.start, args.end],
            "wavlm_layer": layer,
            "stem_frame_count": stem_frames.shape[0],
            "raw_frame_count": raw_frames.shape[0],
            "target_reference_count": len(target_vectors),
            "negative_group_count": len(negative_group_vectors),
            "negative_reference_count": sum(map(len, negative_group_vectors)),
            "output_heads": list(OUTPUTS),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False))
        return 0
    finally:
        verifier.close()


if __name__ == "__main__":
    raise SystemExit(main())
