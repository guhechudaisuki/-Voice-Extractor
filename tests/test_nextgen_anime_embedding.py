from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extractor.nextgen.anime_embedding import AnimeSpeakerOnnx, speechbrain_compatible_fbank


ROOT = Path(__file__).resolve().parents[1]


class AnimeEmbeddingTests(unittest.TestCase):
    @staticmethod
    def waveform() -> torch.Tensor:
        times = torch.arange(16000, dtype=torch.float32) / 16000.0
        return 0.3 * torch.sin(2 * torch.pi * 220 * times) + 0.1 * torch.sin(2 * torch.pi * 610 * times)

    def test_filterbank_has_expected_geometry_and_checkpoint_gain_rule(self):
        wave = self.waveform()
        base = speechbrain_compatible_fbank(wave)
        self.assertEqual(tuple(base.shape), (101, 80))
        self.assertTrue(torch.isfinite(base).all())
        # The upstream model only peak-normalizes when the input exceeds 1.0.
        # Quieter clips retain their actual gain in dB; both louder clips do
        # normalize to the same waveform.
        torch.testing.assert_close(
            speechbrain_compatible_fbank(wave * 2) - base,
            torch.full_like(base, 20.0 * np.log10(2.0)), atol=1e-4, rtol=1e-5,
        )
        torch.testing.assert_close(
            speechbrain_compatible_fbank(wave * 4),
            speechbrain_compatible_fbank(wave * 8), atol=1e-4, rtol=1e-5,
        )

    def test_invalid_waveform_and_rate_fail_closed(self):
        with self.assertRaises(ValueError):
            speechbrain_compatible_fbank(torch.zeros(100))
        with self.assertRaises(ValueError):
            speechbrain_compatible_fbank(torch.zeros(3200))
        with self.assertRaises(ValueError):
            speechbrain_compatible_fbank(torch.full((3200,), float("nan")))
        path = ROOT / "models/anime-speaker-char/anime_speaker_char_ecapa.onnx"
        if path.is_file():
            with self.assertRaises(ValueError):
                AnimeSpeakerOnnx(ROOT / "models", "char").encode(self.waveform(), sample_rate=48000)

    def test_local_onnx_models_produce_unit_embeddings(self):
        if not all((ROOT / "models" / part).is_file() for part in (
            "anime-speaker-char/anime_speaker_char_ecapa.onnx",
            "anime-speaker-va/anime_speaker_va_ecapa.onnx",
        )):
            self.skipTest("Local anime speaker model assets are absent")
        for variant in ("char", "va"):
            with self.subTest(variant=variant):
                vector = AnimeSpeakerOnnx(ROOT / "models", variant).encode(
                    self.waveform(), sample_rate=16000,
                )
                self.assertEqual(vector.shape, (192,))
                self.assertTrue(np.isfinite(vector).all())
                self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
