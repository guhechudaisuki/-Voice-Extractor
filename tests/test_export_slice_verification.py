from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.verify_export_slices import verify_exports
from extractor.audio import write_clip


class ExportSliceVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stem = self.root / "vocals.wav"
        self.output = self.root / "result"
        self.output.mkdir()
        self.audio = self.output / "audio.wav"
        self.manifest = self.output / "manifest.json"
        wave = np.random.default_rng(4).normal(0, 0.05, 44100 * 3)
        sf.write(self.stem, wave, 44100, subtype="PCM_16")
        write_clip(self.stem, self.audio, 0.4, 2.1, sample_rate=16000, normalize_level=True)
        self.row = {"start": 0.4, "end": 2.1, "accepted": True, "audio_file": "audio.wav"}
        self.write_manifest()

    def write_manifest(self):
        self.manifest.write_text(json.dumps({"sentences": [self.row]}), encoding="utf-8")

    def test_resampled_normalized_vocal_slice_passes(self):
        result = verify_exports(self.manifest, self.stem)
        self.assertTrue(result["passed"])
        self.assertEqual(result["verified_count"], 1)
        self.assertGreater(result["clips"][0]["correlation"], 0.999)

    def test_wrong_same_duration_clip_fails_content_check(self):
        write_clip(self.stem, self.audio, 0.8, 2.5, sample_rate=16000)
        result = verify_exports(self.manifest, self.stem)
        self.assertFalse(result["passed"])
        self.assertIn("content", result["clips"][0]["errors"])

    def test_truncated_clip_fails_sample_count(self):
        wave, rate = sf.read(self.audio)
        sf.write(self.audio, wave[:-500], rate)
        result = verify_exports(self.manifest, self.stem)
        self.assertFalse(result["passed"])
        self.assertIn("sample_count", result["clips"][0]["errors"])

    def test_unlisted_old_clip_fails(self):
        (self.output / "stale.wav").write_bytes(self.audio.read_bytes())
        result = verify_exports(self.manifest, self.stem)
        self.assertFalse(result["passed"])
        self.assertEqual(result["unlisted_audio"], ["stale.wav"])

    def test_external_path_is_rejected(self):
        self.row["audio_file"] = "../vocals.wav"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "escapes"):
            verify_exports(self.manifest, self.stem)

    def test_quiet_wrong_tail_fails_even_with_high_whole_clip_correlation(self):
        rng = np.random.default_rng(33)
        common = rng.normal(0, 0.2, 16000)
        source = np.concatenate([common, rng.normal(0, 0.005, 16000)])
        wrong = np.concatenate([common, rng.normal(0, 0.005, 16000)])
        sf.write(self.stem, source, 16000, subtype="PCM_16")
        sf.write(self.audio, wrong, 16000, subtype="PCM_16")
        self.row.update(start=0.0, end=2.0)
        self.write_manifest()
        result = verify_exports(self.manifest, self.stem)
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
