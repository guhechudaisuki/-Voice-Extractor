"""Interval evidence must remain aligned and report model caveats, not decisions."""
from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.full_model_evidence import (  # noqa: E402
    PersonalVADCoverage, SpExPlusEvaluator, WaveformComparison,
)


class DoubleFirstHead:
    def __init__(self):
        self.calls = []

    def __call__(self, source, reference, reference_length):
        assert source.shape[0] == 1
        assert reference_length[0] == reference.shape[1]
        self.calls.append(source.shape[1])
        return (source * 2, source, source)


class FullModelEvidenceTests(unittest.TestCase):
    def test_pvad_coverage_and_source_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            report = root / "report.json"
            sf.write(source, np.zeros(16000 * 3, dtype=np.float32), 16000)
            payload = {"episode": {
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "duration_seconds": 3.0,
                "target_fsm_segments": [[0.5, 1.0], [1.5, 2.5]],
            }}
            report.write_text(json.dumps(payload), encoding="utf-8")
            coverage = PersonalVADCoverage.from_report(report, source)
            result = coverage.evidence(0.75, 2.0)
            self.assertAlmostEqual(result["target_overlap_seconds"], 0.75)
            self.assertAlmostEqual(result["target_coverage_fraction"], 0.6)
            self.assertIsNone(result["decision"])
            payload["episode"]["source_sha256"] = "0" * 64
            report.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                PersonalVADCoverage(report, source)

    def test_voicefilter_exact_samples_and_normalization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            output = root / "output.wav"
            report = root / "report.json"
            # Eight samples per second: the selected second is source=0.25,
            # output=0.5. Other seconds deliberately have different gains.
            x = np.r_[np.ones(8) * 0.125, np.ones(8) * 0.25,
                          np.ones(8) * 0.5]
            y = np.r_[np.ones(8) * 0.125, np.ones(8) * 0.5,
                          np.ones(8) * 0.5]
            sf.write(source, x, 8, subtype="FLOAT")
            sf.write(output, y, 8, subtype="FLOAT")
            report.write_text(json.dumps({
                "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "duration_seconds": 3, "sample_rate": 8,
                "source_peak": 0.5, "playback_global_gain": 0.25,
            }), encoding="utf-8")
            reader = WaveformComparison(source, output, report)
            got = reader.evidence(1.0, 2.0)
            self.assertEqual(got["sample_count"], 8)
            self.assertAlmostEqual(got["output_over_input_db_playback"],
                                   20 * math.log10(2), places=5)
            self.assertAlmostEqual(got["output_over_input_db_model_estimate"],
                                   20 * math.log10(4), places=5)
            self.assertAlmostEqual(got["waveform_cosine"], 1.0)
            with self.assertRaises(ValueError):
                reader.evidence(0, 3.01)
            sf.write(output, y, 16000, subtype="FLOAT")
            with self.assertRaisesRegex(ValueError, "time-aligned"):
                WaveformComparison(source, output)

    def test_voicefilter_resampled_window_retains_whole_file_phase(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            output = root / "output.wav"
            time_axis = np.arange(44100 * 3) / 44100
            first = (0.2 * np.sin(2 * np.pi * 257 * time_axis)).astype(np.float32)
            sf.write(source, np.stack([first, -first], axis=1), 44100, subtype="FLOAT")
            converted = torchaudio.functional.resample(
                torch.from_numpy(first), 44100, 16000,
            ).numpy()
            sf.write(output, converted, 16000, subtype="FLOAT")
            reader = WaveformComparison(source, output)
            got = reader.evidence(1.13, 1.67)
            self.assertEqual(got["sample_count"], round(1.67 * 16000) - round(1.13 * 16000))
            self.assertEqual(got["source_sample_rate"], 44100)
            self.assertGreater(got["waveform_cosine"], 0.99999)
            self.assertAlmostEqual(got["output_over_input_db_playback"], 0.0, places=2)

    def test_spex_exact_interval_respects_first_channel_and_padded_length(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.wav"
            stereo = np.tile(np.array([[0.25, -0.25]], dtype=np.float32), (16000, 1))
            sf.write(source, stereo, 8000, subtype="FLOAT")
            evaluator = SpExPlusEvaluator(
                source, DoubleFirstHead(), torch.ones(1200), torch.device("cpu")
            )
            result = evaluator.evidence(1.0, 1.05)  # 400 samples, padded for model
            self.assertEqual(result["sample_count"], 400)
            self.assertAlmostEqual(result["output_over_input_db"], 20 * math.log10(2))
            self.assertAlmostEqual(result["waveform_cosine"], 1.0)
            self.assertEqual(result["analysis_mode"], "independent_exact_candidate_interval")
            self.assertEqual(result["chunk_seconds"], 4.0)
            self.assertEqual(result["chunk_count"], 1)
            self.assertIsNone(result["decision"])
            with self.assertRaises(ValueError):
                evaluator.evidence(2.0, 2.1)
            wrong_rate = Path(temporary) / "wrong_rate.wav"
            sf.write(wrong_rate, np.zeros(16000), 16000)
            with self.assertRaisesRegex(ValueError, "8 kHz"):
                SpExPlusEvaluator(wrong_rate, DoubleFirstHead(), torch.ones(1200),
                                  torch.device("cpu"))

    def test_spex_long_candidate_is_bounded_and_aggregated(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.wav"
            sf.write(source, np.full(11 * 8000, 0.125, dtype=np.float32),
                     8000, subtype="FLOAT")
            model = DoubleFirstHead()
            evaluator = SpExPlusEvaluator(source, model, torch.ones(1200),
                                          torch.device("cpu"))
            result = evaluator.evidence(0.1, 10.6)
            self.assertEqual(model.calls, [32000, 32000, 20000])
            self.assertEqual(result["chunk_count"], 3)
            self.assertEqual(result["sample_count"], 84000)
            self.assertAlmostEqual(result["output_over_input_db"], 20 * math.log10(2))
            self.assertAlmostEqual(result["waveform_cosine"], 1.0)


if __name__ == "__main__":
    unittest.main()
