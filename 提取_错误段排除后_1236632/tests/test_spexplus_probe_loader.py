"""The SpEx+ probe must reproduce its training data channel convention."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.probe_spexplus_local import load_audio  # noqa: E402


class SpExPlusProbeLoaderTests(unittest.TestCase):
    def test_stereo_uses_first_channel_not_average(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.wav"
            channels = np.tile(np.array([[0.5, -0.5]], dtype=np.float32), (800, 1))
            sf.write(path, channels, 8000)
            loaded = load_audio(path)
        self.assertEqual(loaded.numel(), 800)
        self.assertAlmostEqual(float(loaded.mean()), 0.5, places=3)


if __name__ == "__main__":
    unittest.main()
