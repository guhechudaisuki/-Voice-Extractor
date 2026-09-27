from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from librispeech_t1 import (  # noqa: E402
    assemble_scene, frame_labels, plan_scene, split_speakers,
)


class LibriSpeechT1Tests(unittest.TestCase):
    def test_speaker_split_is_disjoint_and_repeatable(self):
        speakers = {str(i): [Path(f"{i}/{j}.flac") for j in range(8)] for i in range(12)}
        train, validation = split_speakers(speakers, seed=7)
        self.assertFalse(set(train) & set(validation))
        self.assertEqual(set(train) | set(validation), set(speakers))
        self.assertEqual((train, validation), split_speakers(speakers, seed=7))

    def test_reference_is_never_a_query_or_another_speaker(self):
        speakers = {
            str(i): [Path(f"{i}/chapter_{j % 2}/{j:03d}.flac") for j in range(8)]
            for i in range(12)
        }
        pool = sorted(speakers)
        rng = random.Random(9)
        for index in range(60):
            scene = plan_scene(speakers, pool, index, rng)
            queries = set(scene["query_paths"])
            target_refs = set(scene["target_reference_paths"])
            negative_refs = {
                path for group in scene["negative_reference_groups"]
                for path in group["paths"]
            }
            self.assertFalse(queries & target_refs)
            self.assertFalse(queries & negative_refs)
            self.assertEqual(len(target_refs), min(1 + index % 7, 8 - scene["pattern"].count("target")))
            self.assertEqual(len(scene["negative_reference_groups"]), index % 6)
            self.assertTrue(all(Path(path).parts[0] == scene["target_speaker"] for path in target_refs))
            self.assertTrue(all(
                Path(path).parts[0] == group["speaker"]
                for group in scene["negative_reference_groups"] for path in group["paths"]
            ))
            self.assertTrue(all(group["speaker"] != scene["target_speaker"]
                                for group in scene["negative_reference_groups"]))

    def test_full_recordings_are_joined_without_cropping_and_edges_are_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "a.wav"
            second = Path(directory) / "b.wav"
            sf.write(first, np.ones(16000, dtype=np.float32) * 0.1, 16000)
            sf.write(second, np.ones(16000, dtype=np.float32) * 0.2, 16000)
            waveform, intervals = assemble_scene({
                "query_paths": [str(first), str(second)],
                "pattern": ["target", "other"],
                "gap_seconds": [0.4],
            })
            self.assertEqual(waveform.numel(), 38400)
            self.assertEqual(intervals, [(0, 16000, "target"), (22400, 38400, "other")])
            labels = frame_labels(120, intervals)
            self.assertEqual(labels.shape, (120, 4))
            self.assertEqual(labels[25].tolist(), [1.0, 0.0, 1.0, 0.0])
            self.assertEqual(labels[60].tolist(), [0.0, 0.0, 0.0, 0.0])
            self.assertEqual(labels[70, :3].tolist(), [-1.0, -1.0, -1.0])
            self.assertEqual(float(labels[70, 3]), 1.0)

    def test_intra_utterance_silence_is_not_labeled_as_a_speaker(self):
        import torch

        waveform = torch.ones(16000) * 0.1
        waveform[7000:9000] = 0
        labels = frame_labels(50, [(0, 16000, "target")], waveform=waveform)
        self.assertEqual(labels[10, :3].tolist(), [1.0, 0.0, 1.0])
        self.assertEqual(labels[25, :3].tolist(), [-1.0, -1.0, -1.0])


if __name__ == "__main__":
    unittest.main()
