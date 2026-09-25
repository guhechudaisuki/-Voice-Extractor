from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from train_temporal_t1 import auc, operating_point  # noqa: E402


class T1TrainingTests(unittest.TestCase):
    def test_auc_respects_perfect_reverse_and_tied_scores(self):
        labels = np.array([0, 0, 1, 1])
        self.assertEqual(auc(labels, np.array([0.1, 0.2, 0.8, 0.9])), 1.0)
        self.assertEqual(auc(labels, np.array([0.9, 0.8, 0.2, 0.1])), 0.0)
        self.assertEqual(auc(labels, np.array([0.5, 0.5, 0.5, 0.5])), 0.5)
        self.assertIsNone(auc(np.zeros(4), np.arange(4)))

    def test_operating_point_uses_train_negatives_not_validation_threshold(self):
        calibration = (np.array([0, 0, 1]), np.array([0.1, 0.2, 0.9]))
        evaluation = (np.array([0, 0, 1, 1]), np.array([0.15, 0.8, 0.7, 0.95]))
        point = operating_point(calibration, evaluation, negative_quantile=1.0)
        self.assertEqual(point["threshold"], 0.2)
        self.assertEqual(point["target_frame_recall"], 1.0)
        self.assertEqual(point["other_or_silent_frame_false_positive_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
