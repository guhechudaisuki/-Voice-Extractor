from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))

from temporal_identity_model import OUTPUTS, TemporalIdentityHead, temporal_loss  # noqa: E402


class TemporalIdentityModelTests(unittest.TestCase):
    def test_optional_negative_groups_and_variable_lengths(self):
        torch.manual_seed(7)
        model = TemporalIdentityHead(feature_dim=12, projection_dim=8, hidden_dim=6)
        stem = torch.randn(2, 7, 12)
        target = torch.randn(2, 3, 12)
        mask = torch.tensor([[True, True, False], [True, False, False]])
        lengths = torch.tensor([7, 4])
        plain = model(stem, target, target_mask=mask, lengths=lengths)
        self.assertEqual(tuple(plain.shape), (2, 7, len(OUTPUTS)))
        negatives = torch.randn(2, 2, 2, 12)
        negative_mask = torch.tensor([
            [[True, False], [False, False]],
            [[False, False], [False, False]],
        ])
        with_negatives = model(
            stem, target, raw_frames=stem + 0.1,
            target_mask=mask,
            negative_references=negatives,
            negative_mask=negative_mask,
            lengths=lengths,
        )
        self.assertEqual(tuple(with_negatives.shape), (2, 7, len(OUTPUTS)))
        self.assertTrue(torch.isfinite(with_negatives).all())
        with self.assertRaises(ValueError):
            model(stem, target, target_mask=torch.zeros_like(mask))
        with self.assertRaises(ValueError):
            model(stem, target, target_mask=mask.float())
        with self.assertRaises(ValueError):
            model(stem, target, raw_frames=stem[:, :-1])

    def test_reference_order_and_masked_padding_do_not_change_logits(self):
        torch.manual_seed(19)
        model = TemporalIdentityHead(feature_dim=12, projection_dim=8, hidden_dim=6).eval()
        query = torch.randn(1, 8, 12)
        references = torch.randn(1, 3, 12)
        negatives = torch.randn(1, 2, 2, 12)
        negative_mask = torch.tensor([[[True, False], [True, True]]])
        with torch.no_grad():
            baseline = model(
                query, references, negative_references=negatives,
                negative_mask=negative_mask,
            )
            reordered = model(
                query, references[:, [2, 0, 1]],
                negative_references=negatives[:, [1, 0]],
                negative_mask=negative_mask[:, [1, 0]],
            )
            padded = model(
                query,
                torch.cat((references, torch.randn(1, 1, 12) * 100), dim=1),
                target_mask=torch.tensor([[True, True, True, False]]),
                negative_references=negatives,
                negative_mask=negative_mask,
            )
        self.assertTrue(torch.allclose(baseline, reordered, atol=1e-6))
        self.assertTrue(torch.allclose(baseline, padded, atol=1e-6))

    def test_unknown_and_padded_frames_cannot_affect_loss(self):
        logits = torch.zeros(1, 3, len(OUTPUTS))
        labels = torch.tensor([[[1.0, 1.0, 1.0, -1.0], [0.0, 1.0, 1.0, 0.0], [-1.0] * 4]])
        baseline = temporal_loss(logits, labels, torch.tensor([2]))
        altered = logits.clone()
        altered[0, 0, 3] = 100.0
        altered[0, 2] = 100.0
        self.assertAlmostEqual(float(baseline), float(temporal_loss(altered, labels, torch.tensor([2]))), places=6)
        invalid = labels.clone()
        invalid[0, 0, 0] = 2.0
        with self.assertRaises(ValueError):
            temporal_loss(logits, invalid, torch.tensor([2]))

    def test_single_synthetic_batch_can_overfit_multilabel_timeline(self):
        torch.manual_seed(11)
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            dimension = 12
            target_voice = torch.zeros(dimension)
            other_voice = torch.zeros(dimension)
            target_voice[0] = 1.0
            other_voice[1] = 1.0
            stem = torch.stack(
                [target_voice] * 7
                + [other_voice] * 7
                + [target_voice + other_voice] * 5
                + [torch.zeros(dimension)] * 5
            ).unsqueeze(0)
            stem += torch.randn_like(stem) * 0.02
            target_refs = torch.stack([target_voice, target_voice + 0.02]).unsqueeze(0)
            negatives = other_voice.reshape(1, 1, 1, dimension)
            labels = torch.zeros(1, 24, len(OUTPUTS))
            labels[0, :7, 0] = 1
            labels[0, 14:19, 0] = 1
            labels[0, 7:19, 1] = 1
            labels[0, :19, 2] = 1
            labels[0, [7, 14, 19], 3] = 1
            model = TemporalIdentityHead(dimension, 12, 8)
            optimiser = torch.optim.Adam(model.parameters(), lr=0.02)
            for _ in range(220):
                logits = model(
                    stem, target_refs,
                    raw_frames=stem,
                    negative_references=negatives,
                )
                loss = temporal_loss(logits, labels, torch.tensor([24]))
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            self.assertLess(float(loss), 0.08)
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
