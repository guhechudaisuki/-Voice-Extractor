from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extractor.nextgen.artifacts import model_from_metadata
from extractor.nextgen.identity_model import ReferenceFeatures, TemporalIdentityModel


class IdentityAnchorTests(unittest.TestCase):
    def test_uninformative_learned_head_preserves_direct_reference_order(self):
        model = TemporalIdentityModel(feature_dim=2, projection_dim=2, hidden_dim=2)
        for parameter in model.parameters():
            parameter.data.zero_()
        query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        reference = ReferenceFeatures(query[:1], query[:1],
                                      torch.tensor([True]), torch.tensor([1.0]))
        output = model(query, query, reference, (), torch.tensor([True, True]))
        self.assertGreater(float(output.frames[0, 0]), float(output.frames[1, 0]))

    def test_legacy_checkpoint_metadata_retains_v1_forward_behavior(self):
        legacy = TemporalIdentityModel(feature_dim=2, projection_dim=2,
                                       hidden_dim=2, architecture_version=1)
        metadata = {"card": {"architecture": legacy.architecture},
                    "config": {"feature_dim": 2, "projection_dim": 2, "hidden_dim": 2}}
        restored = model_from_metadata(metadata)
        restored.load_state_dict(legacy.state_dict(), strict=True)
        query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        reference = ReferenceFeatures(query[:1], query[:1],
                                      torch.tensor([True]), torch.tensor([1.0]))
        expected = legacy(query, query, reference, (), torch.tensor([True, True]))
        actual = restored(query, query, reference, (), torch.tensor([True, True]))
        self.assertEqual(restored.architecture_version, 1)
        torch.testing.assert_close(actual.frames, expected.frames)
        with self.assertRaisesRegex(ValueError, "disagree"):
            model_from_metadata({**metadata, "config": {**metadata["config"],
                                                      "architecture_version": 2}})


if __name__ == "__main__":
    unittest.main()
