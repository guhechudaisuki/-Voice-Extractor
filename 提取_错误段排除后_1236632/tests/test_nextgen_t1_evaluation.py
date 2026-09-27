from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.evaluate_nextgen_t1 import _frozen_reference_scores


class FrozenReferenceControlTests(unittest.TestCase):
    def test_target_and_exclusion_baselines_use_same_query_tokens(self):
        query = torch.tensor([[1., 0.], [0., 1.]])
        item = {
            "raw": query, "stem": query.clone(),
            "target.raw": query[:1], "target.stem": query[:1].clone(),
            "target.valid": torch.tensor([True]),
            "target.quality": torch.tensor([1.]),
        }
        target, no_exclusion = _frozen_reference_scores(item)
        torch.testing.assert_close(target, torch.tensor([1., 0.]))
        torch.testing.assert_close(no_exclusion, target)
        item.update({
            "excluded.000.raw": query[1:], "excluded.000.stem": query[1:].clone(),
            "excluded.000.valid": torch.tensor([True]),
            "excluded.000.quality": torch.tensor([1.]),
        })
        target, margin = _frozen_reference_scores(item)
        torch.testing.assert_close(target, torch.tensor([1., 0.]))
        torch.testing.assert_close(margin, torch.tensor([1., -1.]))


if __name__ == "__main__":
    unittest.main()
