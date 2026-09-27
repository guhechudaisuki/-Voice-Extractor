from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extractor.nextgen.domain_identity import (
    DomainReferenceBank, ModelViewScore, ReferenceAudio, classify, review_join,
)


def score(variant: str, channel: str, margin: float, role: str = "other") -> ModelViewScore:
    return ModelViewScore(variant, channel, margin, 0.0, role)


class FakeEncoder:
    def encode(self, waveform: torch.Tensor, *, sample_rate: int):
        import numpy as np

        assert sample_rate == 16000
        value = float(waveform[0])
        return np.asarray([value, 1.0 - value], dtype=np.float32)


class DomainIdentityTests(unittest.TestCase):
    def test_two_correlated_channels_alone_do_not_authorize_join(self):
        weak = classify((score("char", "stem", 0.2), score("char", "raw", -0.1),
                         score("va", "stem", -0.1), score("va", "raw", 0.2)))
        # Two positive observations with a stem witness are a proposal, not
        # proof of overall purity. A confirmed change still vetoes the join.
        self.assertEqual(weak.state, "target_supported")
        self.assertFalse(review_join(weak, weak, whole_verified=True,
                                     confirmed_change=True, gap_allowed=True).allowed)
        self.assertFalse(review_join(weak, weak, whole_verified=False,
                                     confirmed_change=False, gap_allowed=True).allowed)

    def test_unresolved_side_and_missing_exclusion_never_authorize_join(self):
        target = classify((score("char", "stem", 0.2), score("char", "raw", 0.1),
                           score("va", "stem", -0.1), score("va", "raw", -0.1)))
        unknown = classify(tuple(ModelViewScore(v, c, 0.9, None, None)
                                 for v in ("char", "va") for c in ("stem", "raw")))
        self.assertEqual(unknown.state, "unresolved")
        self.assertFalse(review_join(target, unknown, whole_verified=True,
                                     confirmed_change=False, gap_allowed=True).allowed)

    def test_agreed_other_role_needs_two_stem_models_and_raw(self):
        other = classify((score("char", "stem", -0.2, "role_a"),
                          score("char", "raw", -0.1, "role_a"),
                          score("va", "stem", -0.3, "role_a"),
                          score("va", "raw", 0.2, "role_b")))
        self.assertEqual(other.state, "other_supported")
        disagreement = classify((score("char", "stem", -0.2, "role_a"),
                                 score("char", "raw", -0.1, "role_a"),
                                 score("va", "stem", -0.3, "role_b"),
                                 score("va", "raw", -0.2, "role_b")))
        self.assertEqual(disagreement.state, "unresolved")

    def test_duplicate_conflicting_role_rejected(self):
        audio = torch.ones(3200)
        encoder = FakeEncoder()
        with self.assertRaises(ValueError):
            DomainReferenceBank({"char": encoder, "va": encoder}, (
                ReferenceAudio("target", audio, audio, 16000),
                ReferenceAudio("other", audio, audio, 16000),
            ))

    def test_reference_crop_tolerates_small_existing_uvr_end_difference(self):
        ReferenceAudio("target", torch.ones(3200), torch.ones(3200 + 560), 16000)
        with self.assertRaises(ValueError):
            ReferenceAudio("target", torch.ones(3200), torch.ones(3200 + 960), 16000)

    def test_sample_rate_must_be_explicit_and_correct(self):
        with self.assertRaises(ValueError):
            ReferenceAudio("target", torch.ones(3200), torch.ones(3200), 44100)
        audio = torch.ones(3200)
        encoder = FakeEncoder()
        bank = DomainReferenceBank({"char": encoder, "va": encoder}, (
            ReferenceAudio("target", audio, audio, 16000),
        ))
        with self.assertRaises(ValueError):
            bank.score(stem=audio, raw=audio, sample_rate=44100)


if __name__ == "__main__":
    unittest.main()
