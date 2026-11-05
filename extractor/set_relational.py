"""Universal permutation-invariant query/reference/cohort relations.

Only the shared acoustic projection is learned. No identity slot, role key or
reference tensor belongs in a checkpoint. Temperature is a separate monotonic
map, fitted across source-disjoint training material.
"""
import torch
import torch.nn.functional as F


class SetRelationalScorer(torch.nn.Module):
    def forward(self, queries, references, competitors=None):
        q, r = F.normalize(queries, dim=-1), F.normalize(references, dim=-1)
        if r.ndim != 2 or not len(r):
            raise ValueError('A nonempty reference set is required')
        similarities = q @ r.T
        local = similarities.topk(min(2, len(r)), dim=-1).values.mean(-1)
        # Reference dispersion determines how far a set centroid can be trusted.
        center = r.mean(0)
        coherence = center.norm().clamp(0., 1.)
        centroid_score = q @ F.normalize(center, dim=-1)
        attention = (8.*similarities).softmax(dim=-1)
        attended = attention @ r
        attended_score = (q*F.normalize(attended,dim=-1)).sum(-1)
        score = local + .1*coherence*(centroid_score-local) + .1*(attended_score-local)
        if competitors is not None and len(competitors):
            rival = (q @ F.normalize(competitors, dim=-1).T).max(-1).values
            score = score - .15 * F.relu(rival - local)
        return score


def calibrated_logits(scores, calibration=None):
    calibration = calibration or {'scale': 20., 'bias': -16.}
    scale, bias = float(calibration['scale']), float(calibration['bias'])
    import math
    if not math.isfinite(scale) or scale <= 0 or not math.isfinite(bias):
        raise ValueError('Global calibration must be finite and strictly monotonic')
    return scores * scale + bias
