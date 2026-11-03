"""Estimate exclusion evidence reliability from paired reference observations."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _rows(values):
    rows = np.asarray(values, dtype=np.float64)
    if rows.ndim != 2 or not len(rows) or not np.isfinite(rows).all():
        raise ValueError("Reference views must be finite non-empty matrices")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(float).eps):
        raise ValueError("Reference views must contain non-zero vectors")
    return rows / norms


def _loo_affinity(rows):
    if len(rows) == 1:
        return np.ones(1)
    peers = rows.sum(axis=0) - rows
    norms = np.linalg.norm(peers, axis=1)
    return np.divide(np.einsum("ij,ij->i", rows, peers), norms,
                     out=np.zeros(len(rows)), where=norms > np.finfo(float).eps)


def audit_exclusion_views(target_views: Mapping, exclusion_views: Mapping) -> dict:
    """Measure within-group coherence and target overlap separately in each view.

    A singleton supplies no internal purity evidence and retains its prior weight.
    The returned weights measure geometric reliability, not probability.
    """
    audit = {"views": {}}
    for channel, values in exclusion_views.items():
        if channel not in target_views:
            continue
        target, excluded = _rows(target_views[channel]), _rows(values)
        if target.shape[1] != excluded.shape[1]:
            raise ValueError("Target and exclusion dimensions differ")
        own = _loo_affinity(excluded)
        reference = _loo_affinity(target)
        target_center = target.mean(axis=0)
        target_center /= max(float(np.linalg.norm(target_center)), np.finfo(float).eps)
        cross = excluded @ target_center
        if len(target) > 1:
            nearest = np.sort(excluded @ target.T, axis=1)[:, -2:].mean(axis=1)
            cross = np.maximum(cross, nearest)
        if len(excluded) > 1:
            variance = float(np.var(own) + np.var(reference))
            # Reference scatter sets the separation scale.
            scale = max(np.sqrt(variance), np.finfo(float).eps)
            separation = (own - cross) / scale
            purity = float(np.mean(0.5 + 0.5 * np.tanh(separation / 2.0)))
            reference_coherence = max(float(np.mean(reference)), np.finfo(float).eps)
            coherence = float(np.clip(float(np.mean(own)) / reference_coherence, 0., 1.))
            weight = coherence * purity
        else:
            purity, coherence, weight = 1., 1., 1.
        audit[channel] = weight
        audit["views"][channel] = {
            "count": len(excluded),
            "coherence": coherence,
            "separation": float(np.mean(own - cross)),
            "target_overlap": float(np.mean(cross)),
            "purity": purity,
            "weight": weight,
            "internal_evidence": len(excluded) > 1,
        }
    return audit


def margin_consensus(gaps: Mapping[str, float], audit: Mapping | None,
                     required_margin: float) -> dict:
    """Pool reliability-weighted margins without erasing corroborated conflicts.

    Discounting a losing view needs two other supporting views. Two views alone
    therefore cannot turn an ambiguous exclusion match into an identity anchor.
    """
    values = {name: float(value) for name, value in gaps.items()
              if value is not None and np.isfinite(value)}
    weights = {name: float(np.clip((audit or {}).get(name, 1.), 0., 1.))
               for name in values}
    total = sum(weights.values())
    pooled = (sum(weights[name] * value for name, value in values.items()) / total
              if total > np.finfo(float).eps else 0.)
    supporting = [name for name, value in values.items() if value >= required_margin]
    opposing = [name for name, value in values.items() if value < required_margin]
    conflict = sum(value < 0. for value in values.values()) > 1
    discounted = bool(opposing and supporting and all(
        weights[name] < min(weights[other] for other in supporting)
        for name in opposing))
    accepted = (bool(values) and not conflict and pooled >= required_margin
                and (not opposing or (len(supporting) >= 2 and discounted)))
    return {"accepted": accepted, "margin": pooled, "weights": weights,
            "supporting_views": supporting, "discounted": discounted,
            "corroborated_conflict": conflict}
