"""Source-disjoint holdout selection and non-regression checkpoint selection."""
import hashlib
import math


def holdout_sources(groups, labels, verified):
    sources = sorted(set(groups), key=lambda value: hashlib.sha256(value.encode()).hexdigest())
    if not verified or len(sources) < 3 or set(labels) != {0, 1}:
        return set()
    # Deterministic rotations find a stratified split without splitting a source.
    size = max(1, len(sources) // 5)
    for width in range(size, max(size + 1, len(sources) // 2 + 1)):
        for offset in range(len(sources)):
            held = {sources[(offset + i) % len(sources)] for i in range(width)}
            if ({label for group, label in zip(groups, labels) if group in held} == {0, 1}
                    and {label for group, label in zip(groups, labels) if group not in held} == {0, 1}):
                return held
    return set()


def classification_metrics(scores, labels):
    import numpy as np
    scores, labels = np.asarray(scores), np.asarray(labels, dtype=bool)
    predicted = scores >= .8
    tp, fp = int((predicted & labels).sum()), int((predicted & ~labels).sum())
    fn, tn = int((~predicted & labels).sum()), int((~predicted & ~labels).sum())
    # Rank-based AUC, including ties, without a quadratic pairwise matrix.
    order = np.argsort(scores, kind="stable")
    _, first, counts = np.unique(scores[order], return_index=True, return_counts=True)
    ranks = np.repeat(first + (counts + 1) / 2., counts)
    positives, negatives = int(labels.sum()), int((~labels).sum())
    auc = ((ranks[labels[order]].sum() - positives * (positives + 1) / 2.)
           / (positives * negatives)) if positives and negatives else None
    return {"auc": float(auc) if auc is not None else None,
            "f1": 2 * tp / max(1, 2 * tp + fp + fn),
            "balanced_accuracy": .5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp))}


def non_regressing(candidate, baseline):
    names = ("loss", "auc", "f1", "balanced_accuracy")
    if any(not isinstance(row.get(name), (float, int)) or not math.isfinite(row[name])
           for row in (candidate, baseline) for name in names):
        return False
    return (candidate["loss"] <= baseline["loss"] + 1e-6
            and all(candidate[name] >= baseline[name] - 1e-6 for name in names[1:]))
