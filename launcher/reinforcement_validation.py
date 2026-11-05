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


def classification_metrics(scores, labels, threshold=.8):
    import numpy as np
    scores, labels = np.asarray(scores), np.asarray(labels, dtype=bool)
    predicted = scores >= threshold
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


def paired_decision_audit(scores, baseline_scores, labels, threshold=.8):
    """Compare the same ordered reference/query pairs, without netting errors.

    Recovering another example never compensates for losing an old true
    positive or admitting an old true negative. Scores remain in memory;
    only counts enter the global model's validation report.
    """
    import numpy as np
    scores, previous, labels = map(np.asarray, (scores, baseline_scores, labels))
    if (scores.ndim != 1 or not scores.size or scores.shape != previous.shape
            or scores.shape != labels.shape or not np.isfinite(scores).all()
            or not np.isfinite(previous).all() or not np.isin(labels, [0, 1]).all()
            or not math.isfinite(threshold)):
        raise ValueError('逐样本验证需要同序、有限且非空的参考配对得分和二元标签')
    positive = labels.astype(bool)
    before, after = previous >= threshold, scores >= threshold
    return dict(pairs=len(scores), threshold=float(threshold),
                protected_positives=int((before & positive).sum()),
                protected_negatives=int((~before & ~positive).sum()),
                lost_positives=int((before & ~after & positive).sum()),
                new_false_positives=int((~before & after & ~positive).sum()),
                recovered_positives=int((~before & after & positive).sum()),
                removed_false_positives=int((before & ~after & ~positive).sum()))


def non_regressing(candidate, baseline):
    names = ("loss", "auc", "f1", "balanced_accuracy")
    if any(not isinstance(row.get(name), (float, int)) or not math.isfinite(row[name])
           for row in (candidate, baseline) for name in names):
        return False
    if 'sample_guard' in baseline:
        before, after = baseline['sample_guard'], candidate.get('sample_guard')
        if (not isinstance(after, dict)
                or any(after.get(key) != before.get(key) for key in
                       ('pairs', 'threshold', 'protected_positives', 'protected_negatives'))
                or after.get('lost_positives') != 0 or after.get('new_false_positives') != 0):
            return False
    return (candidate["loss"] <= baseline["loss"] + 1e-6
            and all(candidate[name] >= baseline[name] - 1e-6 for name in names[1:]))
