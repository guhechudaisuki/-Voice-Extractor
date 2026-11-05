"""Separate source-held-out ranking checks from global monotonic calibration."""
import numpy as np


def _validated(scores, labels):
    x, y = np.asarray(scores, dtype=float), np.asarray(labels, dtype=int)
    if x.ndim != 1 or x.shape != y.shape or not np.isfinite(x).all() or set(y) != {0, 1}:
        raise ValueError('Ranking/calibration requires finite scores and both labels')
    return x, y


def fit_temperature(scores, labels):
    """Convex logistic fit; inputs MUST come only from training sources."""
    x, y = _validated(scores, labels)
    # Small bounded Newton system with a line search, no optimizer dependency.
    design = np.column_stack((x, np.ones(len(x))))
    scale = min(100., 1./max(float(np.std(x)),1e-3))
    theta = np.array([scale, -float(np.mean(x))*scale + np.log(y.mean()/(1.-y.mean()))])
    def loss(t):
        z = design @ t
        return float(np.mean(np.logaddexp(0., z) - y*z) + 1e-6*t[0]**2)
    for _ in range(40):
        z = np.clip(design @ theta, -60, 60)
        p = 1. / (1. + np.exp(-z))
        gradient = design.T @ (p-y) / len(x) + [2e-6*theta[0], 0.]
        hessian = (design.T * (p*(1-p))) @ design / len(x) + np.diag([2e-6, 1e-8])
        delta = np.linalg.solve(hessian, gradient)
        step = 1.
        while step > 1e-6:
            trial = theta - step*delta
            trial[0] = np.clip(trial[0], .01, 200.)
            if loss(trial) <= loss(theta):
                break
            step *= .5
        if step <= 1e-6 or np.linalg.norm(trial-theta) < 1e-7:
            break
        theta = trial
    return {'scale': float(theta[0]), 'bias': float(theta[1]), 'scope': 'universal',
            'method': 'positive_temperature_bias', 'fitted_on': 'training_sources_only'}


def rank_metrics(scores, labels):
    from .reinforcement_validation import classification_metrics
    x, y = _validated(scores, labels)
    order = np.argsort(-x, kind='stable')
    sorted_x, sorted_y = x[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_x)), len(x)-1]
    tp = np.r_[0, np.cumsum(sorted_y)[ends]]
    fp = np.r_[0, np.cumsum(1-sorted_y)[ends]]
    fnr, fpr = 1.-tp/y.sum(), fp/(1-y).sum()
    i = np.argmin(np.abs(fnr-fpr))
    return {'auc': classification_metrics(x,y)['auc'],
            'eer': float((fnr[i]+fpr[i])/2),
            'min_dcf': float(np.min(.5*fnr+.5*fpr))}


def rank_audit(scores, baseline, labels, confidence_margin=.02):
    x, y = _validated(scores, labels)
    old, _ = _validated(baseline, labels)
    # Prefix maxima detect ANY inverted protected pair in O(N log N), not NxN.
    negatives = np.flatnonzero(y == 0)
    negatives = negatives[np.argsort(old[negatives], kind='stable')]
    worst = np.maximum.accumulate(x[negatives])
    limits = np.searchsorted(old[negatives], old[y == 1]-confidence_margin, side='left')
    valid = limits > 0
    inverted = int(np.sum(x[y == 1][valid] <= worst[limits[valid]-1]))
    return {'protected_pairs': int(limits.sum()), 'inverted_positive_queries': inverted,
            'confidence_margin': confidence_margin, 'passed': inverted == 0}


def rank_non_regressing(scores, baseline, labels):
    before, after = rank_metrics(baseline,labels), rank_metrics(scores,labels)
    guard = rank_audit(scores,baseline,labels)
    safe = guard['passed'] and after['auc'] >= before['auc']-1e-9
    safe = safe and after['eer'] <= before['eer']+1e-9 and after['min_dcf'] <= before['min_dcf']+1e-9
    improved = after['eer'] < before['eer']-1e-9 or after['min_dcf'] < before['min_dcf']-1e-9
    return {**after, 'rank_guard': guard, 'non_regressing': bool(safe), 'improved': bool(improved)}
