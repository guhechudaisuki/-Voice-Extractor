"""Symmetric adaptive cohort normalization and regularized Gaussian LLR.

The LLR uses equal class priors. It is an episode estimate, not a measured
probability guarantee. With insufficient independent cohorts the reference
margin is retained and explicitly marked uncalibrated.
"""
import numpy as np


def _unit_rows(values):
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Calibration vectors must be a finite matrix")
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


class AdaptiveLLR:
    def __init__(self, references, cohorts, floor, scale, top_k=20):
        self.references = _unit_rows(references)
        self.cohorts = _unit_rows(cohorts) if len(cohorts) else np.empty((0, self.references.shape[1]))
        self.floor, self.scale, self.top_k = float(floor), max(.04, float(scale)), top_k
        self.center = _unit_rows([self.references.mean(axis=0)])[0]
        self.available = len(self.cohorts) >= 2
        self.positive = self.negative = self.variance = None
        if not self.available:
            return
        # Leave-one-reference-out positives and leave-one-cohort-out negatives
        # avoid self-cosines, including a rival normalizing against itself.
        positive = []
        for i, vector in enumerate(self.references):
            others = np.delete(self.references, i, axis=0)
            if len(others):
                center = _unit_rows([others.mean(axis=0)])[0]
                positive.append(self.normalize(float(vector @ center), vector, center=center))
        negative = [self.normalize(float(vector @ self.center), vector,
                    cohorts=np.delete(self.cohorts, i, axis=0)) for i, vector in enumerate(self.cohorts)]
        # A reference-only prior prevents one sentence from fitting a degenerate
        # likelihood. Trimming bounds the effect of contaminated cohort members.
        prior_pos = self.normalize(min(1., self.floor + self.scale), self.center)
        prior_neg = self.normalize(self.floor - self.scale, self.center)
        def posterior(values, prior):
            values = np.asarray(values)
            if len(values) > 2:
                values = np.clip(values, *np.quantile(values, [.1, .9]))
            return float((values.sum() + 2 * prior) / (len(values) + 2))
        self.positive = posterior(positive, prior_pos)
        self.negative = posterior(negative, prior_neg)
        if len(self.references) == 1:
            # One reference has no leave-one-out positives. Preserve an ordered
            # compact target hypothesis instead of fitting an inverted location.
            self.positive = max(self.positive, self.negative + .500001, .250001)
        residuals = [v-self.positive for v in positive] + [v-self.negative for v in negative]
        self.variance = float((sum(v*v for v in residuals) + 2.) / (len(residuals) + 2))
        self.available = bool(self.positive > self.negative)

    def _stats(self, scores):
        scores = np.sort(scores)[-self.top_k:]
        return float(np.mean(scores)), max(float(np.std(scores)), self.scale)

    def normalize(self, score, vector, *, center=None, cohorts=None):
        cohorts = self.cohorts if cohorts is None else cohorts
        center = self.center if center is None else center
        if not len(cohorts):
            return (score - self.floor) / self.scale
        vector = _unit_rows([vector])[0]
        zm, zs = self._stats(cohorts @ center)
        tm, ts = self._stats(cohorts @ vector)
        return .5 * ((score-zm)/zs + (score-tm)/ts)

    def evaluate(self, score, vector):
        if not self.available or vector is None:
            return (float(score)-self.floor)/self.scale
        z = self.normalize(float(score), vector)
        llr = (self.positive-self.negative) * (z-(self.positive+self.negative)/2) / self.variance
        return float(np.clip(llr, -20., 20.))

    def diagnostics(self):
        return {"method": "as_norm_gaussian_llr" if self.available else "reference_margin",
                "cohort_count": len(self.cohorts), "positive_mean": self.positive,
                "negative_mean": self.negative, "pooled_variance": self.variance,
                "equal_class_prior": True}
