"""Bounded, per-job reuse of exact local identity inputs; never persisted."""
from collections import OrderedDict
import hashlib
import math


class LocalScoreCache:
    def __init__(self, models, maximum_entries=8192):
        if type(maximum_entries) is not int or maximum_entries < 1:
            raise ValueError('Local score cache must have a positive entry limit')
        self.models = models
        self.maximum_entries = maximum_entries
        self.values = OrderedDict()
        self.hits = self.misses = 0

    def __call__(self, wave):
        from .local_identity_classifier import score_window

        # Key the actual samples, not the tensor address or rounded timestamp:
        # shared episode views and padded short phrases must not alias.
        samples = wave.detach().cpu().contiguous().numpy()
        key = (samples.dtype.str, samples.shape,
               hashlib.sha256(memoryview(samples).cast('B')).digest())
        if key in self.values:
            self.hits += 1
            self.values.move_to_end(key)
            return self.values[key]
        self.misses += 1
        value = score_window(self.models, wave)
        if math.isfinite(value):
            self.values[key] = value
            if len(self.values) > self.maximum_entries:
                self.values.popitem(last=False)
        return value
