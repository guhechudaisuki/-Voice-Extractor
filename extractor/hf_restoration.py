"""Retired raw-band restoration: all training audio stays stem-only.

Coherence and Gold quality cannot prove that the raw high band is free of
music or another voice. Saved requests from older clients therefore abstain.
"""
import numpy as np


def restore(stem, raw, sample_rate, *, grade, singing=0., overlap=0.):
    return np.asarray(stem).copy(), {'applied': False, 'reason': 'stem_only_policy'}


def restore_export(audio, destination, row, keep, grade):
    return {'applied': False, 'reason': 'stem_only_policy'}
