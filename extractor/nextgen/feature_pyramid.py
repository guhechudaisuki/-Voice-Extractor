"""Sample-aligned window pyramid, sharing the normal content-addressed cache.

Windows are observations, not independent votes. The trajectory normalizes each
scale separately; a wide window must not mask a local speaker change.
"""
from .embedding_cache import canonical_interval

MICRO_SECONDS = .5
MICRO_HOP = .1
MIN_MICRO_SECONDS = .35


def pyramid_spans(start, end):
    from .cluster_runtime import window_spans

    start, end = canonical_interval(start, end)
    seen = set()
    levels = []
    for name, width, hop in (("micro", MICRO_SECONDS, MICRO_HOP), ("meso", 1.8, .9)):
        spans = []
        for a, b in window_spans(start, end, width=width, hop=hop):
            span = canonical_interval(a, b)
            if span not in seen and b - a >= MIN_MICRO_SECONDS - 1e-9:
                spans.append(span)
                seen.add(span)
        levels.append((name, spans))
    return levels
