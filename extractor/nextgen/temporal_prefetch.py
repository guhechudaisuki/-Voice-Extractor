"""Schedule unchanged exact-window work by engine, entirely within a task."""
from .embedding_cache import canonical_interval
from .feature_pyramid import MIN_MICRO_SECONDS, pyramid_spans


def prefetch_temporal_windows(miner):
    analyzer = miner.analyzer
    # Lightweight test/replay analyzers can retain their scalar implementation.
    if not hasattr(analyzer, '_prefetch_vectors') or not getattr(analyzer, 'encoders', None):
        return
    spans = []
    whole = set()
    for turn in miner.turns:
        if not miner._eligible(turn):
            continue
        if miner.previous and miner.previous[0].membership[turn.index]:
            continue
        whole.add(canonical_interval(turn.start,turn.end))
        spans.append((turn.start,turn.end))
        spans.extend(span for _,level in pyramid_spans(turn.start,turn.end) for span in level)
    if not spans:
        return
    raw_spans = []
    # Do not load raw audio here: its availability is part of the existing
    # quality measurement contract. Reuse only the already aligned RAM track.
    if analyzer.waveform is not None and analyzer.raw_waveform is not None:
        from .quality_fusion import measure_quality
        for start,end in dict.fromkeys(spans):
            key = (start,end)
            if key not in miner.quality_cache:
                miner.quality_cache[key] = measure_quality(analyzer.waveform,start,end,
                    analyzer.raw_waveform,len(miner.competitors))
            if miner.quality_cache[key].get('high_band_loss',0.) > .15:
                raw_spans.append(key)

    def tertiary_needed(start,end):
        # The same eligibility as _check; expanded/contextual profiles can
        # request additional windows later through the unchanged scalar path.
        if end-start < MIN_MICRO_SECONDS-1e-9:
            return False
        if not miner.competitors or (start,end) in whole:
            return True
        profile = miner.profile
        scores,_ = analyzer.scores(*analyzer.embed(start,end),profile)
        anime = analyzer.anime_scores(start,end)
        return (scores[0] >= profile.primary_reference_floor-.06
                or scores[1] >= profile.secondary_reference_floor-.06
                or (anime is not None and (anime[0] >= profile.anime_char_reference_floor
                                          or anime[1] >= profile.anime_va_reference_floor)))

    analyzer.prefetch(spans,raw_spans=raw_spans,tertiary_filter=tertiary_needed)
