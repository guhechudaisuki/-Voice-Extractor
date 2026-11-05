"""Recover complete acoustic units before the shared manifold makes its decision."""
from ..types import CandidateSentence, TimeSpan


def recover_sentences(value, accepted, scored, ledger):
    from .s05_identity import certify_exact
    from ..nextgen.cluster_gate import HARD_VETOES
    from ..subtitle_assistance import restore_subtitle_sentences
    p, verifier, profile = value.pipeline, value.verifier, value.profile
    threshold = max(p.options.speaker_threshold, profile.primary.suggested_threshold)
    progress = lambda _v,m: value.progress(.76,m)
    blocks = list(p._raw_blocked_spans)
    analysis_spans = list(value.spans)

    # Surgery follows a confirmed dual-encoder change, never a subtitle or shot
    # alone. The entire parent was scored first; every child is scored afresh.
    for parent in list(scored):
        span, candidate, _match = parent
        points = [t for t in p._speaker_change_points if span.start < t < span.end]
        if not points:
            continue
        cuts = [span.start, *points, span.end]
        children = [CandidateSentence(a,b,'',diagnostics={
            'surgical_parent_span':[span.start,span.end], 'target_locator_recovery':True})
            for a,b in zip(cuts,cuts[1:]) if b-a >= .35]
        accepted[:] = [c for c in accepted if c is not candidate]
        value.rejected[:] = [c for c in value.rejected if c is not candidate]
        candidate.accepted = False
        candidate.reject_reason = '已在确诊换人点拆分，父句不导出'
        candidate.diagnostics = {'surgical_parent': True, 'span':[span.start,span.end],
            'initial_evidence':dict(candidate.diagnostics)}
        value.rejected.append(candidate)
        accepted.extend(certify_exact(p,children,value.rejected,verifier,profile,
            value.waveform,value.exclusions,ledger=ledger,origin='surgical_exact_audio',scored=scored))
        analysis_spans = [s for s in analysis_spans if s != span]
        analysis_spans.extend(TimeSpan(c.start,c.end) for c in children)

    # Locator windows only select which complete units to re-evaluate. They
    # cannot lend identity to a child or impose their own incomplete edges.
    recall = [c for c in value.rejected if c.reject_reason == '声纹匹配不足'
        and c.diagnostics.get('speaker_tier') == 'recall'
        and not any(c.diagnostics.get(k) for k in HARD_VETOES)]
    if recall:
        located = verifier.locate_target_spans(value.stem,
            [TimeSpan(c.start,c.end) for c in recall],profile,progress=progress)
        for c in recall:
            if not any(min(c.end,s.end)>max(c.start,s.start) for s in located):
                continue
            c.diagnostics['target_locator_recovery'] = True
            checked = certify_exact(p,[c],[],verifier,profile,value.waveform,value.exclusions,
                ledger=ledger,origin='locator_whole_exact_audio')
            if checked:
                accepted.extend(checked)
                value.rejected[:] = [old for old in value.rejected if old is not c]

    p._classifier_target_rescue(accepted,value.rejected,value.waveform,progress=progress)
    p._promote_multimodel_target_subclusters(scored,accepted,value.rejected,verifier,
        profile,value.stem,value.waveform,value.exclusions,progress)

    def join():
        p._merge_verified_target_turns(accepted,value.rejected,verifier,profile,value.waveform,
            threshold,[TimeSpan(c.start,c.end) for c in accepted],value.vad,blocks,progress)
    join()
    p._repair_final_sentence_boundaries(accepted,value.rejected,scored,verifier,profile,
        value.stem,value.waveform,value.exclusions,threshold,blocks,value.vad,progress)
    p._extend_target_clip_over_tail_sliver(accepted,value.rejected,verifier,profile,
        value.waveform,value.exclusions,threshold,progress)
    if p._subtitle_guide is not None and p._subtitle_guide.aligned:
        restore_subtitle_sentences(p,p._subtitle_guide,accepted,value.rejected,value.vad,
            blocks,verifier,profile,value.stem,value.waveform,value.exclusions,threshold,progress)
    join()
    accepted[:] = certify_exact(p,accepted,value.rejected,verifier,profile,value.waveform,
        value.exclusions,ledger=ledger,origin='restored_whole_exact_audio')

    # Cluster complete joined units; do not leave their internal old VAD edges
    # as compulsory cuts. Partial-overlap proposals remain separate evidence.
    for c in accepted:
        contained = [s for s in analysis_spans if c.start <= s.start and s.end <= c.end]
        if contained and not any(min(c.end,s.end)>max(c.start,s.start)
                                 for s in analysis_spans if s not in contained):
            analysis_spans = [s for s in analysis_spans if s not in contained]
            analysis_spans.append(TimeSpan(c.start,c.end))
    return sorted(analysis_spans,key=lambda s:(s.start,s.end))
