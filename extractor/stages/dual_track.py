"""Independent acoustic/cue identity tracks sharing task-local media and engines."""
from dataclasses import replace
import re

from ..types import TimeSpan, CandidateSentence
from ..sentence_policy import positive_conflict
from .s05_identity import ContinuousIdentityStage, IdentityOutput, certify_exact


def cue_units(guide, waveform, *, vad_tools=None, acoustic_vad=(), audit=None):
    """Timing proposals only; no text, actor name or score grants identity."""
    if guide is None or not guide.aligned:
        return []
    from ..acoustic_qc import snap_crop_boundaries_to_valley
    duration = len(waveform)/16000
    wave = waveform.detach().float().cpu().numpy()
    units, limits = [], {}
    for cue in guide.cues:
        if cue.kind != 'speech':
            continue
        # Two dash-led dialogue rows are a multi-speaker caption, not one cue.
        if len(re.findall(r'(?:^|\n)\s*[-–—]\s*\S', cue.text)) > 1:
            continue
        span = guide.cue_span(cue)
        if span is None:
            continue
        a,b = max(0.,span.start), min(duration,span.end)
        if b-a < .35:
            continue
        bounds = (max(0.,a-.25),min(duration,b+.25))
        a,b = snap_crop_boundaries_to_valley(wave,a,b,max_shift=.25,pad_seconds=.04,
            hard_bounds=bounds)
        if b-a >= .35:
            unit = TimeSpan(a,b)
            units.append(unit)
            limits[unit] = bounds
    if vad_tools is not None:
        import torch
        # Silence padding gives FSMN context for soft cues missed by the blind
        # scan. Its result can extend a cue, never trim an existing phoneme.
        missing = [s for s in units if sum(max(0.,min(s.end,v.end)-max(s.start,v.start))
                   for v in acoustic_vad) < .5*s.duration]
        windows = {i:torch.nn.functional.pad(waveform[round(limits[s][0]*16000):round(limits[s][1]*16000)],
                    (8000,8000)) for i,s in enumerate(missing)}
        detected = vad_tools.vad_waveforms(windows) if windows else {}
        replacements = {}
        for i,s in enumerate(missing):
            lo,hi = limits[s]
            origin = round(lo*16000)/16000-.5
            support = [TimeSpan(max(lo,v.start+origin),min(hi,v.end+origin))
                for v in detected.get(i,()) if min(hi,v.end+origin)>max(lo,v.start+origin)]
            if support:
                a,b = min(s.start,min(v.start for v in support)),max(s.end,max(v.end for v in support))
                a,b = snap_crop_boundaries_to_valley(wave,a,b,max_shift=.025,pad_seconds=.04,
                    hard_bounds=(lo,hi))
                replacements[s] = TimeSpan(min(a,s.start),max(b,s.end))
        units = [replacements.get(s,s) for s in units]
        if audit is not None:
            audit['subtitle_vad_retry'] = {'in_memory':True,'windows':len(windows),
                'speech_windows':sum(bool(rows) for rows in detected.values()),
                'identity_authority':False}
        # Retain complete cue envelopes even when a soft tail has no VAD
        # response; exact acoustic identity and purity gates decide admission.
    return sorted(set(units),key=lambda s:(s.start,s.end))


def merge_tracks(first, second, *, maximum_gap, maximum_seconds, blocked, rows, changes,
                 verify_union, gap_verified, rejected=None):
    """Keep exact containment; independently certify every newly formed envelope."""
    candidates = sorted([*first,*second],key=lambda c:(c.start,-c.end))
    output, audit = [], []
    for candidate in candidates:
        physical_veto = any(min(candidate.end,b)>max(candidate.start,a) for a,b in blocked)
        # A newly confirmed local impostor/change supersedes an earlier track
        # certificate. A different parent whole-span verdict does not transfer
        # to a clean child; uncertainty alone supplies no veto.
        # The ownership-aware helper ignores mixed parent/cross-boundary
        # measurements but retains negative whole units inside this candidate.
        identity_veto = positive_conflict(rows,candidate.start,candidate.end,changes)
        if physical_veto or identity_veto:
            decision = 'shared_physical_veto' if physical_veto else 'shared_identity_veto'
            candidate.accepted = False
            candidate.reject_reason = '双路共享污染边界否决' if physical_veto else '双路共享换人或外人证据否决'
            candidate.diagnostics[decision] = True
            candidate.audio_file = candidate.text_file = candidate.video_file = ''
            if rejected is not None:
                rejected.append(candidate)
            audit.append({'span':[candidate.start,candidate.end],'decision':decision})
            continue
        if not output:
            output.append(candidate)
            continue
        left = output[-1]
        if left.start <= candidate.start and left.end >= candidate.end:
            # The full interval already owns its certificate; no new bounds.
            left.diagnostics.setdefault('track_sources',[]).extend(candidate.diagnostics.get('track_sources',[]))
            left.diagnostics['track_sources'] = sorted(set(left.diagnostics['track_sources']))
            if len(left.diagnostics['track_sources']) > 1:
                left.diagnostics['track_provenance'] = 'dual_track_merged'
            audit.append({'parts':[[left.start,left.end],[candidate.start,candidate.end]],'decision':'contained'})
            continue
        gap = candidate.start-left.end
        a,b = left.start,max(left.end,candidate.end)
        conflict = (any(min(b,y)>max(a,x) for x,y in blocked)
                    or positive_conflict(rows,a,b,changes))
        joinable = (gap <= maximum_gap and b-a <= maximum_seconds and not conflict
                    and (gap <= 0 or gap_verified(left,candidate)))
        joined = None
        if joinable:
            proposal = CandidateSentence(a,b,'',diagnostics={
                'target_locator_recovery': True,
                'track_provenance':'dual_track_merged',
                'track_sources':sorted(set(left.diagnostics.get('track_sources',[])
                                          +candidate.diagnostics.get('track_sources',[]))),
                'track_merge_parts':[[left.start,left.end],[candidate.start,candidate.end]]})
            joined = verify_union(proposal)
        if joined is not None:
            output[-1] = joined
            audit.append({'span':[a,b],'decision':'independently_verified_union'})
        elif gap < 0:
            # Do not export duplicate overlapping audio after a failed union.
            output[-1] = max((left,candidate),key=lambda c:c.duration)
            audit.append({'span':[a,b],'decision':'kept_longer_certified_interval'})
        else:
            output.append(candidate)
    return output,audit


def run_tracks(value, guide, pre_singing, *, vad_tools=None):
    from .. import pipeline_common as common
    p = value.pipeline
    p._subtitle_guide = None
    first = ContinuousIdentityStage().run(value)
    for c in first.accepted:
        c.diagnostics.update(track_provenance='track1_acoustic',track_sources=['track1_acoustic'])
    first_report, first_audit = p._cluster_report, p._stage_audit
    first_changes = p._speaker_change_points
    raw_recovered = getattr(p,'_raw_vad_recovered',())
    shared_anchors = p._candidate_cut_anchors
    physical_blocks = p._raw_blocked_spans
    if guide is None or not guide.aligned:
        if not getattr(p.options, 'branch_validation', True):
            from .export_timeline import union_selected
            first.accepted, _ = union_selected(first.accepted, [])
        if first_audit is not None:
            first_audit['tracks'] = {'track1_acoustic':{'accepted_count':len(first.accepted)},
                'track2_subtitle':{'enabled':False},'merged_count':len(first.accepted),
                'branch_validation':getattr(p.options, 'branch_validation', True)}
        return first

    # The subtitle path has independent decisions, filters and episode manifold.
    # Accepted/rejected labels and target prototypes never cross between paths.
    p._subtitle_guide = guide
    p._cluster_report = None
    p._cluster_accepted_spans = None
    p._speaker_change_points = ()
    p._raw_vad_recovered = []
    p._verified_fallback_registry = {}
    p._verified_fallback_anchors = []
    p._stage_audit = {'schema_version':1,'stages':{},'source_duration_seconds':len(value.waveform)/16000}
    track_rejected = []
    cue_spans = cue_units(guide,value.waveform,vad_tools=vad_tools,
        acoustic_vad=value.vad,audit=p._stage_audit)
    p._candidate_cut_anchors = tuple(sorted(set(shared_anchors)
        | {t for s in cue_spans for t in (s.start,s.end)}))
    p._trace_spans('subtitle_cue_units',cue_spans)
    clean = cue_spans
    singing_evidence = list(pre_singing)
    overlap_evidence = []
    if clean and p.options.use_singing_detector:
        detector = common.pooled_encoder(common.SingingDetector,p.device,common.PANNS_MODEL)
        _,residual = detector.clean_spans(value.stem,clean,p.options.singing_threshold,
            progress=lambda v,m:value.progress(.80,'字幕第二路：'+m))
        singing_evidence.extend(residual)
    if singing_evidence:
        clean,removed = p._partition_tainted_spans(clean,singing_evidence,
            minimum_overlap_seconds=.20,waveform=value.waveform)
        track_rejected.extend(CandidateSentence(s.start,s.end,'',reject_reason='检测到有人唱歌') for s in removed)
    if clean and p.options.use_overlap_detector:
        detector = common.pooled_encoder(common.OverlapDetector,p.device,common.OVERLAP_MODEL)
        _,overlap_evidence = detector.clean_spans(value.stem,cue_spans,p.options.overlap_threshold,
            padding=0.,progress=lambda v,m:value.progress(.80,'字幕第二路：'+m))
    overlaps = guide.overlapping_speech_spans()
    clean,removed = p._partition_tainted_spans(clean,[*overlap_evidence,*overlaps],
        minimum_overlap_seconds=.08,waveform=value.waveform)
    track_rejected.extend(CandidateSentence(s.start,s.end,'',reject_reason='检测到多人同时发声') for s in removed)
    p._raw_blocked_spans = tuple([*physical_blocks,*singing_evidence,*overlap_evidence,*overlaps])
    p._trace_spans('residual_singing_evidence',singing_evidence)
    p._trace_spans('overlap_evidence',overlap_evidence)
    p._trace_spans('subtitle_overlap_evidence',overlaps)
    # The cue itself gets evaluated even if blind VAD missed a soft utterance.
    # Relative energy and all acoustic identity gates still reject silence.
    clean = p._merge_short_silence_same_speaker(clean,value.verifier,value.profile,value.waveform,
        progress=lambda v,m:value.progress(.80,'字幕第二路：'+m),
        maximum_silence_seconds=p.options.silence_split_seconds,forbidden_joins=p._raw_blocked_spans)
    review_spans = p._split_micro_window_changes(clean,value.verifier,value.profile,value.waveform,
        progress=lambda v,m:value.progress(.80,'字幕第二路：'+m))
    p._stage_audit['sentence_first'] = {'natural_sentences':len(clean),
        'confirmed_change_points':list(p._speaker_change_points),'anchors_split_preverification':False}
    second_value = replace(value,spans=clean,vad=cue_spans,rejected=track_rejected,
        progress=lambda v,m:value.progress(.80,'字幕第二路：'+m))
    second = ContinuousIdentityStage().run(second_value) if clean else IdentityOutput([],first.ledger)
    second_report, second_audit = p._cluster_report,p._stage_audit
    for c in second.accepted:
        c.diagnostics.update(track_provenance='track2_subtitle',track_sources=['track2_subtitle'])
    for c in track_rejected:
        c.diagnostics['track_provenance'] = 'track2_subtitle'
    # Physical model vetoes are shared. Subtitle overlaps constrain only the
    # subtitle path; they never retroactively cut the pure acoustic path.
    p._raw_blocked_spans = tuple([*physical_blocks,*singing_evidence,*overlap_evidence])
    p._speaker_change_points = tuple(sorted(set(first_changes)|set(p._speaker_change_points)))
    rows = [r for report in (first_report,second_report) if report
            for group in report.get('local_windows',{}).values() for r in group]
    blocks = [(s.start,s.end) for s in p._raw_blocked_spans]
    from ..utterance_assembly import gap_verifier
    gap = gap_verifier(value.waveform,regions=[],sources=[*first.accepted,*second.accepted],blocked=blocks)
    def verify_union(candidate):
        checked = certify_exact(p,[candidate],[],value.verifier,value.profile,value.waveform,
            value.exclusions,ledger=first.ledger,origin='dual_track_union_exact_audio')
        if checked:
            candidate.diagnostics['whole_sentence_protection'] = {
                'span':[candidate.start,candidate.end],'passed':True,
                'method':'independent_union_and_no_positive_impostor'}
        return checked[0] if checked else None
    if getattr(p.options, 'branch_validation', True):
        accepted,merge_audit = merge_tracks(first.accepted,second.accepted,
            maximum_gap=p.options.silence_split_seconds,maximum_seconds=min(20.,p.options.max_sentence_seconds),
            blocked=blocks,rows=rows,changes=p._speaker_change_points,verify_union=verify_union,gap_verified=gap,
            rejected=value.rejected)
    else:
        from .export_timeline import union_selected
        accepted,merge_audit = union_selected(first.accepted,second.accepted)
    value.rejected.extend(track_rejected)
    p._cluster_report = first_report or {}
    p._cluster_report['tracks'] = {'track2_subtitle':second_report}
    p._stage_audit = first_audit
    p._stage_audit['tracks'] = {
        'track1_acoustic':{'accepted_count':len(first.accepted)},
        'track2_subtitle':{'enabled':True,'accepted_count':len(second.accepted),'audit':second_audit},
        'merged_count':len(accepted),'union_decisions':merge_audit}
    p._stage_audit['tracks']['branch_validation'] = getattr(p.options, 'branch_validation', True)
    p._cluster_accepted_spans = [(c.start,c.end) for c in accepted]
    p._subtitle_guide = guide
    p._raw_vad_recovered = raw_recovered
    return IdentityOutput(accepted,first.ledger,review_spans=list(review_spans))
