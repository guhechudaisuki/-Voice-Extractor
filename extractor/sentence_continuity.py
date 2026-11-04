"""Preserve exact acoustic certificates when assembling continuous speech."""
from dataclasses import replace
import math

from .nextgen.cluster_gate import HARD_VETOES


def has_exact_target_certificate(candidate):
    diagnostics = candidate.diagnostics
    audit = diagnostics.get('acoustic_identity_audit') or {}
    return bool(
        diagnostics.get('cluster_identity') is True
        and diagnostics.get('identity_trajectory', {}).get('accepted') is True
        and audit.get('passed') is True
        and audit.get('span') == [candidate.start, candidate.end]
        and not diagnostics.get('review_only')
        and not any(diagnostics.get(key) for key in (*HARD_VETOES, 'pitch_register_mismatch'))
    )


def merge_certified_contiguous(candidates, *, regions, blocked=(),
                               maximum_seconds=20., join_allowed=None,
                               maximum_gap_seconds=0., gap_verified=None):
    """Coalesce certified parts using independent evidence for every added gap.

    Touching parts share a decoded region. Longer gaps require the caller's
    acoustic verification; proximity or subtitle continuity is not evidence.
    """
    groups = []
    for candidate in sorted(candidates, key=lambda c: (c.start, c.end)):
        group = groups[-1] if groups else []
        left = group[-1] if group else None
        gap = candidate.start - left.end if left is not None else math.inf
        # A zero-crossing snap can move a touching boundary by one sample.
        # Require recorded pre-snap coverage as well as the decoded region.
        snap = candidate.diagnostics.get('acoustic_boundary_snap') or {}
        snap_from = snap.get('from') or ()
        snap_gap = (left is not None and 0. < gap <= 1 / 16000 + 1e-9
                    and len(snap_from) == 2 and snap_from[0] <= left.end + 1e-9)
        touching = abs(gap) <= 1e-9 or snap_gap
        verified_gap = (left is not None and not touching
                        and 0. <= gap <= maximum_gap_seconds
                        and gap_verified is not None and gap_verified(left, candidate))
        if (left is not None and has_exact_target_certificate(candidate)
                and all(has_exact_target_certificate(part) for part in group)
                and (touching or verified_gap)
                and candidate.end - group[0].start <= maximum_seconds
                and (verified_gap or any(a <= group[0].start + 1e-9 and b >= candidate.end - 1e-9
                                        for a, b in regions))
                and not any(a < candidate.end and b > group[0].start for a, b in blocked)
                and (join_allowed is None or join_allowed(left, candidate))):
            group.append(candidate)
        else:
            groups.append([candidate])
    output = []
    score_fields = ('speaker_score', 'window_min_score', 'window_p20_score',
                    'speaker_vote_ratio', 'window_vote_ratio')
    for group in groups:
        if len(group) == 1:
            output.extend(group)
            continue
        start, end = group[0].start, group[-1].end
        certificate = {'passed': True, 'span': [start, end],
                       'method': 'certified_contiguous_union'}
        diagnostics = {
            'cluster_identity': True, 'cluster_tier': 'A', 'cluster_mode': 'enforce',
            'cluster_would_accept': True, 'cluster_reason': 'continuous_target_speech',
            'acoustic_parent_span': [start, end],
            'acoustic_identity_audit': certificate,
            'local_identity_audit': dict(certificate),
            'identity_trajectory': {'accepted': True, 'method': 'certified_contiguous_union'},
            'verified_continuity_gaps': [[left.end, right.start] for left, right in zip(group, group[1:])
                                         if right.start > left.end],
            'continuous_parts': [
                {'span': [part.start, part.end], 'speaker_score': part.speaker_score,
                 'certificate': dict(part.diagnostics['acoustic_identity_audit']),
                 'diagnostics': dict(part.diagnostics)}
                for part in group],
        }
        diagnostics['cluster_ids'] = sorted({cluster_id for part in group
                                            for cluster_id in part.diagnostics.get('cluster_ids', ())})
        for name in ('calibrated_fusion_score', 'tier_b_rank_score', 'cohort_cm_p', 'cohort_cm_s'):
            values = [part.diagnostics.get(name) for part in group]
            if all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
                diagnostics[name] = min(values)
        diagnostics['fusion_evidence'] = {'method': 'minimum_measured_component_score'}
        maps = [part.diagnostics.get('identity_heatmap') for part in group]
        if all(isinstance(value, dict) and value.get('frames') for value in maps):
            frames = []
            for part, heatmap in zip(group, maps):
                offset = part.start - start
                frames.extend({**frame, 'offset': frame['offset'] + offset,
                               'end': frame['end'] + offset} for frame in heatmap['frames'])
            diagnostics['identity_heatmap'] = {
                **maps[0], 'source_start': start, 'source_end': end, 'frames': frames}
            for name in ('uncertainty', 'disagreement', 'information_score'):
                diagnostics['identity_heatmap'].pop(name, None)
                values = [heatmap.get(name) for heatmap in maps]
                if all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
                    diagnostics['identity_heatmap'][name] = sum(
                        part.duration * value for part, value in zip(group, values)) / (end - start)
        # Scores remain conservative component scores, not an invented new
        # model evaluation of the concatenated interval.
        output.append(replace(
            group[0], start=start, end=end, diagnostics=diagnostics,
            whisper_text=' '.join(p.whisper_text for p in group if p.whisper_text),
            text=' '.join(p.text for p in group if p.text),
            audio_file='', text_file='', video_file='',
            **{name: min(getattr(part, name) for part in group) for name in score_fields},
        ))
    return output
