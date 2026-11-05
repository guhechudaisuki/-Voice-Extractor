"""Exact sentence evidence, separated from timing-only cut proposals."""
from __future__ import annotations


def positive_conflict(rows, start, end, change_points=()):
    """Only an observation wholly inside a proposal can localize its impostor.

    Pooled scores belong to their measured interval. An overlapping mixed
    parent (or a window crossing the boundary) cannot say which side contains
    another voice; the proposal still needs its own exact acoustic check.
    """
    if any(start < point < end for point in change_points):
        return True
    for row in rows:
        a, b = row.get('span', (0., 0.))
        if b <= a or a < start or b > end:
            continue
        evidence = row.get('identity_evidence') or {}
        if (row.get('micro_state') == 'other' or row.get('hard_veto')
                or row.get('confirmed_speaker_change') or row.get('exclusion_conflict')
                or evidence.get('exclusion_evidence', {}).get('conflict')
                or evidence.get('same_va', {}).get('conflict')
                or evidence.get('vocal_geometry', {}).get('extreme_conflict')):
            return True
    return False


def whole_sentence_evidence(rows, start, end, *, blocked=(), change_points=()):
    """Uncertain phonemes abstain; any positive impostor evidence wins."""
    if any(min(end, b) > max(start, a) for a, b in blocked):
        return None
    if positive_conflict(rows, start, end, change_points):
        return None
    for row in rows:
        raw = row.get('raw_identity_audit') or {}
        raw_verified = (raw.get('passed') is True and raw.get('span') == [start,end]
                        and raw.get('exclusion_checked') is True
                        and raw.get('competitors_checked') is True)
        if (row.get('scale') == 'whole' and row.get('span') == [start, end]
                and row.get('accepted') is True
                and (row.get('formal_whole_verified') is True or raw_verified
                     or (row.get('identity_evidence', {}).get('accepted') is True
                         and row.get('exclusion_margins_passed') is True))):
            # formal_whole_verified is installed only after this exact full
            # audio passed dual identity AND formal excluded-role review.
            # Statistical ambiguity cannot erase it; positive conflicts above
            # remain authoritative, including those inside a protected tail.
            # A calibrated manifold certificate likewise already establishes
            # exclusion and competitor separation in its own evidence space.
            return row
    return None


def exact_certificate(candidate):
    """A flag, copied parent audit, or union of parts is not a new certificate."""
    from .nextgen.cluster_gate import HARD_VETOES
    d = candidate.diagnostics
    if d.get('review_only') or any(d.get(k) for k in (*HARD_VETOES, 'pitch_register_mismatch')):
        return None
    audit = d.get('acoustic_identity_audit') or {}
    local = d.get('local_identity_audit') or {}
    if (local.get('method') == 'subtitle_acoustic_completion' and local.get('passed') is True
            and local.get('span') == [candidate.start,candidate.end]
            and local.get('source_span') in (None, [candidate.start,candidate.end])
            and [candidate.start,candidate.end] in local.get('verified_spans', ())):
        return dict(local, authority='subtitle_acoustic')
    if (audit.get('passed') is not True or audit.get('span') != [candidate.start, candidate.end]
            or audit.get('source_span') not in (None, [candidate.start, candidate.end])
            or audit.get('method') == 'certified_contiguous_union'):
        return None
    if ((d.get('cluster_identity') is True
            and d.get('identity_trajectory', {}).get('accepted') is True)
            or audit.get('authority') == 'multichannel'):
        exact = d.get('exact_multichannel_certificate') or {}
        if (exact.get('passed') is True and not exact.get('positive_conflict')
                and not exact.get('blocked') and exact.get('span') == [candidate.start,candidate.end]):
            return dict(audit, authority='multichannel')
    if audit.get('authority') in ('dual', 'wavlm', 'subtitle_acoustic'):
        return audit
    return None
