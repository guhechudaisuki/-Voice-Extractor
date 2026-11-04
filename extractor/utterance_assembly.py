"""Verify audio between already certified target parts before sentence assembly."""
import math

import torch

from .types import TimeSpan


def strong_island(row, margin=0.):
    values = row.get('domain_margins') or {}
    return (row.get('state') == 'target_supported'
            and not row.get('excluded_role_rejected')
            and not row.get('other_witnesses')
            and all(isinstance(values.get(f'{v}_{c}'), (int, float))
                    and math.isfinite(values[f'{v}_{c}']) and values[f'{v}_{c}'] > margin
                    for v in ('char', 'va') for c in ('stem', 'raw')))


def confirmed_islands(candidates, margin=0.):
    spans = set()
    for candidate in candidates:
        evidence = candidate.diagnostics.get('experimental_final_island_evidence') or {}
        if evidence.get('whole_conflict') or candidate.diagnostics.get('excluded_role_rejected'):
            continue
        for row in evidence.get('islands', ()):
            if strong_island(row, margin):
                spans.add(tuple(row['span']))
    return sorted(spans)


def gap_verifier(waveform, *, regions, sources, blocked=()):
    """Only join silence or independently confirmed speech within a decoded region.

    A wide voiceprint covering both endpoints is insufficient for audible gap
    content. That content also needs exact island evidence from four views.
    """
    islands = confirmed_islands(sources)
    barriers = list(blocked)
    for candidate in sources:
        if candidate.diagnostics.get('excluded_role_rejected') or candidate.diagnostics.get('structural_hard_reject'):
            barriers.append((candidate.start, candidate.end))
    wave = waveform.detach().float().cpu().reshape(-1)

    def verified(left, right):
        start, end = left.end, right.start
        if end <= start:
            return False
        if any(a < end and b > start for a, b in barriers):
            return False
        begin, finish = round(start * 16000), round(end * 16000)
        if begin < 0 or finish > len(wave) or finish <= begin:
            return False
        frame = 160  # 10 ms catches short interjections without averaging a gap.
        # Use the voiced flanks, not the fading/quiet 400 ms next to a pause.
        # Both flanks must place the gap well below their speech level. Keep
        # an absolute ceiling so a quiet third voice cannot inherit a loud
        # parent's identity, and an absolute floor for digital near-silence.
        flank_levels = []
        for a, b in ((max(left.start, left.end - 1.8), left.end),
                     (right.start, min(right.end, right.start + 1.8))):
            levels = [float(chunk.square().mean().sqrt()) for chunk in
                      wave[round(a * 16000):round(b * 16000)].split(frame) if len(chunk)]
            if not levels:
                return False
            flank_levels.append(float(torch.quantile(torch.tensor(levels), .9)))
        speech = min(flank_levels)
        quiet = max(10 ** (-65 / 20), min(10 ** (-32 / 20), speech * 10 ** (-12 / 20)))
        region_covers = any(a <= left.start + 1e-6 and b >= right.end - 1e-6 for a, b in regions)
        for offset in range(begin, finish, frame):
            chunk_end = min(finish, offset + frame)
            chunk = wave[offset:chunk_end]
            if float(chunk.square().mean().sqrt()) <= quiet:
                continue
            a, b = offset / 16000, chunk_end / 16000
            if not region_covers or not any(s <= a + 1e-6 and e >= b - 1e-6 for s, e in islands):
                return False
        return True

    return verified


def recheck_short_exclusion_tails(rejected, accepted, domain_for, *, blocked, margin):
    """Re-evaluate a near-tie short veto inside an independently verified island.

    This only clears the old veto. The normal final cluster, local trajectory
    and arbitration still decide whether any of this audio may be exported.
    """
    islands = confirmed_islands(accepted, margin)
    count = 0
    for candidate in rejected:
        d = candidate.diagnostics
        direct = [d.get(f'excluded_{name}_direct_margin') for name in ('primary', 'secondary')]
        if (not d.get('local_boundary_recovery') or not d.get('excluded_role_rejected')
                or not .35 <= candidate.duration < 1.2
                or any(d.get(k) for k in ('structural_hard_reject', 'excluded_raw_vote',
                                         'excluded_tertiary_conflict', 'excluded_diagonal_vote'))
                or not all(isinstance(v, (int, float)) and math.isfinite(v) and v > -margin for v in direct)
                or any(s.start < candidate.end and s.end > candidate.start for s in blocked)
                or not any(a <= candidate.start + 1e-6 and b >= candidate.end - 1e-6 for a, b in islands)):
            continue
        verdict = domain_for(TimeSpan(candidate.start, candidate.end))
        if verdict is None:
            continue
        scores = {f'{v}_{c}': verdict.score(v, c).margin for v in ('char', 'va') for c in ('stem', 'raw')}
        if not strong_island({'state': verdict.state, 'domain_margins': scores}, margin):
            continue
        d['short_boundary_exclusion_recheck'] = {
            'span': [candidate.start, candidate.end], 'domain_margins': scores,
            'previous_exclusion': {k: v for k, v in d.items() if k.startswith('excluded_')},
            'previous_reason': candidate.reject_reason,
            'result': 'independent_four_view_target_support_requires_final_gate',
        }
        d['excluded_role_rejected'] = False
        candidate.reject_reason = '声纹匹配不足'
        # Preserve the measured generic scores; mark the exact new evidence.
        d['experimental_final_island_evidence'] = {'islands': [{
            'span': [candidate.start, candidate.end], 'state': verdict.state,
            'domain_margins': scores, 'excluded_role_rejected': False}]}
        count += 1
    return count
