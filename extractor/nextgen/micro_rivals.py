"""Resolve short-window evidence for a specific competing speaker."""
from __future__ import annotations


def specific_rival_evidence(rows, floors, margin):
    """Resolve ambiguity without turning weak target evidence into an impostor.

    A half-second phoneme can resemble an unrelated voice in both correlated
    dual encoders, even when only one rival crosses their thresholds. Every
    rival therefore needs an independent encoder (WavLM, or agreeing
    character AND actor views) before becoming positive non-target evidence.
    Confirmation needs absolute support and the same target separation; a
    relative winner among universally weak embeddings is still uncertain.
    """
    candidates = [row for row in rows if all(
        row[channel] > floors[channel]
        and row[channel] - row['target_' + channel] > margin
        for channel in ('primary', 'secondary'))]
    def supports(row, channel):
        view = row.get('independent_channels', {}).get(channel, {})
        score, target = view.get('score'), view.get('target_score')
        return (score is not None and target is not None and channel in floors
                and score > floors[channel] and score - target > margin)

    confirmed = [row for row in candidates
        if supports(row, 'tertiary') or (supports(row, 'anime_char') and supports(row, 'anime_va'))]
    return {'supported': bool(confirmed), 'candidate_count': len(candidates),
            'method': 'independent_rival_resolution',
            'identity_resolved': len(confirmed) == 1,
            'confirmed_ids': [row['cluster_id'] for row in confirmed],
            'candidates': candidates}
