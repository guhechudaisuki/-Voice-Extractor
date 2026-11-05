"""Strict reference recommendations with source timing and acoustic proxy categories."""
import math
import numpy as np
from extractor.timed_text import complete_timing


CATEGORIES = ('neutral', 'happy_high', 'serious_low', 'soft', 'expressive')
GATES = {'duration_seconds': [3.5, 7.5], 'minimum_snr_db': 25.,
         'maximum_internal_pause_seconds': .5, 'maximum_clipping_ratio': .001,
         'minimum_voice_percentile': 95., 'maximum_per_category': 3,
         'complete_genuine_word_timing': True}


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def provenance(row, keep, scope, source_audio):
    """A cropped clip cannot inherit its parent's confidence for recommendations."""
    origin = float(row.get('start') or 0.)
    span = [origin + keep[0], origin + keep[1]]
    diagnostics = row.get('diagnostics') or {}
    navigation = diagnostics.get('review_navigation') or {}
    audit = diagnostics.get('acoustic_identity_audit') or {}
    method, score = None, None
    if audit.get('passed') and audit.get('span') == span:
        method, score = 'exact_acoustic_identity_audit', row.get('speaker_score')
    elif navigation.get('span') == span:
        method, score = 'exact_review_navigation', (navigation.get('reference_scores') or {}).get('primary')
    return {'source_span': span, 'source_audio': source_audio, 'confidence_scope': scope,
            'speaker_confidence': float(score) if finite(score) else None,
            'confidence_method': method,
            'word_timestamps': [dict(w) for w in diagnostics.get('word_timestamps', [])
                if isinstance(w, dict) and finite(w.get('start')) and finite(w.get('end'))
                and span[0] <= w['start'] < w['end'] <= span[1]]}


def longest_internal_pause(wave, rate=16000):
    """20 ms RMS pauses below -20 dB relative to speech; exclude leading/trailing silence."""
    frame = max(1, round(.02 * rate))
    count = len(wave) // frame
    if not count or not np.isfinite(wave).all():
        return None
    rms = np.sqrt(np.mean(np.asarray(wave[:count*frame]).reshape(count, frame)**2, axis=1))
    threshold = max(1e-5, float(np.percentile(rms, 85)) * .1)
    speech = np.flatnonzero(rms > threshold)
    if not len(speech):
        return None
    longest = current = 0
    for silent in rms[speech[0]:speech[-1]+1] <= threshold:
        current = current + 1 if silent else 0
        longest = max(longest, current)
    return longest * frame / rate


def select_golden(records, features):
    """Rank within each source/reference context, never compare unrelated episode scores."""
    scores = {}
    for row in records:
        score, scope = row.get('speaker_confidence'), row.get('confidence_scope')
        if finite(score) and scope and row.get('confidence_method'):
            scores.setdefault(scope, []).append(score)
    thresholds = {scope: float(np.quantile(values, .95)) for scope, values in scores.items()}
    eligible, audits = [], []
    for index, row in enumerate(records):
        qc = row.get('acoustic_qc') or {}
        acoustic = row.get('curation_features') or {}
        reasons = []
        if not row.get('valid_text') or row.get('label') != 'correct':
            reasons.append('not_reviewed_complete_sentence')
        duration = row.get('duration_seconds')
        if not finite(duration) or not 3.5 <= duration <= 7.5:
            reasons.append('duration_outside_3.5_7.5')
        if row.get('quality_grade') not in {'Gold', 'Silver'}:
            reasons.append('quality_grade')
        if not finite(qc.get('snr_db')) or qc['snr_db'] < 25.:
            reasons.append('snr_below_25_or_unknown')
        if not finite(qc.get('clipping_ratio')) or qc['clipping_ratio'] > .001:
            reasons.append('clipping_or_unknown')
        span = row.get('source_span') or []
        words = complete_timing(row.get('text', ''), row.get('word_timestamps'), span) if (
            len(span) == 2 and all(finite(v) for v in span) and span[0] < span[1]) else []
        if not words:
            reasons.append('incomplete_genuine_word_timing')
        pause = acoustic.get('longest_internal_pause_seconds')
        gap = max((b['start'] - a['end'] for a, b in zip(words, words[1:])), default=0.)
        if not finite(pause) or max(pause, gap) > .5 + 1e-9:
            reasons.append('long_internal_pause_or_unknown')
        score, scope = row.get('speaker_confidence'), row.get('confidence_scope')
        threshold = thresholds.get(scope)
        if (threshold is None or not finite(score) or not row.get('confidence_method')
                or score < threshold):
            reasons.append('voice_confidence_outside_top_5_percent_or_unknown')
        if not all(finite(acoustic.get(name)) for name in
                   ('f0_median_hz', 'f0_range_hz', 'syllables_per_second', 'rms_db', 'spectral_tilt_db')):
            reasons.append('incomplete_acoustic_dimensions')
        audits.append({'index': index, 'eligible': not reasons, 'reasons': reasons,
                       'voice_threshold': threshold, 'word_timing_complete': bool(words),
                       'maximum_word_gap_seconds': gap, 'longest_internal_pause_seconds': pause})
        if not reasons:
            eligible.append(index)
    groups = {category: [] for category in CATEGORIES}
    # Five standardized physical measurements are explicit proxies, not inferred
    # emotion labels. Missing categories stay empty instead of relaxing gates.
    for index in eligible:
        f0, width, speed, energy, tilt = features[index]
        if width > .75 or (energy > .75 and speed > .5):
            category = 'expressive'
        elif energy < -.5 and tilt < .25:
            category = 'soft'
        elif f0 > .5 and (speed > 0 or energy > 0):
            category = 'happy_high'
        elif f0 < -.5 and speed <= .5:
            category = 'serious_low'
        else:
            category = 'neutral'
        groups[category].append(index)
    for category, indexes in groups.items():
        indexes.sort(key=lambda i: (-records[i]['speaker_confidence'],
                                   -records[i]['acoustic_qc']['snr_db'], i))
        groups[category] = indexes[:3]
    return {'golden': [i for indexes in groups.values() for i in indexes],
            'golden_categories': groups, 'golden_audit': audits,
            'golden_gates': GATES, 'category_basis': 'five_dimensional_acoustic_proxies_not_emotion_recognition'}
