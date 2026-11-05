"""Exact text recovery from complete, genuine ASR timing; never estimate words."""
import math
import re


def complete_timing(text, entries, span):
    """Return ordered units only if all text and coordinates have real provenance."""
    if not isinstance(entries, list) or not entries:
        return []
    for entry in entries:
        if (not isinstance(entry, dict) or not str(entry.get('text') or '').strip()
                or entry.get('timing_estimated') is not False
                or (entry.get('diagnostics') or {}).get('timing_estimated')
                or not all(isinstance(entry.get(k), (int, float)) and math.isfinite(entry[k])
                           for k in ('start', 'end'))
                or not span[0] <= entry['start'] < entry['end'] <= span[1]):
            return []
    ordered = sorted(entries, key=lambda entry: (entry['start'], entry['end']))
    normalize = lambda value: re.sub(r'\s+', '', value)
    if (''.join(normalize(str(entry['text'])) for entry in ordered) != normalize(text)
            or any(a['end'] > b['start'] + 1e-6 for a, b in zip(ordered, ordered[1:]))):
        return []
    return ordered


def slice_timed_text(text, ordered, span):
    """Preserve original text spacing; a boundary inside any word invalidates it."""
    positions = [i for i, char in enumerate(text) if not char.isspace()]
    selected, cursor = [], 0
    for entry in ordered:
        length = len(re.sub(r'\s+', '', str(entry['text'])))
        start, end = entry['start'], entry['end']
        if min(span[1], end) > max(span[0], start) + 1e-9:
            if start < span[0] - 1e-6 or end > span[1] + 1e-6:
                return ''
            selected.append((cursor, cursor + length))
        cursor += length
    if not selected:
        return ''
    begin, finish = selected[0][0], selected[-1][1]
    return text[positions[begin]:positions[finish] if finish < len(positions) else len(text)].strip()
