"""Word timing refines cuts only inside independently certified acoustic bounds."""
import math


def real_words(words):
    return sorted((w for w in words if isinstance(w,dict) and not w.get('timing_estimated')
                   and all(isinstance(w.get(k),(int,float)) and math.isfinite(w[k]) for k in ('start','end'))
                   and w['end'] > w['start']),key=lambda w:(w['start'],w['end']))


def preserve_word_coverage(original, proposed, words):
    """Reject inward acoustic snaps that would remove an observed word sound.

    This can only restore the pre-snap endpoints. It grants neither identity
    outside the original interval nor permission to cross a pollution fence.
    """
    start, end = proposed
    for word in real_words(words):
        a, b = word['start'], word['end']
        if start > original[0] and min(start, b) > max(original[0], a):
            start = original[0]
        if end < original[1] and min(original[1], b) > max(end, a):
            end = original[1]
    return start, end


def snap_boundaries(start, end, words, *, allowed, blocked=(), waveform=None,
                    sample_rate=16000, maximum=.08):
    """Expand a sliced word by <=80ms; neither timestamps nor text grant identity."""
    original = (float(start),float(end))
    maximum = min(.08,max(0.,float(maximum)))
    def safe(a,b):
        return (any(x-1e-9<=a and b<=y+1e-9 for x,y in allowed)
                and not any(min(b,y)>max(a,x)+1e-9 for x,y in blocked))
    blocked_words=[]
    for word in real_words(words):
        a,b = float(word['start']),float(word['end'])
        if a < start < b and original[0]-a<=maximum+1e-9:
            if safe(a,end): start=a
            else: blocked_words.append([a,b])
        if a < end < b and b-original[1]<=maximum+1e-9:
            if safe(start,b): end=b
            elif [a,b] not in blocked_words: blocked_words.append([a,b])
    # Crossings must lie outward of the complete word, and within the same fence.
    if waveform is not None:
        import numpy as np
        if hasattr(waveform,'detach'): waveform=waveform.detach().cpu().numpy()
        samples=np.asarray(waveform).reshape(-1)
        for edge in (0,1):
            value=(start,end)[edge]
            if abs(value-original[edge])<1e-9: continue
            center=round(value*sample_rate);radius=round(.003*sample_rate)
            indexes=np.arange(max(1,center-radius),min(len(samples),center+radius+1))
            crossings=indexes[samples[indexes]*samples[indexes-1]<=0]/sample_rate
            crossings=sorted(crossings,key=lambda v:abs(v-value))
            for crossing in crossings:
                if (edge==0 and crossing>start) or (edge==1 and crossing<end): continue
                proposed=(float(crossing),end) if edge==0 else (start,float(crossing))
                if abs(crossing-original[edge])<=maximum+1e-9 and safe(*proposed):
                    start,end=proposed;break
    return start,end,{'original_span':list(original),'refined_span':[start,end],
                      'changed':(start,end)!=original,'blocked_words':blocked_words,
                      'maximum_extension_seconds':maximum}


def partial_word_bounds(row, keep, contaminated, waveform=None, sample_rate=16000):
    """Retain the user's pollution fences; report words that cannot safely expand."""
    origin=float(row.get('start') or 0.)
    duration=float(row.get('end') or 0.)-origin
    words=[dict(w,start=w['start']-origin,end=w['end']-origin)
           for w in real_words((row.get('diagnostics') or {}).get('word_timestamps',()))]
    a,b,audit=snap_boundaries(*keep,words,allowed=[(0.,duration)],blocked=contaminated,
                             waveform=waveform,sample_rate=sample_rate)
    return (a,b),audit
