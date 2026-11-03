"""Bounded review telemetry from measured windows, never an identity gate."""
import math
import numpy as np

CHANNELS = ('primary', 'secondary', 'anime_char', 'anime_va', 'tertiary')


def heatmap(trajectory, start, end, max_frames=1500):
    """Project cached evidence to 20 ms display frames; downsample long clips.

    This is window evidence interpolated for display, not a neural frame head.
    Missing channels stay null, never imputed from other encoders.
    """
    step = max(.02, math.ceil((end-start)/max_frames/.02)*.02)
    rows = [r for r in trajectory.rows if r['span'][0] < end and r['span'][1] > start]
    frames=[]
    for a in np.arange(start,end,step):
        b=min(end,a+step); center=(a+b)/2
        active=[r for r in rows if r['span'][0] <= center < r['span'][1]]
        fine=[r for r in active if r.get('scale') == 'micro']
        active=fine or active
        channels={}
        for name in CHANNELS:
            values=[(r.get('identity_evidence') or {}).get('channel_evidence',{}).get(name) for r in active]
            values=[v for v in values if isinstance(v,(int,float)) and math.isfinite(v)]
            channels[name]=round(float(np.mean(values)),4) if values else None
        cells=trajectory._cells(float(a),float(b))
        observed=any(c.observed for c in cells)
        value=sum(c.integral for c in cells)/(b-a) if observed else None
        frames.append({'offset':round(float(a-start),4), 'end':round(float(b-start),4),
                       'evidence':round(value,4) if value is not None else None,
                       'blocked':any(c.blocked for c in cells), 'channels':channels})
    values=[f['evidence'] for f in frames if f['evidence'] is not None]
    uncertainty=float(np.mean([1./(1.+abs(v)) for v in values])) if values else 0.
    disagreement=[]
    for frame in frames:
        votes=[v for v in frame['channels'].values() if v is not None]
        if len(votes)>1:
            positive=sum(v>0 for v in votes)/len(votes)
            disagreement.append(4*positive*(1-positive))
    disagreement=float(np.mean(disagreement)) if disagreement else 0.
    return {'version':1,'kind':'window_evidence_projection','step_seconds':step,
            'source_start':start, 'source_end':end,
            'channels':list(CHANNELS),'frames':frames,
            'uncertainty':uncertainty, 'disagreement':disagreement,
            'information_score':round((uncertainty+disagreement)/2,6)}
