"""Batch-local review ordering. Probabilities never change extraction decisions."""
import numpy as np


def recalibrate(segments):
    def features(seg):
        values = [seg.get('score'),seg.get('calibrated_fusion_score'),
                  *(seg.get('identity_signature') or [None]*5)]
        values = (values+[None]*7)[:7]
        return [float(v) if isinstance(v,(int,float)) and np.isfinite(v) else 0. for v in values]
    labeled = [s for s in segments if s.get('review',{}).get('label') in {'correct','wrong'}]
    labels = np.array([s['review']['label']=='correct' for s in labeled],dtype=float)
    if len(labels)<10 or min(labels.sum(),len(labels)-labels.sum())<3:
        return {'status':'insufficient_labels','message':'至少标注 10 条，含 3 条目标和 3 条非目标；部分错人不作为整句训练标签','scores':[]}
    x = np.asarray([features(s) for s in labeled])
    center,scale = np.mean(x,axis=0),np.maximum(np.std(x,axis=0),.05)
    x = np.column_stack((np.ones(len(x)),np.clip((x-center)/scale,-10,10)))
    weights = np.zeros(x.shape[1])
    # L2 regularized Platt fusion, bounded deterministic Newton updates.
    penalty = np.diag([.01]+[2.]*(x.shape[1]-1))
    for _ in range(30):
        p = 1/(1+np.exp(-np.clip(x@weights,-30,30)))
        gradient = x.T@(p-labels)+penalty@weights
        hessian = x.T@((p*(1-p))[:,None]*x)+penalty
        delta = np.linalg.solve(hessian,gradient)
        weights -= np.clip(delta,-1,1)
        if np.linalg.norm(delta)<1e-6:
            break
    scores=[]
    for seg in segments:
        if seg.get('review',{}).get('label') or (seg.get('accepted') and seg.get('cluster_tier')!='B'):
            continue
        vector=np.r_[1.,np.clip((np.asarray(features(seg))-center)/scale,-10,10)]
        p=float(1/(1+np.exp(-np.clip(vector@weights,-30,30))))
        scores.append({'ordinal':seg['ordinal'],'probability':p,
                       'information':float(4*p*(1-p))})
    return {'status':'recalibrated','labels':len(labels),'scope':'batch_review_order_only',
            'scores':sorted(scores,key=lambda s:(-s['information'],-s['probability'],s['ordinal']))}
