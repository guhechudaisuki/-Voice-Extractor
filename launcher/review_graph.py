"""Transient multiview k-NN graph and deterministic DPP review acquisition."""
import numpy as np
from scipy import sparse


def feature_matrix(segments):
    widths = {}
    for s in segments:
        for name,vector in (s.get('multiview_embeddings') or {}).items():
            if isinstance(vector,list) and vector and np.isfinite(vector).all():
                widths.setdefault(name,len(vector))
    names = sorted(widths)
    rows = []
    for s in segments:
        views = []
        for name in names:
            v = np.asarray((s.get('multiview_embeddings') or {}).get(name,[]),dtype=float)
            if len(v) != widths[name] or not np.isfinite(v).all():
                v = np.zeros(widths[name])
            views.extend(v/max(np.linalg.norm(v),1e-12))
        if not names:
            values = [s.get('score'),s.get('calibrated_fusion_score'),*(s.get('identity_signature') or [])]
            views = [float(v) if isinstance(v,(int,float)) and np.isfinite(v) else 0.
                     for v in (values+[None]*7)[:7]]
        rows.append(views)
    x = np.asarray(rows,dtype=np.float64)
    if not names and len(x):
        x = (x-x.mean(0))/np.maximum(x.std(0),.05)
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12) if len(x) else np.empty((0,0))


def affinity_graph(x, neighbors=12):
    n = len(x)
    if n < 2:
        return sparse.csr_matrix((n,n))
    k = min(neighbors,n-1)
    rows,columns,weights = [],[],[]
    for begin in range(0,n,128):
        scores = x[begin:begin+128]@x.T
        scores[np.arange(len(scores)),np.arange(begin,begin+len(scores))] = -np.inf
        nearest = np.argpartition(-scores,k-1,axis=1)[:,:k]
        for local,js in enumerate(nearest):
            i = begin+local
            js = js[(scores[local,js]>.15) & (np.linalg.norm(x[js],axis=1)>0)]
            rows.extend([i]*len(js)); columns.extend(js.tolist())
            weights.extend(np.exp((scores[local,js]-1.)/.15).tolist())
    graph = sparse.csr_matrix((weights,(rows,columns)),shape=(n,n))
    graph = graph.maximum(graph.T)
    return sparse.diags(1./np.maximum(np.asarray(graph.sum(1)).ravel(),1e-12))@graph


def diverse_order(x, uncertainty, maximum=64):
    """Greedy MAP for an RBF L-ensemble, incremental Cholesky in O(N k²)."""
    n = len(x)
    if not n:
        return []
    quality = np.sqrt(.05+np.asarray(uncertainty))
    residual = quality**2
    factors = np.zeros((min(maximum,n),n))
    selected = []
    for step in range(min(maximum,n)):
        index = int(np.argmax(residual))
        if residual[index] < 1e-10:
            break
        selected.append(index)
        kernel = np.exp(-np.maximum(0.,np.sum((x-x[index])**2,axis=1))/.5)*quality*quality[index]
        factors[step] = (kernel-factors[:step,index]@factors[:step])/np.sqrt(residual[index])
        residual = np.maximum(0.,residual-factors[step]**2)
        residual[selected] = -np.inf
    chosen = set(selected)
    return selected+sorted((i for i in range(n) if i not in chosen),key=lambda i:(-uncertainty[i],i))


def star_projection(segments):
    x = feature_matrix(segments)
    if not len(x):
        return []
    x = x-x.mean(0)
    rng = np.random.default_rng(73)
    basis = rng.normal(size=(x.shape[1],min(2,x.shape[1])))
    for _ in range(24):
        basis,_ = np.linalg.qr(x.T@(x@basis)+1e-9*basis)
    xy = x@basis
    if xy.shape[1]<2:
        xy = np.column_stack((xy,np.zeros(len(x))))
    scale = np.maximum(np.max(np.abs(xy),axis=0),1e-9)
    return [{'ordinal':s['ordinal'],'x':float(p[0]),'y':float(p[1]),
             'cluster_id':s.get('speaker_cluster_id'),
             'cluster_key':s.get('speaker_cluster_key'),
             'cluster_label':s.get('speaker_cluster_label')} for s,p in zip(segments,xy/scale)]
