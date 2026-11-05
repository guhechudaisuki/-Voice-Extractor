"""Task-local graph propagation changes navigation, never labels or exports."""
import numpy as np
from .review_graph import feature_matrix, affinity_graph, diverse_order


def recalibrate(segments):
    labeled = [i for i,s in enumerate(segments) if s.get('review',{}).get('label') in {'correct','wrong'}]
    labels = np.array([segments[i]['review']['label']=='correct' for i in labeled],dtype=float)
    if len(labels)<10 or min(labels.sum(),len(labels)-labels.sum())<3:
        return {'status':'insufficient_labels','message':'至少标注 10 条，含 3 条目标和 3 条非目标；部分错人不作为整句标签','scores':[]}
    x = feature_matrix(segments)
    graph = affinity_graph(x)
    probabilities = np.full(len(segments),.5)
    probabilities[labeled] = labels
    for _ in range(80):
        updated = .95*(graph@probabilities)+.05*.5
        isolated = np.asarray(graph.sum(1)).ravel() == 0
        updated[isolated] = .5
        updated[labeled] = labels
        delta = np.max(np.abs(updated-probabilities))
        probabilities = updated
        if delta<1e-6:
            break
    eligible = [i for i,s in enumerate(segments) if not s.get('review',{}).get('label')
        and (s.get('all_sentence_export') or not s.get('accepted') or s.get('cluster_tier')=='B')]
    uncertainty = 4*probabilities[eligible]*(1-probabilities[eligible])
    order = diverse_order(x[eligible],uncertainty)
    scores = [{'ordinal':segments[eligible[i]]['ordinal'],'probability':float(probabilities[eligible[i]]),
               'information':float(uncertainty[i]),'acquisition_rank':rank}
              for rank,i in enumerate(order)]
    return {'status':'recalibrated','labels':len(labels),'scope':'batch_review_order_only',
            'method':'multiview_knn_label_propagation_dpp','scores':scores}
