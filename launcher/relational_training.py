"""Projection-only set training, with ranking and calibration as separate gates."""
import torch
import torch.nn.functional as F
from extractor.set_relational import SetRelationalScorer, calibrated_logits
from .rank_calibration import fit_temperature, rank_non_regressing, rank_metrics
from .reinforcement_validation import classification_metrics


def train_projection(projection, original, samples, held_out, *, epochs, patience,
                     batch_size, progress, device):
    """Each sample is (query, reference SET, label, source, weight)."""
    scorer = SetRelationalScorer()
    train = [i for i,s in enumerate(samples) if s[3] not in held_out]
    test = [i for i,s in enumerate(samples) if s[3] in held_out]
    def score(indexes, head=projection):
        return torch.stack([scorer(head(samples[i][0].to(device)),
            head(samples[i][1].to(device))) for i in indexes])
    def labels(indexes):
        return torch.tensor([samples[i][2] for i in indexes],device=device,dtype=torch.float32)
    @torch.no_grad()
    def measure(indexes, calibration):
        scores = score(indexes)
        truth = labels(indexes)
        logits = calibrated_logits(scores, calibration)
        weights = torch.tensor([samples[i][4] for i in indexes],device=device)
        loss = F.binary_cross_entropy_with_logits(logits,truth,reduction='none')
        values, y = scores.cpu().tolist(), truth.cpu().tolist()
        probabilities = logits.sigmoid().cpu().tolist()
        # Existing classification report keys remain; threshold is now probability .5.
        metrics = classification_metrics(probabilities,y,threshold=.5)
        return {**metrics, **rank_metrics(values,y), 'loss':float((loss*weights).sum()/weights.sum()),
                'pairs':len(indexes), 'sources':len({samples[i][3] for i in indexes}),
                'positive':int(truth.sum()), 'negative':len(indexes)-int(truth.sum()),
                'scores':values, 'calibration':calibration}
    def calibrate():
        with torch.no_grad():
            return fit_temperature(score(train).cpu().tolist(),labels(train).cpu().tolist())
    calibration = calibrate()
    initial = measure(train,calibration)
    baseline = measure(test,calibration)
    baseline_scores = baseline.pop('scores')
    test_labels = labels(test).cpu().tolist()
    baseline['rank_guard'] = rank_non_regressing(baseline_scores,baseline_scores,test_labels)['rank_guard']
    best, best_state, best_epoch, stale = baseline, None, 0, 0
    optimizer = torch.optim.Adam(projection.parameters(),lr=1e-5)
    generator = torch.Generator().manual_seed(20260930)
    for epoch in range(epochs):
        order = [train[i] for i in torch.randperm(len(train),generator=generator).tolist()]
        for offset in range(0,len(order),batch_size):
            indexes = order[offset:offset+batch_size]
            values, truth = score(indexes), labels(indexes).bool()
            weights = torch.tensor([samples[i][4] for i in indexes],device=device)
            positives, negatives = values[truth], values[~truth]
            if not len(positives) or not len(negatives):
                continue
            # Pairwise ordering has no fixed identity threshold. Source weighting
            # prevents a large reference set or replay source owning the gradient.
            losses = F.softplus((.08-positives[:,None]+negatives[None,:])*10.)
            pair_weights = weights[truth,None]*weights[~truth][None,:]
            loss = (losses*pair_weights).sum()/pair_weights.sum()
            loss = loss + .05*sum((a-b).square().sum()
                for a,b in zip(projection.parameters(),original.parameters()))
            if not torch.isfinite(loss):
                raise ValueError('训练损失无效，原模型未修改')
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(projection.parameters(),1.)
            optimizer.step()
        candidate_calibration = calibrate()
        measured = measure(test,candidate_calibration)
        values = measured.pop('scores')
        gate = rank_non_regressing(values,baseline_scores,test_labels)
        measured['rank_guard'] = gate['rank_guard']
        # At a perfect ROC operating point EER/DCF cannot fall further. Preserve
        # every protected order and allow strictly better held-out calibrated NLL.
        saturated = baseline['eer'] == 0 and baseline['min_dcf'] == 0
        if (gate['non_regressing'] and (gate['improved'] or saturated)
                and measured['loss'] < best['loss']-1e-9
                and measured['f1'] >= baseline['f1']-1e-9
                and measured['balanced_accuracy'] >= baseline['balanced_accuracy']-1e-9):
            best_state = {k:v.detach().cpu().clone() for k,v in projection.state_dict().items()}
            best,best_epoch,stale = measured,epoch+1,0
        else:
            stale += 1
        progress(f"训练 {epoch+1}/{epochs}，留出损失 {measured['loss']:.5f}，最佳轮次 {best_epoch}")
        if stale >= patience:
            break
    if best_state is not None:
        projection.load_state_dict(best_state)
    def report(indexes):
        result = measure(indexes,best['calibration'] if best_state is not None else calibration)
        result.pop('scores')
        return result
    return best_state, initial['loss'], {
        'passed':best_state is not None,'baseline':baseline,'best':best,
        'best_epoch':best_epoch,'epochs_run':epoch+1,'method':'rank_then_global_calibration'}, {
        'training':report(train),'held_out_sources':report(test)}
