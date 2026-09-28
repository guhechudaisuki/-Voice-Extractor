"""Frozen local voice features and conservative negative-window evidence."""

from __future__ import annotations

import math

import torch

from .types import CandidateSentence


def window_starts(sample_count: int, window: int = 8000, hop: int = 4000) -> list[int]:
    if window <= 0 or hop <= 0:
        raise ValueError("Window and hop must be positive")
    if sample_count < window:
        return []
    starts = list(range(0, sample_count - window + 1, hop))
    if starts[-1] != sample_count - window:
        starts.append(sample_count - window)
    return starts


@torch.no_grad()
def pooled_features(encoder, processor, device, wave: torch.Tensor) -> torch.Tensor:
    data = processor(wave.detach().cpu().numpy(), sampling_rate=16000, return_tensors="pt")
    output = encoder.wavlm(
        data.input_values.to(device),
        output_hidden_states=encoder.config.use_weighted_layer_sum,
        return_dict=True,
    )
    if encoder.config.use_weighted_layer_sum:
        layers = torch.stack(output.hidden_states, dim=1)
        weights = encoder.layer_weights.softmax(dim=0)[None, :, None, None]
        frames = (layers * weights).sum(dim=1)
    else:
        frames = output.last_hidden_state
    frames = encoder.projector(frames)
    for layer in encoder.tdnn:
        frames = layer(frames)
    frames = frames[0]
    return torch.cat([frames.mean(dim=0), frames.max(dim=0).values, frames.std(dim=0)])


@torch.no_grad()
def score_window(models: tuple, wave: torch.Tensor) -> float:
    encoder, processor, classifier, _threshold, device = models
    return float(classifier(pooled_features(encoder, processor, device, wave)).item())


def negative_window_evidence(windows: list[dict]) -> float | None:
    """Require two adjacent voiced windows, never a loud whole-clip mean."""
    pairs = []
    for left, right in zip(windows, windows[1:]):
        if not left["voiced"] or not right["voiced"]:
            continue
        if right["start"] > left["end"] or right["start"] <= left["start"]:
            continue
        scores = (left["score"], right["score"])
        if all(score is not None and math.isfinite(score) for score in scores):
            pairs.append(max(scores))
    return min(pairs) if pairs else None


def audit_candidates(
    accepted: list[CandidateSentence], rejected: list[CandidateSentence],
    waveform: torch.Tensor, scorer, threshold: float, *, model_sha256: str = "",
    min_output_seconds: float = 1.2,
) -> tuple[int, list[dict]]:
    """Withdraw a candidate only for calibrated, sustained negative evidence.

    When the veto fires but the clip also contains a sustained positive run
    long enough to stand alone, a trim proposal is returned so the caller can
    re-verify and keep the clean part instead of losing it with the whole
    clip. No timestamps, review labels or transcript text enter the decision.
    The caller must bind the classifier to the current target reference files.
    """
    if not math.isfinite(threshold):
        raise ValueError("Negative threshold must be finite")
    removed = 0
    proposals: list[dict] = []
    for candidate in list(accepted):
        start = max(0, int(candidate.start * 16000))
        end = min(len(waveform), int(candidate.end * 16000))
        wave = waveform[start:end]
        windows = []
        for offset in window_starts(len(wave)):
            local = wave[offset:offset + 8000]
            voiced = bool(torch.isfinite(local).all()) and float(local.square().mean()) >= 1e-8
            score = float(scorer(local)) if voiced else None
            windows.append({
                "start": candidate.start + offset / 16000,
                "end": candidate.start + (offset + 8000) / 16000,
                "score": score if score is not None and math.isfinite(score) else None,
                "voiced": voiced,
            })
        evidence = negative_window_evidence(windows)
        veto = evidence is not None and evidence <= threshold
        candidate.diagnostics["local_classifier_audit"] = {
            "model_sha256": model_sha256, "span": [candidate.start, candidate.end],
            "reject_threshold": threshold, "negative_evidence": evidence,
            "rejected": veto, "windows": windows,
        }
        if not veto:
            continue
        proposal = positive_run_proposal(candidate, windows, threshold,
                                         min_output_seconds)
        candidate.accepted = False
        candidate.reject_reason = "局部声学分类器确认持续非目标声音"
        if proposal is not None:
            candidate.diagnostics["classifier_trim_proposal"] = proposal
        candidate.audio_file = candidate.text_file = candidate.video_file = ""
        accepted.remove(candidate)
        rejected.append(candidate)
        removed += 1
        if proposal is not None:
            proposals.append({
                "source": candidate,
                "span": proposal["span"],
                "parent_span": [candidate.start, candidate.end],
            })
    return removed, proposals


def positive_run_proposal(
    candidate: CandidateSentence, windows: list[dict], threshold: float,
    min_output_seconds: float,
) -> dict | None:
    """Longest sustained above-threshold window run inside a vetoed clip.

    Only windows scoring above the veto threshold count toward the run; the
    returned span covers exactly that run, so the sustained negative region
    the veto found is never carried into the proposed export.
    """
    best: list[dict] = []
    run: list[dict] = []
    for window in windows:
        score = window.get("score")
        if (window.get("voiced") and score is not None
                and math.isfinite(score) and score > threshold):
            run.append(window)
            continue
        if len(run) > len(best):
            best = run
        run = []
    if len(run) > len(best):
        best = run
    if len(best) < 2:
        return None
    span_start, span_end = best[0]["start"], best[-1]["end"]
    if span_end - span_start < min_output_seconds:
        return None
    if span_start <= candidate.start + 0.02 and span_end >= candidate.end - 0.02:
        # The run covers the whole clip; nothing would be removed.
        return None
    return {"span": [span_start, span_end], "windows": len(best)}
