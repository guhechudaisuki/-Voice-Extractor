"""Evaluate research weights on speaker-disjoint English and known anime cases.

This is development evidence, not calibration or a deployable speaker gate.
The anime cases have already informed development and cannot be called blind.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extractor.audio import load_mono  # noqa: E402
from extractor.nextgen.artifacts import ModelCard, model_from_metadata  # noqa: E402
from extractor.nextgen.features import WavLMSpeakerFeatures, file_digest  # noqa: E402
from extractor.nextgen.identity_model import HEADS, ReferenceFeatures, TemporalIdentityModel  # noqa: E402
from extractor.nextgen.prepare_media import load_prepared_scene  # noqa: E402
from extractor.nextgen.timeline import SampleSpan  # noqa: E402
from training.build_nextgen_librispeech import _encoded  # noqa: E402
from training.nextgen_data import audit_records, read_manifest  # noqa: E402
from training.train_nextgen import load_example, reference  # noqa: E402


def _load_research(directory: Path, device: str):
    metadata = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    card = ModelCard(**{**metadata["card"], "reports": tuple(metadata["card"]["reports"])})
    if card.stage != "research" or file_digest(directory / "weights.safetensors") != card.weights_digest:
        raise ValueError("Expected a fingerprinted research checkpoint")
    model = model_from_metadata(metadata)
    model.load_state_dict(load_file(str(directory / "weights.safetensors")), strict=True)
    return model.to(device).eval(), card


def _auc(labels: torch.Tensor, scores: torch.Tensor) -> float | None:
    if labels.shape != scores.shape or labels.ndim != 1 or not torch.isfinite(scores).all():
        raise ValueError("Expected finite one-dimensional scores and matching binary labels")
    if torch.any((labels != 0) & (labels != 1)):
        raise ValueError("AUC requires binary labels")
    positive, negative = scores[labels == 1], scores[labels == 0]
    if not len(positive) or not len(negative):
        return None
    # Exact rank AUC, including ties, without a quadratic frame-pair matrix.
    ordered = negative.sort().values
    lower = torch.searchsorted(ordered, positive, right=False)
    upper = torch.searchsorted(ordered, positive, right=True)
    return float(((lower.double() + upper.double()) * .5).mean() / len(negative))


def _frozen_reference_scores(example: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Same-input no-training control, not a deployable speaker classifier."""
    query = F.normalize(example["raw"].float(), dim=-1)
    target = reference(example, "target")
    target_tokens = F.normalize(target.raw[target.valid].float(), dim=-1)
    if not len(target_tokens):
        raise ValueError("Development reference has no valid speaker token")
    target_max = (query @ target_tokens.T).amax(dim=1)
    groups = sorted({key.split(".")[1] for key in example if key.startswith("excluded.")})
    negative_tokens = [reference(example, f"excluded.{group}") for group in groups]
    negative_tokens = [F.normalize(group.raw[group.valid].float(), dim=-1)
                       for group in negative_tokens if group.valid.any()]
    if not negative_tokens:
        return target_max, target_max
    negative_max = (query @ torch.cat(negative_tokens).T).amax(dim=1)
    return target_max, target_max - negative_max


@torch.inference_mode()
def evaluate_english(model: TemporalIdentityModel, manifest: Path, feature_root: Path,
                     device: str, *, max_scenes: int = 8) -> dict:
    if max_scenes < 1:
        raise ValueError("At least one development scene is required")
    label_rows, score_rows, cosine_rows, margin_rows = [], [], [], []
    records = [record for record in read_manifest(manifest) if record.split == "development"]
    for record in records[:max_scenes]:
        item = load_example(feature_root, record, device)
        groups = sorted({key.split(".")[1] for key in item if key.startswith("excluded.")})
        output = model(item["raw"], item["stem"], reference(item, "target"),
                       tuple(reference(item, f"excluded.{group}") for group in groups),
                       item["allowed"])
        known = item["labels.frames"][:, 0] >= 0
        label_rows.append(item["labels.frames"][known, 0].cpu())
        score_rows.append(output.frames[known, 0].sigmoid().cpu())
        cosine, margin = _frozen_reference_scores(item)
        cosine_rows.append(cosine[known].cpu())
        margin_rows.append(margin[known].cpu())
    if not label_rows:
        raise ValueError("No development records in feature manifest")
    labels, scores = torch.cat(label_rows), torch.cat(score_rows)
    cosine_scores, margin_scores = torch.cat(cosine_rows), torch.cat(margin_rows)
    positive, negative = scores[labels == 1], scores[labels == 0]
    positive_min = float(positive.min()) if len(positive) else None
    negative_max = float(negative.max()) if len(negative) else None
    return {"frame_auc": _auc(labels, scores),
            "frozen_target_cosine_auc": _auc(labels, cosine_scores),
            "frozen_target_minus_exclusion_auc": _auc(labels, margin_scores),
            "scenes_evaluated": min(max_scenes, len(records)),
            "available_development_scenes": len(records),
            "positive_frames": int((labels == 1).sum()),
            "negative_frames": int((labels == 0).sum()),
            "positive_min_sigmoid_score": positive_min,
            "negative_max_sigmoid_score": negative_max,
            "zero_known_error_threshold_exists": (
                positive_min > negative_max if positive_min is not None and negative_max is not None else None),
            "note": "Uncalibrated sigmoid scores on speaker-disjoint English synthetic development scenes; epoch selection already used this split."}


def _reference_pair(encoder: WavLMSpeakerFeatures, raw_path: Path, stem_path: Path):
    raw, stem = load_mono(raw_path, 16000), load_mono(stem_path, 16000)
    length = min(len(raw), len(stem))
    raw, stem = raw[:length], stem[:length]
    raw_tokens, stem_tokens = _encoded(encoder, raw), _encoded(encoder, stem)
    levels = torch.tensor([float(stem[cell.start:cell.end].square().mean().sqrt())
                           for cell in stem_tokens.cells])
    typical = float(torch.quantile(levels, .9))
    valid = levels >= max(.0005, typical * .02)
    if not valid.any():
        raise ValueError(f"No valid voice token in {stem_path}")
    quality = (levels / max(typical, 1e-5)).clamp(0, 1)
    return ReferenceFeatures(raw_tokens.values, stem_tokens.values, valid, quality)


def _join(items: list[ReferenceFeatures]) -> ReferenceFeatures:
    if not items:
        raise ValueError("No paired reference clips found")
    return ReferenceFeatures(*(torch.cat([getattr(item, field) for item in items])
                               for field in ("raw", "stem", "valid", "quality")))


def _anime_reference_groups(encoder: WavLMSpeakerFeatures, work: Path,
                            device: str):
    target_raw = work / "reference_original_voice_clips"
    target_stem = work / "reference_voice_clips"
    names = sorted(path.name for path in target_stem.glob("*.wav"))
    target = _join([_reference_pair(encoder, target_raw / name, target_stem / name)
                    for name in names])
    negative = []
    for directory in sorted((work / "negative_reference_voice_clips").glob("role_*")):
        paired = work / "negative_reference_original_voice_clips" / directory.name
        negative.append(_join([_reference_pair(encoder, paired / path.name, path)
                               for path in sorted(directory.glob("*.wav"))]))
    target_on = ReferenceFeatures(*(value.to(device) for value in (
        target.raw, target.stem, target.valid, target.quality)))
    negative_on = tuple(ReferenceFeatures(*(value.to(device) for value in (
        group.raw, group.stem, group.valid, group.quality))) for group in negative)
    return target_on, negative_on


@torch.inference_mode()
def evaluate_prepared_candidates(model: TemporalIdentityModel,
                                 encoder: WavLMSpeakerFeatures, work: Path,
                                 scene_path: Path, device: str,
                                 probe_spans: tuple[tuple[float, float], ...] = (),
                                 trace_spans: tuple[tuple[float, float], ...] = ()) -> dict:
    """Run the research head on real, bounded new-engine candidates.

    No uncalibrated score is converted to an accepted speaker decision.
    This tests the actual candidate/context/feature integration without a
    missing calibration artifact or invented operating threshold.
    """
    scene = load_prepared_scene(scene_path)
    if scene.audio.source.total_samples > 90 * 16000:
        raise ValueError("Research candidate scene must be at most 90 seconds")
    target, negatives = _anime_reference_groups(encoder, work, device)
    def score(context: SampleSpan, output_span: SampleSpan, *, trace: bool = False) -> dict:
        raw, stem = scene.audio.encode(context, encoder)
        if (raw.source_digest != scene.audio.source.source_sha256
                or raw.backbone_digest != encoder.digest):
            raise ValueError("Prepared query feature source/encoder mismatch")
        allowed = torch.tensor(
            [cell.intersection(output_span) is not None for cell in raw.cells],
            dtype=torch.bool, device=device,
        )
        if not allowed.any():
            raise ValueError("Candidate output has no identity feature frames")
        output = model(raw.values.to(device), stem.values.to(device),
                       target, negatives, allowed)
        probabilities = output.frames.sigmoid()[allowed].cpu()
        row = {
            "output_seconds": [output_span.start / 16000,
                               output_span.end / 16000],
            "frames": len(probabilities),
            "target_p20": round(float(torch.quantile(probabilities[:, 0], .2)), 5),
            "other_p80": round(float(torch.quantile(probabilities[:, 1], .8)), 5),
            "uncalibrated_margin": round(float(
                torch.quantile(probabilities[:, 0], .2)
                - torch.quantile(probabilities[:, 1], .8)
            ), 5),
            "purity_head": round(float(output.purity.sigmoid()), 5),
        }
        if trace:
            bins: dict[int, list[torch.Tensor]] = {}
            for cell, frame, keep in zip(raw.cells, output.frames.sigmoid().cpu(),
                                         allowed.cpu().tolist()):
                if keep:
                    index = max(0, (cell.start + cell.end - 2 * output_span.start)
                                // (2 * 4000))
                    bins.setdefault(index, []).append(frame)
            trace_rows = []
            for index, frames in sorted(bins.items()):
                values = torch.stack(frames)
                trace_rows.append({
                    "start": round((output_span.start + index * 4000) / 16000, 5),
                    "end": round(min(output_span.end,
                                     output_span.start + (index + 1) * 4000) / 16000, 5),
                    "target_mean": round(float(values[:, HEADS.index("target")].mean()), 5),
                    "other_mean": round(float(values[:, HEADS.index("other")].mean()), 5),
                    "change_max": round(float(values[:, HEADS.index("change")].max()), 5),
                })
            row["trace_0_25s"] = trace_rows
        return row
    rows = []
    matched_traces: set[tuple[float, float]] = set()
    for candidate in scene.candidates:
        matching = [
            (start, end) for start, end in trace_spans if
            abs(candidate.output.start / 16000 - start) < 0.02
            and abs(candidate.output.end / 16000 - end) < 0.02
        ]
        matched_traces.update(matching)
        rows.append({**score(candidate.context, candidate.output, trace=bool(matching)),
                     "origin": candidate.origin,
                     "start_complete": candidate.start_complete,
                     "end_complete": candidate.end_complete})
    if len(matched_traces) != len(set(trace_spans)):
        raise ValueError("A requested trace span did not match a prepared candidate")
    probes = []
    for start, end in probe_spans:
        span = SampleSpan(round(start * 16000), round(end * 16000))
        if span.end > scene.audio.source.total_samples:
            raise ValueError("Probe span exceeds prepared source duration")
        probes.append(score(span, span))
    return {
        "warning": "Research scores only; no calibrated identity decision, STT, or audio export.",
        "scene": str(scene_path), "candidate_count": len(rows),
        "candidates": rows, "probes": probes,
    }


@torch.inference_mode()
def evaluate_anime(model: TemporalIdentityModel, encoder: WavLMSpeakerFeatures,
                   work: Path, device: str) -> dict:
    target_on, negative_on = _anime_reference_groups(encoder, work, device)
    raw = load_mono(work / "target_normalized.wav", 16000)
    stem = load_mono(work / "stems/target_vocals.wav", 16000)
    cases = json.loads((ROOT / "evaluation/join_cases.json").read_text(encoding="utf-8"))["cases"]
    rows, positives, negatives = [], [], []
    for case in cases:
        sides = []
        for side_index, (start, end) in enumerate((case["left"], case["right"])):
            begin, finish = round(start * 16000), round(end * 16000)
            raw_part, stem_part = raw[begin:finish], stem[begin:finish]
            length = min(len(raw_part), len(stem_part))
            raw_tokens = _encoded(encoder, raw_part[:length]).values.to(device)
            stem_tokens = _encoded(encoder, stem_part[:length]).values.to(device)
            output = model(raw_tokens, stem_tokens, target_on, negative_on,
                           torch.ones(len(raw_tokens), dtype=torch.bool, device=device))
            probs = output.frames.sigmoid().cpu()
            target_p20 = float(torch.quantile(probs[:, 0], .2))
            other_p80 = float(torch.quantile(probs[:, 1], .8))
            score = target_p20 - other_p80
            (positives if case["same_target"] or side_index == 0 else negatives).append(score)
            sides.append({"span": [start, end],
                          "known_target": bool(case["same_target"] or side_index == 0),
                          "frames": len(raw_tokens),
                          "target_p20": round(target_p20, 5),
                          "other_p80": round(other_p80, 5),
                          "conservative_margin": round(score, 5),
                          "purity": round(float(output.purity.sigmoid()), 5)})
        rows.append({"id": case["id"], "same_target": case["same_target"],
                     "sides": sides})
    return {"rows": rows, "known_target_sides": len(positives),
            "known_other_sides": len(negatives),
            "known_target_min_margin": min(positives),
            "known_other_max_margin": max(negatives),
            "zero_known_error_threshold_exists": min(positives) > max(negatives),
            "note": "First-episode development examples only; no thresholds fitted."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("feature_root", type=Path)
    parser.add_argument("anime_work", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-development-scenes", type=int, default=8)
    parser.add_argument("--prepared-scene", type=Path,
                        help="Also score real prepared candidates (research only)")
    parser.add_argument("--probe-span", action="append", default=[], metavar="START:END",
                        help="Score an exact scene-relative acoustic span for diagnosis")
    parser.add_argument("--trace-span", action="append", default=[], metavar="START:END",
                        help="Record 0.25 s frame summaries for a matching candidate")
    args = parser.parse_args()
    if (args.probe_span or args.trace_span) and not args.prepared_scene:
        parser.error("--probe-span/--trace-span require --prepared-scene")
    def parse_spans(items: list[str]) -> tuple[tuple[float, float], ...]:
        result = []
        for item in items:
            try:
                start, end = (float(value) for value in item.split(":"))
                if not (math.isfinite(start) and math.isfinite(end)
                        and 0 <= start < end):
                    raise ValueError
            except ValueError:
                parser.error("Span must be START:END with 0 <= START < END")
            result.append((start, end))
        return tuple(result)
    probe_spans = parse_spans(args.probe_span)
    trace_spans = parse_spans(args.trace_span)
    model, card = _load_research(args.checkpoint, args.device)
    audit = audit_records(read_manifest(args.feature_root / "manifest.json"), for_training=True)
    if any(audit[field] != getattr(card, field) for field in (
        "dataset_digest", "split_digest", "backbone_digest",
    )):
        raise ValueError("Evaluation manifest does not match the checkpoint's training provenance")
    encoder = WavLMSpeakerFeatures(ROOT / "model/speaker/wavlm-base-plus-sv", device=args.device)
    if encoder.digest != card.backbone_digest:
        raise ValueError("Training and anime evaluation used different acoustic encoders")
    report = {"schema": 1, "status": "research_development_only",
              "weights_digest": card.weights_digest,
              "english": evaluate_english(model, args.feature_root / "manifest.json",
                                          args.feature_root, args.device,
                                          max_scenes=args.max_development_scenes),
              "anime": evaluate_anime(model, encoder, args.anime_work, args.device)}
    if args.prepared_scene:
        report["prepared_scene"] = evaluate_prepared_candidates(
            model, encoder, args.anime_work, args.prepared_scene, args.device,
            probe_spans, trace_spans,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"English frame AUC: {report['english']['frame_auc']}; "
          f"anime known-error separation: {report['anime']['zero_known_error_threshold_exists']}")


if __name__ == "__main__":
    main()
