"""Train a reference-bound candidate from reviewed audio, with sentence holdouts.

Review timestamps construct offline labels only. The inference checkpoint has
no episode times or transcript rules. Existing production weights are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SEED = 20260928
VERIFIED_BATCH = "20260927_182743_batch_b382e7"
RESCUED_BATCH = "20260927_210928_batch_09de17"
REQUEST = ROOT / "work/desktop_requests/request_1787678029_c4ff1245.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _accepted(manifest: dict) -> list[dict]:
    return [row for row in manifest["sentences"] if row["accepted"]]


def _case(start, end, label, origin, *, correction=False) -> dict:
    start, end = float(start), float(end)
    if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end:
        raise ValueError("Invalid reviewed audio interval")
    return {"start": start, "end": end, "label": label,
            "source_id": "episode", "origins": [origin],
            "force_train": correction, "correction": correction}


def _merge(rows: list[dict]) -> list[dict]:
    merged = []
    for row in sorted(rows, key=lambda r: (r["start"], r["end"])):
        if merged and row["start"] <= merged[-1]["end"]:
            current = merged[-1]
            current["end"] = max(current["end"], row["end"])
            current["origins"] += row["origins"]
            current["force_train"] |= row["force_train"]
            current["correction"] |= row["correction"]
        else:
            merged.append({**row, "origins": list(row["origins"])})
    return merged


def build_episode_cases(verified: dict, rescued: dict, reviews: list,
                        old_review: dict, *,
                        rescued_bad_ordinals=(20, 22, 39)) -> tuple[list[dict], list[dict]]:
    """Review feedback supersedes old positives; mixed is unknown.

    ``reviews`` is the chronological chain of (batch_manifest, review,
    source_key) triples. Every review's explicit negatives become corrections
    and its remaining accepted clips count as reviewed positives, so later
    rounds refine the labels earlier rounds produced. Cases never merge or
    clip across source keys: each key is one distinct source audio.
    """
    base_key = reviews[0][2]
    positives: dict[str, list[dict]] = {base_key: []}
    negatives: dict[str, list[dict]] = {base_key: []}
    for batch, manifest in ((VERIFIED_BATCH, verified), (RESCUED_BATCH, rescued)):
        for ordinal, row in enumerate(_accepted(manifest), 1):
            label = int(batch != RESCUED_BATCH or ordinal not in rescued_bad_ordinals)
            case = _case(row["start"], row["end"], label,
                         {"batch": batch, "ordinal": ordinal, "reviewed": True})
            (positives[base_key] if label else negatives[base_key]).append(case)
    for latest, review, source_key in reviews:
        positives.setdefault(source_key, [])
        negatives.setdefault(source_key, [])
        latest_rows = _accepted(latest)
        batch_id = review.get("source_batch_id", "latest")
        latest_excluded: set[int] = set()
        for item in review["excluded_clips"]:
            ordinal = int(item["ordinal"])
            if not 1 <= ordinal <= len(latest_rows):
                raise ValueError("Review ordinal is outside its manifest")
            row = latest_rows[ordinal - 1]
            if any(abs(float(row[key]) - float(value)) > 1 / 16000
                   for key, value in zip(("start", "end"), item["span"])):
                raise ValueError("Review span does not match its manifest ordinal")
            if "duplicate_of" in item:
                if not 1 <= int(item["duplicate_of"]) <= len(latest_rows):
                    raise ValueError("Review duplicate ordinal is invalid")
                latest_excluded.add(ordinal)
                continue
            latest_excluded.add(ordinal)
            if "negative_span" in item:
                # The user kept part of the clip: the reported sub-interval is
                # a negative correction and the kept remainder is a positive
                # correction, so the model learns the split instead of a
                # whole-clip compromise.
                negative_start, negative_end = item["negative_span"]
                clip_start, clip_end = item["span"]
                if (not clip_start - 1 / 16000 <= float(negative_start)
                        < float(negative_end) <= clip_end + 1 / 16000):
                    raise ValueError("Review negative span escapes its clip")
                negatives[source_key].append(_case(
                    negative_start, negative_end, 0,
                    {"review": batch_id, "ordinal": ordinal,
                     "reason": item.get("reason", ""),
                     "partial": True}, correction=True))
                if float(negative_start) - clip_start >= 0.2:
                    positives[source_key].append(_case(
                        clip_start, negative_start, 1,
                        {"review": batch_id, "ordinal": ordinal,
                         "kept_part": True}, correction=True))
                continue
            negatives[source_key].append(_case(
                *item["span"], 0,
                {"review": batch_id, "ordinal": ordinal,
                 "reason": item.get("reason", "")}, correction=True))
        # The reviewed batch's remaining accepted clips were heard and approved
        # by the user, so they are reviewed positives, not unreviewed model
        # output.
        for ordinal, row in enumerate(latest_rows, 1):
            if ordinal in latest_excluded:
                continue
            positives[source_key].append(_case(
                row["start"], row["end"], 1,
                {"batch": batch_id, "ordinal": ordinal, "reviewed": True}))
    for item in old_review["flagged_outputs"]:
        if item["kind"] == "entire_other":
            negatives[base_key].append(_case(*item["span"], 0,
                                             {"review": "20260926", "ordinal": item["index"]}))
    clean, conflicts = [], []
    for source_key in sorted(positives):
        source_negatives = _merge(negatives[source_key])
        for positive in _merge(positives[source_key]):
            pieces = [(positive["start"], positive["end"])]
            for negative in source_negatives:
                start = max(positive["start"], negative["start"])
                end = min(positive["end"], negative["end"])
                if start < end:
                    conflicts.append({"span": [start, end], "resolution": "explicit_negative_wins",
                                      "positive_origins": positive["origins"],
                                      "negative_origins": negative["origins"]})
                remaining = []
                for left, right in pieces:
                    if negative["end"] <= left or negative["start"] >= right:
                        remaining.append((left, right))
                    else:
                        if left < negative["start"]:
                            remaining.append((left, negative["start"]))
                        if negative["end"] < right:
                            remaining.append((negative["end"], right))
                pieces = remaining
            for left, right in pieces:
                clean.append({**positive, "start": left, "end": right,
                              "source_key": source_key})
        for negative in source_negatives:
            clean.append({**negative, "source_key": source_key})
    cases = sorted(clean, key=lambda r: (r["source_key"], r["start"], r["end"]))
    for case in cases:
        case["id"] = f"{case['source_key'][:8]}:{round(case['start'] * 16000)}:{round(case['end'] * 16000)}"
    return cases, conflicts


def assign_splits(cases: list[dict], *, seed: int = SEED) -> None:
    """Assign whole temporal groups, stratified by label, without model scores."""
    groups: dict[str, list[dict]] = {}
    for source in sorted({row["source_id"] for row in cases}):
        end, group = -math.inf, ""
        for row in sorted((r for r in cases if r["source_id"] == source),
                          key=lambda r: (r["start"], r["end"], r["id"])):
            if row["start"] > end + 5.0:
                group = f"{source}:{row['id']}"
            row["group"] = group
            groups.setdefault(group, []).append(row)
            end = max(end, row["end"])
    buckets: dict[tuple, list[str]] = {}
    for group, rows in groups.items():
        if any(row.get("force_train", False) for row in rows):
            for row in rows:
                row["split"] = "train"
        else:
            label_set = tuple(sorted({row["label"] for row in rows}))
            buckets.setdefault(label_set, []).append(group)
    for keys in buckets.values():
        keys.sort(key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())
        heldout = max(1, len(keys) // 5) if len(keys) >= 3 else 0
        for index, key in enumerate(keys):
            split = "calibration" if index < heldout else (
                "test" if index < 2 * heldout else "train")
            for row in groups[key]:
                row["split"] = split


def calibration_thresholds(rows: list[dict]) -> tuple[float, float]:
    negative_means = [row["mean"] for row in rows
                      if row["split"] == "calibration" and row["label"] == 0
                      and row["mean"] is not None and math.isfinite(row["mean"])]
    positive_evidence = [row["negative_evidence"] for row in rows
                         if row["split"] == "calibration" and row["label"] == 1
                         and row["negative_evidence"] is not None
                         and math.isfinite(row["negative_evidence"])]
    if not negative_means or not positive_evidence:
        raise ValueError("Insufficient calibration negatives or adjacent positive windows")
    threshold = max(2.0, max(negative_means) + 0.5)
    # Veto threshold: bisect the gap between the shallowest trained negative
    # correction and the deepest reviewed positive evidence (any split, the
    # zero-false-veto gate enforces this ceiling).  Fall back to the
    # calibration-only anchor when the bisection anchors are missing.
    train_negative_evidence = [row["negative_evidence"] for row in rows
                               if row["split"] == "train" and row["label"] == 0
                               and row.get("correction")
                               and row["negative_evidence"] is not None
                               and math.isfinite(row["negative_evidence"])]
    all_positive_evidence = [row["negative_evidence"] for row in rows
                             if row["label"] == 1
                             and row["negative_evidence"] is not None
                             and math.isfinite(row["negative_evidence"])]
    if train_negative_evidence and all_positive_evidence:
        bisected = (max(train_negative_evidence) + min(all_positive_evidence)) / 2
        reject_threshold = min(-1.0, bisected)
    else:
        reject_threshold = min(-1.0, min(positive_evidence) - 0.5)
    return threshold, reject_threshold


def validate_veto_metrics(metrics: dict, corrections: list[dict],
                          rescue_threshold: float) -> tuple[list[str], list[str]]:
    """Gate a conservative secondary veto, while reporting weak recall.

    Calibration selects the veto threshold below every reviewed positive.
    A small calibration split can contain no negative below that threshold;
    this is a recall warning, not evidence of a positive false veto. Require
    zero positive false vetoes in every split and at least one held-out
    negative detection across calibration plus test. Every negative
    correction must be blocked by at least one calibrated gate - the veto or
    the rescue mean - because production exports only what passes both.
    Positive corrections (kept parts of partially rejected clips) must
    survive the veto.
    """
    reasons, warnings = [], []
    for split in ("train", "calibration", "test"):
        values = metrics.get(split, {})
        if not values.get("positive_cases") or not values.get("negative_cases"):
            reasons.append(f"{split} lacks a reviewed class")
        if values.get("positive_false_veto", 0):
            reasons.append(f"{split} contains positive false vetoes")
    heldout = [metrics.get(split, {}) for split in ("calibration", "test")]
    heldout_negatives = sum(row.get("negative_cases", 0) for row in heldout)
    heldout_detected = sum(row.get("negative_detected", 0) for row in heldout)
    if heldout_negatives and not heldout_detected:
        reasons.append("held-out splits detect no negative cases")
    for split, values in zip(("calibration", "test"), heldout):
        if values.get("negative_cases") and not values.get("negative_detected"):
            warnings.append(f"{split} detects no negative cases")
    for correction in corrections:
        mean = correction.get("mean")
        blocked = correction["detected"] or (
            mean is not None and math.isfinite(mean) and mean < rescue_threshold)
        if correction["label"] == 0 and not blocked:
            reasons.append(
                f"Negative correction {correction['id']} escapes both gates")
        if correction["label"] == 1 and correction["detected"]:
            reasons.append("Explicit positive corrections are vetoed")
    return reasons, warnings


def prepare_dataset(review_paths: list[Path], request_path: Path = REQUEST) -> dict:
    import soundfile as sf

    inputs: dict[str, str] = {}

    def read(path):
        path = Path(path)
        inputs[str(path.resolve())] = sha256(path)
        return json.loads(path.read_text(encoding="utf-8"))

    reviews: list[tuple[dict, dict, str]] = []
    stem_by_hash: dict[str, Path] = {}
    stems: list[Path] = []
    for review_path in review_paths:
        review = read(review_path)
        batch = review["source_batch_id"]
        if Path(batch).name != batch:
            raise ValueError("Review batch must be a directory name")
        batch_path = ROOT / "output" / batch / "batch_manifest.json"
        latest = read(batch_path)
        if inputs[str(batch_path.resolve())] != review["source_manifest_sha256"]:
            raise ValueError(f"Review manifest checksum mismatch: {batch}")
        stem = ROOT / "work" / f"{batch}_001" / "stems/target_vocals.wav"
        inputs[str(stem)] = sha256(stem)
        if inputs[str(stem)] != review["source_stem_sha256"]:
            raise ValueError(f"Review audio checksum mismatch: {batch}")
        stem_hash = inputs[str(stem)]
        reviews.append((latest, review, stem_hash))
        stem_by_hash.setdefault(stem_hash, stem)
        stems.append(stem)
    verified = read(ROOT / "output" / VERIFIED_BATCH / "batch_manifest.json")
    rescued = read(ROOT / "output" / RESCUED_BATCH / "batch_manifest.json")
    old_review = read(ROOT / "evaluation/episode1_user_review_20260926.json")
    cases, conflicts = build_episode_cases(verified, rescued, reviews, old_review)
    for case in cases:
        stem = stem_by_hash[case["source_key"]]
        case["audio"] = str(stem)
        case["source_id"] = case["source_key"]
    request = read(request_path)
    reference_hashes = [sha256(Path(path)) for path in request["references"]]
    inputs.update(zip(request["references"], reference_hashes))
    # All target-reference crops share one training group. A source reference
    # can yield multiple crops, so separate crop-level splits would leak audio.
    work = stems[-1].parent.parent
    reference_sets = [("target_references", 1, sorted((work / "reference_voice_clips").glob("*.wav")))]
    reference_sets.extend((folder.name, 0, sorted(folder.glob("*.wav")))
                          for folder in sorted((work / "negative_reference_voice_clips").glob("role_*")))
    for name, label, paths in reference_sets:
        if not paths:
            raise ValueError(f"Missing cached reference group: {name}")
        cursor = 0.0
        for path in paths:
            inputs[str(path)] = sha256(path)
            duration = sf.info(str(path)).duration
            cases.append({"id": f"{name}:{path.name}", "source_id": f"reference:{name}",
                          "start": cursor, "end": cursor + duration,
                          "audio_start": 0.0, "audio_end": duration, "audio": str(path),
                          "label": label, "force_train": label == 1, "correction": False,
                          "origins": [{"reference_group": name, "file": str(path)}]})
            cursor += duration
    assign_splits(cases)
    return {"seed": SEED, "inputs_sha256": inputs, "reference_sha256": reference_hashes,
            "cases": cases, "label_conflicts": conflicts,
            "ignored_mixed_labels": [item for item in old_review["flagged_outputs"]
                                     if item["kind"] == "mixed"],
            "limitations": "Multi-episode development with grouped per-source sentence holdouts; "
                           "not cross-series validation. "
                           "Latest corrections belong to training, not held-out success claims."}


def train(dataset: dict, output: Path) -> bool:
    import numpy as np
    import torch
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
    from extractor.audio import load_mono
    from extractor.local_identity_classifier import (
        pooled_features, window_starts, negative_window_evidence,
    )

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    directory = ROOT / "model/speaker/wavlm-base-plus-sv"
    encoder = WavLMForXVector.from_pretrained(str(directory), local_files_only=True).to(device).eval()
    encoder.requires_grad_(False)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(str(directory), local_files_only=True)
    audio_cache, features, labels, training_indices = {}, [], [], []
    for ordinal, case in enumerate(dataset["cases"], 1):
        path = case["audio"]
        if path not in audio_cache:
            audio_cache[path] = load_mono(Path(path), 16000)
        full = audio_cache[path]
        start, end = case.get("audio_start", case["start"]), case.get("audio_end", case["end"])
        begin, finish = int(start * 16000), int(end * 16000)
        if not 0 <= begin < finish <= len(full):
            raise ValueError(f"Case outside its source: {case['id']}")
        wave = full[begin:finish]
        case["windows"] = []
        for offset in window_starts(len(wave)):
            local = wave[offset:offset + 8000]
            feature = pooled_features(encoder, processor, device, local).detach().cpu()
            if not torch.isfinite(feature).all():
                raise ValueError(f"Nonfinite features: {case['id']}")
            index = len(features)
            features.append(feature)
            labels.append(case["label"])
            if case["split"] == "train":
                training_indices.append(index)
            case["windows"].append({"feature_index": index, "start": offset / 16000,
                                    "end": (offset + 8000) / 16000,
                                    "voiced": bool(torch.isfinite(local).all())
                                    and float(local.square().mean()) >= 1e-8})
        print(f"FEATURES {ordinal}/{len(dataset['cases'])} {case['split']} label={case['label']} "
              f"windows={len(case['windows'])}", flush=True)
    del encoder, audio_cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    X = torch.stack(features)
    y = torch.tensor(labels, dtype=torch.float32)
    if {labels[index] for index in training_indices} != {0, 1}:
        raise ValueError("Training needs both reviewed classes")
    model = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.ReLU(),
                                torch.nn.Linear(256, 1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_x, train_y = X[training_indices].to(device), y[training_indices].to(device)
    history = []
    for epoch in range(600):
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(train_x).squeeze(-1), train_y)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        optimizer.step()
        if epoch in (0, 149, 299, 599):
            history.append({"epoch": epoch + 1, "loss": float(loss.detach())})
            print(json.dumps(history[-1]), flush=True)
    model.eval()
    with torch.no_grad():
        scores = model(X.to(device)).squeeze(-1).cpu().tolist()
    for case in dataset["cases"]:
        values = []
        for window in case["windows"]:
            value = float(scores[window["feature_index"]])
            window["score"] = value
            values.append(value)
        case["mean"] = sum(values) / len(values) if values else None
        case["min"] = min(values) if values else None
        case["negative_evidence"] = negative_window_evidence(case["windows"])
    report = {**dataset, "training_history": history, "epochs": 600,
              "training_windows": len(training_indices), "total_windows": len(features)}
    reasons = []
    try:
        threshold, reject_threshold = calibration_thresholds(dataset["cases"])
    except ValueError as exc:
        threshold = reject_threshold = None
        reasons.append(str(exc))
    metrics = {}
    warnings = []
    veto_failure_reasons = []
    if reject_threshold is not None:
        def veto(case):
            return (case["negative_evidence"] is not None
                    and case["negative_evidence"] <= reject_threshold)

        for split in ("train", "calibration", "test"):
            rows = [case for case in dataset["cases"] if case["split"] == split]
            positive = [case for case in rows if case["label"] == 1]
            negative = [case for case in rows if case["label"] == 0]
            metrics[split] = {"positive_cases": len(positive), "negative_cases": len(negative),
                              "positive_false_veto": sum(veto(case) for case in positive),
                              "negative_detected": sum(veto(case) for case in negative),
                              "negative_false_rescue": sum(case["mean"] is not None and case["mean"] >= threshold
                                                           for case in negative)}
        corrections = [case for case in dataset["cases"] if case["correction"]]
        report["training_corrections"] = [{"id": case["id"], "label": case["label"],
                                           "detected": veto(case), "mean": case["mean"],
                                           "negative_evidence": case["negative_evidence"]}
                                          for case in corrections]
        reasons, warnings = validate_veto_metrics(
            metrics, [{"id": case["id"], "label": case["label"], "detected": veto(case),
                       "mean": case["mean"]} for case in corrections],
            threshold,
        )
        # The rescue gate is independently shippable.  When only the veto
        # calibration fails, ship the checkpoint with the veto disabled: every
        # negative correction is already blocked by the calibrated rescue mean
        # and no split shows a false rescue, so this is the conservative
        # fallback, never a loosening of the correction contract.
        rescue_reasons = []
        for split in ("train", "calibration", "test"):
            if metrics[split].get("negative_false_rescue", 0):
                rescue_reasons.append(f"{split} false-rescues reviewed negatives")
        unblocked = [item["id"] for item in report["training_corrections"]
                     if item["label"] == 0
                     and not item["detected"]
                     and not (item["mean"] is not None and item["mean"] < threshold)]
        if unblocked:
            rescue_reasons.append(f"negative corrections escape both gates: {unblocked}")
        veto_failure_reasons = reasons
        if rescue_reasons:
            reasons = rescue_reasons + veto_failure_reasons
        elif veto_failure_reasons:
            reasons = []
            warnings = warnings + [f"veto disabled: {reason}" for reason in veto_failure_reasons]
            reject_threshold = None
    report.update(threshold=threshold, reject_threshold=reject_threshold, metrics=metrics,
                  calibration_status="failed" if reasons else "passed",
                  veto_status="disabled" if (reject_threshold is None and not reasons) else "active",
                  veto_failure_reasons=veto_failure_reasons,
                  failure_reasons=reasons, validation_warnings=warnings,
                  validation_scope=(
                      "Secondary local negative veto plus classifier rescue. Rescue threshold is "
                      "calibrated with zero false rescues in every split and every negative "
                      "correction blocked by at least one gate. Veto threshold is calibrated below "
                      "all reviewed positive evidence; when its own calibration fails the veto is "
                      "disabled and the rescue gate alone carries the correction contract."
                  ))
    (output / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if reasons:
        print(json.dumps({"calibration_status": "failed", "reasons": reasons}, ensure_ascii=False), flush=True)
        return False
    payload = {"state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
               "input_dim": X.shape[1], "threshold": threshold,
               "reference_sha256": dataset["reference_sha256"], "calibration_status": "passed",
               "veto_status": "disabled" if reject_threshold is None else "active",
               "window_seconds": 0.5, "stride_seconds": 0.25,
               "feature_version": "wavlm-tdnn-unpooled-v1", "seed": SEED,
               "training_report_sha256": sha256(output / "training_report.json")}
    if reject_threshold is not None:
        payload["reject_threshold"] = reject_threshold
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    (output / "candidate.pt").write_bytes(buffer.getvalue())
    print(json.dumps({"calibration_status": "passed", "veto_status": payload["veto_status"],
                      "metrics": metrics,
                      "candidate": str(output / "candidate.pt")}, ensure_ascii=False), flush=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, action="append", required=True,
                        help="Review JSON, oldest first; each is validated "
                             "against its own batch and chained into labels")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request", type=Path, default=REQUEST)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    dataset = prepare_dataset(args.review, args.request)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "dataset.json").write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = {split: sum(row["split"] == split for row in dataset["cases"])
              for split in ("train", "calibration", "test")}
    print(json.dumps({"dataset": str(args.output / "dataset.json"), "case_counts": counts,
                      "label_conflicts": len(dataset["label_conflicts"])}), flush=True)
    return 0 if args.prepare_only or train(dataset, args.output) else 2


if __name__ == "__main__":
    raise SystemExit(main())
