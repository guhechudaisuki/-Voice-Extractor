"""Bounded reference expansion inside one input file, without prior batches."""
from __future__ import annotations

import json
from pathlib import Path

from .bootstrap_bank import BootstrapVector, deduplicate, save

MIN_SCORE = .75
MIN_MARGIN = .15


def collect_vectors(analyzer, turns, result, blocked_spans=()):
    target = result.target()
    if target is None:
        return [], {"qualified": False, "reason": result.reason}
    # Identity remains tied to the user's original references on every round.
    scores, negatives = analyzer.scores(target.primary_centroid, target.secondary_centroid)
    margins = tuple(scores[i] - max((row[i + 1] for row in negatives), default=-1.) for i in (0, 1))
    qualified = min(scores) >= MIN_SCORE and min(margins) >= MIN_MARGIN
    details = {"qualified": qualified, "original_reference_scores": scores,
               "original_reference_margins": margins}
    if not qualified:
        return [], details
    rows = []
    for turn in turns:
        if (not turn.usable or not result.membership[turn.index]
                or result.cluster_of[turn.index] != target.cluster_id
                or turn.index in result.outliers
                or any(min(turn.end, b) > max(turn.start, a) for a, b in blocked_spans)):
            continue
        supports = result.diagnostics.get("bootstrap_supports", {}).get(str(turn.index), ())
        rows.append(BootstrapVector(analyzer.source_hash, turn.start, turn.end,
                    turn.speech_seconds, turn.primary, turn.secondary, min(scores), tuple(supports)))
    return rows, details


def analyze_with_bootstrap(analyzer, spans, *, output_dir, blocked_spans=(), max_rounds=3,
                           margin=.06, keep_narration=True, require_dominant=False):
    """Only this analyzer's current audio and references are read.

    Reuse encoders, embeddings and fixed cluster candidates. Each round sees
    the previous round's bank; thresholds and original identity never drift.
    """
    if type(max_rounds) is not int or not 1 <= max_rounds <= 10:
        raise ValueError("自举轮数必须为 1–10 的整数")
    turns, candidates, _ = analyzer.analyze(spans, margin=margin, temporal=False,
        keep_narration=keep_narration, require_dominant=require_dominant)
    result, local = analyzer.verify_temporal(turns, candidates, margin=margin)
    vectors, qualification = collect_vectors(analyzer, turns, result, blocked_spans)
    bank = deduplicate(vectors)
    report = {"schema": 1, "scope": "current_file_only", "human_labels_used": False,
              "acoustic_acceptance": "not_established", "self_and_descendants_excluded": True,
              "max_rounds": max_rounds, "seed_turn_count": len(vectors), "seed_vector_count": len(bank),
              "baseline_accepted_turns": sum(result.membership), "rounds": [],
              "thresholds": {"target_score": MIN_SCORE, "target_margin": MIN_MARGIN,
                  "local_margin": margin, "primary_floor": analyzer.profile.primary_reference_floor,
                  "secondary_floor": analyzer.profile.secondary_reference_floor},
              "identity": analyzer.bank_context(), "source_sha256": analyzer.source_hash,
              "qualification": qualification, "status": "no_eligible_evidence", "bank": None}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def save_bank(number):
        name = f"bank_round_{number:02d}.npz"
        meta = save(output_dir / name, bank, source_ids=[analyzer.source_hash],
            context={"identity": report["identity"], "round": number,
                     "scope": report["scope"], "thresholds": report["thresholds"]})
        report.update(bank=name, bank_sha256=meta["sha256"])

    if bank:
        save_bank(0)
    for number in range(1, max_rounds + 1):
        if not bank:
            break
        analyzer.progress(f"当前文件自动扩充参考：第 {number}/{max_rounds} 轮，{len(bank)} 个参考片段")
        before = sum(result.membership)
        used_count, used_hash = len(bank), report["bank_sha256"]
        result, local = analyzer.verify_temporal(turns, candidates, margin=margin,
                                                bank=bank, previous=(result, local))
        vectors, _ = collect_vectors(analyzer, turns, result, blocked_spans)
        bank = deduplicate(vectors)
        growth = sum(result.membership) - before
        save_bank(number)
        report["rounds"].append({"round": number, "used_vectors": used_count,
            "used_bank_sha256": used_hash, "new_accepted_turns": growth,
            "accepted_turns": sum(result.membership), "saved_vectors": len(bank)})
        report["status"] = "max_rounds_reached" if growth else "converged_no_new_turns"
        if not growth:
            break
    report.update(final_accepted_turns=sum(result.membership), vector_count=len(bank),
                  added_turns=sum(result.membership) - report["baseline_accepted_turns"])
    (output_dir / "bootstrap_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    analyzer.bootstrap_provenance = report
    analyzer.progress(f"当前文件参考扩充结束：新增通过 {report['added_turns']} 轮，参考片段 {len(bank)} 个")
    return turns, result, local
