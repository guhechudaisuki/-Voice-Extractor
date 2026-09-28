"""Frozen, review-only three-model experiment on a completed full-episode batch.

The production batch supplies speech boundaries, singing/overlap checks, UVR,
speaker decisions and Japanese STT. Personal VAD 2.0 proposes missed speech
candidates; VoiceFilter and SpEx+ determine which proposals enter the listening
package. None of these three models certifies a speaker or supplies export audio.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.full_model_evidence import (  # noqa: E402
    PersonalVADCoverage, SpExPlusEvaluator, WaveformComparison, file_sha256,
)
from extractor.audio import write_clip  # noqa: E402
from extractor.config import whisper_snapshot  # noqa: E402
from extractor.transcription import WhisperSegmenter  # noqa: E402
from extractor.types import TimeSpan  # noqa: E402

SPEX_ROOT = ROOT / "work" / "spexplus_trial_20260926"
VF_WEIGHT = ROOT / "models" / "conv-voice-filter" / "pytorch_model.bin"
PVAD_WEIGHT = ROOT / "work" / "personal_vad_2_src" / "checkpoints" / "pilot4k_speaker_aware_epoch38" / "best_inference.pt"
SPEX_WEIGHT = (ROOT / "models" / "modelscope_cache" / "alibabasglab"
               / "log_wsj0-2mix_speech_SpEx-plus_2spk" / "checkpoints"
               / "log_wsj0-2mix_speech_SpEx-plus_2spk" / "last_best_checkpoint.pt")


def _overlap(a: dict, b: dict) -> float:
    return max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def eligible_proposal(row: dict, accepted: list[dict], min_duration: float) -> bool:
    """Only a bounded, non-structurally-rejected sentence can be proposed."""
    if row.get("accepted") or row.get("reject_reason") != "声纹匹配不足":
        return False
    duration = row["end"] - row["start"]
    if not min_duration <= duration <= 45.0:
        return False
    if row.get("singing_score", 0) > 0 or row.get("overlap_score", 0) > 0:
        return False
    return all(_overlap(row, item) <= min(0.15, duration * 0.1) for item in accepted)


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def verify_cached_audio_alignment(current_16k: Path, pvad_16k: Path,
                                  spex_8k: Path) -> dict:
    """Check every decoded sample/block, not only a few scene anchors."""
    import torch
    import torchaudio

    max_pvad_error = 0.0
    min_spex_correlation = 1.0
    blocks = compared = 0
    with (sf.SoundFile(current_16k) as current,
          sf.SoundFile(pvad_16k) as personal,
          sf.SoundFile(spex_8k) as spex):
        if (current.samplerate, personal.samplerate, spex.samplerate) != (16000, 16000, 8000):
            raise ValueError("External-model episode caches have unexpected sample rates")
        if current.channels != personal.channels != spex.channels != 1:
            raise ValueError("External-model episode caches must be mono")
        duration = len(current) / 16000
        if abs(duration - len(personal) / 16000) > 0.001 or abs(duration - len(spex) / 8000) > 0.001:
            raise ValueError("External-model episode caches have different durations")
        for block_start in range(0, len(current), 10 * 16000):
            frames = min(10 * 16000, len(current) - block_start)
            original = current.read(frames, dtype="float32")
            pvad_audio = personal.read(frames, dtype="float32")
            spex_audio = spex.read(frames // 2, dtype="float32")
            if len(pvad_audio) != frames or len(spex_audio) != frames // 2:
                raise ValueError("External-model cache ended before the current source")
            pvad_error = float(np.max(np.abs(original - pvad_audio)))
            max_pvad_error = max(max_pvad_error, pvad_error)
            if pvad_error > 1e-4:
                raise ValueError(f"Personal VAD cache differs at source sample {block_start}")
            downsampled = torchaudio.functional.resample(
                torch.from_numpy(original), 16000, 8000,
            ).numpy()
            source_rms = float(np.sqrt(np.mean(downsampled * downsampled)))
            cache_rms = float(np.sqrt(np.mean(spex_audio * spex_audio)))
            if source_rms > 1e-5:
                correlation = float(np.corrcoef(downsampled, spex_audio)[0, 1])
                if (not math.isfinite(correlation) or correlation < 0.98
                        or not 0.9 <= cache_rms / source_rms <= 1.1):
                    raise ValueError(f"SpEx+ cache differs at source sample {block_start}")
                min_spex_correlation = min(min_spex_correlation, correlation)
                compared += 1
            elif cache_rms > 1e-4:
                raise ValueError(f"SpEx+ cache has unmatched signal at source sample {block_start}")
            blocks += 1
    if compared < 3:
        raise ValueError("Not enough non-silent blocks to verify SpEx+ cache")
    return {"block_seconds": 10, "blocks_checked": blocks,
            "spexplus_blocks_compared": compared,
            "max_personal_vad_abs_error": max_pvad_error,
            "min_spexplus_correlation": min_spex_correlation}


def choose_model_proposals(rows: list[dict]) -> tuple[list[dict], dict]:
    """Predeclared review heuristic, not a calibrated identity probability.

    PVAD must cover at least 0.30 s of a complete base speech candidate.
    VF and SpEx+ must each have at least the median response among those PVAD
    candidates. These target-conditioned responses are known to be fallible;
    their only authority is selection for *human review*, not training export.
    """
    pvad_rows = [row for row in rows if row["pvad"]["target_overlap_seconds"] >= 0.30]
    vf_floor = _median([row["voicefilter"]["output_over_input_db_model_estimate"]
                        for row in pvad_rows])
    spex_floor = _median([row["spexplus"]["output_over_input_db"]
                          for row in pvad_rows])
    passing = ([] if vf_floor is None or spex_floor is None else [
        row for row in pvad_rows
        if row["voicefilter"]["output_over_input_db_model_estimate"] >= vf_floor
        and row["spexplus"]["output_over_input_db"] >= spex_floor
    ])
    # Recovery/lattice bookkeeping can leave several proposed versions of
    # one utterance. Prefer its longest complete candidate for listening.
    selected: list[dict] = []
    for row in sorted(passing, key=lambda item: (item["end"] - item["start"]), reverse=True):
        if all(_overlap(row, prior) <= 0.15 for prior in selected):
            selected.append(row)
    return selected, {"pvad_min_overlap_seconds": 0.30,
                      "voicefilter_episode_candidate_median_db": vf_floor,
                      "spexplus_episode_candidate_median_db": spex_floor,
                      "pvad_proposed": len(pvad_rows),
                      "three_model_response_pass": len(passing),
                      "overlapping_alternates_suppressed": len(passing) - len(selected),
                      "three_model_review_selected": len(selected)}


def _resolve_batch(batch_dir: Path) -> tuple[Path, Path, dict]:
    batch_dir = batch_dir.resolve(strict=True)
    batch = json.loads((batch_dir / "batch_manifest.json").read_text(encoding="utf-8"))
    if batch["target_count"] != 1:
        raise ValueError("This experiment requires one full-episode source per batch")
    source_dir = (batch_dir / batch["targets"][0]["output_dir"]).resolve(strict=True)
    if source_dir.parent != batch_dir:
        raise ValueError("Batch manifest output directory escapes the selected batch")
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["accepted_count"] + manifest["rejected_count"] != len(manifest["sentences"]):
        raise ValueError("Incomplete batch sentence ledger")
    def span_keys(rows: list[dict]) -> Counter:
        return Counter((row["start"], row["end"], bool(row["accepted"])) for row in rows)
    if span_keys(manifest["sentences"]) != span_keys(batch["sentences"]):
        raise ValueError("Source and batch candidate ledgers disagree")
    return batch_dir, source_dir, manifest


def _transcribe_new(stem: Path, selected: list[dict], device: str) -> list[str]:
    if not selected:
        return []
    spans = [TimeSpan(row["start"], row["end"]) for row in selected]
    segments = WhisperSegmenter(device).transcribe_spans(stem, spans)
    texts = []
    for index, _row in enumerate(selected):
        matching = [item.text or item.whisper_text for item in segments
                    if item.diagnostics.get("transcription_span_index") == index]
        texts.append(" ".join(text for text in matching if text).strip())
    return texts


def export_review_clip(stem: Path, destination: Path, index: int,
                       row: dict, text: str) -> dict:
    """Export the real UVR stem, never VoiceFilter/SpEx+ generated audio."""
    source = "base_pipeline" if row["base_accepted"] else "three_model_review_proposal"
    name = f"{index:04d}_{row['start']:.2f}-{row['end']:.2f}_{source}_UNVERIFIED"
    wav_path = destination / "audio" / f"{name}.wav"
    text_path = destination / "text" / f"{name}.txt"
    write_clip(stem, wav_path, row["start"], row["end"], sample_rate=16000,
               normalize_level=True)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(text or "[STT_PENDING]", encoding="utf-8")
    return {"start": row["start"], "end": row["end"], "source": source,
            "text": text, "text_status": "ready" if text else "pending",
            "audio": str(wav_path.relative_to(destination)),
            "text_file": str(text_path.relative_to(destination))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pvad-report", type=Path, required=True)
    parser.add_argument("--voicefilter-dir", type=Path, required=True)
    parser.add_argument("--spex-reference", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    batch_dir, source_dir, manifest = _resolve_batch(args.batch)
    if manifest["options"].get("user_excluded_spans"):
        raise ValueError("Label-driven user exclusion marks cannot enter this inference trial")
    destination = args.output.resolve()
    if destination.exists():
        raise FileExistsError(f"Freeze destination already exists: {destination}")
    archive = destination.with_suffix(".zip")
    if archive.exists():
        raise FileExistsError(f"Freeze archive already exists: {archive}")
    work_dir = ROOT / "work" / f"{batch_dir.name}_001"
    raw = work_dir / "target_normalized.wav"
    stem = work_dir / "stems" / "target_vocals.wav"
    if not stem.is_file():
        raise FileNotFoundError(stem)
    current_16k = work_dir / "target_original_vad_16k.wav"
    current_ref = work_dir / "reference_original_voice_clips" / "reference_001_001.wav"
    spex_reference = args.spex_reference or current_ref
    if file_sha256(current_ref) != file_sha256(spex_reference):
        raise ValueError("SpEx+ reference differs from the current batch first reference")
    pvad_report = args.pvad_report.resolve(strict=True)
    pvad_payload = json.loads(pvad_report.read_text(encoding="utf-8"))
    vf_dir = args.voicefilter_dir.resolve(strict=True)
    vf_report_path = vf_dir / "report.json"
    vf_payload = json.loads(vf_report_path.read_text(encoding="utf-8"))
    if pvad_payload.get("reference_paths") != [str(current_ref.resolve())]:
        raise ValueError("Personal VAD was not run with the current first reference")
    if vf_payload.get("reference_sha256") != file_sha256(current_ref):
        raise ValueError("VoiceFilter was not run with the current first reference")
    if vf_payload.get("model_weight_sha256") != file_sha256(VF_WEIGHT):
        raise ValueError("VoiceFilter report does not match its current checkpoint")
    if vf_payload.get("output_sha256") != file_sha256(vf_dir / "FULL_EPISODE_UNVERIFIED.wav"):
        raise ValueError("VoiceFilter output does not match its report")
    pvad_calibration = PVAD_WEIGHT.parent / "target_fsm.json"
    camp_path = (ROOT / "models" / "modelscope_cache" / "iic"
                 / "speech_campplus_sv_zh-cn_16k-common" / "campplus_cn_common.bin")
    if pvad_payload.get("checkpoint_sha256") != file_sha256(PVAD_WEIGHT):
        raise ValueError("Personal VAD report does not match its checkpoint")
    if pvad_payload.get("camplus_sha256") != file_sha256(camp_path):
        raise ValueError("Personal VAD report does not match its CAM++ encoder")
    # The upstream checksum covers LF-normalized JSON on Windows checkouts.
    import hashlib
    normalized_calibration = pvad_calibration.read_bytes().replace(b"\r\n", b"\n")
    if pvad_payload.get("calibration_sha256") != hashlib.sha256(normalized_calibration).hexdigest():
        raise ValueError("Personal VAD report does not match its calibration")
    cache_alignment = verify_cached_audio_alignment(
        current_16k,
        Path(pvad_payload["episode"]["path"]),
        SPEX_ROOT / "episode1_original_8k.wav",
    )
    pvad = PersonalVADCoverage.from_report(pvad_report, current_16k)
    vf = WaveformComparison(
        raw, vf_dir / "FULL_EPISODE_UNVERIFIED.wav", vf_report_path,
    )
    spex = SpExPlusEvaluator.from_paths(
        SPEX_ROOT / "episode1_original_8k.wav", spex_reference, args.device,
    )
    accepted = sorted((row for row in manifest["sentences"] if row["accepted"]),
                      key=lambda row: (row["start"], row["end"]))
    possible = sorted((row for row in manifest["sentences"]
                       if eligible_proposal(row, accepted, manifest["options"]["min_output_seconds"])),
                      key=lambda row: (row["start"], row["end"]))
    candidates = [*accepted, *possible]
    print(f"Candidate ledger: {len(accepted)} base accepted, {len(possible)} eligible rejects", flush=True)
    evidence = []
    for index, row in enumerate(candidates, start=1):
        start, end = float(row["start"]), float(row["end"])
        record = {
            "start": start, "end": end, "base_accepted": bool(row["accepted"]),
            "base_reject_reason": row.get("reject_reason", ""),
            "pvad": pvad.evidence(start, end),
            "voicefilter": vf.evidence(start, end),
            "spexplus": spex.evidence(start, end),
        }
        evidence.append(record)
        if index % 20 == 0 or index == len(candidates):
            print(f"External-model candidates {index}/{len(candidates)}", flush=True)
    extra, policy = choose_model_proposals([row for row in evidence if not row["base_accepted"]])
    extra = sorted(extra, key=lambda row: (row["start"], row["end"]))
    # An external model never changes the actual source waveform.  STT runs
    # only after the complete base sentence boundary has been selected.
    extra_text = _transcribe_new(stem, extra, args.device)
    base_lookup = {(row["start"], row["end"]): row for row in accepted}
    text_lookup = {(row["start"], row["end"]): text for row, text in zip(extra, extra_text)}
    review = sorted([*evidence[:len(accepted)], *extra], key=lambda row: (row["start"], row["end"]))
    destination.mkdir(parents=True)
    audio_dir, text_dir = destination / "audio", destination / "text"
    audio_dir.mkdir()
    text_dir.mkdir()
    exports = []
    for index, row in enumerate(review, start=1):
        key = (row["start"], row["end"])
        text = (base_lookup[key].get("text") or base_lookup[key].get("whisper_text")
                if row["base_accepted"] else text_lookup[key])
        exports.append(export_review_clip(stem, destination, index, row, text))
    weights = {"voicefilter": VF_WEIGHT, "personal_vad_2": PVAD_WEIGHT,
               "spexplus": SPEX_WEIGHT}
    external_inputs = {
        "voicefilter_source": OLD_WORK / "target_normalized.wav",
        "voicefilter_output": VF_ROOT / "FULL_EPISODE_UNVERIFIED.wav",
        "voicefilter_report": VF_ROOT / "report.json",
        "voicefilter_config": VF_WEIGHT.parent / "config.json",
        "personal_vad_source": PVAD_ROOT / "episode1_original_16k.wav",
        "personal_vad_report": PVAD_ROOT / "report_full_episode_single_ref_0.json",
        "personal_vad_calibration": PVAD_WEIGHT.parent / "target_fsm.json",
        "personal_vad_camplus": ROOT / "models" / "modelscope_cache" / "iic"
                                / "speech_campplus_sv_zh-cn_16k-common" / "campplus_cn_common.bin",
        "spexplus_source": SPEX_ROOT / "episode1_original_8k.wav",
        "spexplus_config": SPEX_WEIGHT.parent / "config.yaml",
        "external_reference": args.spex_reference,
    }
    stt_snapshot = whisper_snapshot()
    stt_assets = {path.name: file_sha256(path) for path in sorted(stt_snapshot.iterdir())
                  if path.is_file()}
    freeze = {
        "status": "FROZEN_UNVERIFIED_EXPERIMENT_NOT_TRAINING_AUDIO",
        "batch_manifest": str((batch_dir / "batch_manifest.json").resolve()),
        "batch_manifest_sha256": file_sha256(batch_dir / "batch_manifest.json"),
        "source_manifest": str((source_dir / "manifest.json").resolve()),
        "source_manifest_sha256": file_sha256(source_dir / "manifest.json"),
        "source_sha256": file_sha256(raw), "uvr_stem_sha256": file_sha256(stem),
        "external_cache_alignment": cache_alignment,
        "model_weights_sha256": {key: file_sha256(path) for key, path in weights.items()},
        "external_inputs_sha256": {key: file_sha256(path) for key, path in external_inputs.items()},
        "whisper_snapshot": str(stt_snapshot), "whisper_files_sha256": stt_assets,
        "adapter_sha256": file_sha256(ROOT / "evaluation" / "full_model_evidence.py"),
        "runner_sha256": file_sha256(Path(__file__)),
        "policy": policy,
        "limitations": [
            "All three external models have known false-positive/false-negative cases.",
            "VoiceFilter and SpEx+ response is not a calibrated speaker probability.",
            "Personal VAD may mark fragments; only complete base candidates are proposed.",
            "Extra proposals lack independent final speaker/purity verification.",
            "Do not train a voice clone from this listening package without human audit.",
            "TS-SEP, WeSep and old Personal VAD have no runnable local audio-only weights.",
        ],
        "counts": {"base_accepted": len(accepted), "eligible_rejects": len(possible),
                   "three_model_proposals": len(extra), "listen_files": len(exports)},
        "exports": exports, "evidence": evidence,
    }
    freeze_path = destination / "freeze_manifest.json"
    freeze_path.write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                zipped.write(path, path.relative_to(destination))
    print(f"FROZEN_PREVIEW={destination}", flush=True)
    print(f"FROZEN_ARCHIVE={archive}", flush=True)
    print(f"FREEZE_MANIFEST={freeze_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
