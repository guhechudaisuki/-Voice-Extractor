"""Re-evaluate saved candidates against the global models without separation or STT."""
from dataclasses import fields
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import time


def _model_versions():
    from extractor.model_store import file_sha256
    from launcher.corpus_worker import global_model_paths
    return {name: file_sha256(path) for name, path in global_model_paths().items() if path.is_file()}


def manifest_path(batch):
    path = Path(batch) / 'batch_manifest.json'
    return path if path.is_file() else Path(batch) / 'manifest.json'


def apply_rescore(batch, manifest):
    from extractor.model_store import file_sha256
    path = Path(batch) / 'role_rescore.json'
    if not path.is_file():
        return manifest
    data = json.loads(path.read_text(encoding='utf-8'))
    # Old overlays were bound to private heads and cannot describe global decisions.
    if data.get('schema') != 2 or data.get('model_scope') != 'global':
        return manifest
    if data.get('manifest_sha256') != file_sha256(manifest_path(batch)):
        return manifest
    if _model_versions() != data.get('models'):
        return manifest
    rows = manifest.get('sentences', [])
    for update in data.get('updates', []):
        index = int(update['ordinal']) - 1
        if not 0 <= index < len(rows):
            continue
        row = rows[index]
        if [row.get('start'), row.get('end'), row.get('audio_file')] != update['identity']:
            continue
        row.update(accepted=update['accepted'], reject_reason=update['reject_reason'])
        row['diagnostics'] = copy.deepcopy(update['diagnostics'])
    return manifest


def _publish(path, data):
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix='.rescore.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _classifier(path, wavlm, references):
    import torch
    if not path.is_file():
        return None
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('calibration_status') not in {'passed', 'reinforced'}:
        raise ValueError('全局分类器未通过校准，不能用于重评')
    # Legacy absolute-identity classifiers must not veto unrelated references.
    if payload.get('scoring') != 'reference_cosine_v1' or payload.get('reject_threshold') is None:
        return None
    from extractor.local_identity_classifier import reference_conditioned_classifier
    wavlm._load()
    model = reference_conditioned_classifier(wavlm.encoder, wavlm.processor, wavlm.device, references)
    return model, payload, path


def rescore_batch(batch, *, progress=None):
    if getattr(sys, 'frozen', False):
        from launcher.corpus_worker import invoke
        return invoke('rescore', {'batch': str(Path(batch).resolve())})
    from extractor.model_store import reinforcement_lock
    from launcher.corpus_worker import global_model_paths
    with reinforcement_lock(global_model_paths()['speaker']):
        return _rescore_batch(Path(batch), progress or (lambda _: None))


def _rescore_batch(batch, progress):
    import torch
    from extractor.config import CACHE_ROOT, SV_MODEL, CAMPLUS_MODEL, WESPEAKER_MODEL, ASSET_ROOT
    from extractor.model_store import file_sha256
    from extractor.cached_role_encoder import CachedEncoder, CachedWavLM
    from extractor.speaker import SpeakerVerifier, CAMPlusVerifier, WeSpeakerVerifier
    from extractor.nextgen.anime_embedding import AnimeSpeakerOnnx, _MODEL_FILES
    from extractor.nextgen.cluster_runtime import ClusterAnalyzer
    from extractor.nextgen.cluster_gate import apply_cluster_gate, HARD_VETOES
    from extractor.extraction_guards import VocalTractGeometryGuard
    from extractor.training_context import load_training_context
    from extractor.types import CandidateSentence
    from extractor.audio import load_mono
    from extractor.local_identity_classifier import audit_candidates
    from launcher.review_store import read_review
    from launcher.training_material import reviewable
    from launcher.corpus_worker import global_model_paths
    started = time.perf_counter()
    path = manifest_path(batch)
    original = json.loads(path.read_text(encoding='utf-8'))
    manifest = copy.deepcopy(original)
    reviews = read_review(batch).get('reviews', {})
    groups = {}
    for ordinal, row, clip, relative in reviewable(batch, manifest):
        directory = clip.parent
        while directory != batch and not (directory / '_training/context.json').is_file():
            directory = directory.parent
        groups.setdefault(directory, []).append((ordinal, row))
    if not groups:
        raise ValueError('批次没有可重评的音频候选')
    updates, cache_misses, encoder_loads = [], 0, 0
    paths = global_model_paths()
    versions = _model_versions()
    if 'speaker' not in versions:
        raise ValueError('未找到全局声纹模型，无法重评')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for directory, entries in groups.items():
        context = load_training_context(directory)
        report = json.loads((directory / 'cluster_report.json').read_text(encoding='utf-8'))
        provenance = report.get('provenance') or {}
        inputs = provenance.get('inputs') or []
        stem = directory / '_masters/target_vocals.wav'
        if not stem.is_file():
            stem = Path(inputs[0]['path']) if inputs else Path('')
        if not stem.is_file() or (inputs and file_sha256(stem) != inputs[0]['sha256']):
            raise ValueError('本批次的原始干声母带已清理或改变，无法安全重评；已导出音频和标注保持不变')
        stage = json.loads((directory / 'stage_audit.json').read_text(encoding='utf-8'))['stages']
        spans = stage.get('speaker_turns', {}).get('spans')
        if not spans:
            raise ValueError('此旧批次缺少说话回合时间轴，无法重评')
        progress('读取冻结特征；旧批次缺失项会从已有干声建立缓存，不运行 UVR5/STT')
        encoders = [CachedEncoder(lambda: SpeakerVerifier(device=device), SV_MODEL, device=device),
                    CachedEncoder(lambda: CAMPlusVerifier(device=device), CAMPLUS_MODEL, device=device)]
        anime = [CachedEncoder(lambda v=v: AnimeSpeakerOnnx(ASSET_ROOT/'models', v, device=device),
                    ASSET_ROOT/'models'/_MODEL_FILES[v], device='CUDAExecutionProvider' if device=='cuda' else 'CPUExecutionProvider',
                    implementation=Path(__file__).parents[1]/'extractor/nextgen/anime_embedding.py', anime=True)
                 for v in ('char', 'va')]
        wavlm = CachedWavLM(paths['speaker'].parent, CACHE_ROOT, None, device)
        fourth = CachedEncoder(lambda: WeSpeakerVerifier(device=device), WESPEAKER_MODEL,
            device='CUDAExecutionProvider' if device=='cuda' else 'CPUExecutionProvider', public_method=True)
        analyzer = None
        probes = None
        try:
            analyzer = ClusterAnalyzer(stem, context['reference_paths'], provenance.get('exclusion_groups', ()),
                device=device, encoders=tuple(encoders), anime_encoders=tuple(anime), tertiary_encoder=wavlm, quaternary_encoder=fourth,
                geometry_guard=VocalTractGeometryGuard(context['reference_paths']), progress=progress)
            from extractor.nextgen.rescore_acoustics import AcousticProbeCache
            probes = AcousticProbeCache(analyzer, CACHE_ROOT)
            blocked = list((manifest.get('options') or {}).get('user_excluded_spans', []))
            for name in ('singing_blocked_islands', 'overlap_blocked_islands'):
                blocked.extend(stage.get(name, {}).get('spans', []))
            candidates = []
            for ordinal, row in entries:
                values = {f.name: copy.deepcopy(row[f.name]) for f in fields(CandidateSentence) if f.name in row}
                values.setdefault('whisper_text', '')
                candidate = CandidateSentence(**values)
                candidate.diagnostics['rescore_ordinal'] = ordinal
                # Persisted vetoes belong to the previous model, not this run.
                candidate.diagnostics.pop('local_classifier_audit', None)
                if '局部声学分类器' in candidate.reject_reason:
                    candidate.reject_reason = '声纹匹配不足'
                candidates.append(candidate)
                if any(candidate.diagnostics.get(k) for k in (*HARD_VETOES, 'pitch_register_mismatch')):
                    blocked.append([candidate.start, candidate.end])
            analyzer.temporal_blocked_spans = tuple(blocked)
            classifier = _classifier(paths['classifier'], wavlm, context['reference_paths'])
            if classifier and classifier[1].get('reject_threshold') is not None:
                model, payload, checkpoint = classifier
                def score(wave):
                    with torch.no_grad():
                        return float(model(torch.from_numpy(wavlm.pooled(wave)[1]).to(wavlm.device)).item())
                temporary, withdrawn = list(candidates), []
                audit_candidates(temporary, withdrawn, load_mono(stem, 16000), score,
                    payload['reject_threshold'], model_sha256=file_sha256(checkpoint))
            turns, result, local = analyzer.analyze(spans, require_dominant=False)
            accepted, rejected = [], []
            for candidate in candidates:
                (accepted if candidate.accepted else rejected).append(candidate)
            apply_cluster_gate(accepted, rejected, turns, result, mode='enforce', local=local, blocked_spans=blocked)
            analyzer.wespeaker_arbitrate(accepted, rejected)
            exact = {(c.start, c.end): c for c in [*rejected, *accepted]}
            by_ordinal = {c.diagnostics.get('rescore_ordinal'): c for c in candidates}
            for candidate in [*rejected, *accepted]:
                ordinal = candidate.diagnostics.get('rescore_ordinal')
                original_candidate = by_ordinal.get(ordinal)
                if original_candidate is None or (candidate.start, candidate.end) == (original_candidate.start, original_candidate.end):
                    by_ordinal[ordinal] = candidate
            for ordinal, row in entries:
                # Manual annotations are authoritative. No review label is inferred.
                if reviews.get(str(ordinal), {}).get('label'):
                    continue
                candidate = by_ordinal.get(ordinal) or exact.get((row['start'], row['end']))
                if candidate is None:
                    continue
                diagnostics = copy.deepcopy(candidate.diagnostics)
                # Only the complete saved clip can be promoted without rewriting audio.
                is_accepted = bool(candidate.accepted) and (candidate.start, candidate.end) == (row['start'], row['end'])
                reason = candidate.reject_reason if not candidate.accepted or is_accepted else '重评仅确认部分区间，原片段需复核'
                promoted = is_accepted and not row.get('accepted', False)
                demoted = bool(row.get('accepted')) and not is_accepted
                diagnostics['role_rescore'] = {'model_scope': 'global', 'promoted': promoted,
                                              'demoted': demoted, 'models': versions}
                if is_accepted:
                    diagnostics.update(cluster_tier='A', cluster_removed=False)
                else:
                    diagnostics.update(cluster_tier='B', cluster_removed=True)
                updates.append(dict(ordinal=ordinal, identity=[row['start'], row['end'], row.get('audio_file')],
                    accepted=is_accepted, reject_reason=reason, diagnostics=diagnostics))
            cache_misses += sum(e.feature_cache.misses for e in (*encoders, *anime, fourth)) + wavlm.cache.misses
            encoder_loads += sum(e.model is not None for e in (*encoders, *anime, fourth)) + int(wavlm.encoder is not None)
        finally:
            try:
                if probes is not None:
                    probes.save()
            finally:
                if analyzer is not None:
                    analyzer.close()
                for encoder in (*encoders, *anime, fourth, wavlm):
                    encoder.close()
    data = dict(schema=2, model_scope='global', status='completed', manifest_sha256=file_sha256(path), updates=updates,
                promoted=sum(u['diagnostics']['role_rescore']['promoted'] for u in updates),
                demoted=sum(u['diagnostics']['role_rescore']['demoted'] for u in updates),
                evaluated=len(updates), models=versions,
                cache_misses=cache_misses, encoder_loads=encoder_loads, seconds=round(time.perf_counter()-started, 3),
                scope='saved_acoustic_candidates_no_uvr_no_stt')
    _publish(batch / 'role_rescore.json', data)
    return {k: v for k, v in data.items() if k != 'updates'}
