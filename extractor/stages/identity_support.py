"""identity support responsibilities; composed by ExtractionPipeline."""
from __future__ import annotations

from .. import pipeline_common as common


class IdentitySupportMethods:
    @staticmethod
    def _apply_speaker_match(
        candidate: common.CandidateSentence,
        match,
        profile: common.SpeakerMatchProfile,
        threshold: float,
    ) -> None:
        decision = match.primary
        candidate.speaker_score = decision.score
        candidate.window_min_score = decision.window_min_score
        candidate.window_p20_score = decision.window_p20_score
        candidate.speaker_vote_ratio = decision.vote_ratio
        candidate.window_vote_ratio = decision.window_vote_ratio
        candidate.speaker_threshold = threshold
        candidate.diagnostics.update(
            {
                "speaker_vote_ratio": decision.vote_ratio,
                "speaker_window_p20": decision.window_p20_score,
                "speaker_window_vote_ratio": decision.window_vote_ratio,
                "speaker_threshold": threshold,
                "speaker_reference_floor": profile.primary.reference_floor,
                "speaker_calibration_base": profile.primary.calibration_base,
                "speaker_match_mode": match.match_mode,
                "speaker_tier": match.tier,
                "eres_reference_median": decision.reference_median_score,
                "eres_reference_max": decision.reference_max_score,
                "eres_reference_spread": decision.reference_spread,
                "paired_reference_median": match.paired_reference_median,
            }
        )
        candidate.diagnostics.update(match.diagnostics)
        if match.secondary is not None:
            candidate.diagnostics.update(
                {
                    "camplus_score": match.secondary.score,
                    "camplus_window_p20": match.secondary.window_p20_score,
                    "camplus_window_vote_ratio": match.secondary.window_vote_ratio,
                    "camplus_reference_vote_ratio": match.secondary.vote_ratio,
                    "camplus_reference_median": match.secondary.reference_median_score,
                    "camplus_reference_max": match.secondary.reference_max_score,
                    "camplus_reference_spread": match.secondary.reference_spread,
                }
            )

    def _verify_speaker_span(
        self,
        verifier: common.DualSpeakerVerifier,
        waveform: common.torch.Tensor,
        span: common.TimeSpan,
        profile: common.SpeakerMatchProfile,
        threshold: float,
        *,
        audit_raw: bool = False,
    ) -> common.SpeakerMatchDecision:
        stem_part = self._waveform_span(waveform, span)
        window_seconds = min(1.8, max(1.0, span.duration))
        hop_seconds = min(0.9, max(0.5, span.duration / 2))
        raw_source = self._raw_target_waveform
        raw_profile_ready = (
            profile.raw_primary is not None and profile.raw_secondary is not None
        )
        if (
            (not self.options.use_raw_speaker_rescue and not audit_raw)
            or raw_source is None
            or not raw_profile_ready
        ):
            match = verifier.verify_waveform(
                stem_part,
                profile,
                threshold,
                duration=span.duration,
                window_seconds=window_seconds,
                hop_seconds=hop_seconds,
            )
        else:
            match = self._verify_dual_span(verifier, waveform, span, profile, threshold, audit_raw)
        pitch = common.VocalTractGeometryGuard.condition(self._pitch_diagnostics(waveform, span), (
            match.diagnostics.get('excluded_primary_direct_margin'),
            match.diagnostics.get('excluded_secondary_direct_margin')), self.options.cluster_target_margin)
        match.diagnostics.update(pitch)
        if pitch.get("pitch_register_mismatch"):
            match.accepted = False
            match.tier = "rejected"
            match.merge_only = False
        return match

    def _verify_dual_span(self, verifier, waveform, span, profile, threshold, audit_raw):
        audit_raw = audit_raw or any(min(span.end, s.end) > max(span.start, s.start)
                                    for s in getattr(self, '_raw_vad_recovered', ()))
        stem_part = self._waveform_span(waveform, span)
        raw_part = self._waveform_span(self._raw_target_waveform, span)
        # A raw rescue may never cross an utterance rejected by the singing or
        # overlap gates.  Small boundary rounding differences are tolerated,
        # but any real intersection disables the rescue route.
        clean_gate = not any(
            min(span.end, blocked.end) - max(span.start, blocked.start) > 0.02
            for blocked in self._raw_blocked_spans
        )
        return verifier.verify_dual_channel_waveform(
            stem_part,
            raw_part,
            profile,
            threshold,
            duration=span.duration,
            window_seconds=min(1.8, max(1.0, span.duration)),
            hop_seconds=min(0.9, max(0.5, span.duration / 2)),
            clean_gate=clean_gate,
            allow_raw_rescue=self.options.use_raw_speaker_rescue,
            audit_raw_on_uvr_accept=audit_raw,
            audit_raw_channel=audit_raw,
        )

    def _global_classifier_path(self):
        from ..config import CLASSIFIER_V2_MODEL
        return common.Path(CLASSIFIER_V2_MODEL)

    def _invalidate_changed_tail_scorer(self) -> None:
        """Refresh between extraction jobs; one job keeps one set of weights."""
        classifier_path = self._global_classifier_path()
        stat = classifier_path.stat() if classifier_path.is_file() else None
        speaker_path = common.WAVLM_SV_MODEL / 'pytorch_model.bin'
        speaker_stat = speaker_path.stat() if speaker_path.is_file() else None
        stamp = ((str(classifier_path), stat.st_mtime_ns, stat.st_size) if stat else None,
                 (speaker_stat.st_mtime_ns, speaker_stat.st_size) if speaker_stat else None)
        if getattr(self, '_tail_scorer_stamp', None) != stamp:
            self._tail_scorer_cache = None
            self._local_identity_reject_threshold = None
            self._local_identity_model_sha256 = ''
        self._tail_scorer_stamp = stamp

    def _ensure_tail_scorer(self) -> tuple | None:
        """Load the optional identity classifier and keep it stable for this job."""
        cached = getattr(self, "_tail_scorer_cache", None)
        if cached is not None:
            return cached if cached != "missing" else None
        self._invalidate_changed_tail_scorer()
        classifier_path = self._global_classifier_path()
        self._local_identity_reject_threshold = None
        self._local_identity_model_sha256 = ''
        encoder_dir = common.WAVLM_SV_MODEL
        if not classifier_path.is_file() or not encoder_dir.is_dir():
            self._tail_scorer_cache = "missing"
            return None
        checkpoint_bytes = classifier_path.read_bytes()
        payload = common.torch.load(
            common.io.BytesIO(checkpoint_bytes),
            map_location="cpu",
            weights_only=True,
        )
        import hashlib
        if payload.get('calibration_status') not in {'passed', 'reinforced'}:
            self._tail_scorer_cache = 'missing'
            return None
        universal = payload.get('scoring') in {'reference_cosine_v1', 'set_relational_v2'}
        from ..model_store import active_speaker_path, file_sha256
        from launcher.reinforcement_history import read_history
        projection_path = active_speaker_path(encoder_dir)
        bound_digest = payload.get('projection_sha256')
        rank_generation = (projection_path.is_file() and not bound_digest and
            (read_history(projection_path).get('validation_gate') or {}).get('method') == 'rank_then_global_calibration')
        if ((bound_digest and bound_digest != file_sha256(projection_path))
                or (rank_generation and not bound_digest)):
            common.LOGGER.warning('全局投影与校准版本不匹配，停用局部校准，保留独立声纹门控')
            self._tail_scorer_cache = 'missing'
            return None
        # Old absolute-speaker logits cannot veto an unrelated new reference.
        self._local_identity_reject_threshold = payload.get('reject_threshold') if universal else None
        self._local_identity_model_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
        reference_paths = getattr(self, '_reference_audio_paths', ())
        if not reference_paths:
            self._tail_scorer_cache = 'missing'
            return None
        from ..local_identity_engine import LocalIdentityEngine
        device = common.torch.device(self.device)
        encoder = common.pooled_encoder(LocalIdentityEngine,self.device,
            active_speaker_path(encoder_dir),directory=encoder_dir)
        if payload.get('scoring') == 'set_relational_v2':
            clf = encoder.create_classifier(reference_paths,
                calibration=payload['calibration'], relational=True)
        else:
            clf = encoder.create_classifier(reference_paths)
        threshold = float(payload['threshold']) if universal else 1.0
        self._tail_scorer_cache = (encoder, None, clf, threshold, device)
        return self._tail_scorer_cache

    def _score_local_window(self, models, waveform):
        from ..local_score_cache import LocalScoreCache

        cache = getattr(self, '_local_score_cache', None)
        if cache is None or cache.models is not models:
            cache = self._local_score_cache = LocalScoreCache(models)
        return cache(waveform)

    def _apply_cluster_identity(self, stem, references, negative_reference_groups,
                                verifier, profile, split_result, accepted_turns, rejected,
                                target, job_id, progress):
        """Audit failures are recorded; enforce failures must stop export."""
        if self.options.cluster_identity_mode == "off":
            return
        try:
            from ..nextgen.cluster_runtime import ClusterAnalyzer
            from ..nextgen.cluster_reporting import summarize
            from ..nextgen.cluster_gate import apply_cluster_gate, HARD_VETOES
            verifier._ensure_secondary(profile)
            tertiary = None
            if self.options.cluster_wavlm_channel:
                # An unavailable optional encoder leaves the dual-model gate in place.
                try:
                    tertiary, _ = verifier._tertiary_pair(profile)
                except Exception as exc:
                    common.LOGGER.warning("WavLM 第三通道加载失败，退回双通道：%s", exc)
            quaternary = None
            if self.options.cluster_wespeaker_arbitration:
                try:
                    quaternary, _ = verifier.quaternary_pair(profile)
                except Exception as exc:
                    common.LOGGER.warning("WeSpeaker 第四通道加载失败，跳过仲裁：%s", exc)
            raw_stem = target if self.options.cluster_raw_crosscheck else None
            analyzer = ClusterAnalyzer(
                stem, references, negative_reference_groups,
                device=self.device, cache_enabled=self.options.cluster_embedding_cache,
                encoders=(verifier.primary, verifier.secondary),
                progress=lambda message: progress(0.80, message),
                anime_encoders=verifier.anime_pair(),
                tertiary_encoder=tertiary,
                quaternary_encoder=quaternary,
                raw_stem=raw_stem,
                geometry_guard=self._pitch_guard,
            )
            if getattr(self, '_episode_waveform', None) is not None:
                analyzer.waveform = self._episode_waveform
            if raw_stem is not None and getattr(self, '_raw_target_waveform', None) is not None:
                analyzer.raw_waveform = self._raw_target_waveform
            try:
                spans = [(t.start, t.end) for t in split_result]
                settings = dict(margin=self.options.cluster_target_margin,
                    keep_narration=self.options.cluster_keep_narration,
                    require_dominant=self.options.cluster_require_dominant)
                blocked = (*self.options.user_excluded_spans,
                           *((s.start, s.end) for s in self._raw_blocked_spans))
                if self._pitch_guard is not None:
                    pitch_waveform = common.load_mono(stem, 16000)
                    pitch_blocks = [(span.start, span.end) for span in split_result
                                    if self._pitch_diagnostics(pitch_waveform, span).get("pitch_register_mismatch")]
                    blocked = (*blocked, *pitch_blocks)
                hard_blocked = [(c.start, c.end) for c in [*accepted_turns, *rejected]
                                if any(c.diagnostics.get(key)
                                       for key in (*HARD_VETOES, "pitch_register_mismatch"))]
                stages = (self._stage_audit or {}).get('stages', {})
                detected_blocks = [span for name in (
                    'pre_uvr_singing_mask', 'residual_singing_evidence', 'singing_blocked_islands',
                    'overlap_evidence', 'overlap_blocked_islands')
                    for span in stages.get(name, {}).get('spans', [])]
                analyzer.temporal_blocked_spans = (*blocked, *hard_blocked, *detected_blocks)
                analyzer.confirmed_change_points = self._speaker_change_points
                analyzer.whole_dual_verified_spans = {
                    (c.start, c.end) for c in accepted_turns
                    if c.diagnostics.get('exact_audio_verified')
                    and c.diagnostics.get('acoustic_identity_audit', {}).get('span') == [c.start, c.end]}
                if self.options.cluster_bootstrap:
                    from ..nextgen.bootstrap_training import analyze_with_bootstrap
                    # Training is stricter than export: even small detected
                    # overlaps must not become reusable identity evidence.
                    cluster_turns, cluster_result, local = analyze_with_bootstrap(
                        analyzer, spans, output_dir=self._job_paths(job_id)["output"] / "bootstrap",
                        blocked_spans=analyzer.temporal_blocked_spans,
                        max_rounds=self.options.cluster_bootstrap_max_rounds, **settings)
                else:
                    cluster_turns, cluster_result, local = analyzer.analyze(spans, **settings)
                cluster_result.diagnostics['confirmed_change_points'] = list(self._speaker_change_points)
                direct_selection = (self.options.cluster_identity_mode == 'enforce'
                    and not getattr(self.options, 'branch_validation', True))
                if direct_selection:
                    # The continuous decoder owns this track's selection. The
                    # explicit direct-export mode does not certify new ranges
                    # or inherit a parent candidate's identity/ASR diagnostics.
                    import math
                    selected = sorted(set(tuple(span) for span in
                        (cluster_result.diagnostics.get('accepted_spans') or ())))
                    if any(not all(math.isfinite(v) for v in (a,b)) or not 0 <= a < b
                           for a,b in selected):
                        raise ValueError('Continuous decoder returned invalid selected intervals')
                    previous = {(c.start,c.end) for c in accepted_turns}
                    accepted_turns[:] = [common.CandidateSentence(a,b,'',accepted=True,
                        diagnostics={'branch_validation':False, 'timeline_selection':{
                            'span':[a,b], 'method':'continuous_decoder',
                            'additional_validation':'disabled'}}) for a,b in selected]
                    selected_keys = set(selected)
                    rejected[:] = [c for c in rejected if (c.start,c.end) not in selected_keys]
                    for candidate in rejected:
                        candidate.accepted = False
                    changes = {'removed':len(previous - selected_keys),
                        'added':len(selected_keys - previous), 'borderline':0,
                        'selected':len(selected), 'branch_validation':False}
                else:
                    changes = apply_cluster_gate(
                        accepted_turns, rejected, cluster_turns, cluster_result,
                        mode=self.options.cluster_identity_mode,
                        min_seconds=self.options.min_output_seconds,
                        blocked_spans=blocked, local=local,
                        island_arbitration=analyzer.wespeaker_island_evidence,
                        keep_rejected=self.options.keep_rejected,
                    )
                # STT 精裁阶段要用已验证干净区间判定每个文本分片的归属。
                self._cluster_accepted_spans = list(
                    cluster_result.diagnostics.get("accepted_spans") or [])
                if self.options.cluster_identity_mode == "enforce" and not direct_selection:
                    # An exact multichannel decision has already consulted
                    # target, rivals and exclusions on this range. Evaluate it
                    # before optional single-encoder arbitration, which cannot
                    # overturn that certificate merely for a weak target cosine.
                    protected = []
                    arbitration_candidates = []
                    for candidate in accepted_turns:
                        certificate = analyzer.certify_exact_interval(candidate.start, candidate.end)
                        candidate.diagnostics['exact_multichannel_certificate'] = certificate
                        exact_pass = (certificate.get('span') == [candidate.start, candidate.end]
                            and certificate.get('passed') is True
                            and not certificate.get('positive_conflict') and not certificate.get('blocked')
                            and not any(candidate.diagnostics.get(key) for key in HARD_VETOES))
                        if exact_pass:
                            protected.append(candidate)
                            candidate.diagnostics['whole_sentence_protection'] = {
                                'span': [candidate.start, candidate.end], 'passed': True,
                                'method': 'exact_multichannel_and_no_positive_impostor'}
                        else:
                            arbitration_candidates.append(candidate)
                    arbitrated = analyzer.wespeaker_arbitrate(arbitration_candidates, rejected)
                    survivor_ids = {id(c) for c in [*protected, *arbitration_candidates]}
                    accepted_turns[:] = [c for c in accepted_turns if id(c) in survivor_ids]
                    if arbitrated:
                        changes = {**changes,
                                   "wespeaker_demoted": arbitrated}
                self._cluster_report = summarize(cluster_result, cluster_turns, [])
                self._cluster_report.update(
                    mode=self.options.cluster_identity_mode, changes=changes,
                    branch_validation=getattr(self.options, 'branch_validation', True),
                    local_windows=local, provenance=analyzer.provenance(),
                    meta={"source_stem": str(stem), "source_name": target.name, "job_id": job_id},
                )
                self._trace_spans("cluster_target_members", [
                    t for t in cluster_turns if cluster_result.membership[t.index]])
                if direct_selection:
                    progress(0.80, f'分流检验关闭：保留连续解码器选中的 {len(accepted_turns)} 个区间')
                else:
                    progress(0.80, f"簇级判决 {self.options.cluster_identity_mode}："
                             f"目标簇 {cluster_result.target_cluster_id}，"
                             f"通过 {sum(cluster_result.membership)} 轮，撤下 {changes['removed']} 段，"
                             f"身份未决 {changes.get('borderline', 0)} 段不导出")
            finally:
                analyzer.close()

        except Exception as exc:
            if self.options.cluster_identity_mode != "audit":
                raise
            self._cluster_report = {
                "mode": "audit", "reason": "audit_failed", "clusters": [],
                "target_cluster_id": None, "error": f"{type(exc).__name__}: {exc}",
                "meta": {"source_stem": str(stem), "source_name": target.name, "job_id": job_id},
            }
            progress(0.80, f"簇审计失败，继续原提取流程：{type(exc).__name__}: {exc}")
