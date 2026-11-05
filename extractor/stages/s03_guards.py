"""s03 guards responsibilities; composed by ExtractionPipeline."""
from __future__ import annotations

from .. import pipeline_common as common


class S03GuardsMethods:
    @staticmethod
    def _partition_tainted_spans(
        spans: list[common.TimeSpan],
        tainted: list[common.TimeSpan],
        *,
        minimum_overlap_seconds: float,
        minimum_fraction: float = 0.0,
        waveform: common.torch.Tensor | None = None,
    ) -> tuple[list[common.TimeSpan], list[common.TimeSpan]]:
        """Carve confirmed pollution and keep clean remnants of at least 350 ms.

        Detector evidence is already thresholded. Legacy overlap limits remain
        accepted by this interface but cannot license contaminated samples.
        """
        return common.carve_tainted_spans(spans, tainted, waveform)

    @staticmethod
    def _snap_target_spans_to_speech(
        target_spans: list[common.TimeSpan],
        speech_spans: list[common.TimeSpan],
        minimum_overlap_seconds: float = 0.10,
    ) -> list[common.TimeSpan]:
        """Return complete speaker units touched by target-locator windows.

        Each unit stays independent.  Joining the first and last touched unit
        here can silently cross a confirmed speaker boundary and export a
        sequential two-speaker conversation as one target sentence.
        """

        snapped: dict[tuple[float, float], common.TimeSpan] = {}
        ordered_speech = sorted(speech_spans, key=lambda item: (item.start, item.end))
        for target in target_spans:
            for speech in ordered_speech:
                overlap = min(target.end, speech.end) - max(target.start, speech.start)
                if overlap >= min(minimum_overlap_seconds, speech.duration * 0.25):
                    snapped[(round(speech.start, 5), round(speech.end, 5))] = speech
        return sorted(snapped.values(), key=lambda item: (item.start, item.end))

    def _split_micro_window_changes(self, spans, verifier, profile, waveform, progress):
        output, cuts = [], []
        splitter = common.LocalSpeakerTurnSplitter(
            verifier.primary, secondary=verifier.secondary,
            secondary_factory=verifier._boundary_secondary_factory)

        def embed_windows(windows):
            return verifier.primary._embeddings_from_waveforms([
                common.torch.as_tensor(window, dtype=common.torch.float32) for window in windows])

        for index, span in enumerate(spans):
            points = common.detect_speaker_change_valleys(
                waveform, span.start, span.end, embed_windows=embed_windows,
                target_centroid=profile.primary.centroid)
            points = sorted(set(points) | {point for point in getattr(self,'_candidate_cut_anchors',())
                if span.start < point < span.end})
            points = splitter.confirm_boundary_proposals(waveform, span, points)
            edges = [span.start, *points, span.end]
            output.extend(common.TimeSpan(start, end) for start, end in zip(edges, edges[1:]))
            cuts.extend(points)
            progress((index + 1) / max(1, len(spans)), f"微窗换人检测 {index + 1}/{len(spans)}")
        self._speaker_change_points = tuple(sorted(cuts))
        return output

    def _enforce_atomic_boundaries(self, accepted, rejected, verifier, profile, waveform, exclusions):
        """Recheck recovery proposals that cross an original sentence or speaker cut."""
        output = []
        threshold = max(self.options.speaker_threshold, profile.primary.suggested_threshold)
        for candidate in accepted:
            parent = common.TimeSpan(candidate.start, candidate.end)
            from ..sentence_policy import positive_conflict
            local = (self._cluster_report or {}).get('local_windows', {})
            rows = [r for group in local.values() for r in group]
            confirmed = (self._stage_audit or {}).get('sentence_first', {}).get('confirmed_change_points', ())
            conflict = positive_conflict(rows, parent.start, parent.end, confirmed)
            parts = common.subtitle_atomic_spans(self._subtitle_guide, [parent]) if conflict else [parent]
            # A proposed micro cut is not new identity evidence. The final
            # multichannel trajectory has already checked this exact interval;
            # rechecking its short phonemes can wrongly revoke a whole sentence.
            blocked = [*self._raw_blocked_spans,
                       *(common.TimeSpan(*span) for span in self.options.user_excluded_spans)]
            if (not conflict and parts == [parent] and common.has_exact_target_certificate(candidate)
                    and not any(span.start < parent.end and span.end > parent.start
                                for span in blocked)):
                output.append(candidate)
                continue
            separated = []
            for part in parts:
                edges = [part.start, *(point for point in self._speaker_change_points
                                      if part.start + 1e-6 < point < part.end - 1e-6), part.end]
                separated.extend(common.TimeSpan(start, end) for start, end in zip(edges, edges[1:]))
            parts, _removed = common.carve_tainted_spans(
                separated, blocked, waveform,
                minimum_seconds=common.MIN_TARGET_SECONDS, padding=0.)
            if len(parts) == 1 and parts[0] == parent:
                output.append(candidate)
                continue
            if (not _removed and parts and all(self._identity_join_allowed(
                    left, right, verifier, profile, waveform)
                    for left, right in zip(parts, parts[1:]))):
                output.append(candidate)
                continue
            for part in parts:
                child = common.CandidateSentence(part.start, part.end, "")
                child.diagnostics["atomic_parent_span"] = [parent.start, parent.end]
                child.diagnostics["acoustic_parent_span"] = [part.start, part.end]
                if part.duration + 1e-6 < common.MIN_TARGET_SECONDS:
                    child.reject_reason = "讲话片段短于训练下限"
                    rejected.append(child)
                    continue
                match = self._verify_speaker_span(verifier, waveform, part, profile, threshold, audit_raw=True)
                exclusion = verifier.exclusion_audit(match, profile, exclusions) or {}
                self._apply_speaker_match(child, match, profile, threshold)
                child.diagnostics.update(exclusion)
                if match.accepted and not exclusion.get("excluded_role_rejected"):
                    child.accepted = True
                    child.diagnostics["atomic_identity_audit"] = {
                        "passed": True, "span": [part.start, part.end],
                        "speaker_match_mode": match.match_mode,
                        "exclusion_checked": True,
                    }
                    # This child has just passed on its own exact waveform.
                    # Preserve that authority for the next export-length gate;
                    # the parent's certificate cannot authorize this interval.
                    child.diagnostics["acoustic_identity_audit"] = {
                        "passed": True, "span": [part.start, part.end],
                        "method": "independently_verified_atomic_span",
                        "authority": "wavlm" if getattr(match, "tier", None) == "tertiary" else "dual",
                        "exclusion_checked": True,
                    }
                    child.diagnostics["exact_audio_verified"] = True
                    output.append(child)
                else:
                    child.reject_reason = "原子子句身份复核未通过"
                    rejected.append(child)
        accepted[:] = output

    def _pitch_diagnostics(self, waveform, span):
        if self._pitch_guard is None:
            return {}
        key = (span.start, span.end)
        if key not in self._pitch_span_cache:
            self._pitch_span_cache[key] = self._pitch_guard.evaluate(self._waveform_span(waveform, span))
        diagnostics = getattr(span, 'diagnostics', {}) or {}
        return common.VocalTractGeometryGuard.condition(self._pitch_span_cache[key], (
            diagnostics.get('excluded_primary_direct_margin'),
            diagnostics.get('excluded_secondary_direct_margin')), self.options.cluster_target_margin)

    def _filter_pitch_register(self, accepted, rejected, waveform) -> int:
        withheld = []
        for candidate in accepted:
            diagnostics = self._pitch_diagnostics(waveform, candidate)
            candidate.diagnostics.update(diagnostics)
            if diagnostics.get("pitch_register_mismatch"):
                candidate.accepted = False
                candidate.reject_reason = "声纹边缘证据与声道几何不一致"
                candidate.diagnostics.update(cluster_tier="B", cluster_reason=candidate.reject_reason)
                withheld.append(candidate)
        withheld_ids = {id(candidate) for candidate in withheld}
        accepted[:] = [candidate for candidate in accepted if id(candidate) not in withheld_ids]
        rejected.extend(withheld)
        return len(withheld)

    def _apply_user_marked_exclusions(
        self,
        accepted_turns: list[common.CandidateSentence],
        rejected: list[common.CandidateSentence],
        excluded_spans: common.Iterable[common.TimeSpan],
        verifier: common.DualSpeakerVerifier,
        profile: common.SpeakerMatchProfile,
        stem_waveform: common.torch.Tensor,
        exclusion_profiles: list[common.ExclusionSpeakerProfile],
        threshold: float,
    ) -> int:
        """Trim accepted turns at the spans the user marked as wrong.

        A turn overlapping a marked span keeps the part outside every mark
        when that remainder independently passes the formal dual-model
        verification and the exclusion audit; otherwise the whole turn is
        deleted.  The marked part itself is never exported.
        """
        from copy import deepcopy

        spans = [span for span in excluded_spans if span.end > span.start]
        if not spans:
            return 0
        removed = 0
        for candidate in list(accepted_turns):
            span = common.TimeSpan(candidate.start, candidate.end)
            if not any(
                min(span.end, mark.end) - max(span.start, mark.start) > 0.05
                for mark in spans
            ):
                continue
            pieces = [span]
            for mark in spans:
                nxt = []
                for piece in pieces:
                    if min(piece.end, mark.end) - max(piece.start, mark.start) <= 0.05:
                        nxt.append(piece)
                        continue
                    if piece.start < mark.start - 0.05:
                        nxt.append(common.TimeSpan(piece.start, mark.start))
                    if mark.end < piece.end - 0.05:
                        nxt.append(common.TimeSpan(mark.end, piece.end))
                pieces = nxt
            for piece in pieces:
                if piece.duration + 1e-6 < self.options.min_output_seconds:
                    continue
                match = self._verify_speaker_span(
                    verifier, stem_waveform, piece, profile, threshold,
                    audit_raw=True,
                )
                exclusion = verifier.exclusion_audit(
                    match, profile, exclusion_profiles,
                ) or {}
                formal_ok = (
                    match.accepted
                    and not exclusion.get("excluded_role_rejected")
                )
                if not formal_ok:
                    continue
                trimmed = common.CandidateSentence(piece.start, piece.end, "")
                self._apply_speaker_match(trimmed, match, profile, threshold)
                trimmed.diagnostics.update(exclusion)
                trimmed.diagnostics["user_marked_trim"] = {
                    "original_span": [span.start, span.end],
                    "kept_span": [piece.start, piece.end],
                    "verified": formal_ok,
                    "provenance": {"diagnostics": deepcopy(candidate.diagnostics)},
                }
                accepted_turns.append(trimmed)
            accepted_turns.remove(candidate)
            candidate.accepted = False
            candidate.audio_file = ""
            candidate.text_file = ""
            candidate.video_file = ""
            candidate.reject_reason = "用户标记排除区间，已按标记删除"
            candidate.diagnostics["user_marked_exclusion"] = True
            rejected.append(candidate)
            removed += 1
        return removed
