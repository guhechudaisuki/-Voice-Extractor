"""cluster recovery responsibilities; composed by ExtractionPipeline."""
from __future__ import annotations

from .. import pipeline_common as common


class ClusterRecoveryMethods:
    @staticmethod
    def _promote_confirmed_recall_candidates(
        scored_turns: list[
            tuple[common.TimeSpan, common.CandidateSentence, common.SpeakerMatchDecision | None]
        ],
        accepted_turns: list[common.CandidateSentence],
        rejected: list[common.CandidateSentence],
    ) -> int:
        """Promote only recall turns independently confirmed window-by-window.

        This is intentionally separate from adaptive/local-seed recall.  It
        handles target utterances elsewhere in the episode, but requires the
        target locator to cover almost the entire turn and both independent
        speaker models to support every local window.  A similar-sounding
        whole-turn embedding alone can never pass this gate.
        """

        match_by_id = {
            id(candidate): match
            for _span, candidate, match in scored_turns
            if match is not None
        }
        retained: list[common.CandidateSentence] = []
        promoted = 0
        for candidate in rejected:
            if candidate.reject_reason != "声纹匹配不足":
                retained.append(candidate)
                continue
            match = match_by_id.get(id(candidate))
            secondary = match.secondary if match is not None else None
            if match is None or secondary is None or match.tier != "recall":
                retained.append(candidate)
                continue

            coverage = float(candidate.diagnostics.get("target_coverage", 0.0))
            independent_window_consensus = (
                candidate.window_vote_ratio >= 0.75
                and secondary.window_vote_ratio >= 0.75
                and candidate.window_p20_score >= 0.50
                and secondary.window_p20_score >= 0.46
            )
            direct_reference_evidence = (
                match.primary.score >= 0.58
                and secondary.score >= 0.50
                and match.primary.reference_median_score >= 0.50
                and secondary.reference_median_score >= 0.44
                and match.paired_reference_median >= 0.49
            )
            if (
                coverage < 0.90
                or not independent_window_consensus
                or not direct_reference_evidence
            ):
                retained.append(candidate)
                continue

            candidate.reject_reason = ""
            candidate.diagnostics.update(
                {
                    "confirmed_recall": True,
                    "confirmed_recall_coverage_floor": 0.90,
                    "confirmed_recall_window_vote_floor": 0.75,
                }
            )
            accepted_turns.append(candidate)
            promoted += 1

        rejected[:] = retained
        return promoted

    @staticmethod
    def _promote_global_target_cluster(
        scored_turns: list[
            tuple[common.TimeSpan, common.CandidateSentence, common.SpeakerMatchDecision | None]
        ],
        accepted_turns: list[common.CandidateSentence],
        rejected: list[common.CandidateSentence],
        extra_seed_items: common.Iterable[tuple[common.CandidateSentence, common.SpeakerMatchDecision]] = (),
    ) -> int:
        """Recover target turns belonging to the episode-wide target cluster.

        Reference recordings can differ from an episode in channel, emotion,
        and vocal effort.  Strong turns already found inside the episode are a
        cleaner domain-matched representation.  Both embedding spaces must
        independently agree with that cluster and the target-window locator
        must cover nearly the full turn.
        """

        accepted_ids = {id(candidate) for candidate in accepted_turns}
        match_by_id = {
            id(candidate): match
            for _span, candidate, match in scored_turns
            if match is not None
        }
        seed_items = [
            (candidate, match)
            for _span, candidate, match in scored_turns
            if id(candidate) in accepted_ids
            and match is not None
            and match.accepted
            and match.secondary is not None
            and match.primary.embedding is not None
            and match.secondary.embedding is not None
        ]
        seed_items.extend(
            (candidate, match)
            for candidate, match in extra_seed_items
            if match.accepted
            and match.secondary is not None
            and match.primary.embedding is not None
            and match.secondary.embedding is not None
        )
        if len(seed_items) < 3:
            return 0

        primary_centroid = common.torch.nn.functional.normalize(
            common.torch.stack([match.primary.embedding for _candidate, match in seed_items]).mean(dim=0),
            dim=0,
        )
        secondary_centroid = common.torch.nn.functional.normalize(
            common.torch.stack([match.secondary.embedding for _candidate, match in seed_items]).mean(dim=0),
            dim=0,
        )

        retained: list[common.CandidateSentence] = []
        promoted = 0
        for candidate in rejected:
            if candidate.reject_reason != "声纹匹配不足":
                retained.append(candidate)
                continue
            match = match_by_id.get(id(candidate))
            secondary = match.secondary if match is not None else None
            if (
                match is None
                or secondary is None
                or match.tier != "recall"
                or match.primary.embedding is None
                or secondary.embedding is None
            ):
                retained.append(candidate)
                continue

            coverage = float(candidate.diagnostics.get("target_coverage", 0.0))
            primary_cluster_score = float(match.primary.embedding @ primary_centroid)
            secondary_cluster_score = float(secondary.embedding @ secondary_centroid)
            cluster_consensus = (
                primary_cluster_score >= 0.78
                and secondary_cluster_score >= 0.72
            )
            direct_evidence = (
                match.primary.score >= 0.58
                and secondary.score >= 0.47
                and match.paired_reference_median >= 0.47
                and match.primary.reference_max_score >= 0.54
                and secondary.reference_max_score >= 0.44
            )
            if coverage < 0.85 or not cluster_consensus or not direct_evidence:
                retained.append(candidate)
                continue

            candidate.reject_reason = ""
            candidate.diagnostics.update(
                {
                    "global_target_cluster": True,
                    "global_target_primary_score": round(primary_cluster_score, 5),
                    "global_target_secondary_score": round(secondary_cluster_score, 5),
                }
            )
            accepted_turns.append(candidate)
            promoted += 1

        rejected[:] = retained
        return promoted

    def _build_contrastive_edge_candidates(
        self,
        scored_turns: list[
            tuple[common.TimeSpan, common.CandidateSentence, common.SpeakerMatchDecision | None]
        ],
        accepted_turns: list[common.CandidateSentence],
        verifier: common.DualSpeakerVerifier,
        profile: common.SpeakerMatchProfile,
        waveform: common.torch.Tensor,
        target_spans: list[common.TimeSpan],
        effective_threshold: float,
    ) -> list[tuple[common.TimeSpan, common.CandidateSentence, common.SpeakerMatchDecision]]:
        """Recover locator cores with a short, decisively non-target edge.

        The normal boundary scanner needs enough audio on both sides of a cut.
        A rapid reply shorter than that context can therefore remain attached
        to an otherwise complete target utterance.  This path trusts neither
        locator coverage nor a single verifier: the core and residual must be
        separated by one locator edge and show a strong three-model contrast.
        WeSpeaker performs a fourth independent check in the later consensus
        pass before any candidate can be exported.
        """

        if not target_spans:
            return []
        occupied = [common.TimeSpan(item.start, item.end) for item in accepted_turns]
        output: list[tuple[common.TimeSpan, common.CandidateSentence, common.SpeakerMatchDecision]] = []
        seen: set[tuple[float, float]] = set()
        for source_span, source, _source_match in scored_turns:
            source_coverage = float(
                source.diagnostics.get(
                    "target_coverage",
                    self._target_coverage(source_span, target_spans),
                )
            )
            if (
                source.reject_reason != "声纹匹配不足"
                or not 0.65 <= source_coverage <= 0.92
                or source_span.duration > min(12.0, self.options.max_sentence_seconds)
            ):
                continue

            clipped = sorted(
                (
                    common.TimeSpan(
                        max(source_span.start, target.start),
                        min(source_span.end, target.end),
                    )
                    for target in target_spans
                    if min(source_span.end, target.end)
                    - max(source_span.start, target.start)
                    >= 0.10
                ),
                key=lambda item: (item.start, item.end),
            )
            merged: list[common.TimeSpan] = []
            for item in clipped:
                if merged and item.start <= merged[-1].end + 0.05:
                    merged[-1] = common.TimeSpan(
                        merged[-1].start,
                        max(merged[-1].end, item.end),
                    )
                else:
                    merged.append(item)
            if len(merged) != 1:
                continue

            located = merged[0]
            shares_start = abs(located.start - source_span.start) <= 0.05
            shares_end = abs(located.end - source_span.end) <= 0.05
            if shares_start == shares_end:
                continue
            if shares_start:
                core_span = common.TimeSpan(source_span.start, located.end)
                residual_span = common.TimeSpan(located.end, source_span.end)
                edge = "tail"
            else:
                core_span = common.TimeSpan(located.start, source_span.end)
                residual_span = common.TimeSpan(source_span.start, located.start)
                edge = "head"
            if (
                core_span.duration < max(1.20, self.options.min_output_seconds)
                or not 0.25 <= residual_span.duration <= 0.90
                or self._target_coverage(core_span, target_spans) < 0.98
                or self._target_coverage(
                    core_span, verifier.tertiary_target_spans
                )
                < 0.35
                or any(
                    min(core_span.end, existing.end)
                    - max(core_span.start, existing.start)
                    > 0.10
                    for existing in occupied
                )
            ):
                continue
            key = (round(core_span.start, 5), round(core_span.end, 5))
            if key in seen:
                continue

            core_match = self._verify_speaker_span(
                verifier,
                waveform,
                core_span,
                profile,
                effective_threshold,
            )
            residual_match = self._verify_speaker_span(
                verifier,
                waveform,
                residual_span,
                profile,
                effective_threshold,
            )
            promoted_match = verifier.promote_contrastive_edge_with_tertiary(
                self._waveform_span(waveform, core_span),
                self._waveform_span(waveform, residual_span),
                profile,
                core_match,
                residual_match,
                core_span.duration,
                residual_span.duration,
            )
            if promoted_match is None:
                continue

            candidate = common.CandidateSentence(core_span.start, core_span.end, "")
            self._apply_speaker_match(
                candidate,
                promoted_match,
                profile,
                effective_threshold,
            )
            candidate.reject_reason = "声纹匹配不足"
            candidate.diagnostics.update(
                {
                    "contrastive_edge_trim": True,
                    "contrastive_edge": edge,
                    "original_turn_start": source_span.start,
                    "original_turn_end": source_span.end,
                    "contrastive_residual_start": residual_span.start,
                    "contrastive_residual_end": residual_span.end,
                    "target_coverage": round(
                        self._target_coverage(core_span, target_spans), 5
                    ),
                    "tertiary_target_coverage": round(
                        self._target_coverage(
                            core_span, verifier.tertiary_target_spans
                        ),
                        5,
                    ),
                }
            )
            for name in ("speaker_turn_index", "speech_block_index"):
                if name in source.diagnostics:
                    candidate.diagnostics[name] = source.diagnostics[name]
            output.append((core_span, candidate, promoted_match))
            seen.add(key)
        return output
