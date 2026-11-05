"""candidate recovery responsibilities; composed by ExtractionPipeline."""
from __future__ import annotations

from .. import pipeline_common as common


class CandidateRecoveryMethods:
    def _build_adaptive_speaker_profile(
        self,
        accepted_turns: list[common.CandidateSentence],
        verifier: common.DualSpeakerVerifier,
        profile: common.SpeakerMatchProfile,
        waveform: common.torch.Tensor,
        threshold: float,
    ) -> tuple[common.SpeakerMatchProfile, common.CAMPlusProfile, float, float] | None:
        """Add only strict, domain-matched target turns as temporary anchors."""

        anchors = [
            candidate
            for candidate in accepted_turns
            if 2.20 <= candidate.duration <= 15.0
            and (
                candidate.duration <= 4.50
                or candidate.diagnostics.get("post_target_silence_merge")
            )
            and candidate.diagnostics.get("speaker_tier")
            in {"short_strong", "strong", "balanced"}
            and float(candidate.diagnostics.get("target_coverage", 0.0)) >= 0.90
            and float(candidate.diagnostics.get("speech_ratio", 1.0)) >= 0.80
            and int(candidate.diagnostics.get("merged_recall_edge_count", 0)) == 0
        ]
        if len(anchors) < 3:
            return None

        primary_embeddings: list[common.torch.Tensor] = []
        secondary_embeddings: list[common.torch.Tensor] = []
        primary_original_scores: list[float] = []
        secondary_original_scores: list[float] = []
        for candidate in anchors:
            match = self._verify_speaker_span(
                verifier,
                waveform,
                common.TimeSpan(candidate.start, candidate.end),
                profile,
                threshold,
            )
            if (
                not match.accepted
                or match.primary.embedding is None
                or match.secondary is None
                or match.secondary.embedding is None
            ):
                continue
            primary_embeddings.append(match.primary.embedding)
            secondary_embeddings.append(match.secondary.embedding)
            primary_original_scores.append(match.primary.score)
            secondary_original_scores.append(match.secondary.score)
        if len(primary_embeddings) < 3:
            return None

        secondary_profile = verifier._ensure_secondary(profile)
        primary_all = common.torch.cat(
            [profile.primary.embeddings, common.torch.stack(primary_embeddings)],
            dim=0,
        )
        secondary_all = common.torch.cat(
            [secondary_profile.embeddings, common.torch.stack(secondary_embeddings)],
            dim=0,
        )
        primary_centroid = common.torch.nn.functional.normalize(primary_all.mean(dim=0), dim=0)
        secondary_centroid = common.torch.nn.functional.normalize(secondary_all.mean(dim=0), dim=0)
        primary_anchor_gains = (
            common.torch.stack(primary_embeddings) @ primary_centroid
            - common.torch.tensor(primary_original_scores)
        )
        secondary_anchor_gains = (
            common.torch.stack(secondary_embeddings) @ secondary_centroid
            - common.torch.tensor(secondary_original_scores)
        )
        primary_gain_floor = max(
            0.10,
            min(0.14, float(common.torch.quantile(primary_anchor_gains, 0.10))),
        )
        secondary_gain_floor = max(
            0.08,
            min(0.11, float(common.torch.quantile(secondary_anchor_gains, 0.10))),
        )
        original_indexes = list(
            profile.primary.reference_indexes
            or tuple(range(len(profile.primary.embeddings)))
        )
        indexes = tuple(
            [*original_indexes, *(10_000 + index for index in range(len(primary_embeddings)))]
        )
        adaptive_primary = common.SpeakerProfile(
            embeddings=primary_all,
            centroid=primary_centroid,
            reference_scores=[float(value) for value in primary_all @ primary_centroid],
            suggested_threshold=profile.primary.suggested_threshold,
            reference_floor=profile.primary.reference_floor,
            calibration_base=profile.primary.calibration_base,
            reference_indexes=indexes,
        )
        adaptive_secondary = common.CAMPlusProfile(
            embeddings=secondary_all,
            centroid=secondary_centroid,
            reference_scores=[float(value) for value in secondary_all @ secondary_centroid],
            reference_indexes=indexes,
        )
        return (
            common.SpeakerMatchProfile(
                primary=adaptive_primary,
                reference_paths=profile.reference_paths,
                base_threshold=profile.base_threshold,
            ),
            adaptive_secondary,
            round(primary_gain_floor, 5),
            round(secondary_gain_floor, 5),
        )

    @staticmethod
    def _needs_boundary_recovery(match: common.SpeakerMatchDecision, duration: float) -> bool:
        """Select near-target turns and audit every long accepted turn."""

        secondary = match.secondary
        if secondary is None:
            return False
        # Original-channel rescue is evidence for a separator-damaged turn,
        # not a structural identity decision.  Even a short rescued span must
        # pass the local boundary audit; otherwise a brief A-B switch could be
        # exported simply because its averaged raw embedding looked strong.
        if match.tier == "raw_rescue":
            return duration >= 1.20
        if duration < 2.50:
            return False
        if not match.accepted:
            return (
                duration >= 3.0
                and match.primary.score >= 0.50
                and secondary.score >= 0.50
                and match.primary.reference_median_score >= 0.40
                and secondary.reference_median_score >= 0.40
            )
        # A whole-turn embedding can hide a second speaker in the middle. The
        # local pass is mandatory for longer accepted turns.
        return True

    def _recover_target_segments(
        self,
        span: common.TimeSpan,
        boundaries: list[common.SpeakerBoundary],
        verifier: common.DualSpeakerVerifier,
        waveform: common.torch.Tensor,
        profile: common.SpeakerMatchProfile,
        exclusion_profiles: list[common.ExclusionSpeakerProfile],
        threshold: float,
        progress: common.ProgressCallback,
        recovery_index: int,
        recovery_count: int,
        *,
        scored_parts: list[
            tuple[common.TimeSpan, common.CandidateSentence, common.SpeakerMatchDecision | None]
        ] | None = None,
    ) -> tuple[list[tuple[common.CandidateSentence, common.SpeakerMatchDecision]], list[common.CandidateSentence]] | None:
        """Split a suspicious turn at every confirmed boundary and rescore it.

        The old recovery pass only kept a target suffix. That leaves an already
        accepted prefix (or an additional speaker later in the turn) in the
        exported file. Every resulting continuous segment is now verified on
        its own; unverified pieces are rejected instead of being joined back.
        """

        minimum = self.options.min_sentence_seconds
        candidates = sorted(
            {
                round(boundary.time, 3)
                for boundary in boundaries
                if boundary.confidence >= 0.90
                and span.start + minimum <= boundary.time <= span.end - minimum
            }
        )
        if not candidates:
            return None

        cuts = [span.start, *candidates, span.end]
        parts = [common.TimeSpan(start, end) for start, end in zip(cuts, cuts[1:])]
        accepted: list[tuple[common.CandidateSentence, common.SpeakerMatchDecision]] = []
        rejected: list[common.CandidateSentence] = []
        for part_index, part in enumerate(parts, start=1):
            match: common.SpeakerMatchDecision | None = None
            candidate = common.CandidateSentence(part.start, part.end, "")
            candidate.speaker_threshold = threshold
            candidate.diagnostics.update(
                {
                    "local_boundary_recovery": True,
                    "original_turn_start": span.start,
                    "original_turn_end": span.end,
                    "recovery_part_index": part_index,
                    "recovery_part_count": len(parts),
                    "recovery_boundaries": candidates,
                }
            )
            if part.duration < minimum:
                candidate.reject_reason = "说话回合过短"
            else:
                match = self._verify_speaker_span(
                    verifier, waveform, part, profile, threshold
                )
                original_match_tier = match.tier
                self._apply_speaker_match(candidate, match, profile, threshold)
                exclusion = verifier.exclusion_audit(
                    match,
                    profile,
                    exclusion_profiles,
                )
                if exclusion is not None:
                    candidate.diagnostics.update(exclusion)
                if exclusion and exclusion.get("excluded_role_rejected"):
                    candidate.reject_reason = (
                        f"更接近{exclusion['excluded_role']}，已按排除角色删除"
                    )
                if not match.accepted:
                    rescued_match = None
                    if not candidate.reject_reason and (
                        match.tier == "recall" or exclusion_profiles
                    ):
                        rescued_match = verifier.promote_local_with_tertiary(
                            self._waveform_span(waveform, part),
                            profile,
                            match,
                            part.duration,
                        )
                    if rescued_match is not None:
                        match = rescued_match
                        self._apply_speaker_match(
                            candidate,
                            match,
                            profile,
                            threshold,
                        )
                        candidate.diagnostics["local_edge_only"] = (
                            original_match_tier == "rejected"
                        )
                    elif not candidate.reject_reason:
                        candidate.reject_reason = "声纹匹配不足"
            progress(
                0.72
                + 0.055
                * (
                    (recovery_index - 1) + part_index / max(1, len(parts))
                )
                / max(1, recovery_count),
                f"局部换人复核 {recovery_index}/{recovery_count}，"
                f"片段 {part_index}/{len(parts)}",
            )
            # Keep scoped evidence independently of later candidate replacement.
            # A failed target match is unresolved, not proof of another person.
            candidate.diagnostics["local_identity_evidence"] = [{
                "span": [part.start, part.end],
                "state": (
                    "other" if candidate.diagnostics.get("excluded_role_rejected")
                    else "unresolved" if candidate.reject_reason else "target"
                ),
                "method": "boundary_part_verification",
                "parent_span": [span.start, span.end],
            }]
            if scored_parts is not None:
                scored_parts.append((part, candidate, match))
            if candidate.reject_reason:
                rejected.append(candidate)
            else:
                assert match is not None
                accepted.append((candidate, match))

        # Rejoin consecutive locally confirmed target pieces.  They were split
        # only to audit the change points; exporting each phonetic fragment
        # separately would recreate the incomplete-sentence problem.
        merged_accepted: list[tuple[common.CandidateSentence, common.SpeakerMatchDecision]] = []
        cursor = 0
        while cursor < len(accepted):
            group = [accepted[cursor]]
            cursor += 1
            while (
                cursor < len(accepted)
                and accepted[cursor][0].start - group[-1][0].end <= 0.02
            ):
                group.append(accepted[cursor])
                cursor += 1
            if len(group) == 1:
                merged_accepted.extend(group)
                continue
            joined_span = common.TimeSpan(group[0][0].start, group[-1][0].end)
            joined_match = self._verify_speaker_span(
                verifier,
                waveform,
                joined_span,
                profile,
                threshold,
            )
            joined_candidate: common.CandidateSentence | None = None
            joined_exclusion = verifier.exclusion_audit(
                joined_match,
                profile,
                exclusion_profiles,
            )
            if (
                not joined_match.accepted
                and not (
                    joined_exclusion
                    and joined_exclusion.get("excluded_role_rejected")
                )
            ):
                joined_match = (
                    verifier.promote_local_with_tertiary(
                        self._waveform_span(waveform, joined_span),
                        profile,
                        joined_match,
                        joined_span.duration,
                    )
                    or joined_match
                )
            if joined_match.accepted and not (
                joined_exclusion
                and joined_exclusion.get("excluded_role_rejected")
            ):
                joined_candidate = common.CandidateSentence(
                    joined_span.start, joined_span.end, ""
                )
                self._apply_speaker_match(
                    joined_candidate, joined_match, profile, threshold
                )
                joined_candidate.diagnostics.update(
                    {
                        "local_boundary_recovery": True,
                        "original_turn_start": span.start,
                        "original_turn_end": span.end,
                        "local_target_parts_merged": len(group),
                        "recovery_boundaries": candidates,
                        "local_identity_evidence": [
                            evidence for item, _match in group
                            for evidence in item.diagnostics["local_identity_evidence"]
                        ],
                    }
                )
                if joined_exclusion is not None:
                    joined_candidate.diagnostics.update(joined_exclusion)
                merged_accepted.append((joined_candidate, joined_match))
            else:
                merged_accepted.extend(group)

            if scored_parts is not None and joined_candidate is not None:
                # The merged sentence is the candidate that later recovery
                # checks should use as a domain-matched anchor. Keep its
                # independently rescored match instead of forcing those
                # checks to use the original mixed parent turn.
                scored_parts.append((joined_span, joined_candidate, joined_match))
        accepted = merged_accepted

        if not accepted:
            # A confirmed boundary with no independently matching piece is
            # safer to discard as a mixed turn than to export the original span.
            discarded = common.CandidateSentence(span.start, span.end, "")
            discarded.reject_reason = "疑似混合说话人，保守舍弃"
            discarded.diagnostics.update(
                {
                    "local_boundary_recovery": True,
                    "recovery_boundaries": candidates,
                }
            )
            # The parent summary cannot replace its local evidence. A later
            # recovery may overlap only one of these parts.
            return [], [*rejected, discarded]
        return accepted, rejected

    @staticmethod
    def _expand_recall_candidates(
        scored_turns: list[
            tuple[common.TimeSpan, common.CandidateSentence, common.SpeakerMatchDecision | None]
        ],
        accepted_turns: list[common.CandidateSentence],
        rejected: list[common.CandidateSentence],
        profile: common.SpeakerMatchProfile,
        extra_seed_items: common.Iterable[tuple[common.CandidateSentence, common.SpeakerMatchDecision]] = (),
    ) -> int:
        """Recover target turns that are acoustically close to confirmed seeds.

        A fixed lower threshold would admit other speakers globally. Instead,
        use accepted turns from the same silence-delimited speech block as
        local seeds and require both verifier embeddings to stay close to that
        block centroid. Only candidates rejected for speaker mismatch are
        eligible; overlap, singing, short-turn and confirmed mixed-speaker
        decisions stay hard rejects.
        """

        accepted_ids = {id(candidate) for candidate in accepted_turns}
        extra_seed_items = list(extra_seed_items)
        seed_blocks = {
            int(candidate.diagnostics["speech_block_index"])
            for candidate in accepted_turns
            if "speech_block_index" in candidate.diagnostics
        }
        seed_items = [
            (candidate, match)
            for _span, candidate, match in scored_turns
            if id(candidate) in accepted_ids
            and match is not None
            and match.secondary is not None
            and match.primary.embedding is not None
            and match.secondary.embedding is not None
        ]
        seed_items.extend(
            (candidate, match)
            for candidate, match in extra_seed_items
            if match.secondary is not None
            and match.primary.embedding is not None
            and match.secondary.embedding is not None
        )
        seed_blocks.update(
            int(candidate.diagnostics["speech_block_index"])
            for candidate, _match in extra_seed_items
            if isinstance(candidate.diagnostics.get("speech_block_index"), int)
        )
        if len(seed_items) < 3:
            return 0

        primary_seeds = common.torch.stack(
            [match.primary.embedding for _candidate, match in seed_items]
        )
        secondary_seeds = common.torch.stack(
            [match.secondary.embedding for _candidate, match in seed_items]
        )
        primary_centroid = common.torch.nn.functional.normalize(primary_seeds.mean(dim=0), dim=0)
        secondary_centroid = common.torch.nn.functional.normalize(secondary_seeds.mean(dim=0), dim=0)
        primary_seed_scores = primary_seeds @ primary_centroid
        secondary_seed_scores = secondary_seeds @ secondary_centroid

        # The floors follow the observed target-speaker variation, with hard
        # lower bounds so a weak seed set cannot turn into an open gate.
        # Keep a hard floor, but allow substantially more natural variation
        # than the first adaptive pass. The candidate still needs direct
        # evidence from both speaker models below, so this is not a global
        # threshold reduction.
        primary_floor = max(
            0.64,
            float(common.torch.quantile(primary_seed_scores, 0.10)) - 0.18,
        )
        secondary_floor = max(
            0.58,
            float(common.torch.quantile(secondary_seed_scores, 0.10)) - 0.18,
        )
        candidate_by_id = {
            id(candidate): match
            for _span, candidate, match in scored_turns
            if match is not None
        }
        retained_rejected: list[common.CandidateSentence] = []
        promoted = 0
        for candidate in rejected:
            # Keep all non-speaker rejects permanently excluded.
            if candidate.reject_reason != "声纹匹配不足":
                retained_rejected.append(candidate)
                continue
            match = candidate_by_id.get(id(candidate))
            candidate_block = candidate.diagnostics.get("speech_block_index")
            if (
                not isinstance(candidate_block, int)
                or candidate_block not in seed_blocks
            ):
                retained_rejected.append(candidate)
                continue
            # Coverage comes from the independent sliding-window locator.  A
            # low-coverage turn can contain another speaker even when its
            # whole-turn embedding looks close, so keep the recall gate high.
            if float(candidate.diagnostics.get("target_coverage", 0.0)) < 0.85:
                retained_rejected.append(candidate)
                continue
            if (
                match is None
                or match.secondary is None
                or match.primary.embedding is None
                or match.secondary.embedding is None
            ):
                retained_rejected.append(candidate)
                continue
            # A rejected dual-model decision is not eligible for recall. Only
            # the explicit weak tier, which has strong primary/reference
            # evidence, may enter this block-local recovery pass.
            if match.tier not in {"weak", "recall"}:
                retained_rejected.append(candidate)
                continue

            primary_seed_score = float(match.primary.embedding @ primary_centroid)
            secondary_seed_score = float(match.secondary.embedding @ secondary_centroid)
            primary_reference_max = float(
                candidate.diagnostics.get("eres_reference_max", 0.0)
            )
            secondary_reference_max = float(
                candidate.diagnostics.get("camplus_reference_max", 0.0)
            )
            reference_floor = float(profile.primary.reference_floor)
            reference_evidence = (
                primary_reference_max >= max(0.52, reference_floor - 0.11)
                and secondary_reference_max >= max(0.48, reference_floor - 0.13)
            )
            window_evidence = (
                candidate.window_p20_score >= primary_floor - 0.20
                and match.secondary.window_p20_score >= secondary_floor - 0.20
                and candidate.window_vote_ratio >= 0.50
                and match.secondary.window_vote_ratio >= 0.50
            )
            direct_evidence = (
                match.primary.score >= 0.54
                and match.secondary.score >= 0.50
                and match.paired_reference_median
                >= max(0.46, reference_floor - 0.14)
            )
            if (
                primary_seed_score < primary_floor
                or secondary_seed_score < secondary_floor
                or not reference_evidence
                or not window_evidence
                or not direct_evidence
            ):
                retained_rejected.append(candidate)
                continue

            candidate.reject_reason = ""
            candidate.diagnostics.update(
                {
                    "adaptive_recall": True,
                    "adaptive_recall_primary_seed": round(primary_seed_score, 5),
                    "adaptive_recall_secondary_seed": round(secondary_seed_score, 5),
                    "adaptive_recall_primary_floor": round(primary_floor, 5),
                    "adaptive_recall_secondary_floor": round(secondary_floor, 5),
                }
            )
            accepted_turns.append(candidate)
            promoted += 1

        rejected[:] = retained_rejected
        return promoted

    @staticmethod
    def _wavlm_same_speaker_floor(profile) -> float:
        """Calibrate edge continuity from the reference speaker itself."""

        embeddings = profile.embeddings
        if embeddings.shape[0] < 2:
            return max(0.82, float(profile.acceptance_floor))
        pairwise = embeddings @ embeddings.T
        mask = common.torch.triu(
            common.torch.ones_like(pairwise, dtype=common.torch.bool), diagonal=1
        )
        values = pairwise[mask]
        if values.numel() == 0:
            return max(0.82, float(profile.acceptance_floor))
        return max(
            float(profile.acceptance_floor),
            float(common.torch.quantile(values, 0.20)),
        )
