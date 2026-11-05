"""multimodel recovery responsibilities; composed by ExtractionPipeline."""
from __future__ import annotations

from .. import pipeline_common as common


class MultimodelRecoveryMethods:
    def _promote_multimodel_target_subclusters(
        self,
        scored_turns: list[
            tuple[common.TimeSpan, common.CandidateSentence, common.SpeakerMatchDecision | None]
        ],
        accepted_turns: list[common.CandidateSentence],
        rejected: list[common.CandidateSentence],
        verifier: common.DualSpeakerVerifier,
        profile: common.SpeakerMatchProfile,
        audio_path: common.Path,
        waveform: common.torch.Tensor,
        exclusion_profiles: list[common.ExclusionSpeakerProfile],
        progress: common.ProgressCallback,
    ) -> int:
        """Recover clean speaker-only rejects with deterministic model votes.

        Accepted episode turns remain separate target prototypes instead of
        being averaged into one centroid. Both base models need support from
        two prototypes, and all three models must agree on at least one of the
        same prototypes. Scores from unrelated models are never averaged.
        """

        raw_anchors = sorted(
            {
                (round(item.start, 5), round(item.end, 5)): item
                for item in accepted_turns
                if item.duration >= max(1.80, self.options.min_output_seconds)
                and not item.diagnostics.get("local_edge_only")
            }.values(),
            key=lambda item: (item.start, item.end),
        )
        matches = {
            id(candidate): match
            for span, candidate, match in scored_turns
            if match is not None
            and span.start == candidate.start and span.end == candidate.end
            and match.secondary is not None
            and match.primary.embedding is not None
            and match.secondary.embedding is not None
        }
        boundary_clean_residuals = [
            candidate
            for candidate in rejected
            if candidate.reject_reason == "疑似混合说话人，目标声纹不连续"
            and max(1.80, self.options.min_sentence_seconds)
            <= candidate.duration
            <= min(15.0, self.options.max_sentence_seconds)
            and not any(
                candidate.start + 0.05
                < float(boundary)
                < candidate.end - 0.05
                for boundary in candidate.diagnostics.get(
                    "recovery_boundaries", []
                )
            )
        ]
        for candidate in boundary_clean_residuals:
            if id(candidate) not in matches:
                matches[id(candidate)] = self._verify_speaker_span(
                    verifier,
                    waveform,
                    common.TimeSpan(candidate.start, candidate.end),
                    profile,
                    profile.primary.suggested_threshold,
                )
        eligible = [
            candidate
            for candidate in rejected
            if candidate.reject_reason
            in {"声纹匹配不足", "疑似混合说话人，目标声纹不连续"}
            and max(1.80, self.options.min_sentence_seconds)
            <= candidate.duration
            <= min(15.0, self.options.max_sentence_seconds)
            and id(candidate) in matches
        ]
        if not raw_anchors or not eligible:
            return 0

        assert verifier.secondary is not None
        # S03 already compares each proposed change with same-side baselines
        # in both encoders. A fresh dense sweep of cross-phoneme cosine lows
        # is neither new evidence of another speaker nor a valid hard veto.
        # Keep the confirmed points in this task's memory and reuse them for
        # anchors and recovery proposals alike; all promoted ranges still go
        # through exact-audio certification and the shared manifold.
        confirmed_points = tuple(getattr(self, "_speaker_change_points", ()))

        def confirmed_boundaries(candidate):
            return [float(point) for point in confirmed_points
                    if candidate.start < point < candidate.end]

        def reject_candidate(candidate, reason):
            candidate.accepted = False
            candidate.reject_reason = reason
            candidate.audio_file = candidate.text_file = candidate.video_file = ""

        anchors: list[common.CandidateSentence] = []
        anchor_matches: list[common.SpeakerMatchDecision] = []
        for anchor in raw_anchors:
            anchor_internal_boundaries = confirmed_boundaries(anchor)
            anchor.diagnostics["multi_model_anchor_boundary_count"] = len(
                anchor_internal_boundaries
            )
            # Keep the legacy raw-cosine array shape; the confirmation pass
            # retains points, not those unrelated cross-phoneme measurements.
            anchor.diagnostics["multi_model_anchor_boundaries"] = []
            anchor.diagnostics["multi_model_anchor_confirmed_boundary_points"] = anchor_internal_boundaries
            anchor.diagnostics["multi_model_anchor_boundary_source"] = "dual_same_side_confirmation"
            decisive_boundary = bool(anchor_internal_boundaries)
            anchor.diagnostics["multi_model_anchor_decisive_boundary"] = (
                decisive_boundary
            )
            if decisive_boundary:
                reject_candidate(anchor, "多模型复核确认内部换人")
                anchor.diagnostics["structural_hard_reject"] = True
                if anchor in accepted_turns:
                    accepted_turns.remove(anchor)
                if anchor not in rejected:
                    rejected.append(anchor)
                continue
            anchor_match = self._verify_speaker_span(
                verifier,
                waveform,
                common.TimeSpan(anchor.start, anchor.end),
                profile,
                profile.primary.suggested_threshold,
            )
            if (
                not anchor_match.accepted
                or anchor_match.secondary is None
                or anchor_match.primary.embedding is None
                or anchor_match.secondary.embedding is None
            ):
                continue
            anchor_exclusion = verifier.exclusion_audit(
                anchor_match,
                profile,
                exclusion_profiles,
            )
            if anchor_exclusion and anchor_exclusion.get("excluded_role_rejected"):
                continue
            anchors.append(anchor)
            anchor_matches.append(anchor_match)
        if len(anchors) < 2:
            return 0

        anchor_waveforms = [
            self._waveform_span(waveform, common.TimeSpan(item.start, item.end))
            for item in anchors
        ]
        primary_anchors = common.torch.stack(
            [match.primary.embedding for match in anchor_matches]
        )
        secondary_anchors = common.torch.stack(
            [match.secondary.embedding for match in anchor_matches]
        )
        fourth, fourth_reference = verifier.quaternary_pair(profile)
        fourth_waveforms = [
            *anchor_waveforms,
            *[
                self._waveform_span(
                    waveform,
                    common.TimeSpan(candidate.start, candidate.end),
                )
                for candidate in eligible
            ],
            *[
                self._waveform_span(
                    waveform,
                    common.TimeSpan(
                        float(candidate.diagnostics["contrastive_residual_start"]),
                        float(candidate.diagnostics["contrastive_residual_end"]),
                    ),
                )
                for candidate in eligible
                if candidate.diagnostics.get("contrastive_edge_trim")
            ],
        ]
        progress(0.795, "多模型声纹裁决：正在准备 WeSpeaker 批量复核")
        fourth_embeddings = fourth.embeddings_from_waveforms(
            fourth_waveforms,
            progress=lambda completed, total: progress(
                0.795,
                f"多模型声纹裁决：WeSpeaker {completed}/{total}",
            ),
        )
        unfiltered_anchor_count = len(anchors)
        fourth_anchors = fourth_embeddings[:unfiltered_anchor_count]
        candidate_end = unfiltered_anchor_count + len(eligible)
        fourth_candidates = fourth_embeddings[
            unfiltered_anchor_count:candidate_end
        ]
        contrastive_candidates = [
            candidate
            for candidate in eligible
            if candidate.diagnostics.get("contrastive_edge_trim")
        ]
        fourth_residuals = {
            id(candidate): embedding
            for candidate, embedding in zip(
                contrastive_candidates,
                fourth_embeddings[candidate_end:],
            )
        }
        fourth_anchor_reference = fourth_anchors @ fourth_reference.embeddings.T
        fourth_anchor_keep = fourth_anchor_reference.max(dim=1).values >= 0.30
        for anchor, score, keep in zip(
            anchors,
            fourth_anchor_reference.max(dim=1).values,
            fourth_anchor_keep,
        ):
            anchor.diagnostics["wespeaker_anchor_reference_max"] = round(
                float(score), 5
            )
            anchor.diagnostics["wespeaker_anchor_reference_valid"] = bool(keep)
        keep_indexes = fourth_anchor_keep.nonzero().flatten()
        if keep_indexes.numel() < 2:
            return 0
        anchors = [anchors[int(index)] for index in keep_indexes]
        primary_anchors = primary_anchors[keep_indexes]
        secondary_anchors = secondary_anchors[keep_indexes]
        fourth_anchors = fourth_anchors[keep_indexes]

        def continuity_parts(candidate: common.CandidateSentence) -> list[common.torch.Tensor]:
            value = self._waveform_span(
                waveform,
                common.TimeSpan(candidate.start, candidate.end),
            )
            size = round(1.20 * 16000)
            hop = round(0.60 * 16000)
            if value.numel() <= size:
                return [value]
            starts = list(range(0, value.numel() - size + 1, hop))
            last = value.numel() - size
            if starts[-1] != last:
                starts.append(last)
            return [value[start : start + size] for start in starts]

        anchor_window_waveforms: list[common.torch.Tensor] = []
        anchor_window_ranges: list[tuple[int, int]] = []
        for anchor in anchors:
            start_index = len(anchor_window_waveforms)
            anchor_window_waveforms.extend(continuity_parts(anchor))
            anchor_window_ranges.append(
                (start_index, len(anchor_window_waveforms))
            )
        anchor_primary_windows = verifier.primary._embeddings_from_waveforms(
            anchor_window_waveforms
        )
        anchor_secondary_windows = verifier.secondary._embeddings_from_waveforms(
            anchor_window_waveforms
        )
        anchor_fourth_windows = fourth.embeddings_from_waveforms(
            anchor_window_waveforms
        )
        continuity_anchor_indexes = list(range(len(anchors)))
        while len(continuity_anchor_indexes) >= 2:
            removed_indexes: list[int] = []
            for anchor_index in continuity_anchor_indexes:
                anchor = anchors[anchor_index]
                start_index, end_index = anchor_window_ranges[anchor_index]
                comparison_indexes = [
                    index
                    for index in continuity_anchor_indexes
                    if index != anchor_index
                ]
                common_windows = (
                    (
                        anchor_primary_windows[start_index:end_index]
                        @ primary_anchors[comparison_indexes].T
                        >= 0.38
                    )
                    & (
                        anchor_secondary_windows[start_index:end_index]
                        @ secondary_anchors[comparison_indexes].T
                        >= 0.30
                    )
                    & (
                        anchor_fourth_windows[start_index:end_index]
                        @ fourth_anchors[comparison_indexes].T
                        >= 0.40
                    )
                ).any(dim=1)
                continuity_ratio = float(common_windows.float().mean())
                anchor.diagnostics["multi_model_anchor_continuity"] = round(
                    continuity_ratio, 5
                )
                decisive_boundary = bool(confirmed_boundaries(anchor))
                anchor.diagnostics["multi_model_anchor_decisive_boundary"] = (
                    decisive_boundary
                )
                if continuity_ratio < 0.70 or decisive_boundary:
                    removed_indexes.append(anchor_index)
            if not removed_indexes:
                break
            for anchor_index in removed_indexes:
                anchor = anchors[anchor_index]
                # A low episode-cluster continuity score makes this a poor
                # recovery prototype, but is not sufficient to discard a
                # directly verified target turn. Decisive speaker boundaries
                # were already hard-rejected before profile construction.
                anchor.diagnostics["multi_model_anchor_excluded"] = True
            removed_set = set(removed_indexes)
            continuity_anchor_indexes = [
                index
                for index in continuity_anchor_indexes
                if index not in removed_set
            ]
        if len(continuity_anchor_indexes) < 2:
            return 0
        continuity_anchor_indexes_tensor = common.torch.tensor(
            continuity_anchor_indexes,
            dtype=common.torch.long,
        )
        anchors = [anchors[index] for index in continuity_anchor_indexes]
        primary_anchors = primary_anchors[continuity_anchor_indexes_tensor]
        secondary_anchors = secondary_anchors[continuity_anchor_indexes_tensor]
        fourth_anchors = fourth_anchors[continuity_anchor_indexes_tensor]
        if not eligible:
            return 0

        provisional: list[common.CandidateSentence] = []
        for candidate, fourth_embedding in zip(eligible, fourth_candidates):
            match = matches[id(candidate)]
            assert match.secondary is not None
            assert match.primary.embedding is not None
            assert match.secondary.embedding is not None
            primary_top = common.torch.topk(
                match.primary.embedding @ primary_anchors.T,
                k=2,
            ).values
            secondary_top = common.torch.topk(
                match.secondary.embedding @ secondary_anchors.T,
                k=2,
            ).values
            fourth_top = common.torch.topk(
                fourth_embedding @ fourth_anchors.T,
                k=2,
            ).values
            primary_second = float(primary_top[1])
            secondary_second = float(secondary_top[1])
            fourth_first = float(fourth_top[0])
            fourth_second = float(fourth_top[1])
            common_anchor_mask = (
                (match.primary.embedding @ primary_anchors.T >= 0.64)
                & (match.secondary.embedding @ secondary_anchors.T >= 0.58)
                & (fourth_embedding @ fourth_anchors.T >= 0.60)
            )
            common_anchor_count = int(common_anchor_mask.sum())
            fourth_direct = float(
                (fourth_reference.embeddings @ fourth_embedding).max()
            )
            direct_reference_evidence = (
                match.primary.reference_max_score >= 0.42
                or match.secondary.reference_max_score >= 0.42
                or fourth_direct >= 0.42
            )
            base_consensus = (
                primary_second >= 0.64
                and secondary_second >= 0.58
            )
            wespeaker_dominant_consensus = (
                float(primary_top[0]) >= 0.66
                and float(secondary_top[0]) >= 0.64
                and fourth_second >= 0.70
                and common_anchor_count >= 1
            )
            fourth_model_rescue = (
                float(primary_top[0]) >= 0.60
                and float(secondary_top[0]) >= 0.56
                and fourth_first >= 0.72
                and fourth_second >= 0.68
            )
            base_consensus = (
                base_consensus
                or wespeaker_dominant_consensus
                or fourth_model_rescue
            )
            contrastive_edge = bool(
                candidate.diagnostics.get("contrastive_edge_trim")
            )
            fourth_support = fourth_first >= (0.60 if contrastive_edge else 0.64)
            contrastive_fourth_passed = not contrastive_edge
            if contrastive_edge:
                residual_fourth = fourth_residuals[id(candidate)]
                residual_fourth_score = float(
                    residual_fourth @ fourth_reference.centroid
                )
                residual_fourth_direct = float(
                    (fourth_reference.embeddings @ residual_fourth).max()
                )
                fourth_score_margin = float(
                    fourth_embedding @ fourth_reference.centroid
                ) - residual_fourth_score
                fourth_direct_margin = fourth_direct - residual_fourth_direct
                contrastive_fourth_passed = (
                    fourth_direct >= 0.55
                    and residual_fourth_direct <= 0.35
                    and fourth_score_margin >= 0.20
                    and fourth_direct_margin >= 0.20
                )
                candidate.diagnostics.update(
                    {
                        "contrastive_residual_wespeaker_score": round(
                            residual_fourth_score, 5
                        ),
                        "contrastive_residual_wespeaker_reference_max": round(
                            residual_fourth_direct, 5
                        ),
                        "contrastive_wespeaker_score_margin": round(
                            fourth_score_margin, 5
                        ),
                        "contrastive_wespeaker_direct_margin": round(
                            fourth_direct_margin, 5
                        ),
                        "contrastive_edge_fourth_passed": (
                            contrastive_fourth_passed
                        ),
                    }
                )
            candidate.diagnostics.update(
                {
                    "multi_model_consensus_checked": True,
                    "episode_eres_top1": round(float(primary_top[0]), 5),
                    "episode_eres_top2": round(primary_second, 5),
                    "episode_camplus_top1": round(float(secondary_top[0]), 5),
                    "episode_camplus_top2": round(secondary_second, 5),
                    "episode_wespeaker_top1": round(fourth_first, 5),
                    "episode_wespeaker_top2": round(fourth_second, 5),
                    "multi_model_base_consensus": base_consensus,
                    "multi_model_wespeaker_dominant_consensus": (
                        wespeaker_dominant_consensus
                    ),
                    "multi_model_fourth_model_rescue": fourth_model_rescue,
                    "multi_model_wespeaker_support": fourth_support,
                    "multi_model_common_anchor_count": common_anchor_count,
                    "multi_model_direct_reference": direct_reference_evidence,
                }
            )
            if (
                not base_consensus
                or not fourth_support
                or (common_anchor_count < 1 and not fourth_model_rescue)
                or not direct_reference_evidence
                or not contrastive_fourth_passed
            ):
                continue

            boundary_residual = (
                candidate.reject_reason
                == "疑似混合说话人，目标声纹不连续"
            )
            if boundary_residual or contrastive_edge:
                exclusion = verifier.multimodel_exclusion_audit(
                    match,
                    profile,
                    fourth_embedding,
                    fourth_reference,
                    exclusion_profiles,
                )
            else:
                exclusion = verifier.exclusion_audit(
                    match,
                    profile,
                    exclusion_profiles,
                )
            if exclusion is not None:
                candidate.diagnostics.update(exclusion)
            if exclusion and exclusion.get("excluded_role_rejected"):
                reject_candidate(candidate, f"更接近{exclusion['excluded_role']}，已按排除角色删除")
                continue
            if (
                boundary_residual
                and exclusion
                and not exclusion.get("excluded_all_models_clear", False)
            ):
                reject_candidate(candidate, "边界残片未通过三模型排除人物净胜复核")
                candidate.diagnostics["structural_hard_reject"] = True
                continue

            overlapping = [
                existing
                for existing in accepted_turns
                if min(candidate.end, existing.end)
                - max(candidate.start, existing.start)
                > 0.10
            ]
            if overlapping and not all(
                candidate.start <= existing.start + 0.25
                and candidate.end >= existing.end - 0.25
                for existing in overlapping
            ):
                continue
            provisional.append(candidate)

        continuity_ready: list[common.CandidateSentence] = []
        continuity_waveforms: list[common.torch.Tensor] = []
        continuity_ranges: list[tuple[int, int]] = []
        for candidate in provisional:
            start_index = len(continuity_waveforms)
            continuity_waveforms.extend(continuity_parts(candidate))
            continuity_ranges.append((start_index, len(continuity_waveforms)))

        if continuity_waveforms:
            progress(0.795, "多模型候选首尾连续性复核")
            primary_windows = verifier.primary._embeddings_from_waveforms(
                continuity_waveforms
            )
            secondary_windows = verifier.secondary._embeddings_from_waveforms(
                continuity_waveforms
            )
            fourth_windows = fourth.embeddings_from_waveforms(
                continuity_waveforms
            )
            for candidate, (start_index, end_index) in zip(
                provisional,
                continuity_ranges,
            ):
                common_windows = (
                    (
                        primary_windows[start_index:end_index]
                        @ primary_anchors.T
                        >= 0.38
                    )
                    & (
                        secondary_windows[start_index:end_index]
                        @ secondary_anchors.T
                        >= 0.30
                    )
                    & (
                        fourth_windows[start_index:end_index]
                        @ fourth_anchors.T
                        >= 0.40
                    )
                ).any(dim=1)
                continuity_ratio = float(common_windows.float().mean())
                candidate.diagnostics.update(
                    {
                        "multi_model_continuity_ratio": round(
                            continuity_ratio, 5
                        ),
                        "multi_model_continuity_windows": len(common_windows),
                        "multi_model_continuity_passed": int(
                            common_windows.sum()
                        ),
                    }
                )
                edge_tolerant = (
                    continuity_ratio >= 0.55
                    and int(
                        candidate.diagnostics.get(
                            "multi_model_common_anchor_count", 0
                        )
                    )
                    >= 1
                    and float(candidate.diagnostics.get("episode_eres_top2", 0.0))
                    >= 0.72
                    and float(
                        candidate.diagnostics.get("episode_camplus_top2", 0.0)
                    )
                    >= 0.60
                    and float(
                        candidate.diagnostics.get("episode_wespeaker_top1", 0.0)
                    )
                    >= 0.65
                )
                fourth_model_edge_tolerant = (
                    continuity_ratio >= 0.55
                    and bool(
                        candidate.diagnostics.get(
                            "multi_model_fourth_model_rescue"
                        )
                    )
                    and float(
                        candidate.diagnostics.get("episode_wespeaker_top2", 0.0)
                    )
                    >= 0.68
                )
                edge_tolerant = edge_tolerant or fourth_model_edge_tolerant
                candidate.diagnostics["multi_model_edge_tolerant"] = edge_tolerant
                if continuity_ratio < 0.70 and not edge_tolerant:
                    # A failed recovery vote is uncertainty, not positive
                    # evidence for a competing speaker. Leave this available
                    # to the later exact multichannel decoder.
                    reject_candidate(candidate, "声纹匹配不足")
                    candidate.diagnostics["multi_model_continuity_inconclusive"] = True
                    continue
                continuity_ready.append(candidate)

        promoted_ids: set[int] = set()
        promoted = 0
        for candidate in continuity_ready:
            boundaries = confirmed_boundaries(candidate)
            candidate.diagnostics["multi_model_internal_boundary_count"] = len(
                boundaries
            )
            decisive_boundary = bool(boundaries)
            candidate.diagnostics["multi_model_decisive_boundary"] = (
                decisive_boundary
            )
            if decisive_boundary:
                reject_candidate(candidate, "多模型复核确认内部换人")
                candidate.diagnostics["structural_hard_reject"] = True
                continue
            overlapping = [
                existing
                for existing in accepted_turns
                if min(candidate.end, existing.end)
                - max(candidate.start, existing.start)
                > 0.10
            ]
            self._attach_verified_fallbacks(candidate, overlapping)
            for existing in overlapping:
                accepted_turns.remove(existing)
            candidate.accepted = True
            candidate.reject_reason = ""
            candidate.diagnostics["multi_model_target_recovery"] = True
            accepted_turns.append(candidate)
            promoted_ids.add(id(candidate))
            promoted += 1

        if promoted_ids:
            rejected[:] = [
                candidate
                for candidate in rejected
                if id(candidate) not in promoted_ids
            ]
        return promoted
