"""sentence repair responsibilities; composed by ExtractionPipeline."""
from __future__ import annotations

from .. import pipeline_common as common


class SentenceRepairMethods:
    def _extend_target_clip_over_tail_sliver(
        self,
        accepted_turns: list[common.CandidateSentence],
        rejected: list[common.CandidateSentence],
        verifier: common.DualSpeakerVerifier,
        profile: common.SpeakerMatchProfile,
        stem_waveform: common.torch.Tensor,
        exclusion_profiles: list[common.ExclusionSpeakerProfile],
        threshold: float,
        progress: common.ProgressCallback,
    ) -> int:
        """Absorb an unscorable edge sliver and its quiet gap into the
        adjacent accepted clip.

        A sliver shorter than 0.25 s cannot be scored by any identity tooling
        in this project, so the binding evidence is: the sliver sits directly
        against the accepted clip, a quiet gap bounds it on the far side, the
        joined span passes formal dual-model verification, and the exclusion
        audit does not reject the joined span.  The quiet gap is the natural
        sentence edge, so the other speaker's speech beyond it is never
        absorbed.
        """
        if not accepted_turns or not rejected:
            return 0
        from ..nextgen.internal_pauses import locate_quiet_gaps

        extended_count = 0
        for candidate in list(rejected):
            if candidate.diagnostics.get("structural_hard_reject"):
                continue
            span = common.TimeSpan(candidate.start, candidate.end)
            samples = self._waveform_span(
                stem_waveform, span
            ).detach().cpu().numpy()
            if samples.size == 0:
                continue
            gaps = list(locate_quiet_gaps(
                samples,
                16000,
                lower_seconds=self.options.silence_min_seconds,
                upper_seconds=self.options.silence_split_seconds,
            ))
            if not gaps:
                continue
            head_neighbors = [
                item for item in accepted_turns
                if abs(item.end - span.start) <= 0.05
            ]
            tail_neighbors = [
                item for item in accepted_turns
                if abs(span.end - item.start) <= 0.05
            ]
            if head_neighbors:
                neighbor = head_neighbors[0]
                head_side = True
            elif tail_neighbors:
                neighbor = tail_neighbors[0]
                head_side = False
            else:
                continue
            boundary = gaps[0] if head_side else gaps[-1]
            if head_side:
                sliver_seconds = boundary.start / 16000.0
            else:
                sliver_seconds = span.duration - boundary.end / 16000.0
            if sliver_seconds > 0.25:
                continue
            if head_side:
                extension_end = span.start + boundary.end / 16000.0
                extended = common.TimeSpan(neighbor.start, extension_end)
            else:
                extension_start = span.start + boundary.start / 16000.0
                extended = common.TimeSpan(extension_start, neighbor.end)
            if extended.duration > self.options.max_sentence_seconds:
                continue
            extended_match = self._verify_speaker_span(
                verifier,
                stem_waveform,
                extended,
                profile,
                threshold,
                audit_raw=True,
            )
            if not extended_match.accepted:
                continue
            extended_exclusion = verifier.exclusion_audit(
                extended_match, profile, exclusion_profiles,
            ) or {}
            if extended_exclusion.get("excluded_role_rejected"):
                continue
            neighbor.start, neighbor.end = extended.start, extended.end
            neighbor.diagnostics["tail_sliver_extension"] = {
                "source_turn": [span.start, span.end],
                "sliver_seconds": round(sliver_seconds, 3),
                "extended_span": [extended.start, extended.end],
            }
            if head_side:
                remainder_start = extended.end
                if candidate.end - remainder_start > 0.01:
                    candidate.start = remainder_start
                else:
                    rejected.remove(candidate)
            else:
                remainder_end = extended.start
                if remainder_end - candidate.start > 0.01:
                    candidate.end = remainder_end
                else:
                    rejected.remove(candidate)
            extended_count += 1
        return extended_count

    @staticmethod
    def _frame_db_levels(
        waveform: common.torch.Tensor,
        span: common.TimeSpan,
        *,
        frame_seconds: float = 0.02,
        hop_seconds: float = 0.01,
    ) -> tuple[list[float], list[float]]:
        start_sample = max(0, round(span.start * 16000))
        end_sample = min(waveform.numel(), round(span.end * 16000))
        frame = max(1, round(frame_seconds * 16000))
        hop = max(1, round(hop_seconds * 16000))
        times: list[float] = []
        levels: list[float] = []
        cursor = start_sample
        while cursor + frame <= end_sample:
            part = waveform[cursor : cursor + frame]
            rms = common.torch.sqrt(common.torch.mean(part.square()) + 1e-12)
            times.append(cursor / 16000)
            levels.append(float(20.0 * common.torch.log10(rms + 1e-12)))
            cursor += hop
        return times, levels

    @classmethod
    def _quiet_gap_ratio(
        cls,
        waveform: common.torch.Tensor,
        gap: common.TimeSpan,
    ) -> tuple[float, float]:
        """Measure whether a proposed join really contains a silent pause."""

        if gap.duration < 0.06:
            return 0.0, -120.0
        _times, levels = cls._frame_db_levels(waveform, gap)
        context = common.TimeSpan(max(0.0, gap.start - 0.40), gap.end + 0.40)
        _context_times, context_levels = cls._frame_db_levels(waveform, context)
        if not levels or not context_levels:
            return 0.0, -120.0
        speech_level = float(common.torch.quantile(common.torch.tensor(context_levels), 0.75))
        quiet_floor = min(-38.0, speech_level - 15.0)
        ratio = sum(level <= quiet_floor for level in levels) / len(levels)
        return float(ratio), float(quiet_floor)

    @staticmethod
    def _shared_source_speech_block(
        left: common.TimeSpan,
        right: common.TimeSpan,
        source_speech_spans: common.Iterable[common.TimeSpan],
        *,
        tolerance: float = 0.03,
    ) -> common.TimeSpan | None:
        """Return the original VAD block containing both proposed sentence parts."""

        merged_start = min(left.start, right.start)
        merged_end = max(left.end, right.end)
        return next(
            (
                block
                for block in source_speech_spans
                if block.start <= merged_start + tolerance
                and merged_end <= block.end + tolerance
            ),
            None,
        )

    @classmethod
    def _refine_boundary_to_quiet_gap(
        cls,
        waveform: common.torch.Tensor,
        candidate: common.TimeSpan,
        boundary: float,
        *,
        target_side: str,
        radius: float = 0.50,
    ) -> tuple[float, list[dict[str, float]]]:
        """Snap a coarse identity change to the adjacent short acoustic pause."""

        search = common.TimeSpan(
            max(candidate.start, boundary - radius),
            min(candidate.end, boundary + radius),
        )
        times, levels = cls._frame_db_levels(waveform, search)
        if not times:
            return boundary, []
        speech_level = float(common.torch.quantile(common.torch.tensor(levels), 0.75))
        quiet_floor = max(-55.0, min(-38.0, speech_level - 18.0))
        quiet = [level <= quiet_floor for level in levels]
        runs: list[dict[str, float]] = []
        index = 0
        while index < len(quiet):
            if not quiet[index]:
                index += 1
                continue
            first = index
            while index + 1 < len(quiet) and quiet[index + 1]:
                index += 1
            start = times[first]
            end = times[index] + 0.02
            if end - start >= 0.06:
                runs.append(
                    {
                        "start": round(start, 5),
                        "end": round(end, 5),
                        "duration": round(end - start, 5),
                        "threshold_db": round(quiet_floor, 3),
                    }
                )
            index += 1

        if target_side == "left":
            following = [run for run in runs if run["start"] >= boundary - 0.02]
            refined = following[0]["end"] if following else boundary
        else:
            preceding = [run for run in runs if run["end"] <= boundary + 0.02]
            refined = preceding[-1]["end"] if preceding else boundary
        refined = min(candidate.end - 0.20, max(candidate.start + 0.20, refined))
        return float(refined), runs

    def _repair_final_sentence_boundaries(
        self,
        accepted_turns: list[common.CandidateSentence],
        rejected: list[common.CandidateSentence],
        scored_turns: list[
            tuple[common.TimeSpan, common.CandidateSentence, common.SpeakerMatchDecision | None]
        ],
        verifier: common.DualSpeakerVerifier,
        profile: common.SpeakerMatchProfile,
        audio_path: common.Path,
        waveform: common.torch.Tensor,
        exclusion_profiles: list[common.ExclusionSpeakerProfile],
        threshold: float,
        blocked_join_spans: common.Iterable[common.TimeSpan],
        source_speech_spans: common.Iterable[common.TimeSpan],
        progress: common.ProgressCallback,
    ) -> tuple[int, int]:
        """Repair only proven edge omissions and mixed recovery candidates.

        Short edges are never exported or identified independently. A missing
        edge must remain inside the same original VAD block as the accepted
        core, and the complete sentence must pass the regular dual-model and
        exclusion audits. Internal cropping is limited to a recovery candidate
        with one independently stable identity boundary.
        """

        if not accepted_turns:
            return 0, 0
        tertiary, tertiary_profile = verifier._tertiary_pair(profile)
        same_speaker_floor = self._wavlm_same_speaker_floor(tertiary_profile)
        cropped = 0
        expanded = 0
        boundary_splitter: common.LocalSpeakerTurnSplitter | None = None

        crop_candidates = [
            candidate
            for candidate in accepted_turns
            if candidate.diagnostics.get("multi_model_target_recovery")
            and int(
                candidate.diagnostics.get(
                    "multi_model_internal_boundary_count", 0
                )
                or 0
            )
            == 1
            and candidate.diagnostics.get("speaker_tier")
            not in {"short_strong", "strong", "balanced"}
            and candidate.duration >= self.options.min_output_seconds * 2
        ]
        if crop_candidates:
            assert verifier.secondary is not None
            boundary_splitter = common.LocalSpeakerTurnSplitter(
                verifier.primary,
                secondary=verifier.secondary,
            )
            boundaries = boundary_splitter.detect_multiscale_speaker_boundaries(
                audio_path,
                [common.TimeSpan(item.start, item.end) for item in crop_candidates],
                progress=lambda value, message: progress(
                    0.40 * value,
                    f"恢复候选边界终审：{message}",
                ),
            )
            for candidate in list(crop_candidates):
                local = [
                    boundary
                    for boundary in boundaries
                    if candidate.start + 0.20
                    <= boundary.time
                    <= candidate.end - 0.20
                ]
                if len(local) != 1:
                    continue
                boundary = local[0]
                left = common.TimeSpan(candidate.start, boundary.time)
                right = common.TimeSpan(boundary.time, candidate.end)
                parts = [
                    self._waveform_span(waveform, left),
                    self._waveform_span(waveform, right),
                ]
                primary_embeddings = verifier.primary._embeddings_from_waveforms(parts)
                secondary_embeddings = verifier.secondary._embeddings_from_waveforms(parts)
                tertiary_embeddings = tertiary._embeddings_from_waveforms(parts)
                pair_similarity = float(
                    tertiary_embeddings[0] @ tertiary_embeddings[1]
                )
                if pair_similarity >= same_speaker_floor:
                    continue
                secondary_profile = verifier._ensure_secondary(profile)
                left_scores = (
                    float(primary_embeddings[0] @ profile.primary.centroid),
                    float(secondary_embeddings[0] @ secondary_profile.centroid),
                    float(tertiary_embeddings[0] @ tertiary_profile.centroid),
                )
                right_scores = (
                    float(primary_embeddings[1] @ profile.primary.centroid),
                    float(secondary_embeddings[1] @ secondary_profile.centroid),
                    float(tertiary_embeddings[1] @ tertiary_profile.centroid),
                )
                margins = tuple(
                    left_score - right_score
                    for left_score, right_score in zip(left_scores, right_scores)
                )
                left_wins = all(value > 0.0 for value in margins)
                right_wins = all(value < 0.0 for value in margins)
                if not (left_wins or right_wins):
                    continue
                if (
                    abs(margins[0]) < 0.08
                    or abs(margins[1]) < 0.08
                    or abs(margins[2]) < 0.15
                ):
                    continue
                target_side = "left" if left_wins else "right"
                refined, quiet_runs = self._refine_boundary_to_quiet_gap(
                    waveform,
                    common.TimeSpan(candidate.start, candidate.end),
                    boundary.time,
                    target_side=target_side,
                )
                retained_span = (
                    common.TimeSpan(candidate.start, refined)
                    if target_side == "left"
                    else common.TimeSpan(refined, candidate.end)
                )
                discarded_span = (
                    common.TimeSpan(refined, candidate.end)
                    if target_side == "left"
                    else common.TimeSpan(candidate.start, refined)
                )
                if retained_span.duration + 1e-6 < self.options.min_output_seconds:
                    continue
                retained_match = self._verify_speaker_span(
                    verifier,
                    waveform,
                    retained_span,
                    profile,
                    threshold,
                )
                exclusion = verifier.exclusion_audit(
                    retained_match,
                    profile,
                    exclusion_profiles,
                )
                if exclusion and exclusion.get("excluded_role_rejected"):
                    continue

                replacement = common.CandidateSentence(
                    retained_span.start,
                    retained_span.end,
                    "",
                )
                replacement.diagnostics.update(candidate.diagnostics)
                self._apply_speaker_match(
                    replacement,
                    retained_match,
                    profile,
                    threshold,
                )
                replacement.diagnostics.update(
                    {
                        "final_identity_boundary_crop": True,
                        "final_identity_parent_span": [
                            candidate.start,
                            candidate.end,
                        ],
                        "final_identity_boundary": boundary.to_dict(),
                        "final_identity_refined_boundary": round(refined, 5),
                        "final_identity_target_side": target_side,
                        "final_identity_wavlm_pair": round(pair_similarity, 5),
                        "final_identity_wavlm_same_floor": round(
                            same_speaker_floor, 5
                        ),
                        "final_identity_target_margins": [
                            round(value, 5) for value in margins
                        ],
                        "final_identity_quiet_runs": quiet_runs,
                    }
                )
                if exclusion is not None:
                    replacement.diagnostics.update(exclusion)
                accepted_turns.remove(candidate)
                accepted_turns.append(replacement)
                discarded = common.CandidateSentence(
                    discarded_span.start,
                    discarded_span.end,
                    "",
                    reject_reason="检测到连续换人，已裁掉非目标人物一侧",
                )
                discarded.diagnostics.update(
                    {
                        "final_identity_boundary_discard": True,
                        "final_identity_parent_span": [
                            candidate.start,
                            candidate.end,
                        ],
                    }
                )
                rejected.append(discarded)
                cropped += 1

        accepted_turns.sort(key=lambda item: (item.start, item.end))
        accepted_ids = {id(candidate) for candidate in accepted_turns}
        edge_pool = [
            (candidate, match)
            for _span, candidate, match in scored_turns
            if match is not None
            and not match.accepted
            and id(candidate) not in accepted_ids
            and candidate.duration >= self.options.min_sentence_seconds
            and candidate.duration <= min(8.0, self.options.max_sentence_seconds)
        ]
        blocked = tuple(blocked_join_spans)
        source_blocks = tuple(source_speech_spans)
        used_edges: set[int] = set()
        for core in list(accepted_turns):
            core_span = common.TimeSpan(core.start, core.end)
            plans: list[tuple[float, common.CandidateSentence, common.TimeSpan]] = []
            for edge, _match in edge_pool:
                if id(edge) in used_edges:
                    continue
                edge_span = common.TimeSpan(edge.start, edge.end)
                if any(
                    min(edge_span.end, existing.end)
                    - max(edge_span.start, existing.start)
                    > 0.10
                    for existing in accepted_turns
                ):
                    continue
                left_gap = core_span.start - edge_span.end
                right_gap = edge_span.start - core_span.end
                if 0.06 <= left_gap <= self.options.silence_split_seconds:
                    merged_span = common.TimeSpan(edge_span.start, core_span.end)
                    gap_span = common.TimeSpan(edge_span.end, core_span.start)
                    plans.append((left_gap, edge, merged_span))
                elif 0.06 <= right_gap <= self.options.silence_split_seconds:
                    merged_span = common.TimeSpan(core_span.start, edge_span.end)
                    gap_span = common.TimeSpan(core_span.end, edge_span.start)
                    plans.append((right_gap, edge, merged_span))
            if not plans:
                continue
            for gap, edge, merged_span in sorted(plans, key=lambda item: item[0]):
                edge_span = common.TimeSpan(edge.start, edge.end)
                left, right = sorted((core_span, edge_span), key=lambda item: item.start)
                if not self._identity_join_allowed(left, right, verifier, profile, waveform):
                    continue
                gap_span = (
                    common.TimeSpan(edge.end, core.start)
                    if edge.end <= core.start
                    else common.TimeSpan(core.end, edge.start)
                )
                if any(
                    min(gap_span.end, dirty.end)
                    - max(gap_span.start, dirty.start)
                    > 0.01
                    for dirty in blocked
                ):
                    continue
                quiet_ratio, quiet_floor = self._quiet_gap_ratio(
                    waveform,
                    gap_span,
                )
                if quiet_ratio < 0.50:
                    continue
                source_block = self._shared_source_speech_block(
                    core_span,
                    edge_span,
                    source_blocks,
                )
                if source_block is None:
                    continue
                initial_match = self._verify_speaker_span(
                    verifier,
                    waveform,
                    merged_span,
                    profile,
                    threshold,
                )
                if not initial_match.accepted:
                    continue
                initial_exclusion = verifier.exclusion_audit(
                    initial_match,
                    profile,
                    exclusion_profiles,
                )
                if initial_exclusion and initial_exclusion.get(
                    "excluded_role_rejected"
                ):
                    continue

                internal_boundary: common.SpeakerBoundary | None = None
                discarded_internal_span: common.TimeSpan | None = None
                if edge_span.duration >= 1.40:
                    if boundary_splitter is None:
                        assert verifier.secondary is not None
                        boundary_splitter = common.LocalSpeakerTurnSplitter(
                            verifier.primary,
                            secondary=verifier.secondary,
                        )
                    internal_boundaries = (
                        boundary_splitter.detect_multiscale_speaker_boundaries(
                            audio_path,
                            [edge_span],
                            contexts=(0.50, 0.60, 0.70),
                            cluster_seconds=0.25,
                            minimum_context_votes=3,
                            scan_hop_seconds=0.05,
                            primary_candidate_threshold=0.82,
                            minimum_similarity_drop=0.04,
                            minimum_separation_seconds=0.15,
                            progress=lambda value, message: progress(
                                0.55 + 0.25 * value,
                                f"新增句尾换人复核：{message}",
                            ),
                        )
                    )
                    decisive = [
                        boundary
                        for boundary in internal_boundaries
                        if edge_span.start + 0.45
                        <= boundary.time
                        <= edge_span.end - 0.45
                        and boundary.scale_votes >= 3
                        and boundary.primary_similarity <= 0.20
                        and boundary.secondary_similarity is not None
                        and boundary.secondary_similarity <= 0.10
                    ]
                    if decisive:
                        edge_on_left = edge_span.end <= core_span.start
                        internal_boundary = (
                            decisive[-1] if edge_on_left else decisive[0]
                        )
                        if edge_on_left:
                            discarded_internal_span = common.TimeSpan(
                                edge_span.start,
                                internal_boundary.time,
                            )
                            edge_span = common.TimeSpan(
                                internal_boundary.time,
                                edge_span.end,
                            )
                            merged_span = common.TimeSpan(
                                internal_boundary.time,
                                core_span.end,
                            )
                        else:
                            discarded_internal_span = common.TimeSpan(
                                internal_boundary.time,
                                edge_span.end,
                            )
                            edge_span = common.TimeSpan(
                                edge_span.start,
                                internal_boundary.time,
                            )
                            merged_span = common.TimeSpan(
                                core_span.start,
                                internal_boundary.time,
                            )
                        if edge_span.duration < self.options.min_sentence_seconds:
                            continue

                core_audio = self._waveform_span(waveform, core_span)
                edge_audio = self._waveform_span(waveform, edge_span)
                edge_embeddings = tertiary._embeddings_from_waveforms(
                    [core_audio, edge_audio]
                )
                pair_similarity = float(edge_embeddings[0] @ edge_embeddings[1])
                edge_target_score = float(
                    edge_embeddings[1] @ tertiary_profile.centroid
                )
                # A rejected neighboring turn must never inherit the accepted
                # core's identity.  The edge is a separate utterance after a
                # quiet gap, so it needs both an independent target decision
                # and local same-speaker continuity before it can be joined.
                if pair_similarity < same_speaker_floor:
                    continue
                edge_match = self._verify_speaker_span(
                    verifier,
                    waveform,
                    edge_span,
                    profile,
                    threshold,
                )
                if not edge_match.accepted:
                    continue
                edge_exclusion = verifier.exclusion_audit(
                    edge_match,
                    profile,
                    exclusion_profiles,
                )
                if edge_exclusion and edge_exclusion.get("excluded_role_rejected"):
                    continue
                merged_match = self._verify_speaker_span(
                    verifier,
                    waveform,
                    merged_span,
                    profile,
                    threshold,
                )
                if not merged_match.accepted:
                    continue
                identity_match = merged_match
                exclusion = verifier.exclusion_audit(
                    identity_match,
                    profile,
                    exclusion_profiles,
                )
                if exclusion and exclusion.get("excluded_role_rejected"):
                    continue

                replacement = common.CandidateSentence(
                    merged_span.start,
                    merged_span.end,
                    "",
                )
                replacement.diagnostics.update(core.diagnostics)
                self._apply_speaker_match(
                    replacement,
                    identity_match,
                    profile,
                    threshold,
                )
                replacement.diagnostics.update(
                    {
                        "final_same_speaker_edge_recovery": True,
                        "final_same_speaker_core_span": [core.start, core.end],
                        "final_same_speaker_edge_span": [edge.start, edge.end],
                        "final_same_speaker_gap_seconds": round(gap, 5),
                        "final_same_speaker_gap_quiet_ratio": round(
                            quiet_ratio, 5
                        ),
                        "final_same_speaker_gap_quiet_floor_db": round(
                            quiet_floor, 3
                        ),
                        "final_same_speaker_wavlm_pair": round(
                            pair_similarity, 5
                        ),
                        "final_same_speaker_wavlm_floor": round(
                            same_speaker_floor, 5
                        ),
                        "final_same_speaker_edge_target": round(
                            edge_target_score, 5
                        ),
                        "final_same_speaker_source_block": [
                            source_block.start,
                            source_block.end,
                        ],
                        "final_same_speaker_internal_boundary": (
                            internal_boundary.to_dict()
                            if internal_boundary is not None
                            else None
                        ),
                        "final_target_core_anchored_extension": False,
                        "post_target_silence_merge": True,
                    }
                )
                if exclusion is not None:
                    replacement.diagnostics.update(exclusion)
                accepted_turns.remove(core)
                accepted_turns.append(replacement)
                used_edges.add(id(edge))
                rejected[:] = [
                    item for item in rejected if id(item) != id(edge)
                ]
                if (
                    discarded_internal_span is not None
                    and discarded_internal_span.duration >= 0.20
                ):
                    discarded_internal = common.CandidateSentence(
                        discarded_internal_span.start,
                        discarded_internal_span.end,
                        "",
                        reject_reason="新增句尾内部检测到换人，已裁掉非目标人物一侧",
                    )
                    discarded_internal.diagnostics.update(
                        {
                            "final_same_speaker_internal_discard": True,
                            "final_same_speaker_internal_boundary": (
                                internal_boundary.to_dict()
                                if internal_boundary is not None
                                else None
                            ),
                        }
                    )
                    rejected.append(discarded_internal)
                expanded += 1
                break

        accepted_turns.sort(key=lambda item: (item.start, item.end))
        progress(
            1.0,
            f"边界终审完成：补全 {expanded} 段，裁切 {cropped} 段",
        )
        return expanded, cropped
