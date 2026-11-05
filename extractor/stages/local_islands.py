"""local islands responsibilities; composed by ExtractionPipeline."""
from __future__ import annotations

from .. import pipeline_common as common


class LocalIslandsMethods:
    def _experimental_audit_final_islands(
        self,
        accepted_turns: list[common.CandidateSentence],
        rejected: list[common.CandidateSentence],
        verifier: common.DualSpeakerVerifier,
        profile: common.SpeakerMatchProfile,
        stem_waveform: common.torch.Tensor,
        stem_path: common.Path,
        raw_path: common.Path,
        threshold: float,
        target_reference_clips: list[common.Path],
        raw_reference_clips: list[common.Path],
        negative_reference_clips: list[list[common.Path]],
        raw_negative_reference_clips: list[list[common.Path]],
        exclusion_profiles: list[common.ExclusionSpeakerProfile],
        blocked_spans: common.Iterable[common.TimeSpan],
        progress: common.ProgressCallback,
    ) -> tuple[int, int]:
        """Resolve local identity conflicts before STT.

        A parent's high speaker score cannot supply identity to a separate
        short speech island. Independently conflicting islands withdraw the
        parent. Only exact target-supported child spans can be reintroduced.
        Without exclusion references this check abstains.
        """
        if (not accepted_turns or not negative_reference_clips
                or self._raw_target_waveform is None):
            return 0, 0
        from ..nextgen.anime_embedding import AnimeSpeakerOnnx
        from ..nextgen.domain_identity import DomainReferenceBank, ReferenceAudio
        from ..nextgen.internal_pauses import (
            locate_quiet_gaps, locate_terminal_short_tail_gap,
        )
        from ..nextgen.purity_consensus import (
            eligible_identity_island, local_other_witnesses, target_runs,
            whole_identity_conflict,
        )
        # Default-on now, so a missing optional asset must abstain instead of
        # failing the one-click run.
        if not all(
            (common.ASSET_ROOT / "models" / name).is_file()
            for name in (
                "anime-speaker-char/anime_speaker_char_ecapa.onnx",
                "anime-speaker-va/anime_speaker_va_ecapa.onnx",
            )
        ):
            return 0, 0

        raw_waveform = self._raw_target_waveform
        references: list[ReferenceAudio] = []

        def add_references(role: str, stems: list[common.Path], raws: list[common.Path]) -> None:
            by_stem = {path.name: path for path in stems}
            by_raw = {path.name: path for path in raws}
            if not by_stem or by_stem.keys() != by_raw.keys():
                raise ValueError("局部身份终审需要成对的原音/UVR 参考音频")
            for name in sorted(by_stem):
                references.append(ReferenceAudio(
                    role=role,
                    stem=common.load_mono(by_stem[name], 16000),
                    raw=common.load_mono(by_raw[name], 16000),
                    sample_rate=16000,
                ))

        add_references("target", target_reference_clips, raw_reference_clips)
        if len(negative_reference_clips) != len(raw_negative_reference_clips):
            raise ValueError("局部身份终审的排除人物参考组未对齐")
        for index, (stems, raws) in enumerate(zip(
            negative_reference_clips, raw_negative_reference_clips,
        ), 1):
            add_references(f"role_{index:03d}", stems, raws)
        bank = DomainReferenceBank(
            dict(zip(("char", "va"), verifier.anime_pair())),
            references,
        )
        blocked = tuple(blocked_spans)

        def domain_for(span: common.TimeSpan):
            if not eligible_identity_island(span.start, span.end):
                return None
            try:
                return bank.score(
                    stem=self._waveform_span(stem_waveform, span),
                    raw=self._waveform_span(raw_waveform, span),
                    sample_rate=16000,
                )
            except ValueError:
                return None

        def alternate_audio(path: common.Path, span: common.TimeSpan) -> common.torch.Tensor:
            # Resampling after bounded source decoding is a legitimate second
            # preprocessing view. It is used only to *abstain* when the main
            # character-stem sign is unstable and the production verifier is
            # already near an exclusion tie.
            import librosa
            import soundfile as sf

            with sf.SoundFile(path) as source:
                rate = source.samplerate
                source.seek(round(span.start * rate))
                samples = source.read(
                    round(span.duration * rate), dtype="float32", always_2d=True,
                ).mean(axis=1)
            if rate != 16000:
                samples = librosa.resample(
                    samples, orig_sr=rate, target_sr=16000,
                )
            return common.torch.from_numpy(samples.copy())

        def margins(verdict) -> dict[str, float | None]:
            return {
                f"{variant}_{channel}": verdict.score(variant, channel).margin
                for variant in ("char", "va")
                for channel in ("stem", "raw")
            }

        total = len(accepted_turns)
        withheld = 0
        rescued = 0
        superseded_ids: set[int] = set()
        for index, candidate in enumerate(list(accepted_turns), 1):
            if id(candidate) in superseded_ids:
                # A verified child superseded this fragment after the snapshot
                # was taken; its rejection is already recorded.
                continue
            parent = common.TimeSpan(candidate.start, candidate.end)
            whole_domain = domain_for(parent)
            whole_match = self._verify_speaker_span(
                verifier, stem_waveform, parent, profile, threshold, audit_raw=True,
            )
            whole_exclusion = verifier.exclusion_audit(
                whole_match, profile, exclusion_profiles,
            ) or {}
            whole_margins = (
                margins(whole_domain) if whole_domain is not None else None
            )
            alternate_margins = None
            direct_margin = whole_exclusion.get(
                "excluded_primary_direct_margin"
            )
            if (whole_domain is not None and direct_margin is not None
                    and float(direct_margin) <= 0.02):
                try:
                    alternate = bank.score(
                        stem=alternate_audio(stem_path, parent),
                        raw=alternate_audio(raw_path, parent),
                        sample_rate=16000,
                    )
                    alternate_margins = margins(alternate)
                except ValueError:
                    pass
            whole_conflict = bool(
                whole_margins is not None and alternate_margins is not None
                and whole_identity_conflict(
                    whole_margins, alternate_margins, direct_margin,
                )
            )
            samples = self._waveform_span(stem_waveform, parent).detach().cpu().numpy()
            gaps = list(locate_quiet_gaps(
                samples, 16000,
                lower_seconds=self.options.silence_min_seconds,
                upper_seconds=self.options.silence_split_seconds,
            ))
            terminal_gap = locate_terminal_short_tail_gap(
                samples, 16000,
                lower_seconds=self.options.silence_min_seconds,
                upper_seconds=self.options.silence_split_seconds,
            )
            if terminal_gap is not None and not any(
                terminal_gap.start < gap.end and gap.start < terminal_gap.end
                for gap in gaps
            ):
                gaps.append(terminal_gap)
                gaps.sort(key=lambda gap: gap.start)
            else:
                terminal_gap = None
            terminal_tail_start = (
                candidate.start + terminal_gap.end / 16000.0
                if terminal_gap is not None else None
            )
            island_spans: list[common.TimeSpan] = []
            cursor = candidate.start
            for gap in gaps:
                gap_start = candidate.start + gap.start / 16000.0
                gap_end = candidate.start + gap.end / 16000.0
                if gap_start > cursor:
                    island_spans.append(common.TimeSpan(cursor, gap_start))
                cursor = gap_end
            if cursor < candidate.end:
                island_spans.append(common.TimeSpan(cursor, candidate.end))

            island_rows: list[dict] = []
            suspect_spans: set[tuple[float, float]] = set()
            for island in island_spans:
                terminal_tail = (
                    terminal_tail_start is not None
                    and abs(island.start - terminal_tail_start) <= 0.02
                    and abs(island.end - candidate.end) <= 0.02
                )
                domain = None if terminal_tail else domain_for(island)
                state = (
                    "unresolved_terminal_short_tail" if terminal_tail
                    else domain.state if domain is not None else "unresolved_short"
                )
                witnesses: tuple[str, ...] = ()
                local_exclusion: dict = {}
                local_margins = margins(domain) if domain is not None else None
                if domain is not None:
                    local_match = self._verify_speaker_span(
                        verifier, stem_waveform, island, profile, threshold,
                        audit_raw=True,
                    )
                    local_exclusion = verifier.exclusion_audit(
                        local_match, profile, exclusion_profiles,
                    ) or {}
                    witnesses = local_other_witnesses(
                        local_margins, local_exclusion,
                    )
                if witnesses:
                    suspect_spans.add((island.start, island.end))
                if terminal_tail:
                    # A terminal syllable shorter than the acoustic flank
                    # cannot inherit the earlier speaker's identity. It may
                    # be the target or somebody else; neither is presumed.
                    suspect_spans.add((island.start, island.end))
                island_rows.append({
                    "span": [island.start, island.end],
                    "state": state,
                    "domain_margins": local_margins,
                    "excluded_role": local_exclusion.get("excluded_role"),
                    "excluded_primary_margin": local_exclusion.get(
                        "excluded_primary_margin"
                    ),
                    "excluded_role_rejected": local_exclusion.get(
                        "excluded_role_rejected", False
                    ),
                    "other_witnesses": list(witnesses),
                    "terminal_tail_unresolved": terminal_tail,
                })
            candidate.diagnostics["experimental_final_island_evidence"] = {
                "whole_domain_margins": whole_margins,
                "whole_alternate_margins": alternate_margins,
                "whole_excluded_primary_direct_margin": whole_exclusion.get(
                    "excluded_primary_direct_margin"
                ),
                "whole_conflict": whole_conflict,
                "islands": island_rows,
                "terminal_short_tail_gap": (
                    [candidate.start + terminal_gap.start / 16000.0,
                     candidate.start + terminal_gap.end / 16000.0]
                    if terminal_gap is not None else None
                ),
            }
            if not whole_conflict and not suspect_spans:
                continue

            accepted_turns.remove(candidate)
            candidate.reject_reason = "最终局部身份冲突，原混合片段已撤回"
            candidate.diagnostics["experimental_final_island_consensus"] = {
                "whole_conflict": whole_conflict,
                "suspect_islands": [list(span) for span in sorted(suspect_spans)],
            }
            rejected.append(candidate)
            withheld += 1
            if whole_conflict and not suspect_spans:
                continue

            for left, right in target_runs(
                island_rows, suspect_spans,
                maximum_gap_seconds=self.options.silence_split_seconds,
            ):
                child_span = common.TimeSpan(left, right)
                child = common.CandidateSentence(left, right, "")
                child.diagnostics["final_purity_parent_span"] = [
                    candidate.start, candidate.end,
                ]
                contained: list[common.CandidateSentence] = []
                if child_span.duration + 1e-6 < self.options.min_output_seconds:
                    child.reject_reason = "局部目标子句短于当前导出下限"
                elif any(
                    min(child_span.end, mask.end)
                    - max(child_span.start, mask.start) > 0.01
                    for mask in blocked
                ):
                    child.reject_reason = "局部目标子句跨越歌声或多人遮罩"
                else:
                    overlapping = [
                        other for other in accepted_turns
                        if common.reserves_audio(other) and min(child_span.end, other.end)
                        - max(child_span.start, other.start) > 0.10
                    ]
                    if any(
                        other.start < child_span.start - 0.02
                        or other.end > child_span.end + 0.02
                        for other in overlapping
                    ):
                        child.reject_reason = "局部目标子句与已保留回合重叠"
                    else:
                        contained = overlapping
                    if not child.reject_reason:
                        child_match = self._verify_speaker_span(
                            verifier, stem_waveform, child_span, profile, threshold,
                            audit_raw=True,
                        )
                        child_exclusion = verifier.exclusion_audit(
                            child_match, profile, exclusion_profiles,
                        ) or {}
                        child_domain = domain_for(child_span)
                        if (not child_match.accepted
                                or child_exclusion.get("excluded_role_rejected")
                                or child_domain is None
                                or child_domain.state != "target_supported"):
                            child.reject_reason = "局部目标子句身份未获独立确认"
                            if (not child_match.accepted
                                    and not child_exclusion.get("excluded_role_rejected")
                                    and child_domain is not None
                                    and child_domain.state == "target_supported"):
                                # The anime-domain models confirm the exact child;
                                # only the legacy voiceprint gate missed it. Hand
                                # it to the standard classifier rescue instead of
                                # a dead-end reason, so evidence trained on
                                # user-reviewed audio gets the final say.
                                child.reject_reason = "声纹匹配不足"
                        else:
                            # The child passed on its own audio. Accepted turns
                            # it fully covers are narrower stale fragments; the
                            # complete verified clause replaces them instead of
                            # being vetoed by them.
                            for stale in contained:
                                accepted_turns.remove(stale)
                                stale.accepted = False
                                stale.reject_reason = "被更完整的局部目标子句取代"
                                stale.diagnostics["superseded_by_child"] = [
                                    child.start, child.end,
                                ]
                                superseded_ids.add(id(stale))
                                rejected.append(stale)
                            self._apply_speaker_match(
                                child, child_match, profile, threshold,
                            )
                            child.diagnostics.update(child_exclusion)
                            child.diagnostics["local_identity_audit"] = {
                                "span": [left, right],
                                "passed": True,
                                "method": "experimental_island_consensus_child",
                            }
                            child.diagnostics["experimental_purity_child"] = True
                            accepted_turns.append(child)
                            rescued += 1
                if child.reject_reason:
                    rejected.append(child)
            progress(0.80, f"实验性局部身份终审：{index}/{total}")
        accepted_turns.sort(key=lambda item: (item.start, item.end))
        from ..utterance_assembly import recheck_short_exclusion_tails
        reconsidered = recheck_short_exclusion_tails(
            rejected, accepted_turns, domain_for,
            blocked=blocked, margin=max(.06, self.options.cluster_target_margin))
        if reconsidered:
            progress(0.80, f"短尾音独立复核：{reconsidered} 段解除近似声纹否决，交最终身份筛选")
        return withheld, rescued

    def _experimental_reassemble_internal_other_tails(
        self,
        accepted_turns: list[common.CandidateSentence],
        rejected: list[common.CandidateSentence],
        verifier: common.DualSpeakerVerifier,
        profile: common.SpeakerMatchProfile,
        stem_waveform: common.torch.Tensor,
        threshold: float,
        target_reference_clips: list[common.Path],
        raw_reference_clips: list[common.Path],
        negative_reference_clips: list[list[common.Path]],
        raw_negative_reference_clips: list[list[common.Path]],
        exclusion_profiles: list[common.ExclusionSpeakerProfile],
        blocked_spans: common.Iterable[common.TimeSpan],
    ) -> tuple[int, int]:
        """Reassemble local speech after VAD missed a short intervening voice.

        An energy trough only proposes a boundary.  Both sides are checked by
        independent frozen char/va models on stem and original audio.  A
        replacement is exported only after the exact new span passes the
        ordinary full verifier and supplied exclusion-person audit.  With no
        exclusion references the witness abstains and this path does nothing.
        """
        if not negative_reference_clips or self._raw_target_waveform is None:
            return 0, 0
        from ..nextgen.anime_embedding import AnimeSpeakerOnnx
        from ..nextgen.domain_identity import DomainReferenceBank, ReferenceAudio
        from ..nextgen.internal_pauses import locate_quiet_gaps

        raw_waveform = self._raw_target_waveform
        blocked = tuple(blocked_spans)
        possible: list[tuple[common.CandidateSentence, float, float, float]] = []
        for candidate in accepted_turns:
            span = common.TimeSpan(candidate.start, candidate.end)
            samples = self._waveform_span(stem_waveform, span).detach().cpu().numpy()
            for gap in locate_quiet_gaps(
                samples, 16000,
                lower_seconds=self.options.silence_min_seconds,
                upper_seconds=self.options.silence_split_seconds,
            ):
                gap_start = candidate.start + gap.start / 16000.0
                gap_end = candidate.start + gap.end / 16000.0
                tail_seconds = candidate.end - gap_end
                if 0.20 <= tail_seconds <= 0.80:
                    possible.append((candidate, gap_start, gap_end, gap.length / 16000.0))
        if not possible:
            return 0, 0

        def add_references(
            destination: list[ReferenceAudio], role: str,
            stem_paths: list[common.Path], raw_paths: list[common.Path],
        ) -> None:
            stems = {path.name: path for path in stem_paths}
            raws = {path.name: path for path in raw_paths}
            if not stems or stems.keys() != raws.keys():
                raise ValueError("Anime local audit needs paired raw/stem reference clips")
            for name in sorted(stems):
                destination.append(ReferenceAudio(
                    role=role,
                    stem=common.load_mono(stems[name], 16000),
                    raw=common.load_mono(raws[name], 16000),
                    sample_rate=16000,
                ))

        references: list[ReferenceAudio] = []
        add_references(references, "target", target_reference_clips, raw_reference_clips)
        if len(negative_reference_clips) != len(raw_negative_reference_clips):
            raise ValueError("Anime local audit negative reference groups are mispaired")
        for index, (stem_group, raw_group) in enumerate(
            zip(negative_reference_clips, raw_negative_reference_clips), 1,
        ):
            add_references(references, f"role_{index:03d}", stem_group, raw_group)
        bank = DomainReferenceBank(
            dict(zip(("char", "va"), verifier.anime_pair())),
            references,
        )

        def local_state(span: common.TimeSpan) -> str:
            try:
                return bank.score(
                    stem=self._waveform_span(stem_waveform, span),
                    raw=self._waveform_span(raw_waveform, span),
                    sample_rate=16000,
                ).state
            except ValueError:
                return "unresolved"

        replaced = 0
        withheld = 0
        for candidate, gap_start, gap_end, gap_seconds in possible:
            if not any(item is candidate for item in accepted_turns):
                continue
            left = common.TimeSpan(candidate.start, gap_start)
            analysis_end = min(
                raw_waveform.numel() / 16000.0,
                candidate.end + 0.12,
                max(candidate.end, gap_end + 0.28),
            )
            if analysis_end - gap_end < 0.20:
                continue
            tail = common.TimeSpan(gap_end, analysis_end)
            if local_state(left) != "target_supported" or local_state(tail) != "other_supported":
                continue

            # Try the clean cropped core first, then an independently
            # target-supported neighboring rejected speech island.  No weak
            # side inherits identity merely from a high-scoring whole clip.
            choices: list[tuple[common.TimeSpan, common.CandidateSentence | None]] = []
            predecessors = sorted(
                (
                    edge for edge in rejected
                    if edge.reject_reason == "声纹匹配不足"
                    and self.options.silence_min_seconds
                    <= candidate.start - edge.end
                    <= self.options.silence_split_seconds
                    and edge.start < candidate.start
                ),
                key=lambda edge: edge.end,
                reverse=True,
            )
            for edge in predecessors:
                edge_span = common.TimeSpan(edge.start, edge.end)
                if local_state(edge_span) == "target_supported":
                    choices.append((common.TimeSpan(edge.start, gap_start), edge))
            choices.append((left, None))

            replacement: common.CandidateSentence | None = None
            used_edge: common.CandidateSentence | None = None
            for proposed, edge in choices:
                if proposed.duration + 1e-6 < self.options.min_output_seconds:
                    continue
                if any(
                    min(proposed.end, blocked_span.end)
                    - max(proposed.start, blocked_span.start) > 0.01
                    for blocked_span in blocked
                ):
                    continue
                if any(
                    (item.diagnostics.get("excluded_role_rejected")
                     or item.diagnostics.get("structural_hard_reject"))
                    and min(proposed.end, item.end) - max(proposed.start, item.start) > 0.08
                    for item in rejected if item is not edge
                ):
                    continue
                if any(
                    item is not candidate
                    and min(proposed.end, item.end) - max(proposed.start, item.start) > 0.10
                    for item in accepted_turns
                ):
                    continue
                match = self._verify_speaker_span(
                    verifier, stem_waveform, proposed, profile, threshold,
                    audit_raw=True,
                )
                if not match.accepted:
                    continue
                exclusion = verifier.exclusion_audit(match, profile, exclusion_profiles)
                if exclusion and exclusion.get("excluded_role_rejected"):
                    continue
                replacement = common.CandidateSentence(proposed.start, proposed.end, "")
                replacement.diagnostics.update(candidate.diagnostics)
                self._apply_speaker_match(replacement, match, profile, threshold)
                replacement.diagnostics.update({
                    "experimental_internal_reassembly": True,
                    "internal_pause_parent_span": [candidate.start, candidate.end],
                    "internal_pause_gap_span": [round(gap_start, 5), round(gap_end, 5)],
                    "internal_pause_gap_seconds": round(gap_seconds, 5),
                    "internal_pause_tail_analysis_span": [round(tail.start, 5), round(tail.end, 5)],
                    "internal_pause_predecessor_span": ([edge.start, edge.end] if edge else None),
                    "local_identity_audit": {"span": [proposed.start, proposed.end], "passed": True,
                                             "method": "char_va_local_parts_plus_whole_verifier"},
                })
                if exclusion is not None:
                    replacement.diagnostics.update(exclusion)
                used_edge = edge
                break

            accepted_turns.remove(candidate)
            candidate.reject_reason = "内部停顿后短音支持其他人物，原混合片段已撤回"
            rejected.append(candidate)
            if replacement is None:
                withheld += 1
                continue
            accepted_turns.append(replacement)
            if used_edge is not None:
                rejected[:] = [item for item in rejected if item is not used_edge]
            rejected.append(common.CandidateSentence(
                gap_end, candidate.end, "",
                reject_reason="内部停顿后短音支持其他人物，未并入目标句",
                diagnostics={"internal_pause_tail_other_supported": True,
                             "parent_span": [candidate.start, candidate.end]},
            ))
            replaced += 1
        accepted_turns.sort(key=lambda item: (item.start, item.end))
        return replaced, withheld

    def _experimental_rescue_adjacent_domain_sides(
        self,
        accepted_turns: list[common.CandidateSentence],
        rejected: list[common.CandidateSentence],
        verifier: common.DualSpeakerVerifier,
        profile: common.SpeakerMatchProfile,
        stem_waveform: common.torch.Tensor,
        stem_path: common.Path,
        raw_path: common.Path,
        threshold: float,
        target_reference_clips: list[common.Path],
        raw_reference_clips: list[common.Path],
        negative_reference_clips: list[list[common.Path]],
        raw_negative_reference_clips: list[list[common.Path]],
        exclusion_profiles: list[common.ExclusionSpeakerProfile],
        blocked_spans: common.Iterable[common.TimeSpan],
        vad_spans: list[common.TimeSpan],
    ) -> int:
        """Recover a rejected neighboring span using independent domain evidence.

        The former's identity is checked on its *own exact audio*. Neither the
        core nor the high-scoring concatenation can lend it target identity.
        Multiscale changes, singing/overlap and final exclusion still veto.
        """
        if (not negative_reference_clips or self._raw_target_waveform is None
                or verifier.secondary is None):
            return 0
        from ..anime_identity import AnimeDomainWitness

        edges = [
            item for item in rejected
            if item.reject_reason == "声纹匹配不足"
            and item.duration >= 0.85
            and not item.diagnostics.get("excluded_role_rejected")
            and not item.diagnostics.get("structural_hard_reject")
        ]
        pairs: list[tuple[common.CandidateSentence, common.CandidateSentence, common.TimeSpan]] = []
        queued: set[tuple[int, int]] = set()

        def queue_adjacent_edges(core: common.CandidateSentence) -> None:
            if not self._install_acoustic_identity_certificate(core):
                return
            for edge in edges:
                gap = (core.start - edge.end if edge.end <= core.start
                       else edge.start - core.end if core.end <= edge.start
                       else -1.0)
                if not 0 <= gap <= self.options.silence_split_seconds:
                    continue
                proposed = common.TimeSpan(min(core.start, edge.start),
                                    max(core.end, edge.end))
                if proposed.duration > min(20.0, self.options.max_sentence_seconds):
                    continue
                key = (id(core), id(edge))
                if key in queued:
                    continue
                queued.add(key)
                pairs.append((core, edge, proposed))

        for core in tuple(accepted_turns):
            queue_adjacent_edges(core)
        if not pairs:
            return 0
        models_root = common.ASSET_ROOT / "models"
        if not all((models_root / name).is_file() for name in (
            "anime-speaker-char/anime_speaker_char_ecapa.onnx",
            "anime-speaker-va/anime_speaker_va_ecapa.onnx",
        )):
            return 0
        witness = AnimeDomainWitness(
            models_root, target_reference_clips, raw_reference_clips,
            negative_reference_clips, raw_negative_reference_clips,
            encoders=verifier.anime_pair(),
        )
        splitter = common.LocalSpeakerTurnSplitter(
            verifier.primary, secondary=verifier.secondary,
        )
        raw_waveform = self._raw_target_waveform
        forbidden = tuple(blocked_spans)
        rescued = 0
        for pair_index, (core, edge, proposed) in enumerate(pairs, 1):
            if core not in accepted_turns or edge not in rejected:
                continue
            # A whole-span embedding can hide a third short voice.  Do
            # not join across speech in the *gap* between two independently
            # checked sides. Rejections wholly inside the audited core may
            # be alternate proposals, not a third physical speech island.
            gap = (common.TimeSpan(edge.end, core.start) if edge.end <= core.start
                   else common.TimeSpan(core.end, edge.start))
            if any(
                min(gap.end, island.end) - max(gap.start, island.start) > 0.03
                for island in vad_spans
            ):
                continue
            if any(
                item is not edge and item.reject_reason
                and min(gap.end, item.end)
                - max(gap.start, item.start) > 0.03
                for item in rejected
            ):
                continue
            if any(
                min(proposed.end, block.end)
                - max(proposed.start, block.start) > 0.01
                for block in forbidden
            ):
                continue
            if any(
                item is not core and
                min(proposed.end, item.end)
                - max(proposed.start, item.start) > 0.10
                for item in accepted_turns
            ):
                continue
            if any(
                item is not edge
                and (item.diagnostics.get("excluded_role_rejected")
                     or item.diagnostics.get("structural_hard_reject")
                     or item.reject_reason in ("检测到有人唱歌", "检测到多人同时发声"))
                and min(proposed.end, item.end)
                - max(proposed.start, item.start) > 0.01
                for item in rejected
            ):
                continue
            try:
                side_states = tuple(
                    witness.score(
                        self._waveform_span(stem_waveform, part),
                        self._waveform_span(raw_waveform, part),
                    ).state
                    for part in (
                        common.TimeSpan(edge.start, edge.end),
                        common.TimeSpan(core.start, core.end),
                    )
                )
            except ValueError as error:
                if str(error) not in (
                    "Silent audio cannot provide speaker identity evidence",
                    "Expected at least 0.2 s of finite mono audio at 16 kHz",
                ):
                    raise
                continue
            if side_states != ("target_supported", "target_supported"):
                continue
            match = self._verify_speaker_span(
                verifier, stem_waveform, proposed, profile, threshold,
                audit_raw=True,
            )
            if not match.accepted:
                continue
            exclusion = verifier.exclusion_audit(
                match, profile, exclusion_profiles,
            )
            if exclusion and exclusion.get("excluded_role_rejected"):
                continue
            local_span = [common.TimeSpan(0.0, proposed.duration)]
            if (splitter.detect_multiscale_speaker_boundaries(
                    self._waveform_span(stem_waveform, proposed), local_span)
                    or splitter.detect_multiscale_speaker_boundaries(
                        self._waveform_span(raw_waveform, proposed), local_span)):
                continue
            merged = common.CandidateSentence(proposed.start, proposed.end, "")
            self._apply_speaker_match(merged, match, profile, threshold)
            merged.diagnostics.update({
                "experimental_adjacent_domain_rescue": True,
                "rescued_side_span": [edge.start, edge.end],
                "audited_core_span": [core.start, core.end],
                "local_identity_audit": {
                    "span": [proposed.start, proposed.end],
                    "passed": True,
                    "method": "char_va_both_sides_plus_multiscale_change_review",
                },
                "speech_ratio": round(
                    common.speech_ratio(vad_spans, proposed.start, proposed.end), 5,
                ),
            })
            merged.diagnostics["speech_seconds"] = round(
                merged.duration * merged.diagnostics["speech_ratio"], 5,
            )
            if exclusion is not None:
                merged.diagnostics.update(exclusion)
            accepted_turns.remove(core)
            accepted_turns.append(merged)
            rejected.remove(edge)
            rescued += 1
            queue_adjacent_edges(merged)
        accepted_turns.sort(key=lambda item: (item.start, item.end))
        return rescued

    def _classifier_target_rescue(
        self,
        accepted_turns: list[common.CandidateSentence],
        rejected: list[common.CandidateSentence],
        stem_waveform: common.torch.Tensor,
        progress: common.ProgressCallback,
        *,
        scorer=None,
        threshold: float | None = None,
    ) -> int:
        """Accept identity-gate rejects that the anime-domain classifier
        confidently scores as the target speaker.

        The classifier is trained only on user-verified spans; its vote is
        additional evidence for fragments the four voiceprint models left
        below threshold, never a replacement for the formal gates. Overlap-
        and singing-blocked turns, structurally rejected turns and
        user-marked spans stay excluded.
        """
        if scorer is None:
            loaded = self._ensure_tail_scorer()
            if loaded is None:
                return 0
            threshold = loaded[3]

            def scorer(span: common.TimeSpan, wave: common.torch.Tensor) -> float | None:
                return self._score_local_window(loaded, wave)

        from ..local_identity_classifier import window_starts

        sr = 16000
        window = int(0.5 * sr)
        excluded_spans = [
            common.TimeSpan(start, end)
            for start, end in self.options.user_excluded_spans
        ]
        excluded_spans.extend(self._raw_blocked_spans)
        accepted_count = 0
        for candidate in list(rejected):
            if candidate.reject_reason != "声纹匹配不足":
                continue
            if candidate.diagnostics.get("structural_hard_reject"):
                continue
            if candidate.diagnostics.get("excluded_role_rejected"):
                continue
            if not (
                common.MIN_TARGET_SECONDS - 1e-9
                <= candidate.duration
                <= self.options.max_sentence_seconds
            ):
                continue
            span = common.TimeSpan(candidate.start, candidate.end)
            if any(
                min(span.end, excluded.end) - max(span.start, excluded.start) > 1e-9
                for excluded in excluded_spans
            ):
                continue
            if any(
                min(span.end, accepted.end) - max(span.start, accepted.start) > 0.10
                for accepted in accepted_turns
            ):
                # Already covered by an accepted clip; exporting it again
                # would duplicate half of an existing sentence.
                continue
            wave = self._waveform_span(stem_waveform, span)
            if wave.numel() < round(common.MIN_TARGET_SECONDS * sr) - 1:
                continue
            scores = []
            # score_window pads its encoder input, never the source interval.
            for pos in window_starts(wave.numel()) or [0]:
                value = scorer(
                    common.TimeSpan(span.start + pos / sr, min(span.end, span.start + (pos + window) / sr)),
                    wave[pos:pos + window],
                )
                if value is None or not common.math.isfinite(value):
                    scores = []
                    break
                scores.append(value)
            if not scores:
                continue
            mean_score = sum(scores) / len(scores)
            if mean_score < threshold:
                continue
            candidate.reject_reason = ""
            candidate.diagnostics["classifier_target_rescue"] = {
                "mean_score": round(mean_score, 3),
                "threshold": threshold,
                "windows": len(scores),
            }
            accepted_turns.append(candidate)
            rejected.remove(candidate)
            accepted_count += 1
        if accepted_count:
            progress(
                0.80,
                f"分类器目标救援：恢复 {accepted_count} 段声纹不足碎片",
            )
        return accepted_count
