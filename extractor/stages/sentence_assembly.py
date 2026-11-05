"""sentence assembly responsibilities; composed by ExtractionPipeline."""
from __future__ import annotations

from .. import pipeline_common as common


class SentenceAssemblyMethods:
    @classmethod
    def _coalesce_short_sentences(
        cls,
        sentences: list[common.CandidateSentence],
        minimum_seconds: float,
        max_gap: float = 0.65,
    ) -> list[common.CandidateSentence]:
        """Merge tiny ASR fragments inside one already verified speaker turn."""

        output = list(sorted(sentences, key=lambda item: (item.start, item.end)))
        index = 0
        while index < len(output):
            current = output[index]
            if current.duration >= minimum_seconds:
                index += 1
                continue
            if index + 1 < len(output):
                following = output[index + 1]
                if following.start - current.end <= max_gap:
                    current.end = max(current.end, following.end)
                    current.whisper_text = cls._join_transcript_text(
                        current.whisper_text, following.whisper_text
                    )
                    current.text = cls._join_transcript_text(current.text, following.text)
                    current.diagnostics["stt_short_merged"] = True
                    current.diagnostics["stt_merged_span"] = [following.start, following.end]
                    del output[index + 1]
                    continue
            if index > 0:
                previous = output[index - 1]
                if current.start - previous.end <= max_gap:
                    previous.end = max(previous.end, current.end)
                    previous.whisper_text = cls._join_transcript_text(
                        previous.whisper_text, current.whisper_text
                    )
                    previous.text = cls._join_transcript_text(previous.text, current.text)
                    previous.diagnostics["stt_short_merged"] = True
                    previous.diagnostics["stt_merged_span"] = [current.start, current.end]
                    del output[index]
                    index = max(0, index - 1)
                    continue
            index += 1
        return output

    @staticmethod
    def _merge_adjacent_target_turns(
        turns: list[common.CandidateSentence],
        minimum_seconds: float,
        max_gap: float = 0.25,
    ) -> list[common.CandidateSentence]:
        """Rejoin target turns split by an over-sensitive local boundary."""

        output = list(sorted(turns, key=lambda item: (item.start, item.end)))
        index = 0
        while index + 1 < len(output):
            left, right = output[index], output[index + 1]
            gap = max(0.0, right.start - left.end)
            left_index = left.diagnostics.get("speaker_turn_index")
            right_index = right.diagnostics.get("speaker_turn_index")
            adjacent_turns = (
                isinstance(left_index, int)
                and isinstance(right_index, int)
                and right_index == left_index + 1
            )
            if (
                gap <= max_gap
                and adjacent_turns
                and left.diagnostics.get("speech_block_index")
                == right.diagnostics.get("speech_block_index")
            ):
                left.end = max(left.end, right.end)
                left.diagnostics["speaker_turns_merged"] = True
                left.diagnostics["merged_turn_span"] = [right.start, right.end]
                # Keep conservative metadata when two verified turns are joined.
                left.speaker_score = min(left.speaker_score, right.speaker_score)
                left.window_p20_score = min(left.window_p20_score, right.window_p20_score)
                left.speaker_vote_ratio = min(left.speaker_vote_ratio, right.speaker_vote_ratio)
                left.window_vote_ratio = min(left.window_vote_ratio, right.window_vote_ratio)
                del output[index + 1]
                continue
            index += 1
        return output

    @staticmethod
    def _target_coverage(span: common.TimeSpan, target_spans: list[common.TimeSpan]) -> float:
        if span.duration <= 0.0 or not target_spans:
            return 0.0
        covered = sum(
            max(0.0, min(span.end, target.end) - max(span.start, target.start))
            for target in target_spans
        )
        return min(1.0, covered / span.duration)

    def _merge_short_silence_same_speaker(
        self,
        spans: list[common.TimeSpan],
        verifier: common.DualSpeakerVerifier,
        profile: common.SpeakerMatchProfile,
        waveform: common.torch.Tensor,
        progress: common.ProgressCallback,
        *,
        maximum_silence_seconds: float = 0.85,
        primary_same_floor: float = 0.76,
        secondary_same_floor: float = 0.64,
        forbidden_joins: common.Iterable[common.TimeSpan] = (),
        strict_flags: list[bool] | None = None,
    ) -> list[common.TimeSpan]:
        """Join nearby verified speech without crossing unconfirmed identities."""
        if strict_flags is not None and len(strict_flags) != len(spans):
            raise ValueError("strict_flags must align with spans")
        order = sorted(range(len(spans)), key=lambda index: (spans[index].start, spans[index].end))
        spans = [spans[index] for index in order]
        if strict_flags is not None:
            strict_flags = [strict_flags[index] for index in order]
        if len(spans) < 2:
            return spans
        forbidden_joins = tuple(forbidden_joins)
        parts = [self._waveform_span(waveform, span) for span in spans]
        progress(0.0, f"短静音声纹核验：ERes2Net 0/{len(parts)}")
        primary_embeddings = verifier.primary._embeddings_from_waveforms(
            parts,
            progress=lambda completed, total: progress(
                0.50 * completed / max(1, total),
                f"短静音声纹核验：ERes2Net {completed}/{total}",
            ),
        )
        verifier._ensure_secondary(profile)
        assert verifier.secondary is not None
        progress(0.50, f"短静音声纹核验：CAM++ 0/{len(parts)}")
        secondary_embeddings = verifier.secondary._embeddings_from_waveforms(
            parts,
            progress=lambda completed, total: progress(
                0.50 + 0.50 * completed / max(1, total),
                f"短静音声纹核验：CAM++ {completed}/{total}",
            ),
        )

        output: list[common.TimeSpan] = [spans[0]]
        for index in range(1, len(spans)):
            previous_source = spans[index - 1]
            current = spans[index]
            gap = max(0.0, current.start - previous_source.end)
            primary_similarity = float(primary_embeddings[index - 1] @ primary_embeddings[index])
            secondary_similarity = float(
                secondary_embeddings[index - 1] @ secondary_embeddings[index]
            )
            blocked_join = gap > 0.0 and any(
                min(current.start, blocked.end) - max(previous_source.end, blocked.start)
                > 0.01
                and not (
                    blocked.start <= previous_source.start + 0.02
                    and blocked.end >= current.end - 0.02
                )
                for blocked in forbidden_joins
            )
            both_strict = (
                strict_flags is not None
                and bool(strict_flags[index - 1])
                and bool(strict_flags[index])
            )
            same_speaker = (
                gap <= maximum_silence_seconds
                and current.end-output[-1].start <= min(20., self.options.max_sentence_seconds)
                and not blocked_join
                and (
                    both_strict
                    or (
                        primary_similarity >= primary_same_floor
                        and secondary_similarity >= secondary_same_floor
                    )
                )
            )
            if same_speaker:
                same_speaker = self._identity_join_allowed(
                    previous_source, current, verifier, profile, waveform)
            if same_speaker:
                output[-1] = common.TimeSpan(output[-1].start, current.end)
            else:
                output.append(current)
        progress(1.0, f"短静音声纹核验完成：{len(spans)} 段合并为 {len(output)} 段")
        return output

    def _identity_join_allowed(self, left, right, verifier, profile, waveform) -> bool:
        if any(min(right.end, block.end) - max(left.start, block.start) > 1 / 16000
               for block in self._raw_blocked_spans):
            return False
        if any(left.start < point < right.end for point in self._speaker_change_points):
            return False
        local_edges = (common.TimeSpan(max(left.start, left.end - 1.8), left.end),
                       common.TimeSpan(right.start, min(right.end, right.start + 1.8)))
        if any(self._pitch_diagnostics(waveform, span).get("pitch_register_mismatch")
               for span in (left, right, *local_edges)):
            return False
        guide = self._subtitle_guide
        if guide is None or not guide.crosses_cue_boundary(left, right):
            return True
        guard = self._subtitle_join_guard
        return guard is not None and guard.allows(
            left, right, waveform, self._raw_target_waveform, verifier, profile,
            max(self.options.speaker_threshold, profile.primary.suggested_threshold))

    def _merge_verified_target_turns(
        self,
        accepted_turns: list[common.CandidateSentence],
        rejected: list[common.CandidateSentence],
        verifier: common.DualSpeakerVerifier,
        profile: common.SpeakerMatchProfile,
        waveform: common.torch.Tensor,
        threshold: float,
        target_spans: list[common.TimeSpan],
        vad_spans: list[common.TimeSpan],
        forbidden_joins: common.Iterable[common.TimeSpan],
        progress: common.ProgressCallback,
    ) -> int:
        """Join short-silence fragments only around a formally accepted core.

        Edge fragments never become output by themselves.  They may only repair
        a clipped edge when they are adjacent to a strict target turn,
        both speaker models regard the neighboring audio as the same voice, and
        the complete joined sentence passes formal verification again.
        """

        if not accepted_turns:
            return 0
        accepted_ids = {id(candidate) for candidate in accepted_turns}
        possible_edges = [
            candidate
            for candidate in rejected
            if candidate.reject_reason == "声纹匹配不足"
            and (
                candidate.diagnostics.get("speaker_tier") == "recall"
                or (
                    candidate.duration >= 1.20
                    and float(candidate.diagnostics.get("eres_score", 0.0)) >= 0.50
                    and float(candidate.diagnostics.get("camplus_score", 0.0)) >= 0.50
                    and float(
                        candidate.diagnostics.get("paired_reference_median", 0.0)
                    )
                    >= 0.40
                )
            )
            and not any(
                min(candidate.end, accepted.end) - max(candidate.start, accepted.start)
                > 0.10
                for accepted in accepted_turns
            )
        ]
        edge_by_id: dict[int, common.CandidateSentence] = {}
        for accepted in accepted_turns:
            left = [
                candidate
                for candidate in possible_edges
                if 0.0
                <= accepted.start - candidate.end
                <= self.options.silence_split_seconds
            ]
            right = [
                candidate
                for candidate in possible_edges
                if 0.0
                <= candidate.start - accepted.end
                <= self.options.silence_split_seconds
            ]
            if left:
                candidate = max(left, key=lambda item: item.end)
                edge_by_id[id(candidate)] = candidate
            if right:
                candidate = min(right, key=lambda item: item.start)
                edge_by_id[id(candidate)] = candidate
        recall_edges = list(edge_by_id.values())
        pool = sorted(
            [*accepted_turns, *recall_edges],
            key=lambda item: (item.start, item.end),
        )
        if len(pool) < 2:
            return 0
        # A single VAD island may still contain several speaker turns.  The
        # island-wide join blocker is intentionally ignored when it encloses
        # both selected parts, so separately rejected speech inside their gap
        # must be represented explicitly or the final whole-span score can
        # hide a brief third voice.
        pool_ids = {id(candidate) for candidate in pool}
        unselected_gap_speech: list[common.TimeSpan] = []
        for previous, current in zip(pool, pool[1:]):
            if current.start <= previous.end + 0.01:
                continue
            for candidate in rejected:
                if (
                    id(candidate) in pool_ids
                    or candidate.reject_reason == "有效讲话不足或接近空段"
                ):
                    continue
                if (
                    min(candidate.end, current.start)
                    - max(candidate.start, previous.end)
                    > 0.01
                ):
                    unselected_gap_speech.append(
                        common.TimeSpan(candidate.start, candidate.end)
                    )
        merged_spans = self._merge_short_silence_same_speaker(
            [common.TimeSpan(candidate.start, candidate.end) for candidate in pool],
            verifier,
            profile,
            waveform,
            progress,
            maximum_silence_seconds=self.options.silence_split_seconds,
            forbidden_joins=[*forbidden_joins, *unselected_gap_speech],
            strict_flags=[id(candidate) in accepted_ids for candidate in pool],
        )

        output: list[common.CandidateSentence] = []
        used_accepted_ids: set[int] = set()
        used_recall_ids: set[int] = set()
        merged_count = 0
        for merged_span in merged_spans:
            members = [
                candidate
                for candidate in pool
                if merged_span.start - 0.02 <= candidate.start
                and candidate.end <= merged_span.end + 0.02
            ]
            strict_members = [
                candidate for candidate in members if id(candidate) in accepted_ids
            ]
            if not strict_members:
                continue
            if len(members) == 1:
                output.append(strict_members[0])
                used_accepted_ids.add(id(strict_members[0]))
                continue
            coverage = self._target_coverage(merged_span, target_spans)
            if (
                merged_span.duration > min(20.0, self.options.max_sentence_seconds)
                or coverage < 0.30
            ):
                output.extend(strict_members)
                used_accepted_ids.update(id(candidate) for candidate in strict_members)
                continue
            match = self._verify_speaker_span(
                verifier,
                waveform,
                merged_span,
                profile,
                threshold,
            )
            if not match.accepted:
                output.extend(strict_members)
                used_accepted_ids.update(id(candidate) for candidate in strict_members)
                continue

            merged = common.CandidateSentence(merged_span.start, merged_span.end, "")
            self._apply_speaker_match(merged, match, profile, threshold)
            merged.diagnostics.update(
                {
                    "post_target_silence_merge": True,
                    "merged_strict_target_count": len(strict_members),
                    "merged_recall_edge_count": len(members) - len(strict_members),
                    "target_coverage": round(coverage, 5),
                    "speech_ratio": round(
                        common.speech_ratio(vad_spans, merged_span.start, merged_span.end),
                        5,
                    ),
                }
            )
            merged.diagnostics["speech_seconds"] = round(
                merged.duration * merged.diagnostics["speech_ratio"],
                5,
            )
            output.append(merged)
            used_accepted_ids.update(id(candidate) for candidate in strict_members)
            used_recall_ids.update(
                id(candidate)
                for candidate in members
                if id(candidate) not in accepted_ids
            )
            merged_count += 1

        output.extend(
            candidate
            for candidate in accepted_turns
            if id(candidate) not in used_accepted_ids
        )
        accepted_turns[:] = sorted(
            {
                (round(candidate.start, 5), round(candidate.end, 5)): candidate
                for candidate in output
            }.values(),
            key=lambda item: (item.start, item.end),
        )
        if used_recall_ids:
            rejected[:] = [
                candidate for candidate in rejected if id(candidate) not in used_recall_ids
            ]
        return merged_count

    @staticmethod
    def _waveform_span(waveform: common.torch.Tensor, span: common.TimeSpan, sample_rate: int = 16000) -> common.torch.Tensor:
        start = max(0, int(round(span.start * sample_rate)))
        end = min(waveform.numel(), int(round(span.end * sample_rate)))
        return waveform[start:end]
