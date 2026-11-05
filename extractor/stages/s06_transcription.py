"""s06 transcription responsibilities; composed by ExtractionPipeline."""
from __future__ import annotations

from .. import pipeline_common as common


class S06TranscriptionMethods:
    _HALLUCINATION_PATTERN = common.re.compile(
        r"^(?:谢谢[观看收]*|感谢[观听]*[看听]*|字幕[由提供做].*|译[制作者]*|仅供学习|请订阅|"
        r"关注[我频]*[频道]*|下[集期]再见|官方?字幕|SN?S\d+|[Ss]ub\s?[Bb]y|"
        r"ご視聴ありがとうございました|チャンネル登録|次回もお楽しみに|"
        r"amara\.org|youtube|niconico|bilibili|[（(]?完[)）]?|[。.．…\-—~～*✦✧ ]+)$"
    )

    @staticmethod
    def _join_transcript_text(left: str, right: str) -> str:
        """Join adjacent ASR fragments without inserting spaces in CJK text."""

        left = (left or "").strip()
        right = (right or "").strip()
        if not left:
            return right
        if not right:
            return left
        cjk = r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]"
        return f"{left}{right}" if common.re.search(cjk, left[-1:] + right[:1]) else f"{left} {right}"

    def _subtitle_fallback_fragment(self, turn):
        if self._subtitle_guide is None or turn.diagnostics.get("pitch_register_mismatch"):
            return None
        fallback = self._subtitle_guide.text_fallback(common.TimeSpan(turn.start, turn.end))
        if fallback is None:
            return None
        # A cue can be a translation. Preserve it as a review hint, never as
        # invented ASR text in a training manifest when recognition is empty.
        turn.diagnostics['subtitle_review_hint'] = fallback
        return None

    @staticmethod
    def _text_similarity(left: str, right: str) -> float:
        """去标点空格后的归一化字符 LCS 相似度（STT 分片 ↔ 字幕 Cue）。"""
        import re as _re
        clean = _re.compile(r"[\s\W_]+", _re.UNICODE)
        a = clean.sub("", str(left or ""))
        b = clean.sub("", str(right or ""))
        if not a or not b:
            return 0.0
        # O(len*len) DP；分片与字幕都是短句，代价可忽略。
        previous = [0] * (len(b) + 1)
        for i in range(1, len(a) + 1):
            current = [0] * (len(b) + 1)
            ai = a[i - 1]
            for j in range(1, len(b) + 1):
                if ai == b[j - 1]:
                    current[j] = previous[j - 1] + 1
                else:
                    current[j] = max(previous[j], current[j - 1])
            previous = current
        return 2.0 * previous[len(b)] / (len(a) + len(b))

    @staticmethod
    def _has_kana(text: str) -> bool:
        return bool(common.re.search(r"[\u3040-\u309f\u30a0-\u30ff]", str(text or "")))

    def _fragment_clean_evidence(
        self, fragment: common.CandidateSentence, guide, span_cache: dict,
    ) -> tuple[bool, str]:
        """判断一个 STT 分片是否带有"目标正文"证据。

        证据二选一：
        1. 分片区间落在簇级滑窗已验证的干净区间内（声纹证据）；
        2. 分片与对齐字幕 Cue 时间重叠 >=50% 且文本相似度 >=0.55
           （字幕是原子单人台词，时间+文本双重对齐）。
        """
        start, end = fragment.start, fragment.end
        for a, b in self._cluster_accepted_spans or []:
            if a <= start + 0.06 and b >= end - 0.06:
                return True, "cluster_clean_span"
        if guide is not None and guide.aligned:
            key = (round(start, 2), round(end, 2))
            if key in span_cache:
                return span_cache[key]
            best = None
            for cue in guide.cues:
                span = guide.cue_span(cue)
                if span is None:
                    continue
                overlap = min(end, span.end) - max(start, span.start)
                if overlap < 0.5 * max(0.01, end - start):
                    continue
                similarity = self._text_similarity(fragment.whisper_text, cue.text)
                if best is None or similarity > best[0]:
                    best = (similarity, cue)
            result = (best is not None and best[0] >= 0.55, "subtitle_aligned")
            span_cache[key] = result
            return result
        return False, "no_evidence"

    def _trim_turn_to_clean_fragments(
        self, turn, sentences: list[common.CandidateSentence], guide,
    ) -> tuple[float, float, list[common.CandidateSentence]] | None:
        """多分片候选句精裁：只保留声纹/字幕双证据通过的最长连续分片运行。

        返回 (start, end, kept_fragments)；无可保留运行时返回 None。
        """
        span_cache: dict = {}
        flags = []
        for fragment in sentences:
            clean, _method = self._fragment_clean_evidence(fragment, guide, span_cache)
            flags.append(clean)
        if not any(flags):
            return None
        best_run, current = None, []
        for fragment, clean in zip(sentences, flags):
            if clean:
                current.append(fragment)
                if best_run is None or len(current) > len(best_run):
                    best_run = list(current)
            else:
                current = []
        if best_run is None:
            return None
        return best_run[0].start, best_run[-1].end, best_run

    def _correct_text_with_subtitle(
        self, text: str, language: str, start: float, end: float, guide,
    ) -> tuple[str, dict | None]:
        """字幕正字：日文候选与日文字幕 Cue 文本高匹配时，用字幕原文修正 STT 错字。"""
        if guide is None or not guide.aligned or language != "ja":
            return text, None
        best = None
        for cue in guide.cues:
            if not self._has_kana(cue.text):
                continue
            span = guide.cue_span(cue)
            if span is None:
                continue
            overlap = min(end, span.end) - max(start, span.start)
            if overlap < 0.5 * max(0.01, end - start):
                continue
            similarity = self._text_similarity(text, cue.text)
            if best is None or similarity > best[0]:
                best = (similarity, cue)
        if best is None or best[0] < 0.70:
            return text, None
        corrected = common.clean_dialogue_text(best[1].text)
        if not corrected:
            return text, None
        return corrected, {
            "subtitle_text_correction": {
                "similarity": round(best[0], 4),
                "stt_text": text,
                "subtitle_text": corrected,
            }
        }

    def _finalize_target_transcripts(self, accepted_turns, transcribed, rejected, raw_original):
        """Attach text to acoustically accepted spans without surrendering audio ownership."""
        accepted = []
        by_turn_fragments = {}
        by_turn: dict[int, list[common.CandidateSentence]] = {}
        for sentence in transcribed:
            turn_index = int(sentence.diagnostics.get("transcription_span_index", -1))
            by_turn.setdefault(turn_index, []).append(sentence)
        for index, turn in enumerate(accepted_turns):
            self._install_acoustic_identity_certificate(turn)
            sentences = by_turn.get(index, [])
            if not sentences:
                fallback = self._subtitle_fallback_fragment(turn)
                if fallback is not None:
                    sentences = [fallback]
            sentences.sort(key=lambda item: (item.start, item.end))
            # Whisper 静音幻觉过滤：纯水印/频道宣传类文本分片不参与拼接。
            hallucinated = [
                fragment for fragment in sentences
                if self._HALLUCINATION_PATTERN.match(
                    str(fragment.whisper_text or "").strip())
            ]
            if hallucinated:
                turn.diagnostics["stt_hallucination_filtered"] = [
                    {"span": [f.start, f.end], "text": f.whisper_text}
                    for f in hallucinated]
                sentences = [f for f in sentences if f not in hallucinated]
            text = ""
            languages: list[str] = []
            for fragment in sentences:
                text = self._join_transcript_text(text, fragment.whisper_text)
                if fragment.language and fragment.language != "auto":
                    languages.append(fragment.language)
            text = text.strip()
            if not text:
                fallback = self._subtitle_fallback_fragment(turn)
                if fallback is None:
                    turn.diagnostics['stt_empty_text_preserved'] = True
                else:
                    sentences = [fallback]
                    text = fallback.whisper_text
                    languages = [fallback.language]
            fragment_count = len(sentences)
            stt_identity_veto = self._stt_fragment_identity_veto(
                turn,
                fragment_count,
            )
            audio_start, audio_end = turn.start, turn.end
            if stt_identity_veto == "local_identity_unverified":
                # Keep only text-aligned portions with independent identity evidence.
                trimmed = self._trim_turn_to_clean_fragments(
                    turn, sentences, self._subtitle_guide)
                if trimmed is not None:
                    trim_start, trim_end, kept = trimmed
                    coverage = (trim_end - trim_start) / max(
                        1e-6, turn.duration)
                    if coverage < 0.85:
                        if trim_end - trim_start + 1e-6 < self.options.min_output_seconds:
                            turn.reject_reason = "局部目标子句短于当前导出下限"
                            turn.diagnostics["stt_fragment_count"] = fragment_count
                            turn.diagnostics["stt_identity_veto"] = stt_identity_veto
                            rejected.append(turn)
                            continue
                        audio_start, audio_end = trim_start, trim_end
                        text = ""
                        for fragment in kept:
                            text = self._join_transcript_text(text, fragment.whisper_text)
                        text = text.strip()
                        turn.diagnostics["stt_trimmed_span"] = [audio_start, audio_end]
                        turn.diagnostics["stt_changed_audio_boundary"] = True
                else:
                    turn.reject_reason = "局部身份尚未确认，暂不解除多片段导出保护"
                    turn.diagnostics["stt_fragment_count"] = fragment_count
                    turn.diagnostics["stt_identity_veto"] = stt_identity_veto
                    rejected.append(turn)
                    continue
            elif stt_identity_veto is not None:
                turn.reject_reason = "声学身份复核未通过"
                turn.diagnostics["stt_fragment_count"] = fragment_count
                turn.diagnostics["stt_identity_veto"] = stt_identity_veto
                rejected.append(turn)
                continue
            if audio_end - audio_start + 1e-6 < self.options.min_output_seconds:
                if (audio_end - audio_start >= common.MIN_TARGET_SECONDS - 1e-9
                        and self._install_acoustic_identity_certificate(turn)):
                    turn.diagnostics['certified_short_export'] = True
                else:
                    turn.accepted = False
                    turn.reject_reason = "目标短语音保留待复核"
                    turn.diagnostics.update(cluster_tier='B', review_only=True)
                    rejected.append(turn)
                    continue

            language = (
                max(set(languages), key=languages.count)
                if languages
                else "auto"
            )
            sentence_language = common._language_from_text(text, language)
            corrected, correction_info = self._correct_text_with_subtitle(
                text, sentence_language, audio_start, audio_end,
                self._subtitle_guide)
            sentence = common.CandidateSentence(
                start=audio_start,
                end=audio_end,
                whisper_text=text,
                language=sentence_language,
                text=corrected,
                accepted=True,
            )
            if correction_info:
                sentence.diagnostics.update(correction_info)
            sentence.speaker_score = turn.speaker_score
            sentence.window_min_score = turn.window_min_score
            sentence.window_p20_score = turn.window_p20_score
            sentence.speaker_vote_ratio = turn.speaker_vote_ratio
            sentence.window_vote_ratio = turn.window_vote_ratio
            sentence.speaker_threshold = turn.speaker_threshold
            sentence.overlap_score = turn.overlap_score
            sentence.singing_score = turn.singing_score
            sentence.speech_score = turn.speech_score
            sentence.diagnostics.update(turn.diagnostics)
            words = [word for fragment in sentences
                     for word in fragment.diagnostics.get('word_timestamps',())]
            sentence.diagnostics['word_timestamps'] = words
            sentence.diagnostics['aligned_raw_audio'] = str(raw_original.resolve())
            sentence.diagnostics['word_boundary_allowed_span'] = [turn.start, turn.end]
            if any(fragment.diagnostics.get("transcription_silence_padding_retry") for fragment in sentences):
                sentence.diagnostics["transcription_silence_padding_retry"] = True
            sentence.diagnostics.update(
                {
                    "stt_fragment_count": len(sentences),
                    "stt_fragment_boundary_only": len(sentences) >= 2,
                    "audio_boundary_source": (
                        "stt_subtitle_aligned"
                        if turn.diagnostics.get("stt_trimmed_span")
                        else "silence_and_speaker"),
                    "stt_changed_audio_boundary": bool(
                        turn.diagnostics.get("stt_trimmed_span")),
                }
            )
            sentence_fragments = [fragment for fragment in sentences
                                  if audio_start <= fragment.start and fragment.end <= audio_end]
            by_turn_fragments[(sentence.start, sentence.end)] = sentence_fragments
            accepted.append(sentence)
        return accepted, by_turn_fragments

    @staticmethod
    def _stt_fragment_identity_veto(
        candidate: common.CandidateSentence,
        fragment_count: int,
    ) -> str | None:
        """Do not relax legacy export unless local identity was verified.

        No veto is not positive evidence of a pure speaker. The legacy
        multi-fragment hold remains for unresolved turns until acoustic
        certification is available. It is not a claim that Whisper detected
        another speaker. A certificate belongs to one exact audio interval.
        """
        diagnostics = candidate.diagnostics
        if diagnostics.get("multi_model_anchor_decisive_boundary"):
            return "pre_stt_decisive_speaker_boundary"
        if any(diagnostics.get(key) for key in (
            "structural_hard_reject", "excluded_role_rejected",
            "final_same_speaker_internal_discard", "final_identity_boundary_discard",
            "pitch_register_mismatch",
        )):
            return "pre_stt_acoustic_veto"
        # Exclusion as a recovery prototype is NOT evidence of another person.
        audit = diagnostics.get("local_identity_audit", {})
        if (isinstance(audit, dict) and audit.get("passed") is True
                and audit.get("span") == [candidate.start, candidate.end]):
            return None
        acoustic_audit = diagnostics.get("acoustic_identity_audit", {})
        if (isinstance(acoustic_audit, dict)
                and acoustic_audit.get("passed") is True
                and acoustic_audit.get("span") == [candidate.start, candidate.end]):
            return None
        if fragment_count < 2 or diagnostics.get("post_target_silence_merge"):
            return None
        return "local_identity_unverified"

    @staticmethod
    def _install_acoustic_identity_certificate(candidate: common.CandidateSentence) -> bool:
        """Certify an exact pre-STT span after the acoustic gates have run.

        STT may return multiple text chunks for one clean utterance.  This
        certificate is deliberately narrower than a generic speaker match: it
        is issued only by paths that performed a final local/boundary audit and
        have no unresolved change or exclusion evidence.  It never expands a
        candidate or treats transcript segmentation as speaker evidence.
        """
        diagnostics = candidate.diagnostics
        hard_vetoes = (
            "structural_hard_reject",
            "excluded_role_rejected",
            "final_same_speaker_internal_discard",
            "final_identity_boundary_discard",
            "multi_model_anchor_decisive_boundary",
            "pitch_register_mismatch",
        )
        if any(diagnostics.get(key) for key in hard_vetoes):
            return False
        if diagnostics.get("boundary_suppressed_recovery"):
            return False
        from ..sentence_policy import exact_certificate
        if exact_certificate(candidate) is not None:
            # STT attaches words to an already independently certified span;
            # never erase the actual multichannel/WavLM/dual authority here.
            return True
        existing = diagnostics.get("local_identity_audit")
        if (isinstance(existing, dict) and existing.get("passed") is True
                and existing.get("span") == [candidate.start, candidate.end]):
            return True

        method: str | None = None
        atomic_audit = diagnostics.get("atomic_identity_audit", {})
        if (diagnostics.get('cluster_identity') is True
                and diagnostics.get('identity_trajectory', {}).get('accepted') is True):
            method = 'cluster_identity_trajectory'
        elif (atomic_audit.get("passed") is True
                and atomic_audit.get("span") == [candidate.start, candidate.end]):
            method = "independently_verified_atomic_span"
        elif diagnostics.get("post_target_silence_merge"):
            method = "verified_target_silence_merge"
        elif diagnostics.get("final_identity_boundary_crop"):
            method = "verified_identity_boundary_crop"
        elif diagnostics.get("target_locator_recovery"):
            # Locator recovery is accepted here only when the whole exact span
            # passed the formal verifier and the local scan did not produce an
            # unresolved boundary-suppressed candidate.
            if (
                float(diagnostics.get("target_coverage", 0.0)) >= 0.80
                and float(diagnostics.get("speech_ratio", 0.0)) >= 0.80
            ):
                method = "target_locator_full_span_acoustic_review"
        elif diagnostics.get("local_boundary_recovery") and not diagnostics.get(
            "local_edge_only"
        ):
            # A recovery part bounded by confirmed change cuts and verified on
            # its own exact audio has completed the local/boundary audit. Its
            # recorded evidence must cover exactly this span and contain no
            # unresolved or exclusion state.
            all_entries = [
                entry
                for entry in diagnostics.get("local_identity_evidence") or []
                if isinstance(entry, dict)
            ]
            target_entries = [
                entry
                for entry in all_entries
                if entry.get("state") == "target"
                and isinstance(entry.get("span"), (list, tuple))
                and len(entry["span"]) == 2
            ]
            if target_entries and len(target_entries) == len(all_entries):
                start = min(float(entry["span"][0]) for entry in target_entries)
                end = max(float(entry["span"][1]) for entry in target_entries)
                if (
                    start >= candidate.start - 0.02
                    and start <= candidate.start + 0.02
                    and end >= candidate.end - 0.02
                    and end <= candidate.end + 0.02
                ):
                    method = "verified_boundary_part"
        elif diagnostics.get("boundary_audit") == "clean":
            method = "clean_local_boundary_audit"

        if method is None:
            return False
        diagnostics["acoustic_identity_audit"] = {
            "span": [candidate.start, candidate.end],
            "passed": True,
            "method": method,
        }
        return True

    def _attach_verified_fallbacks(
        self,
        evicting: common.CandidateSentence,
        evicted: common.Iterable[common.CandidateSentence],
    ) -> None:
        """Remember verified spans that a wider accepted candidate replaces.

        The wider candidate takes over export duty, but if it later fails the
        final multi-fragment review, the exact spans that already passed the
        acoustic gates must not be lost with it.  Anchors are kept alive so
        id()-keyed registry entries cannot be reused by other objects.
        """
        items = [item for item in evicted if item is not evicting]
        if not items:
            return
        self._verified_fallback_anchors.append(evicting)
        bucket = self._verified_fallback_registry.setdefault(id(evicting), [])
        for item in items:
            if all(item is not existing for existing in bucket):
                bucket.append(item)

    def _collect_verified_fallbacks(
        self,
        accepted_turns: list[common.CandidateSentence],
    ) -> int:
        """Queue evicted verified sub-spans as independent pre-STT turns.

        Each fallback is transcribed on its own exact interval.  It is only
        exported when its wider parent is withheld; the post-STT dedup removes
        it when the parent survives, so this never duplicates an export.
        """
        if not self._verified_fallback_registry:
            return 0
        added = 0
        accepted_ids = {id(candidate) for candidate in accepted_turns}
        worklist: list[tuple[common.CandidateSentence, common.CandidateSentence, frozenset[int]]] = [
            (candidate, candidate, frozenset({id(candidate)}))
            for candidate in accepted_turns
        ]
        # Parents withheld after registration (e.g. by the final island
        # consensus) stay in the rejected ledger.  Their verified sub-spans
        # are exactly the children the consensus wants to keep, so they must
        # not be lost together with the parent.  Anchors are scan sources
        # only: they must not occupy the seen set, or the normal eviction
        # chain would lose its ancestor context.
        for anchor in self._verified_fallback_anchors:
            if id(anchor) not in accepted_ids:
                worklist.append((anchor, anchor, frozenset({id(anchor)})))
        seen = set(accepted_ids)
        index = 0
        while index < len(worklist):
            parent, source, chain = worklist[index]
            index += 1
            for evicted in self._verified_fallback_registry.get(id(source), []):
                if id(evicted) in seen:
                    continue
                if id(source) not in accepted_ids and not (
                    evicted.start >= source.start + 0.05
                    or evicted.end <= source.end - 0.05
                ):
                    # The withheld parent's final-review evidence covers this
                    # exact audio; only a genuinely narrower verified child
                    # may survive it.
                    continue
                # Overlap with the candidate's own eviction chain is expected
                # (each fallback sits inside the parent it backs up).  Only
                # unrelated accepted turns may veto the rescue.
                if any(
                    min(evicted.end, existing.end)
                    - max(evicted.start, existing.start) > 0.10
                    and id(existing) not in chain
                    for existing in accepted_turns
                ):
                    continue
                fallback = common.CandidateSentence(evicted.start, evicted.end, "")
                fallback.speaker_score = evicted.speaker_score
                fallback.window_min_score = evicted.window_min_score
                fallback.window_p20_score = evicted.window_p20_score
                fallback.speaker_vote_ratio = evicted.speaker_vote_ratio
                fallback.window_vote_ratio = evicted.window_vote_ratio
                fallback.speaker_threshold = evicted.speaker_threshold
                fallback.overlap_score = evicted.overlap_score
                fallback.singing_score = evicted.singing_score
                fallback.speech_score = evicted.speech_score
                fallback.diagnostics = dict(evicted.diagnostics)
                fallback.diagnostics["verified_subspan_fallback"] = True
                fallback.diagnostics["fallback_parent_span"] = [
                    parent.start,
                    parent.end,
                ]
                accepted_turns.append(fallback)
                seen.add(id(fallback))
                worklist.append(
                    (
                        fallback,
                        evicted,
                        chain | {id(fallback), id(evicted)},
                    )
                )
                added += 1
        return added

    @staticmethod
    def _dedup_verified_fallback_exports(
        accepted: list[common.CandidateSentence],
        rejected: list[common.CandidateSentence],
    ) -> int:
        """Keep a fallback export only when its wider parent was withheld."""
        kept: list[common.CandidateSentence] = []
        removed = 0
        for sentence in accepted:
            if sentence.diagnostics.get("verified_subspan_fallback"):
                parent_start, parent_end = (
                    sentence.diagnostics.get("fallback_parent_span")
                    or [sentence.start, sentence.end]
                )
                parent_alive = any(
                    existing is not sentence
                    and not existing.diagnostics.get("verified_subspan_fallback")
                    and existing.start <= float(parent_start) + 0.25
                    and existing.end >= float(parent_end) - 0.25
                    for existing in accepted
                )
                if parent_alive:
                    sentence.accepted = False
                    sentence.reject_reason = "完整父回合已通过复核，后备子回合不再重复导出"
                    rejected.append(sentence)
                    removed += 1
                    continue
            kept.append(sentence)
        if removed:
            accepted[:] = kept
        return removed

    def _refine_cropped_sentences(
        self, candidates: list[common.CandidateSentence], waveform: common.torch.Tensor,
        blocked_spans: common.Iterable[common.TimeSpan] = (),
    ) -> None:
        """Refine artificial cuts only, within the verified parent and barriers."""
        marked = [candidate for candidate in candidates if any(
            candidate.diagnostics.get(key)
            for key in ("cluster_cropped", "cluster_span_synthesized", "stt_trimmed_span"))]
        if not marked:
            return
        wave = waveform.detach().float().cpu().numpy().reshape(-1)
        duration = len(wave) / 16000
        barriers = [*blocked_spans, *(common.TimeSpan(*span) for span in self.options.user_excluded_spans)]
        for candidate in marked:
            original = (candidate.start, candidate.end)
            diagnostics = candidate.diagnostics
            protection = diagnostics.get('whole_sentence_protection') or {}
            if protection.get('passed') and protection.get('span') == list(original):
                continue
            if (diagnostics.get("overlong_split_from")
                    or diagnostics.get("word_boundary_refinement")
                    or diagnostics.get("acoustic_snap_applied_span") == list(original)):
                continue
            parent = diagnostics.get("acoustic_parent_span", original)
            # A trimmed STT identity region must not expand back into discarded text.
            if diagnostics.get("stt_trimmed_span"):
                parent = diagnostics["stt_trimmed_span"]
            lower, upper = max(0., float(parent[0])), min(duration, float(parent[1]))
            certificates = [key for key in ("local_identity_audit", "acoustic_identity_audit")
                if isinstance(diagnostics.get(key), dict)
                and diagnostics[key].get("passed") is True
                and diagnostics[key].get("span") == list(original)]
            if certificates:
                # An exact identity certificate permits only a subinterval.
                # Expanding to a quiet valley outside it requires new speaker
                # verification; this acoustic-only refinement cannot supply it.
                lower, upper = max(lower, original[0]), min(upper, original[1])
            if any(block.start < candidate.end and block.end > candidate.start for block in barriers):
                diagnostics["acoustic_snap_skipped"] = "candidate_overlaps_hard_barrier"
                continue
            for block in barriers:
                if block.end <= candidate.start:
                    lower = max(lower, block.end)
                elif block.start >= candidate.end:
                    upper = min(upper, block.start)
            # Prevent two separately transcribed neighbours from gaining shared audio.
            for other in candidates:
                if other is candidate:
                    continue
                if other.end <= candidate.start:
                    lower = max(lower, other.end)
                elif other.start >= candidate.end:
                    upper = min(upper, other.start)
            if not lower <= candidate.start < candidate.end <= upper:
                continue
            start, end = common.snap_crop_boundaries_to_valley(
                wave, *original, hard_bounds=(lower, upper))
            from ..word_boundaries import preserve_word_coverage
            start, end = preserve_word_coverage(
                original, (start, end), diagnostics.get('word_timestamps', ()))
            if end - start + 1e-6 < self.options.min_output_seconds:
                continue
            diagnostics["acoustic_snap_applied_span"] = [start, end]
            if (start, end) != original:
                candidate.start, candidate.end = start, end
                for key in certificates:
                    diagnostics[key] = {**diagnostics[key], "span": [start, end],
                                        "source_span": list(original)}
                diagnostics["acoustic_boundary_snap"] = {
                    "from": list(original), "to": [start, end], "hard_bounds": [lower, upper]}
