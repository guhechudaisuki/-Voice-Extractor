"""runner responsibilities; composed by ExtractionPipeline."""
from __future__ import annotations

from .. import pipeline_common as common


class RunnerMethods:
    def run(
        self,
        references: common.Iterable[str | common.Path],
        target: str | common.Path,
        negative_references: common.Iterable[common.Iterable[str | common.Path]] | None = None,
        progress: common.ProgressCallback | None = None,
        job_id: str | None = None,
        create_archive: bool = True,
        subtitle: str | common.Path | None = None,
    ) -> common.PipelineResult:
        progress = progress or common._noop_progress
        references = [common.Path(path) for path in references if path]
        negative_reference_groups = [
            [common.Path(path) for path in group if path]
            for group in (negative_references or [])
        ]
        negative_reference_groups = [
            group for group in negative_reference_groups if group
        ]
        target = common.Path(target)
        if not references:
            raise ValueError("请至少提供一段参考音频")
        if not target.exists():
            raise FileNotFoundError(target)
        from ..nextgen.features import file_digest

        self._reference_audio_paths = list(references)
        self._batch_bank = []
        self._local_score_cache = None
        self._raw_vad_recovered = []
        reference_sha256 = sorted(file_digest(path) for path in references)
        self._invalidate_changed_tail_scorer()
        if reference_sha256 != getattr(self, "_reference_sha256", None):
            self._tail_scorer_cache = None
            self._local_identity_reject_threshold = None
            self._local_identity_model_sha256 = ""
            self._reference_sha256 = reference_sha256
        if subtitle is None:
            # Prefer a matching subtitle alongside the source media.
            discovered = common.discover_subtitle_for_target(target)
            if discovered is not None:
                progress(0.01, f"自动发现同目录字幕：{discovered.name}")
                subtitle = discovered
        self._subtitle_guide = common.SubtitleGuide.load(common.Path(subtitle)) if subtitle else None
        guide_for_tracks = self._subtitle_guide
        self._pitch_guard = None
        self._pitch_span_cache = {}
        self._subtitle_join_guard = None
        self._speaker_change_points = ()
        self._candidate_cut_anchors = ()
        self._stage_audit = {"schema_version": 1, "stages": {}}
        self._cluster_report = None
        job_id = job_id or common.time.strftime("%Y%m%d_%H%M%S") + "_" + common.uuid.uuid4().hex[:6]
        paths = self._job_paths(job_id)
        paths["root"].mkdir(parents=True, exist_ok=True)
        singing_detector: common.SingingDetector | None = None
        feature_context = common.ExitStack()
        try:
            progress(0.01, "检查本地模型")
            normalized_refs, reference_durations, reference_precleaned = self._prepare_references(
                references, paths, progress)
            normalized_negative_groups, negative_duration_groups = (
                self._prepare_negative_references(
                    negative_reference_groups,
                    paths,
                    progress,
                )
            )
            # Address the original media before any decode, masking or model work.
            from ..model_store import file_sha256
            from ..config import UVR_MODEL
            source_sha = file_sha256(target)
            try:
                uvr_model_sha256 = file_sha256(UVR_MODEL)
            except OSError:
                uvr_model_sha256 = "missing_uvr_model"
            cache_settings = {
                "uvr_model_sha256": uvr_model_sha256,
                "overlap_threshold": self.options.overlap_threshold,
            }
            cache_args = (source_sha, self.options.singing_threshold,
                          self.options.use_singing_detector, self.options.use_overlap_detector)
            cache_dir = common.media_cache_dir(*cache_args, **cache_settings)
            persist_cache = common.os.environ.get('VOICE_EXTRACT_PERSIST_CACHE') == '1'
            cache_hit = common.media_cache_hit(*cache_args, **cache_settings) if persist_cache else None
            cache_data = common.load_media_cache(cache_hit) if cache_hit is not None else {}
            self._media_cache_hit = bool(cache_data)
            self._media_cache_data = cache_data
            self._media_cache_dir = cache_dir
            self._media_cache_src_sha = source_sha
            pre_singing_spans: list[common.TimeSpan] = []
            if cache_data:
                target_duration = cache_data["target_duration"]
                # The unmasked mono source also serves all later raw-channel reads.
                target_normalized = cache_hit / "target_raw_16k.wav"
                raw_original = target_normalized
                stem = cache_hit / "target_vocals.wav"
                target_for_separator = None
                pre_singing_spans = [common.TimeSpan(*span) for span in cache_data['pre_singing_spans']]
                progress(0.16, "媒体缓存命中：已恢复音轨与检测结果")
            else:
                target_normalized = common.normalize_audio(
                    target, paths["normalized_target"], sample_rate=44100, stereo=True)
                target_duration = common.probe_duration(target_normalized)
                # Preserve the original identity channel before separation.
                raw_original = common.normalize_audio(
                    target_normalized, paths["root"] / "target_raw_16k.wav",
                    sample_rate=16000, stereo=False)
                target_for_separator = target_normalized
                if self.options.use_singing_detector:
                    singing_detector = common.pooled_encoder(common.SingingDetector, self.device, common.PANNS_MODEL)
                    _non_singing, pre_singing_spans = singing_detector.clean_spans(
                        target_normalized, [common.TimeSpan(0., target_duration)],
                        self.options.singing_threshold,
                        progress=lambda value, message: progress(.12 + .04 * value, message))
                    if pre_singing_spans:
                        target_for_separator = common.mute_spans(
                            target_normalized, paths['singing_removed_target'], pre_singing_spans)
                common.pad_for_separator(target_for_separator)
            self._stage_audit["source_duration_seconds"] = round(target_duration, 5)
            self._trace_spans('pre_uvr_singing_mask', pre_singing_spans)
            reference_precleaned_stems: dict[int, common.Path] = {}
            separation_items = []
            paths["reference_stems"].mkdir(parents=True, exist_ok=True)
            for index, path in enumerate(normalized_refs, start=1):
                stem_path = paths["reference_stems"] / f"reference_{index:03d}_vocals.wav"
                if reference_precleaned[index - 1]:
                    # Reuse the clean mono reference without another separation.
                    common.shutil.copyfile(path, stem_path)
                    reference_precleaned_stems[index - 1] = stem_path
                    continue
                separation_items.append((path, stem_path))
            negative_item_count = 0
            negative_precleaned_stems: dict[int, common.Path] = {}
            negative_total = 0
            for group_index, group in enumerate(normalized_negative_groups, start=1):
                group_dir = paths["negative_stems"] / f"role_{group_index:03d}"
                for reference_index, path in enumerate(group, start=1):
                    destination = group_dir / f"reference_{reference_index:03d}_vocals.wav"
                    if (group_index, reference_index) in getattr(self, "_negative_reference_precleaned", ()):
                        group_dir.mkdir(parents=True, exist_ok=True)
                        common.shutil.copyfile(path, destination)
                        negative_precleaned_stems[negative_total] = destination
                        negative_total += 1
                        continue
                    separation_items.append(
                        (
                            path,
                            destination,
                        )
                    )
                    negative_item_count += 1
                    negative_total += 1
            if not cache_data:
                separation_items.append((target_for_separator, paths["stems"] / "target_vocals.wav"))
            try:
                separated = []
                if separation_items:
                    from .s02_separation import SeparationStage, SeparationInput
                    separated = SeparationStage().run(SeparationInput(
                        separation_items, self.device,
                        lambda value, message: progress(0.16 + 0.24 * value, message)))
            except ValueError as exc:
                if "有效信号" not in str(exc):
                    raise
                # A fully silent target has no sentence by definition. Return
                # the same empty artifact shape as a music-only target.
                output_dir = paths["output"]
                output_dir.mkdir(parents=True, exist_ok=True)
                manifest, transcript, archive = self._write_outputs(
                    [],
                    [],
                    target_normalized,
                    target.name,
                    paths,
                    progress,
                    create_archive=create_archive,
                )
                progress(1.0, "完成：目标音频没有人声")
                return common.PipelineResult(job_id, output_dir, archive, [], [], manifest, transcript)
            reference_stems: list[common.Path] = []
            separated_cursor = 0
            for index in range(len(normalized_refs)):
                if index in reference_precleaned_stems:
                    reference_stems.append(reference_precleaned_stems[index])
                else:
                    reference_stems.append(separated[separated_cursor])
                    separated_cursor += 1
            negative_separated = iter(separated[
                separated_cursor : separated_cursor + negative_item_count
            ])
            negative_flat_stems = [
                negative_precleaned_stems[index] if index in negative_precleaned_stems
                else next(negative_separated) for index in range(negative_total)]
            if not cache_data:
                stem = separated[-1]
            for reference_stem, original_duration in zip(reference_stems, reference_durations):
                common.trim_audio_in_place(reference_stem, original_duration)
            negative_stem_groups: list[list[common.Path]] = []
            negative_cursor = 0
            for durations in negative_duration_groups:
                count = len(durations)
                group_stems = negative_flat_stems[
                    negative_cursor : negative_cursor + count
                ]
                negative_cursor += count
                for reference_stem, original_duration in zip(group_stems, durations):
                    common.trim_audio_in_place(reference_stem, original_duration)
                negative_stem_groups.append(group_stems)
            if not cache_data:
                common.trim_audio_in_place(stem, target_duration)

            vad_tools = common.FunASRTools(self.device)
            target_vad_inputs = []
            raw_vad_target = None
            if not cache_data:
                # Cache both VAD domains regardless of the current export mode.
                raw_vad_target = raw_original
                target_vad_inputs = [stem, raw_vad_target]
            vad_map = vad_tools.vad_many(
                [*reference_stems,
                 *(item for group in negative_stem_groups for item in group),
                 *target_vad_inputs],
                progress=lambda value, message: progress(0.40 + 0.08 * value, message))
            if cache_data:
                vad_map[stem] = [common.TimeSpan(*span) for span in cache_data["initial_vad"]]
                raw_target_vad = [common.TimeSpan(*span) for span in cache_data["raw_initial_vad"]]
            else:
                raw_target_vad = list(vad_map.get(raw_vad_target, []))
            initial_vad = list(vad_map.get(stem, []))
            self._trace_spans("raw_initial_vad", raw_target_vad)
            self._trace_spans("initial_vad", vad_map.get(stem, []))
            if self._subtitle_guide is not None:
                guide = self._subtitle_guide
                guide.calibrate(vad_map.get(stem, []))
                if guide.aligned:
                    progress(0.48, f"字幕时间轴已对齐：偏移 {guide.offset:+.2f} 秒，{len(guide.anchors)} 个锚点")
                    self._candidate_cut_anchors = tuple(sorted({edge
                        for cue in guide.cues if (cue_span := guide.cue_span(cue)) is not None
                        for edge in (cue_span.start, cue_span.end)}))
                    progress(0.48, "字幕已对齐：纯声学主线独立运行，字幕台词交给第二路审核")
                else:
                    progress(0.48, "字幕与音频时间轴缺少可靠对应，已回退到原始切分流程")
            self._subtitle_guide = None
            self._candidate_cut_anchors = ()
            self._trace_spans("subtitle_assisted_vad", vad_map.get(stem, []))
            reference_clips = self._make_reference_clips(reference_stems, vad_map, paths)
            # Build a second, time-aligned reference domain from the original
            # normalized audio.  Reusing the UVR VAD spans keeps clip ordering
            # and identity anchors identical while preserving spectral detail
            # that the separator may have removed.
            raw_reference_vad = {
                raw_path: list(vad_map.get(stem_path, []))
                for raw_path, stem_path in zip(normalized_refs, reference_stems)
            }
            raw_reference_clips = self._make_reference_clips(
                normalized_refs,
                raw_reference_vad,
                paths,
                output_dir=paths["raw_reference_clips"],
                prefix="reference",
            )
            negative_reference_clips = [
                self._make_reference_clips(
                    group,
                    vad_map,
                    paths,
                    output_dir=paths["negative_clips"] / f"role_{group_index:03d}",
                    prefix="negative",
                )
                for group_index, group in enumerate(negative_stem_groups, start=1)
            ]
            raw_negative_reference_clips: list[list[common.Path]] = []
            for group_index, (raw_group, stem_group) in enumerate(
                zip(normalized_negative_groups, negative_stem_groups),
                start=1,
            ):
                raw_group_vad = {
                    raw_path: list(vad_map.get(stem_path, []))
                    for raw_path, stem_path in zip(raw_group, stem_group)
                }
                raw_negative_reference_clips.append(
                    self._make_reference_clips(
                        raw_group,
                        raw_group_vad,
                        paths,
                        output_dir=paths["raw_negative_clips"]
                        / f"role_{group_index:03d}",
                        prefix="negative",
                    )
                )
            self._pitch_guard = common.PitchRegisterGuard(reference_clips)
            self._subtitle_join_guard = common.SubtitleJoinGuard(
                common.ASSET_ROOT / "models", reference_clips, raw_reference_clips)
            # Keep VAD boundaries until independent speaker evidence permits a join.
            combined_vad, raw_recovered = common.recover_raw_vad(vad_map.get(stem, []), raw_target_vad,
                                                        pre_singing_spans)
            vad_spans = list(combined_vad)
            self._raw_vad_recovered = raw_recovered
            self._trace_spans('raw_vad_recovered', raw_recovered)
            self._trace_spans("atomic_speech_islands", vad_spans)
            subtitle_track_enabled = (guide_for_tracks is not None and guide_for_tracks.aligned)
            if not vad_spans and not subtitle_track_enabled:
                if persist_cache and not cache_data and common._cache_waveforms_ready(stem, raw_original):
                    common.save_media_cache(cache_dir, source_sha, stem, raw_original, {
                        "target_duration": target_duration, "detector_spans": [],
                        "initial_vad": [[v.start, v.end] for v in initial_vad],
                        "raw_initial_vad": [[v.start, v.end] for v in raw_target_vad],
                        "pre_singing_spans": [[v.start, v.end] for v in pre_singing_spans],
                        "residual_singing": [], "singing_blocks": [],
                        "overlap_evidence": [], "overlap_blocks": []})
                manifest, transcript, archive = self._write_outputs(
                    [],
                    [],
                    stem,
                    target.name,
                    paths,
                    progress,
                    create_archive=create_archive,
                )
                progress(1.0, "完成：目标音频未检测到讲话")
                return common.PipelineResult(
                    job_id,
                    paths["output"],
                    archive,
                    [],
                    [],
                    manifest,
                    transcript,
                )

            self._verified_fallback_registry = {}
            self._verified_fallback_anchors = []
            accepted_turns: list[common.CandidateSentence] = []
            rejected: list[common.CandidateSentence] = []
            # Evidence is reusable only for the exact detector input islands.
            # A new subtitle can recover speech outside old VAD coverage; then
            # re-evaluate that evidence without invalidating the UVR/audio cache.
            detector_spans = [[span.start, span.end] for span in vad_spans]
            reuse_evidence = bool(cache_data) and cache_data["detector_spans"] == detector_spans
            if self.options.use_singing_detector and not reuse_evidence and singing_detector is None:
                singing_detector = common.pooled_encoder(common.SingingDetector, self.device, common.PANNS_MODEL)
            overlap_detector = (common.pooled_encoder(common.OverlapDetector, self.device, common.OVERLAP_MODEL)
                                if self.options.use_overlap_detector and not reuse_evidence else None)
            verifier = common.DualSpeakerVerifier(
                self.device,
                status=lambda message: progress(0.49, message),
                cache_enabled=self.options.cluster_embedding_cache,
            )
            try:
                profile = verifier.build_channel_profile(
                    reference_clips,
                    raw_reference_clips,
                    self.options.speaker_threshold,
                )
                exclusion_profiles: list[common.ExclusionSpeakerProfile] = []
                if negative_reference_clips:
                    progress(
                        0.49,
                        f"建立 {len(negative_reference_clips)} 个排除角色声纹边界",
                    )
                    exclusion_profiles = verifier.build_channel_exclusion_profiles(
                        negative_reference_clips,
                        profile,
                        raw_reference_groups=raw_negative_reference_clips,
                    )
                target_waveform = common.load_mono(stem, 16000)
                raw_target_waveform = common.load_mono(target_normalized, 16000)
                # Normalize tiny decoder/resampler length differences without
                # shifting the shared timeline.  Large mismatches remain
                # visible to the per-span dual-channel duration guard.
                if abs(raw_target_waveform.numel() - target_waveform.numel()) <= 1600:
                    if raw_target_waveform.numel() < target_waveform.numel():
                        raw_target_waveform = common.torch.nn.functional.pad(
                            raw_target_waveform,
                            (0, target_waveform.numel() - raw_target_waveform.numel()),
                        )
                    elif raw_target_waveform.numel() > target_waveform.numel():
                        raw_target_waveform = raw_target_waveform[
                            : target_waveform.numel()
                        ]
                self._raw_target_waveform = raw_target_waveform
                self._episode_waveform = target_waveform
                from .s04_features import FrameFeatureStage
                feature_buses = feature_context.enter_context(
                    FrameFeatureStage().run((target_waveform, raw_target_waveform)))
                # Filter each smallest silence-delimited island before any
                # joining.  Otherwise one overlap elsewhere in a long merged
                # block incorrectly deletes a clean target utterance.
                clean_atomic_spans = list(vad_spans)
                blocked_join_spans: list[common.TimeSpan] = []
                residual_singing: list[common.TimeSpan] = []
                singing_blocks: list[common.TimeSpan] = []
                if self.options.use_singing_detector:
                    if reuse_evidence:
                        residual_singing = [common.TimeSpan(*span) for span in cache_data["residual_singing"]]
                    else:
                        _post_clean, residual_singing = singing_detector.clean_spans(
                            stem, clean_atomic_spans, self.options.singing_threshold,
                            progress=lambda value, message: progress(0.49 + 0.03 * value, message))
                    singing_evidence = [*pre_singing_spans, *residual_singing]
                    clean_atomic_spans, singing_blocks = self._partition_tainted_spans(
                        clean_atomic_spans,
                        singing_evidence,
                        minimum_overlap_seconds=0.20,
                        minimum_fraction=0.15,
                        waveform=target_waveform,
                    )
                    blocked_join_spans.extend(singing_blocks)
                    rejected.extend(
                        common.CandidateSentence(
                            span.start,
                            span.end,
                            "",
                            reject_reason="检测到有人唱歌",
                            singing_score=self.options.singing_threshold,
                        )
                        for span in singing_blocks
                    )
                self._trace_spans("residual_singing_evidence", residual_singing)
                self._trace_spans("singing_blocked_islands", singing_blocks)

                overlap_evidence: list[common.TimeSpan] = []
                overlap_blocks: list[common.TimeSpan] = []
                subtitle_overlaps = (
                    self._subtitle_guide.overlapping_speech_spans()
                    if self._subtitle_guide is not None and self._subtitle_guide.aligned else [])
                if clean_atomic_spans and (self.options.use_overlap_detector or subtitle_overlaps):
                    if self.options.use_overlap_detector:
                        if reuse_evidence:
                            overlap_evidence = [common.TimeSpan(*span) for span in cache_data["overlap_evidence"]]
                        else:
                            _overlap_clean, overlap_evidence = overlap_detector.clean_spans(
                                stem, vad_spans, self.options.overlap_threshold,
                                padding=0.0,
                                progress=lambda value, message: progress(0.52 + 0.03 * value, message))
                    clean_atomic_spans, overlap_blocks = self._partition_tainted_spans(
                        clean_atomic_spans,
                        [*overlap_evidence, *subtitle_overlaps],
                        minimum_overlap_seconds=0.08,
                        minimum_fraction=0.04,
                        waveform=target_waveform,
                    )
                    blocked_join_spans.extend(overlap_blocks)
                    rejected.extend(
                        common.CandidateSentence(
                            span.start,
                            span.end,
                            "",
                            reject_reason="检测到多人同时发声",
                            overlap_score=self.options.overlap_threshold,
                        )
                        for span in overlap_blocks
                    )
                self._trace_spans("overlap_evidence", overlap_evidence)
                self._trace_spans("subtitle_overlap_evidence", subtitle_overlaps)
                self._trace_spans("overlap_blocked_islands", overlap_blocks)
                if persist_cache and not cache_data:
                    front_stages = {"target_duration": target_duration,
                                    "detector_spans": detector_spans}
                    front_stages.update({
                        name: [[span.start, span.end] for span in spans]
                        for name, spans in (
                            ('pre_singing_spans', pre_singing_spans),
                            ("initial_vad", initial_vad), ("raw_initial_vad", raw_target_vad),
                            ("residual_singing", residual_singing), ("singing_blocks", singing_blocks),
                            ("overlap_evidence", overlap_evidence), ("overlap_blocks", overlap_blocks))})
                    if common._cache_waveforms_ready(stem, raw_original):
                        common.save_media_cache(cache_dir, source_sha, stem, raw_original, front_stages)

                # Raw VAD contributes coordinates and identity evidence only.
                # Analysis and export continue to use the separated vocal stem.
                raw_recovered = [common.TimeSpan(max(raw.start, clean.start), min(raw.end, clean.end))
                    for raw in raw_recovered for clean in clean_atomic_spans
                    if min(raw.end, clean.end) - max(raw.start, clean.start) >= common.MIN_TARGET_SECONDS - 1e-9]
                self._raw_vad_recovered = raw_recovered
                self._raw_blocked_spans = tuple([
                    *blocked_join_spans,
                    *(common.TimeSpan(max(0., s.start-.03), s.end+.03)
                      for s in [*pre_singing_spans, *residual_singing, *overlap_evidence, *subtitle_overlaps])])
                clean_atomic_spans = self._merge_short_silence_same_speaker(
                    clean_atomic_spans, verifier, profile, target_waveform,
                    progress=lambda v,m: progress(.54+.005*v,m),
                    maximum_silence_seconds=self.options.silence_split_seconds,
                    forbidden_joins=self._raw_blocked_spans)
                if self.options.use_shot_boundaries:
                    from ..shot_boundaries import shot_cut_proposals
                    try:
                        shots = shot_cut_proposals(target, source_sha)
                    except (OSError, RuntimeError, ValueError) as exc:
                        common.LOGGER.warning('镜头切点不可用，继续声学切分：%s',exc)
                        shots = []
                    self._candidate_cut_anchors = tuple(sorted(set(self._candidate_cut_anchors) | set(shots)))
                    self._stage_audit['shot_boundaries'] = {'proposals':shots,'identity_authority':False}
                # Shots and acoustic valleys are proposals to the same dual
                # encoder confirmer; only confirmed changes permit surgery.
                single_speaker_parts = self._split_micro_window_changes(
                    clean_atomic_spans, verifier, profile, target_waveform,
                    progress=lambda value, message: progress(0.54 + .01 * value, message))
                review_pool = list(single_speaker_parts)
                self._candidate_cut_anchors = tuple(sorted(set(self._candidate_cut_anchors)
                    | set(self._speaker_change_points)))
                self._stage_audit['sentence_first'] = {
                    'natural_sentences': len(clean_atomic_spans),
                    'cut_anchors': list(self._candidate_cut_anchors),
                    'confirmed_change_points': list(self._speaker_change_points),
                    'anchors_split_preverification': False}
                self._trace_spans('raw_vad_recovered_clean', raw_recovered)
                self._trace_spans("clean_speech_islands", clean_atomic_spans)

                progress(
                    0.55,
                    f"最小语音岛过滤完成：剩余 {len(clean_atomic_spans)} 段；"
                    f"舍弃歌声 {sum(item.reject_reason == '检测到有人唱歌' for item in rejected)} 段、"
                    f"多人重叠 {sum(item.reject_reason == '检测到多人同时发声' for item in rejected)} 段",
                )
                if not clean_atomic_spans and not subtitle_track_enabled:
                    manifest, transcript, archive = self._write_outputs(
                        [],
                        rejected,
                        stem,
                        target.name,
                        paths,
                        progress,
                        create_archive=create_archive,
                    )
                    progress(1.0, "完成：过滤后没有单人讲话")
                    return common.PipelineResult(
                        job_id,
                        paths["output"],
                        archive,
                        [],
                        rejected,
                        manifest,
                        transcript,
                    )

                self._raw_blocked_spans = tuple([
                    *blocked_join_spans,
                    *(common.TimeSpan(max(0., span.start - .03), span.end + .03)
                      for span in [*pre_singing_spans, *residual_singing,
                                   *overlap_evidence, *subtitle_overlaps]),
                ])
                from ..stages.s05_identity import IdentityInput
                from .dual_track import run_tracks
                identity_result = run_tracks(IdentityInput(
                    self, target_waveform, list(clean_atomic_spans), vad_spans, verifier,
                    profile, exclusion_profiles, rejected, stem, reference_clips,
                    negative_reference_clips, target, source_sha, job_id, progress),
                    guide_for_tracks, pre_singing_spans, vad_tools=vad_tools)
                self._subtitle_guide = guide_for_tracks
                accepted_turns = identity_result.accepted
                review_pool.extend(identity_result.review_spans)
                self._stage_audit['shared_features'] = [
                    {'computed_frames': bus.computed_frames, 'requested_frames': bus.reused_frames}
                    for bus in feature_buses]
            finally:
                verifier.close()

            if not self.options.branch_validation:
                from .export_timeline import attach_transcripts
                accepted = list(accepted_turns)
                if accepted:
                    progress(.80, f"直接合流：对 {len(accepted)} 个通过区间转写并保留完整音频")
                    fragments = common.WhisperSegmenter(self.device).transcribe_spans(
                        stem, [common.TimeSpan(c.start,c.end) for c in accepted],
                        progress=lambda value,message: progress(.80+.13*value,message))
                    attach_transcripts(accepted, fragments)
            else:
                accepted: list[common.CandidateSentence] = []
                by_turn_fragments: dict[tuple[float, float], list[common.CandidateSentence]] = {}
                segmenter = None
                if accepted_turns:
                    accepted_turns.sort(key=lambda item: (item.start, item.end))
                    progress(0.80, f"筛选完成：仅对 {len(accepted_turns)} 个目标人物回合执行 STT")
                    segmenter = common.WhisperSegmenter(self.device)
                    transcribed = segmenter.transcribe_spans(
                        stem,
                        [common.TimeSpan(candidate.start, candidate.end) for candidate in accepted_turns],
                        progress=lambda value, message: progress(0.80 + 0.13 * value, message),
                    )
                    accepted, by_turn_fragments = self._finalize_target_transcripts(
                        accepted_turns, transcribed, rejected, raw_original)
                    fallback_deduped = self._dedup_verified_fallback_exports(
                        accepted,
                        rejected,
                    )
                    if fallback_deduped:
                        progress(
                            0.92,
                            f"后备子回合去重：{fallback_deduped} 段由完整父回合覆盖，"
                            "不重复导出",
                        )
                    self._deduplicate_content_clips(
                        accepted, target_waveform,
                        progress=lambda _value, message: progress(0.92, message),
                        rejected=rejected,
                    )
                    from ..transcript_finalization import finalize_transcripts
                    accepted = finalize_transcripts(
                        accepted, target_waveform, by_turn_fragments, stem=stem,
                        clips_dir=paths["candidate_clips"], vad_tools=vad_tools,
                        refine_crops=self._refine_cropped_sentences,
                        blocked=[*blocked_join_spans, *pre_singing_spans, *residual_singing,
                                 *overlap_evidence, *(common.TimeSpan(*s) for s in self.options.user_excluded_spans)],
                        progress=lambda value, message: progress(0.92 + 0.02 * value, message))
                else:
                    progress(0.92, "筛选完成：没有目标人物回合，跳过 STT")
                self._refine_cropped_sentences(
                    accepted, target_waveform,
                    [*blocked_join_spans, *pre_singing_spans, *residual_singing, *overlap_evidence])
                from ..stages.s05_identity import certify_exact
                accepted = certify_exact(self, accepted, rejected, verifier, profile,
                    target_waveform, exclusion_profiles, ledger=identity_result.ledger,
                    origin='post_stt_exact_audio')
                self._stage_audit['candidate_ledger'] = identity_result.ledger.to_dict()
                self._filter_pitch_register(accepted, rejected, target_waveform)

            if self.options.export_all_sentences:
                from .remainder_export import remaining_spans
                from .export_timeline import attach_transcripts
                from ..review_navigation import annotate_navigation
                import soundfile as sf
                # Partition on the same sample grid used by the WAV writer.
                # The dry master can have a higher rate than analysis audio.
                extra_spans = remaining_spans(review_pool, accepted, blocked=[
                    *self._raw_blocked_spans,
                    *(common.TimeSpan(*s) for s in self.options.user_excluded_spans)],
                    sample_rate=sf.info(str(stem)).samplerate)
                remaining = [common.CandidateSentence(s.start,s.end,'',accepted=False,
                    reject_reason='剩余单句（待人工复核）',diagnostics={
                        'all_sentence_export':True, 'remaining_sentence_export':True,
                        'review_only':True, 'export_group':'remaining',
                        'speaker_gate_bypassed':True,
                        'branch_validation':self.options.branch_validation}) for s in extra_spans]
                if remaining:
                    progress(.94, f'导出全部：转写目标结果之外的 {len(remaining)} 个单句')
                    fragments = common.WhisperSegmenter(self.device).transcribe_spans(
                        stem, extra_spans, progress=lambda value,message: progress(.94+.01*value,message))
                    attach_transcripts(remaining,fragments)
                    from ..acoustic_qc import split_overlong_sentences
                    by_span = {i:[] for i in range(len(remaining))}
                    for fragment in fragments:
                        by_span.setdefault(fragment.diagnostics.get('transcription_span_index',-1),[]).append(fragment)
                    remaining = split_overlong_sentences(remaining,target_waveform.cpu().numpy(),
                        {(c.start,c.end):by_span[i] for i,c in enumerate(remaining)})
                    annotate_navigation(remaining,target_waveform,verifier,profile)
                rejected.extend(remaining)
                self._trace_spans('remaining_single_speech',remaining)
                self._stage_audit['remaining_export'] = {
                    'count':len(remaining), 'excluded_target_spans':[[c.start,c.end] for c in accepted],
                    'target_voice_filter':False}
            self._trace_spans("final_accepted", accepted)
            from ..training_context import save_training_context
            save_training_context(paths["output"], target, reference_clips, identity_references=references)
            manifest, transcript, archive = self._write_outputs(
                accepted,
                rejected,
                stem,
                target.name,
                paths,
                progress,
                create_archive=create_archive,
                source_media=target,
            )
            progress(1.0, f"完成：保留 {len(accepted)} 句，舍弃 {len(rejected)} 句")
            return common.PipelineResult(job_id, paths["output"], archive, accepted, rejected, manifest, transcript)
        finally:
            feature_context.close()
            self._episode_waveform = None
            common.MODEL_POOL.clear_task_state()
            self._batch_bank = []
            self._tail_scorer_cache = None
            self._reference_sha256 = []
            self._reference_audio_paths = []
            self._cluster_accepted_spans = None
            self._verified_fallback_registry = {}
            self._verified_fallback_anchors = []
            self._raw_target_waveform = None
            self._local_score_cache = None
            self._raw_blocked_spans = ()
            self._subtitle_guide = None
            self._pitch_guard = None
            self._pitch_span_cache.clear()
            self._speaker_change_points = ()
            self._candidate_cut_anchors = ()
            if self._subtitle_join_guard is not None:
                self._subtitle_join_guard.close()
                self._subtitle_join_guard = None
            self._stage_audit = None
            self._cluster_report = None
            if singing_detector is not None:
                singing_detector.close()
            if self.options.cleanup_work:
                common.shutil.rmtree(paths["root"], ignore_errors=True)
