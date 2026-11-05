"""s07 export responsibilities; composed by ExtractionPipeline."""
from __future__ import annotations

from .. import pipeline_common as common


class S07ExportMethods:
    def _write_outputs(
        self,
        accepted: list[common.CandidateSentence],
        rejected: list[common.CandidateSentence],
        stem: common.Path,
        target_name: str,
        paths: dict[str, common.Path],
        progress: common.ProgressCallback,
        *,
        create_archive: bool = True,
        source_media: common.Path | None = None,
    ) -> tuple[common.Path, common.Path, common.Path]:
        # Recovery can move the same object through several pools. Normalize
        # that state before any audio is written, so a stale accepted flag in
        # a rejected row cannot masquerade as an actual exported sentence.
        invalid_accepted = [c for c in accepted if not c.accepted or c.reject_reason]
        accepted[:] = [c for c in accepted if c.accepted and not c.reject_reason]
        accepted_ids = {id(c) for c in accepted}
        rejected[:] = list({id(c): c for c in [*rejected, *invalid_accepted]
                           if id(c) not in accepted_ids}.values())
        for candidate in rejected:
            candidate.accepted = False
            candidate.audio_file = candidate.text_file = candidate.video_file = ""
        output_dir = paths["output"]
        output_dir.mkdir(parents=True, exist_ok=True)
        common.annotate_raw_recovery([*accepted, *rejected], getattr(self, '_raw_vad_recovered', ()))
        from ..audio_mastering import prepare_masters
        export_stem, raw_master, export_rate = prepare_masters(
            stem, source_media, output_dir, self.options.export_sample_rate)
        for candidate in [*accepted, *rejected]:
            candidate.diagnostics['aligned_vocal_audio'] = str(export_stem.resolve())
            if raw_master:
                candidate.diagnostics['aligned_raw_audio'] = str(raw_master.resolve())
            candidate.diagnostics['export_sample_rate'] = export_rate
        cluster_report = getattr(self, "_cluster_report", None)
        if cluster_report is not None:
            cluster_report["final_output_count"] = len(accepted)
            (output_dir / "cluster_report.json").write_text(
                common.json.dumps(cluster_report, ensure_ascii=False, indent=2), encoding="utf-8")
        audio_dir = output_dir / "audio"
        text_dir = output_dir / "text"
        video_dir = output_dir / "video"
        audio_dir.mkdir(exist_ok=True)
        text_dir.mkdir(exist_ok=True)
        video_source = None
        if self.options.export_video_clips and source_media is not None:
            try:
                if common.has_video_stream(source_media):
                    video_source = source_media
                    video_dir.mkdir(exist_ok=True)
            except (OSError, RuntimeError) as exc:
                common.LOGGER.warning("视频流检测失败，跳过视频片段：%s", exc)
        for index, candidate in enumerate(accepted, start=1):
            stem_name = f"{index:04d}_{common._safe_name(candidate.text or candidate.whisper_text)}"
            audio_path = audio_dir / f"{stem_name}.wav"
            text_path = text_dir / f"{stem_name}.txt"
            common.write_clip(
                export_stem,
                audio_path,
                candidate.start,
                candidate.end,
                sample_rate=export_rate,
                normalize_level=True,
            )
            common._record_acoustic_quality(candidate, audio_path)
            text_path.write_text(candidate.text or candidate.whisper_text, encoding="utf-8")
            candidate.audio_file = str(audio_path.relative_to(output_dir))
            candidate.text_file = str(text_path.relative_to(output_dir))
            if video_source is not None:
                video_path = video_dir / f"{stem_name}.mp4"
                common.write_video_clip(
                    video_source,
                    video_path,
                    candidate.start,
                    candidate.end,
                )
                candidate.video_file = str(video_path.relative_to(output_dir))
            progress(0.94 + 0.05 * index / max(1, len(accepted)), f"导出句子 {index}/{len(accepted)}")

        remaining = [c for c in rejected if self.options.export_all_sentences
                     and c.diagnostics.get('remaining_sentence_export')]
        if remaining:
            remaining_audio = output_dir / 'remaining_audio'
            remaining_text = output_dir / 'remaining_text'
            remaining_audio.mkdir(exist_ok=True)
            remaining_text.mkdir(exist_ok=True)
            for index,candidate in enumerate(remaining,1):
                name=f'{index:04d}_{common._safe_name(candidate.text or candidate.whisper_text)}'
                audio_path=remaining_audio / (name+'.wav')
                text_path=remaining_text / (name+'.txt')
                common.write_clip(export_stem,audio_path,candidate.start,candidate.end,
                                  sample_rate=export_rate,normalize_level=True)
                common._record_acoustic_quality(candidate,audio_path)
                text_path.write_text(candidate.text or candidate.whisper_text,encoding='utf-8')
                candidate.audio_file=str(audio_path.relative_to(output_dir))
                candidate.text_file=str(text_path.relative_to(output_dir))
            (output_dir/'remaining_manifest.json').write_text(common.json.dumps({
                'purpose':'remaining_single_speech_review', 'target_voice_filter':False,
                'count':len(remaining), 'sentences':[common._manifest_record(c) for c in remaining]},
                ensure_ascii=False,indent=2),encoding='utf-8')

        if self.options.keep_rejected and rejected:
            rejected_dir = output_dir / "rejected_audio"
            rejected_dir.mkdir(exist_ok=True)
            for index, candidate in enumerate(rejected, start=1):
                if candidate.diagnostics.get('remaining_sentence_export'):
                    continue
                rejected_path = rejected_dir / f"{index:04d}_{common._safe_name(candidate.reject_reason, 'rejected')}.wav"
                common.write_clip(export_stem, rejected_path, candidate.start, candidate.end, sample_rate=export_rate)
                common._record_acoustic_quality(candidate, rejected_path)
                candidate.audio_file = str(rejected_path.relative_to(output_dir))
        subtitle_guide = getattr(self, "_subtitle_guide", None)
        if subtitle_guide is not None:
            for candidate in [*accepted, *rejected]:
                subtitle_guide.annotate(candidate)
        stage_audit = self._stage_audit
        if stage_audit is not None:
            (output_dir / "stage_audit.json").write_text(
                common.json.dumps(stage_audit, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        accepted_records = [common._manifest_record(candidate) for candidate in accepted]
        rejected_records = [common._manifest_record(candidate) for candidate in rejected]
        records = [*accepted_records, *rejected_records]
        reject_summary: dict[str, int] = {}
        for candidate in rejected:
            reason = candidate.reject_reason or "未知原因"
            reject_summary[reason] = reject_summary.get(reason, 0) + 1
        manifest = output_dir / "manifest.json"
        manifest.write_text(
            common.json.dumps(
                {
                    "target": target_name,
                    "accepted_count": len(accepted),
                    "remaining_count": len(remaining),
                    "rejected_count": len(rejected),
                    "reject_summary": reject_summary,
                    "options": common.asdict(self.options),
                    **({"stage_audit_file": "stage_audit.json"} if stage_audit is not None else {}),
                    **({"subtitle_assistance": subtitle_guide.report} if subtitle_guide is not None else {}),
                    "sentences": records,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        csv_path = output_dir / "manifest.csv"
        fields = [
            "audio_file",
            "text_file",
            "video_file",
            "text",
            "language",
            "start",
            "end",
            "speaker_score",
            "window_min_score",
            "window_p20_score",
            "speaker_vote_ratio",
            "window_vote_ratio",
            "speaker_threshold",
            "overlap_score",
            "singing_score",
            "quality_grade",
            "snr_db",
            "accepted",
            "reject_reason",
        ]
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = common.csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)

        srt_path = output_dir / "transcript.srt"
        srt_parts: list[str] = []
        for index, candidate in enumerate(accepted, start=1):
            srt_parts.append(
                f"{index}\n{common._srt_time(candidate.start)} --> {common._srt_time(candidate.end)}\n{candidate.text or candidate.whisper_text}\n"
            )
        srt_path.write_text("\n".join(srt_parts), encoding="utf-8")

        archive_path = common.OUTPUT_ROOT / f"{output_dir.name}.zip"
        if not create_archive:
            return manifest, srt_path, archive_path
        # Never expose a partially written archive to Explorer or the GUI.
        # Windows may try to open the path as soon as it appears, so build the
        # ZIP beside the final path and publish it only after ZipFile closes.
        partial_archive = archive_path.with_name(archive_path.name + ".part")
        partial_archive.unlink(missing_ok=True)
        with common.zipfile.ZipFile(partial_archive, "w", compression=common.zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(output_dir.rglob("*")):
                if file.is_file() and '_training' not in file.relative_to(output_dir).parts:
                    archive.write(file, file.relative_to(output_dir))
        partial_archive.replace(archive_path)
        return manifest, srt_path, archive_path

    def run_many(
        self,
        references: common.Iterable[str | common.Path],
        targets: common.Iterable[str | common.Path],
        negative_references: common.Iterable[common.Iterable[str | common.Path]] | None = None,
        progress: common.ProgressCallback | None = None,
        subtitles: dict[str | common.Path, str | common.Path] | None = None,
    ) -> common.BatchPipelineResult:
        """Process target files sequentially and create one batch download."""
        progress = progress or common._noop_progress
        reference_paths = [common.Path(path) for path in references if path]
        negative_reference_groups = [
            [common.Path(path) for path in group if path]
            for group in (negative_references or [])
        ]
        negative_reference_groups = [
            group for group in negative_reference_groups if group
        ]
        target_paths = [common.Path(path) for path in targets if path]
        if not reference_paths:
            raise ValueError("请至少提供一段参考音频")
        if not target_paths:
            raise ValueError("请至少提供一段待提取音频")
        missing = [str(path) for path in target_paths if not path.exists()]
        if missing:
            raise FileNotFoundError("待提取音频不存在：" + "，".join(missing))
        subtitle_paths = common.validate_subtitle_bindings(target_paths, subtitles)

        batch_id = common.time.strftime("%Y%m%d_%H%M%S") + "_batch_" + common.uuid.uuid4().hex[:6]
        batch_dir = common.OUTPUT_ROOT / batch_id
        batch_dir.mkdir(parents=True, exist_ok=False)
        # Each batch starts with an empty pool of confirmed references.
        self._batch_bank = []
        results: list[common.PipelineResult] = []
        target_count = len(target_paths)

        try:
            for target_index, target in enumerate(target_paths, start=1):
                prefix = f"目标文件 {target_index}/{target_count} [{target.name}]"

                def report_target(value: float, message: str, *, _prefix: str = prefix) -> None:
                    overall = ((target_index - 1) + min(1.0, max(0.0, value))) / target_count
                    progress(overall, f"{_prefix} · {message}")

                report_target(0.0, "准备处理")
                destination_name = f"{target_index:03d}_{common._safe_name(target.stem, 'target')}"
                destination = batch_dir / destination_name
                child_job_id = f"{batch_id}_{target_index:03d}"
                result = self.run(
                    reference_paths,
                    target,
                    negative_references=negative_reference_groups,
                    progress=report_target,
                    job_id=child_job_id,
                    create_archive=False,
                    subtitle=subtitle_paths.get(target.resolve()),
                )
                if destination.exists():
                    common.shutil.rmtree(destination)
                common.shutil.move(str(result.output_dir), str(destination))

                old_output_dir = result.output_dir
                old_manifest_rel = result.manifest_path.relative_to(old_output_dir)
                old_transcript_rel = result.transcript_path.relative_to(old_output_dir)
                for sentence in [*result.accepted, *result.rejected]:
                    if sentence.audio_file:
                        sentence.audio_file = str(common.Path(destination_name) / sentence.audio_file)
                    if sentence.text_file:
                        sentence.text_file = str(common.Path(destination_name) / sentence.text_file)
                    if sentence.video_file:
                        sentence.video_file = str(common.Path(destination_name) / sentence.video_file)
                result.output_dir = destination
                result.manifest_path = destination / old_manifest_rel
                result.transcript_path = destination / old_transcript_rel
                result.archive_path.unlink(missing_ok=True)
                results.append(result)
                report_target(1.0, f"文件完成，保留 {len(result.accepted)} 句")

            records: list[dict] = []
            target_summaries: list[dict] = []
            srt_parts: list[str] = []
            srt_index = 1
            for target, result in zip(target_paths, results):
                target_summaries.append(
                    {
                        "target": target.name,
                        "output_dir": result.output_dir.name,
                        "accepted_count": len(result.accepted),
                        "remaining_count": sum(bool(c.diagnostics.get('remaining_sentence_export'))
                                               for c in result.rejected),
                        "rejected_count": len(result.rejected),
                        "reject_summary": {
                            reason: sum((item.reject_reason or "未知原因") == reason for item in result.rejected)
                            for reason in sorted({item.reject_reason or "未知原因" for item in result.rejected})
                        },
                    }
                )
                for sentence in [*result.accepted, *result.rejected]:
                    record = common._manifest_record(sentence)
                    record["target"] = target.name
                    records.append(record)
                for sentence in result.accepted:
                    srt_parts.append(
                        f"{srt_index}\n{common._srt_time(sentence.start)} --> {common._srt_time(sentence.end)}\n"
                        f"[{target.name}] {sentence.text or sentence.whisper_text}\n"
                    )
                    srt_index += 1

            manifest_path = batch_dir / "batch_manifest.json"
            manifest_path.write_text(
                common.json.dumps(
                    {
                        "batch_id": batch_id,
                        "target_count": target_count,
                        "accepted_count": sum(len(result.accepted) for result in results),
                        "remaining_count": sum(bool(c.diagnostics.get('remaining_sentence_export'))
                            for result in results for c in result.rejected),
                        "rejected_count": sum(len(result.rejected) for result in results),
                        "options": common.asdict(self.options),
                        "targets": target_summaries,
                        "sentences": records,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            csv_path = batch_dir / "batch_manifest.csv"
            fields = [
                "target",
                "audio_file",
                "text_file",
                "video_file",
                "text",
                "language",
                "start",
                "end",
                "speaker_score",
                "window_min_score",
                "window_p20_score",
                "speaker_vote_ratio",
                "window_vote_ratio",
                "speaker_threshold",
                "overlap_score",
                "singing_score",
                "quality_grade",
                "snr_db",
                "accepted",
                "reject_reason",
            ]
            with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = common.csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(records)

            transcript_path = batch_dir / "batch_transcript.srt"
            transcript_path.write_text("\n".join(srt_parts), encoding="utf-8")
            archive_path = common.OUTPUT_ROOT / f"{batch_id}.zip"
            temporary_archive = archive_path.with_suffix(".zip.tmp")
            temporary_archive.unlink(missing_ok=True)
            with common.zipfile.ZipFile(temporary_archive, "w", compression=common.zipfile.ZIP_DEFLATED) as archive:
                for file in batch_dir.rglob("*"):
                    # A batch download is the single archive for all targets;
                    # never nest a child archive inside it.
                    if file.is_file() and file.suffix.lower() != ".zip" and '_training' not in file.relative_to(batch_dir).parts:
                        archive.write(file, file.relative_to(batch_dir))
            with common.zipfile.ZipFile(temporary_archive, "r") as check:
                bad_entry = check.testzip()
                if bad_entry is not None:
                    raise RuntimeError(f"批次压缩包校验失败：{bad_entry}")
            temporary_archive.replace(archive_path)
            progress(1.0, f"批次完成：{target_count} 个目标文件")
            return common.BatchPipelineResult(
                batch_id=batch_id,
                output_dir=batch_dir,
                archive_path=archive_path,
                results=results,
                manifest_path=manifest_path,
                transcript_path=transcript_path,
            )
        except Exception:
            if not results:
                common.shutil.rmtree(batch_dir, ignore_errors=True)
            raise

    def _deduplicate_content_clips(
        self,
        accepted_turns: list[common.CandidateSentence],
        stem_waveform: common.torch.Tensor,
        progress: common.ProgressCallback,
        *,
        rejected: list[common.CandidateSentence] | None = None,
    ) -> int:
        from ..content_dedup import deduplicate_content

        removed = deduplicate_content(
            accepted_turns, stem_waveform.detach().cpu().numpy(),
        )
        if rejected is not None:
            rejected.extend(removed)
        if removed:
            progress(0.93, f"音频去重：移除 {len(removed)} 个重复片段，保留完整句边界")
        return len(removed)
