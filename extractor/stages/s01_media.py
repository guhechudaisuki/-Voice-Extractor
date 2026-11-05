"""s01 media responsibilities; composed by ExtractionPipeline."""
from __future__ import annotations

from .. import pipeline_common as common


class S01MediaMethods:
    def _job_paths(self, job_id: str) -> dict[str, common.Path]:
        root = common.WORK_ROOT / job_id
        return {
            "root": root,
            "normalized_refs": root / "references_normalized",
            "reference_stems": root / "reference_stems",
            "reference_clips": root / "reference_voice_clips",
            "raw_reference_clips": root / "reference_original_voice_clips",
            "negative_refs": root / "negative_references_normalized",
            "negative_stems": root / "negative_reference_stems",
            "negative_clips": root / "negative_reference_voice_clips",
            "raw_negative_clips": root / "negative_reference_original_voice_clips",
            "normalized_target": root / "target_normalized.wav",
            "raw_vad_target": root / "target_original_vad_16k.wav",
            "singing_removed_target": root / "target_singing_removed.wav",
            "stems": root / "stems",
            "candidate_clips": root / "candidate_clips",
            "output": common.OUTPUT_ROOT / job_id,
        }

    @staticmethod
    def _is_precleaned_reference(path: common.Path) -> bool:
        """已是 16kHz 单声道人声短切片（如本工具切出的 cue 参考）→ 免二次 UVR。"""
        import soundfile as sf
        try:
            info = sf.info(str(path))
        except (OSError, RuntimeError):
            return False
        duration = info.frames / max(1, info.samplerate)
        return info.samplerate == 16000 and info.channels == 1 and duration < 10.0

    def _prepare_references(
        self,
        references: list[common.Path],
        paths: dict[str, common.Path],
        progress: common.ProgressCallback,
    ) -> tuple[list[common.Path], list[float], list[bool]]:
        paths["normalized_refs"].mkdir(parents=True, exist_ok=True)
        normalized: list[common.Path] = []
        durations: list[float] = []
        precleaned: list[bool] = []
        for index, reference in enumerate(references, start=1):
            destination = paths["normalized_refs"] / f"reference_{index:03d}.wav"
            if self._is_precleaned_reference(reference):
                # Repeated separation can damage a clean reference's high frequencies.
                common.shutil.copyfile(reference, destination)
                durations.append(common.pad_for_separator(destination))
                precleaned.append(True)
            else:
                common.normalize_audio(reference, destination, sample_rate=44100, stereo=True)
                durations.append(common.pad_for_separator(destination))
                precleaned.append(False)
            normalized.append(destination)
            progress(0.04 + 0.08 * index / max(1, len(references)), f"规范化参考音频 {index}/{len(references)}")
        return normalized, durations, precleaned

    def _prepare_negative_references(
        self,
        reference_groups: list[list[common.Path]],
        paths: dict[str, common.Path],
        progress: common.ProgressCallback,
    ) -> tuple[list[list[common.Path]], list[list[float]]]:
        normalized_groups: list[list[common.Path]] = []
        duration_groups: list[list[float]] = []
        self._negative_reference_precleaned: set[tuple[int, int]] = set()
        total = sum(len(group) for group in reference_groups)
        completed = 0
        for group_index, group in enumerate(reference_groups, start=1):
            normalized: list[common.Path] = []
            durations: list[float] = []
            group_dir = paths["negative_refs"] / f"role_{group_index:03d}"
            group_dir.mkdir(parents=True, exist_ok=True)
            for reference_index, reference in enumerate(group, start=1):
                destination = group_dir / f"reference_{reference_index:03d}.wav"
                if self._is_precleaned_reference(reference):
                    common.shutil.copyfile(reference, destination)
                    self._negative_reference_precleaned.add((group_index, reference_index))
                else:
                    common.normalize_audio(reference, destination, sample_rate=44100, stereo=True)
                durations.append(common.pad_for_separator(destination))
                normalized.append(destination)
                completed += 1
                progress(
                    0.12,
                    f"规范化排除角色 {group_index}：{reference_index}/{len(group)} "
                    f"（总计 {completed}/{max(1, total)}）",
                )
            normalized_groups.append(normalized)
            duration_groups.append(durations)
        return normalized_groups, duration_groups

    def _make_reference_clips(
        self,
        reference_stems: list[common.Path],
        spans_by_path: dict[common.Path, list[common.TimeSpan]],
        paths: dict[str, common.Path],
        output_dir: common.Path | None = None,
        prefix: str = "reference",
    ) -> list[common.Path]:
        output: list[common.Path] = []
        output_dir = output_dir or paths["reference_clips"]
        output_dir.mkdir(parents=True, exist_ok=True)
        for reference_index, stem in enumerate(reference_stems, start=1):
            spans = self._merge_vad_spans(spans_by_path.get(stem, []), gap=0.35)
            clip_index = 0
            for span in spans:
                start = max(0.0, span.start - 0.05)
                end = min(common.probe_duration(stem), span.end + 0.05)
                cursor = start
                while end - cursor >= 0.8:
                    clip_end = min(end, cursor + 8.0)
                    if clip_end - cursor < 0.8:
                        break
                    clip_index += 1
                    destination = output_dir / (
                        f"{prefix}_{reference_index:03d}_{clip_index:03d}.wav"
                    )
                    common.write_clip(stem, destination, cursor, clip_end, sample_rate=16000)
                    output.append(destination)
                    cursor = clip_end
            if clip_index == 0:
                # Short references are still useful; the verifier pads them safely.
                destination = output_dir / f"{prefix}_{reference_index:03d}_001.wav"
                common.write_clip(stem, destination, 0.0, common.probe_duration(stem), sample_rate=16000)
                output.append(destination)
        if not output:
            raise ValueError("参考音频中未检测到有效讲话")
        return output

    @staticmethod
    def _merge_vad_spans(spans: list[common.TimeSpan], gap: float = 0.18) -> list[common.TimeSpan]:
        if not spans:
            return []
        spans = sorted(spans, key=lambda item: item.start)
        merged: list[common.TimeSpan] = [spans[0]]
        for span in spans[1:]:
            previous = merged[-1]
            if span.start <= previous.end + gap:
                merged[-1] = common.TimeSpan(previous.start, max(previous.end, span.end))
            else:
                merged.append(span)
        return merged
