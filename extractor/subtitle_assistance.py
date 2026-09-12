"""Audio-verified subtitle proposals. Subtitle text never enters transcription."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from .audio import write_clip
from .speaker import LocalSpeakerTurnSplitter
from .subtitles import SubtitleGuide, _intersection
from .types import CandidateSentence, TimeSpan


def split_at_subtitle_pauses(pipeline, guide, speech, waveform, progress):
    """Refine overlong VAD blocks only at acoustically supported cue edges."""
    cuts = []
    duration = waveform.numel() / 16000
    edges = sorted({round(edge, 4) for cue in guide.cues
                    if (span := guide.cue_span(cue)) is not None
                    for edge in (span.start, span.end)})
    minimum_pause = max(.06, pipeline.options.silence_min_seconds)
    for index, edge in enumerate(edges):
        # A subtitle edge near an existing VAD boundary cannot improve it.
        container = next((p for p in speech if p.start + .55 < edge < p.end - .55), None)
        if container is None:
            continue
        _, runs = pipeline._refine_boundary_to_quiet_gap(
            waveform, TimeSpan(max(0, edge - 1), min(duration, edge + 1)),
            edge, target_side="left", radius=.65,
        )
        choices = sorted([
            (abs((r["start"] + r["end"]) / 2 - edge), r) for r in runs
            if r["duration"] >= minimum_pause and container.start < r["start"]
            and r["end"] < container.end
        ], key=lambda row: row[0])
        if not choices or choices[0][0] > .45:
            continue
        if len(choices) > 1 and choices[1][0] - choices[0][0] < .10:
            continue
        run = choices[0][1]
        gap = TimeSpan(run["start"], run["end"])
        if not any(_intersection(gap, old) > 0 for old in cuts):
            cuts.append(gap)
        progress((index + 1) / max(1, len(edges)), f"字幕边界对齐 {index + 1}/{len(edges)}")
    output = []
    applied = []
    for part in speech:
        start = part.start
        for gap in sorted(cuts, key=lambda p: p.start):
            if gap.start - start < .55 or part.end - gap.end < .55:
                continue
            # Guard only inside silence, without converting an upper-bound
            # hard gap or lower-bound verified gap into an unconditional join.
            guard = min(.02, max(0, (gap.duration - pipeline.options.silence_min_seconds) / 2))
            if gap.duration > pipeline.options.silence_split_seconds:
                guard = min(guard, max(0, (gap.duration - pipeline.options.silence_split_seconds - .001) / 2))
            output.append(TimeSpan(start, gap.start + guard))
            start = gap.end - guard
            applied.append([gap.start, gap.end])
        output.append(TimeSpan(start, part.end))
    guide.report["acoustic_subtitle_splits"] = applied
    return output


def retry_subtitle_vad(guide, speech, duration, stem, work_dir, vad_tools, progress):
    """Retry uncovered cues using the existing VAD on level-normalized UVR audio."""
    windows = guide.retry_windows(speech, duration)
    guide.report.update(vad_retry_windows=len(windows), vad_recovered_spans=[])
    if not windows:
        return list(speech)
    recovered = []
    with TemporaryDirectory(prefix="subtitle_vad_", dir=work_dir) as directory:
        clips = []
        for index, window in enumerate(windows):
            clip = Path(directory) / f"{index:04d}.wav"
            write_clip(stem, clip, window.start, window.end, sample_rate=16000, normalize_level=True)
            clips.append(clip)
        found = vad_tools.vad_many(clips, progress=progress)
        for window, clip in zip(windows, clips):
            for local in found.get(clip, []):
                # A detection touching the retry window may be an amputated
                # syllable. Do not turn an arbitrary subtitle edge into a cut.
                if local.start < .04 or local.end > window.duration - .04:
                    continue
                part = TimeSpan(window.start + local.start, window.start + local.end)
                if part.duration < .55:
                    continue
                if any(_intersection(part, old) > 0 for old in [*speech, *recovered]):
                    continue
                supported = any(
                    span is not None and _intersection(part, span) >= part.duration * .5
                    for span in (guide.cue_span(cue) for cue in guide.cues)
                )
                if supported:
                    recovered.append(part)
    guide.report["vad_recovered_spans"] = [[p.start, p.end] for p in recovered]
    return sorted([*speech, *recovered], key=lambda p: (p.start, p.end))


def _uncovered(span: TimeSpan, cores: list[TimeSpan]) -> list[TimeSpan]:
    cursor = span.start
    result = []
    for core in sorted(cores, key=lambda p: p.start):
        if core.end <= cursor or core.start >= span.end:
            continue
        if core.start > cursor:
            result.append(TimeSpan(cursor, min(core.start, span.end)))
        cursor = max(cursor, core.end)
    if cursor < span.end:
        result.append(TimeSpan(cursor, span.end))
    return result


def restore_subtitle_sentences(
    pipeline, guide: SubtitleGuide, accepted, rejected, speech, blocked,
    verifier, profile, stem, waveform, exclusion_profiles, threshold, progress,
):
    """Add complete-cue proposals without changing the existing identity policy.

    Each added voiced part must earn its own identity; neither subtitle text
    nor a longer accepted core can lend that identity to a neighboring voice.
    Every veto keeps the original outputs intact.
    """
    proposals = guide.groups(speech, pipeline.options.silence_split_seconds)
    audit = []
    guide.report["completion_proposals"] = audit
    guide.report["completed_sentences"] = 0
    if not accepted:
        return 0
    forbidden = [*blocked, *[
        TimeSpan(c.start, c.end) for c in rejected
        if any(c.diagnostics.get(key) for key in (
            "structural_hard_reject", "final_identity_boundary_discard",
            "final_same_speaker_internal_discard", "excluded_role_rejected",
        ))
    ]]
    splitter = None
    restored = 0
    for index, (cue, parts) in enumerate(proposals):
        span = TimeSpan(parts[0].start, parts[-1].end)
        cores = [c for c in accepted if _intersection(span, TimeSpan(c.start, c.end)) > .02]
        if not cores or any(c.start < span.start - .001 or c.end > span.end + .001 for c in cores):
            continue
        if len(cores) == 1 and abs(cores[0].start - span.start) < .001 and abs(cores[0].end - span.end) < .001:
            continue
        record = {"cue_index": cue.index, "span": [span.start, span.end], "result": "pending"}
        audit.append(record)
        progress((index + 1) / max(1, len(proposals)), f"字幕完整句核验 {index + 1}/{len(proposals)}")
        if span.duration > pipeline.options.max_sentence_seconds:
            record["result"] = "too_long"
            continue
        if any(_intersection(span, dirty) > .01 for dirty in forbidden):
            record["result"] = "blocked_audio"
            continue
        core_spans = [TimeSpan(c.start, c.end) for c in cores]
        additions = [extra for part in parts for extra in _uncovered(part, core_spans)]
        # Do not pad a too-short addition with target speech to make it pass.
        if any(extra.duration < verifier.SHORT_MIN_DURATION for extra in additions):
            record["result"] = "unverifiable_short_edge"
            continue

        checked = {}

        def verify(part):
            key = (part.start, part.end)
            if key not in checked:
                match = pipeline._verify_speaker_span(verifier, waveform, part, profile, threshold)
                exclusion = verifier.exclusion_audit(match, profile, exclusion_profiles)
                checked[key] = (match, exclusion)
            match, exclusion = checked[key]
            return match.accepted and not (exclusion and exclusion.get("excluded_role_rejected"))

        if not all(verify(part) for part in [*parts, *additions]):
            record["result"] = "independent_identity_rejected"
            continue
        if len(parts) > 1:
            merged = pipeline._merge_short_silence_same_speaker(
                parts, verifier, profile, waveform, progress=lambda _v, _m: None,
                maximum_silence_seconds=pipeline.options.silence_split_seconds,
                forbidden_joins=forbidden,
            )
            if len(merged) != 1:
                record["result"] = "speaker_continuity_rejected"
                continue
        if additions:
            tertiary, tertiary_profile = verifier._tertiary_pair(profile)
            anchor = max(core_spans, key=lambda p: p.duration)
            embeddings = tertiary._embeddings_from_waveforms([
                pipeline._waveform_span(waveform, part) for part in [anchor, *additions]
            ])
            floor = pipeline._wavlm_same_speaker_floor(tertiary_profile)
            if any(float(embeddings[0] @ embedding) < floor for embedding in embeddings[1:]):
                record["result"] = "edge_continuity_rejected"
                continue
        if not verify(span):
            record["result"] = "whole_sentence_rejected"
            continue
        if splitter is None:
            verifier._ensure_secondary(profile)
            splitter = LocalSpeakerTurnSplitter(verifier.primary, secondary=verifier.secondary)
        boundaries = splitter.detect_multiscale_speaker_boundaries(
            stem, [span], progress=lambda _v, _m: None,
        )
        if any(span.start + .20 < boundary.time < span.end - .20 for boundary in boundaries):
            record["result"] = "internal_speaker_boundary"
            continue

        replacement = CandidateSentence(span.start, span.end, "")
        replacement.diagnostics.update(deepcopy(max(cores, key=lambda c: c.duration).diagnostics))
        match, exclusion = checked[(span.start, span.end)]
        pipeline._apply_speaker_match(replacement, match, profile, threshold)
        if exclusion:
            replacement.diagnostics.update(exclusion)
        replacement.diagnostics.update(
            subtitle_completion=True,
            subtitle_original_spans=[[c.start, c.end] for c in cores],
            subtitle_added_spans=[[p.start, p.end] for p in additions],
            post_target_silence_merge=len(parts) > 1,
        )
        core_ids = {id(c) for c in cores}
        accepted[:] = [c for c in accepted if id(c) not in core_ids]
        accepted.append(replacement)
        rejected[:] = [c for c in rejected if not (span.start <= c.start and c.end <= span.end)]
        restored += 1
        record["result"] = "restored"
    accepted.sort(key=lambda c: (c.start, c.end))
    guide.report["completed_sentences"] = restored
    return restored
