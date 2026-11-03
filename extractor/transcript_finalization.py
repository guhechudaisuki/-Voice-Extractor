"""Finalize genuine word timings before splitting, with acoustic fences intact."""
from .audio import write_clip
from .word_boundaries import real_words, snap_boundaries
from .acoustic_qc import split_overlong_sentences


def finalize_transcripts(candidates, waveform, fragments, *, stem, clips_dir,
                         vad_tools, refine_crops, blocked=(), progress=None):
    by_candidate = {id(c): fragments.get((c.start, c.end), []) for c in candidates}
    chinese = {}
    for index, candidate in enumerate(candidates):
        if candidate.language == 'zh':
            path = clips_dir / f'zh_stt_{index:04d}.wav'
            write_clip(stem, path, candidate.start, candidate.end, sample_rate=16000)
            chinese[path] = candidate
    if chinese:
        try:
            refined = vad_tools.transcribe_files(list(chinese), progress=progress)
            for path, candidate in chinese.items():
                candidate.text = refined.get(path) or candidate.whisper_text
                words = real_words(getattr(vad_tools, 'word_timestamps', {}).get(path, []))
                if words:
                    candidate.diagnostics['word_timestamps'] = [dict(word,
                        start=word['start'] + candidate.start, end=word['end'] + candidate.start)
                        for word in words if 0 <= word['start'] < word['end'] <= candidate.duration + .001]
        finally:
            for path in chinese:
                path.unlink(missing_ok=True)
    # Energy valleys propose cuts first; word protection is the final authority
    # within the independently certified identity interval, never outside it.
    refine_crops(candidates, waveform, blocked)
    barriers = [(span.start, span.end) for span in blocked]
    for candidate in candidates:
        allowed = candidate.diagnostics.get('word_boundary_allowed_span', [candidate.start, candidate.end])
        lower, upper = allowed
        trimmed = candidate.diagnostics.get('stt_trimmed_span')
        if trimmed:
            lower, upper = max(lower, trimmed[0]), min(upper, trimmed[1])
        for name in ('local_identity_audit', 'acoustic_identity_audit'):
            certificate = candidate.diagnostics.get(name)
            if isinstance(certificate, dict) and certificate.get('passed'):
                span = certificate.get('source_span') or certificate.get('span')
                if span:
                    lower, upper = max(lower, span[0]), min(upper, span[1])
        for other in candidates:
            if other is candidate:
                continue
            if other.end <= candidate.start:
                lower = max(lower, other.end)
            elif other.start >= candidate.end:
                upper = min(upper, other.start)
        candidate.start, candidate.end, audit = snap_boundaries(
            candidate.start, candidate.end, candidate.diagnostics.get('word_timestamps', []),
            allowed=[(lower, upper)], blocked=barriers, waveform=waveform)
        candidate.diagnostics['word_boundary_refinement'] = audit
    rekeyed = {(c.start, c.end): by_candidate[id(c)] for c in candidates}
    return split_overlong_sentences(candidates, waveform.detach().float().cpu().numpy(), rekeyed)
