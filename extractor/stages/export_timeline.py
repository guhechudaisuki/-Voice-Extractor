"""Direct selection union and transcription that cannot change audio coverage."""
from copy import deepcopy
from collections import Counter

from ..types import CandidateSentence


def union_selected(first, second):
    """Union only overlapping/touching selected intervals, never fill a gap.

    This is an explicit export selection, not a new acoustic certificate.
    Evidence of each source interval remains auditable without being inherited.
    """
    groups = []
    for candidate in sorted([*first,*second],key=lambda c:(c.start,c.end)):
        if not candidate.accepted or candidate.end <= candidate.start:
            continue
        if groups and candidate.start <= max(c.end for c in groups[-1]):
            groups[-1].append(candidate)
        else:
            groups.append([candidate])
    output, audit = [], []
    for group in groups:
        start,end=min(c.start for c in group),max(c.end for c in group)
        sources=sorted({source for c in group for source in c.diagnostics.get('track_sources',())})
        provenance='dual_track_merged' if len(sources)>1 else sources[0] if sources else 'track1_acoustic'
        parts=[{'span':[c.start,c.end], 'track_sources':c.diagnostics.get('track_sources',[]),
                'selection':deepcopy(c.diagnostics)} for c in group]
        candidate=CandidateSentence(start,end,'',accepted=True,diagnostics={
            'branch_validation':False, 'export_group':'target',
            'track_provenance':provenance, 'track_sources':sources,
            'timeline_union':{'span':[start,end],'method':'selected_interval_union',
                              'additional_validation':'disabled','parts':parts}})
        output.append(candidate)
        audit.append({'span':[start,end],'decision':'direct_selected_union',
                      'parts':[[c.start,c.end] for c in group]})
    return output,audit


def attach_transcripts(candidates, fragments):
    """Attach actual ASR text/words by input index; keep every audio fence."""
    from .s06_transcription import S06TranscriptionMethods
    grouped={}
    for fragment in fragments:
        index=int(fragment.diagnostics.get('transcription_span_index',-1))
        grouped.setdefault(index,[]).append(fragment)
    for index,candidate in enumerate(candidates):
        parts=sorted(grouped.get(index,()),key=lambda c:(c.start,c.end))
        text=''
        for part in parts:
            text=S06TranscriptionMethods._join_transcript_text(text,part.whisper_text or part.text)
        languages=Counter(p.language for p in parts if p.language and p.language!='auto')
        candidate.text=candidate.whisper_text=text.strip()
        candidate.language=languages.most_common(1)[0][0] if languages else 'auto'
        candidate.diagnostics.update(transcript_pending=not bool(text.strip()),
            stt_fragment_count=len(parts),word_timestamps=[word for part in parts
                for word in part.diagnostics.get('word_timestamps',())])
    return candidates
