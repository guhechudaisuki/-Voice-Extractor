"""One continuous identity decoder followed by exact-interval certification."""
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Callable, TYPE_CHECKING
import torch

from .base import Stage
from ..audio import speech_ratio
from ..types import CandidateSentence, TimeSpan
from ..speaker import DualSpeakerVerifier, SpeakerMatchProfile, ExclusionSpeakerProfile
from ..nextgen.ledger import Candidate, CandidateLedger, Decision, EvidenceKind, LocalEvidence, State
from ..nextgen.timeline import SourceTimeline
from ..model_pool import resource_key
from ..config import SV_MODEL, CAMPLUS_MODEL

if TYPE_CHECKING:
    from ..pipeline import ExtractionPipeline


@dataclass
class IdentityInput:
    pipeline: 'ExtractionPipeline'
    waveform: torch.Tensor
    spans: list[TimeSpan]
    vad: list[TimeSpan]
    verifier: DualSpeakerVerifier
    profile: SpeakerMatchProfile
    exclusions: list[ExclusionSpeakerProfile]
    rejected: list[CandidateSentence]
    stem: Path
    references: list[Path]
    negative_references: list[list[Path]]
    target: Path
    source_sha256: str
    job_id: str
    progress: Callable[[float, str], None]


@dataclass
class IdentityOutput:
    accepted: list[CandidateSentence]
    ledger: CandidateLedger
    review_spans: list[TimeSpan] = field(default_factory=list)


def certify_exact(pipeline, candidates, rejected, verifier, profile, waveform, exclusions,
                  *, ledger=None, origin='exact_final_audio', scored=None):
    """No boundary edit, parent score or context prior can skip this review."""
    threshold = max(pipeline.options.speaker_threshold, profile.primary.suggested_threshold)
    blocks = [*pipeline._raw_blocked_spans,
              *(TimeSpan(*s) for s in pipeline.options.user_excluded_spans)]
    retained = []
    for candidate in candidates:
        span = TimeSpan(candidate.start, candidate.end)
        from ..sentence_policy import exact_certificate
        from ..recall import MIN_TARGET_SECONDS
        certificate = exact_certificate(candidate)
        from ..nextgen.cluster_gate import HARD_VETOES
        exact = candidate.diagnostics.get('exact_multichannel_certificate') or {}
        local_veto = (exact.get('span') == [span.start,span.end]
                      and (exact.get('positive_conflict') or exact.get('blocked')))
        blocked = (any(candidate.diagnostics.get(key) for key in HARD_VETOES)
                   or local_veto or any(min(span.end,b.end) > max(span.start,b.start) for b in blocks))
        minimum = MIN_TARGET_SECONDS if certificate else getattr(pipeline.options, 'min_output_seconds', 0.)
        too_short = span.duration + 1e-6 < minimum
        invalid = not (0 <= span.start < span.end <= len(waveform) / 16000 + 1e-6)
        match = None if blocked or invalid else pipeline._verify_speaker_span(
            verifier, waveform, span, profile, threshold, audit_raw=True)
        # Rechecking a WavLM certificate must retain its stricter exclusion
        # contract even when this fresh dual decision has no tertiary tier.
        exclusion_options = {'tertiary_recovery': True} if (
            certificate and certificate.get('authority') == 'wavlm') else {}
        exclusion = (verifier.exclusion_audit(match, profile, exclusions,
            **exclusion_options) or {}) if match is not None else {}
        if (match is not None and not match.accepted and certificate is None
                and not exclusion.get('excluded_role_rejected')
                and candidate.diagnostics.get('target_locator_recovery')):
            rescued = verifier.promote_local_with_tertiary(
                pipeline._waveform_span(waveform,span),profile,match,span.duration)
            if rescued is not None:
                match = rescued
                exclusion = verifier.exclusion_audit(match,profile,exclusions,tertiary_recovery=True) or {}
                too_short = span.duration + 1e-6 < MIN_TARGET_SECONDS
        if scored is not None:
            scored.append((span,candidate,match))
        passed = bool(match is not None and (match.accepted or certificate) and not too_short
                      and not exclusion.get('excluded_role_rejected'))
        if match is not None:
            pipeline._apply_speaker_match(candidate, match, profile, threshold)
        candidate.diagnostics.update(exclusion)
        candidate.accepted = passed
        candidate.diagnostics['acoustic_identity_audit'] = {
            'span': [span.start, span.end], 'passed': passed, 'method': origin,
            'authority': certificate['authority'] if certificate else 'wavlm' if getattr(match,'tier',None)=='tertiary' else 'dual',
            'prior_method': certificate.get('method') if certificate else None,
            'exclusion_checked': match is not None}
        candidate.diagnostics['exact_audio_verified'] = passed
        if passed:
            candidate.reject_reason = ''
            retained.append(candidate)
        else:
            candidate.reject_reason = ('污染边界否决' if blocked else '无效音频区间' if invalid
                else '排除角色复核失败' if exclusion.get('excluded_role_rejected')
                else '说话回合过短' if too_short else '声纹匹配不足')
            candidate.audio_file = candidate.text_file = candidate.video_file = ''
            if not any(old is candidate for old in rejected):
                rejected.append(candidate)
        if ledger is not None and invalid:
            candidate.diagnostics['invalid_proposal'] = {'reason':'outside_decoded_audio'}
        if ledger is not None and not invalid:
            interval = ledger.source.from_seconds(span.start, span.end)
            proposal = Candidate(ledger.source.source_sha256, 16000, interval, interval,
                                 (interval,), origin)
            key = ledger.add(proposal)
            evidence = ledger.observe(LocalEvidence(ledger.source.source_sha256,16000,interval,
                EvidenceKind.TARGET if passed else EvidenceKind.OTHER if exclusion.get('excluded_role_rejected')
                else EvidenceKind.UNCERTAIN, ledger.reference_digest, ledger.model_digest, origin))
            ledger.decide(Decision(key, State.ACCEPTED if passed else State.REJECTED,
                'exact_audio_pass' if passed else candidate.reject_reason, (evidence,), 'continuous-v1'))
            candidate.diagnostics['ledger_candidate_key'] = key
    return retained


class ContinuousIdentityStage(Stage[IdentityInput, IdentityOutput]):
    name = 'continuous_identity'

    def run(self, value):
        p = value.pipeline
        flow = []
        if p._stage_audit is not None:
            p._stage_audit['identity_flow'] = flow

        def record_step(step, accepted):
            reasons = {}
            for candidate in value.rejected:
                reason = candidate.reject_reason or 'unspecified'
                reasons[reason] = reasons.get(reason, 0) + 1
            flow.append({'step': step, 'accepted_count': len(accepted),
                'accepted_seconds': round(sum(c.duration for c in accepted), 5),
                'accepted_spans': [[c.start, c.end] for c in accepted],
                'rejected_count': len(value.rejected), 'reject_summary': reasons})

        digest = hashlib.sha256(repr(resource_key('dual-encoder',SV_MODEL,CAMPLUS_MODEL)).encode()).hexdigest()
        references = hashlib.sha256(''.join(sorted(p._reference_sha256)).encode()).hexdigest()
        ledger = CandidateLedger(SourceTimeline(value.source_sha256,16000,len(value.waveform)), references,digest)
        if not value.spans:
            if p._stage_audit is not None:
                p._stage_audit['candidate_ledger'] = ledger.to_dict()
            return IdentityOutput([],ledger)
        proposals = [CandidateSentence(s.start,s.end,'',diagnostics={
            'audio_boundary_source':'acoustic_proposal',
            'speech_ratio':speech_ratio(value.vad,s.start,s.end),
            'speech_seconds':speech_ratio(value.vad,s.start,s.end)*s.duration}) for s in value.spans]
        value.progress(.63, f'连续证据场：登记 {len(proposals)} 个独立区间')
        scored = []
        accepted = certify_exact(p,proposals,value.rejected,value.verifier,value.profile,
            value.waveform,value.exclusions,ledger=ledger,origin='initial_exact_audio',scored=scored)
        record_step('initial_exact', accepted)
        from .sentence_recovery import recover_sentences
        analysis_spans = recover_sentences(value,accepted,scored,ledger)
        record_step('sentence_recovery', accepted)
        # TemporalMiner uses EpisodeManifold for reference/rival relations and
        # IdentityTrajectory for multiscale decoding. No earlier heuristic
        # rescue may promote a candidate around this one shared field.
        p._apply_cluster_identity(value.stem,value.references,value.negative_references,
            value.verifier,value.profile,analysis_spans,accepted,value.rejected,
            value.target,value.job_id,value.progress)
        record_step('cluster_identity', accepted)
        if not getattr(p.options, 'branch_validation', True):
            # Selection is explicit export policy, not an invented acoustic
            # certificate. Off/audit retain the independently accepted path;
            # enforce uses the continuous decoder's own selected intervals.
            record_step('direct_timeline_selection', accepted)
            p._trace_spans('identity_accepted_before_stt', accepted)
            if p._stage_audit is not None:
                p._stage_audit['candidate_ledger'] = ledger.to_dict()
            return IdentityOutput(sorted(accepted,key=lambda c:(c.start,c.end)),ledger)
        p._apply_user_marked_exclusions(accepted,value.rejected,
            [TimeSpan(*s) for s in p.options.user_excluded_spans],value.verifier,
            value.profile,value.waveform,value.exclusions,p.options.speaker_threshold)
        record_step('user_exclusions', accepted)
        p._enforce_atomic_boundaries(accepted,value.rejected,value.verifier,value.profile,
                                    value.waveform,value.exclusions)
        record_step('atomic_boundaries', accepted)
        models = p._ensure_tail_scorer()
        if models is not None and p._local_identity_reject_threshold is not None:
            from ..local_identity_classifier import audit_candidates
            _, trims = audit_candidates(accepted,value.rejected,value.waveform,
                lambda wave: p._score_local_window(models,wave),p._local_identity_reject_threshold,
                model_sha256=p._local_identity_model_sha256)
            # Trims are proposals only. The same exact-audio gate sees them.
            accepted.extend(CandidateSentence(*t['span'],'',diagnostics={'classifier_veto_trim':{
                'parent_span': t['parent_span'], 'kept_span':t['span']}}) for t in trims)
        record_step('local_classifier', accepted)
        accepted = certify_exact(p,accepted,value.rejected,value.verifier,value.profile,
            value.waveform,value.exclusions,ledger=ledger)
        record_step('exact_after_guards', accepted)
        from ..sentence_continuity import merge_certified_contiguous
        from ..utterance_assembly import gap_verifier
        regions = (p._cluster_report or {}).get('diagnostics', {}).get('trajectory_spans', ())
        blocked = [(s.start,s.end) for s in p._raw_blocked_spans]
        parts = list(accepted)
        merged = merge_certified_contiguous(parts,regions=regions,blocked=blocked,
            maximum_seconds=min(20.,p.options.max_sentence_seconds),
            maximum_gap_seconds=p.options.silence_split_seconds,
            gap_verified=gap_verifier(value.waveform,regions=regions,sources=parts,blocked=blocked))
        accepted = []
        original_ids = {id(c) for c in parts}
        for candidate in merged:
            if id(candidate) in original_ids:
                accepted.append(candidate)
                continue
            checked = certify_exact(p,[candidate],value.rejected,value.verifier,value.profile,
                value.waveform,value.exclusions,ledger=ledger,origin='joined_exact_audio')
            accepted.extend(checked or [c for c in parts
                if candidate.start <= c.start and c.end <= candidate.end])
        record_step('contiguous_merge', accepted)
        p._refine_cropped_sentences(accepted,value.waveform,p._raw_blocked_spans)
        accepted = certify_exact(p,accepted,value.rejected,value.verifier,value.profile,
            value.waveform,value.exclusions,ledger=ledger,origin='pre_stt_exact_audio')
        record_step('pre_stt_exact', accepted)
        p._filter_pitch_register(accepted,value.rejected,value.waveform)
        record_step('pitch_filter', accepted)
        p._trace_spans('identity_accepted_before_stt',accepted)
        if p._stage_audit is not None:
            p._stage_audit['candidate_ledger'] = ledger.to_dict()
        return IdentityOutput(sorted(accepted,key=lambda c:(c.start,c.end)),ledger)
