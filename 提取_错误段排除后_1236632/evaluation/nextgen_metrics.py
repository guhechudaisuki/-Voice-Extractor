"""Sample-based acoustic evaluation; ground truth never enters runtime code."""
from __future__ import annotations

from dataclasses import dataclass

from extractor.nextgen.ledger import Candidate
from extractor.nextgen.timeline import SampleSpan, SourceTimeline, union_length


@dataclass(frozen=True)
class AcousticTruth:
    source: SourceTimeline
    reviewed: tuple[SampleSpan, ...]
    target_utterances: tuple[SampleSpan, ...]
    other: tuple[SampleSpan, ...] = ()
    singing: tuple[SampleSpan, ...] = ()
    overlap: tuple[SampleSpan, ...] = ()
    human_uncertain: tuple[SampleSpan, ...] = ()

    def __post_init__(self) -> None:
        for spans in (self.reviewed, self.target_utterances, self.other, self.singing,
                      self.overlap, self.human_uncertain):
            for span in spans:
                self.source.validate(span)
        for span in (*self.target_utterances, *self.other, *self.singing, *self.overlap, *self.human_uncertain):
            if coverage(span, self.reviewed) != span.length:
                raise ValueError("Truth extends outside manually reviewed source regions")
        for target in self.target_utterances:
            if any(target.intersection(risk) for risk in (*self.other, *self.singing, *self.overlap,
                                                         *self.human_uncertain)):
                raise ValueError("Eligible single-speaker truth contradicts exclusion/unknown labels")
        if sum(span.length for span in self.target_utterances) != union_length(list(self.target_utterances)):
            raise ValueError("Overlapping eligible utterance labels")


def coverage(span: SampleSpan, regions: tuple[SampleSpan, ...]) -> int:
    return union_length([part for region in regions if (part := region.intersection(span)) is not None])


def evaluate(truth: AcousticTruth, outputs: tuple[Candidate, ...], *,
             candidates: tuple[Candidate, ...] = (), unresolved: tuple[Candidate, ...] = ()) -> dict:
    for row in (*outputs, *candidates, *unresolved):
        if (row.source_sha256, row.sample_rate) != (truth.source.source_sha256, truth.source.sample_rate):
            raise ValueError("Evaluation source mismatch")
        truth.source.validate(row.output)
    contamination = (*truth.other, *truth.singing, *truth.overlap)
    clean = [row for row in outputs if row.start_complete and row.end_complete
             and not any(row.output.intersection(span) for span in (*contamination, *truth.human_uncertain))
             and coverage(row.output, truth.reviewed) == row.output.length]
    complete = [target for target in truth.target_utterances if any(row.output.contains(target) for row in clean)]
    missed = [target for target in truth.target_utterances if target not in complete]
    # Reachability upper bound is geometry only, never an oracle purity claim.
    reachable = [target for target in truth.target_utterances
                 if any(row.output.contains(target) for row in candidates)]
    output_spans = tuple(row.output for row in outputs)
    union_output = union_length(list(output_spans))
    total_target = union_length(list(truth.target_utterances))
    return {
        "scope": "full_source" if union_length(list(truth.reviewed)) == truth.source.total_samples
                 else "manually_reviewed_subset_only",
        "sample_rate": truth.source.sample_rate,
        "eligible_utterances": len(truth.target_utterances), "complete_correct_utterances": len(complete),
        "complete_correct_samples": union_length(complete), "eligible_samples": total_target,
        "complete_utterance_recall": len(complete) / len(truth.target_utterances) if truth.target_utterances else None,
        "complete_sample_recall": union_length(complete) / total_target if total_target else None,
        "partial_target_samples": union_length([part for row in outputs for target in truth.target_utterances
                                                 if (part := row.output.intersection(target)) is not None]),
        "contaminated_outputs": sum(any(row.output.intersection(span) for span in contamination) for row in outputs),
        "other_samples": union_length([part for row in outputs for span in truth.other
                                        if (part := row.output.intersection(span)) is not None]),
        "singing_samples": union_length([part for row in outputs for span in truth.singing
                                          if (part := row.output.intersection(span)) is not None]),
        "overlap_samples": union_length([part for row in outputs for span in truth.overlap
                                          if (part := row.output.intersection(span)) is not None]),
        "duplicate_samples": sum(span.length for span in output_spans) - union_output,
        "outside_reviewed_samples": union_output - union_length([
            part for row in outputs for region in truth.reviewed
            if (part := row.output.intersection(region)) is not None]),
        "human_uncertain_output_samples": union_length([
            part for row in outputs for region in truth.human_uncertain
            if (part := row.output.intersection(region)) is not None]),
        "machine_unresolved_eligible_samples": union_length([
            part for row in unresolved for target in truth.target_utterances
            if (part := row.output.intersection(target)) is not None]),
        "geometry_reachable_utterances": len(reachable),
        "missed": [{"start": row.start, "end": row.end} for row in missed],
    }
