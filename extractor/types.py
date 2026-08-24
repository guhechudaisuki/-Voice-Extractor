from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TimeSpan:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class CandidateSentence:
    start: float
    end: float
    whisper_text: str
    language: str = "auto"
    text: str = ""
    speaker_score: float = 0.0
    window_min_score: float = 0.0
    window_p20_score: float = 0.0
    speaker_vote_ratio: float = 0.0
    window_vote_ratio: float = 0.0
    speaker_threshold: float = 0.0
    overlap_score: float = 0.0
    singing_score: float = 0.0
    speech_score: float = 0.0
    accepted: bool = False
    reject_reason: str = ""
    audio_file: str = ""
    text_file: str = ""
    video_file: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineResult:
    job_id: str
    output_dir: Path
    archive_path: Path
    accepted: list[CandidateSentence]
    rejected: list[CandidateSentence]
    manifest_path: Path
    transcript_path: Path


@dataclass
class BatchPipelineResult:
    batch_id: str
    output_dir: Path
    archive_path: Path
    results: list[PipelineResult]
    manifest_path: Path
    transcript_path: Path

    @property
    def accepted(self) -> list[CandidateSentence]:
        return [sentence for result in self.results for sentence in result.accepted]

    @property
    def rejected(self) -> list[CandidateSentence]:
        return [sentence for result in self.results for sentence in result.rejected]
