"""Sample-exact source coordinates and explicitly verified view alignment."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from fractions import Fraction


def require_digest(value: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Expected a lowercase SHA-256 digest")


@dataclass(frozen=True, order=True)
class SampleSpan:
    start: int
    end: int

    def __post_init__(self) -> None:
        if any(type(v) is not int for v in (self.start, self.end)):
            raise TypeError("Source boundaries must be integer samples")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("Expected a nonempty nonnegative interval")

    @property
    def length(self) -> int:
        return self.end - self.start

    def contains(self, other: SampleSpan) -> bool:
        return self.start <= other.start and other.end <= self.end

    def intersection(self, other: SampleSpan) -> SampleSpan | None:
        start, end = max(self.start, other.start), min(self.end, other.end)
        return SampleSpan(start, end) if start < end else None


@dataclass(frozen=True)
class SourceTimeline:
    source_sha256: str
    sample_rate: int
    total_samples: int

    def __post_init__(self) -> None:
        require_digest(self.source_sha256)
        if any(type(v) is not int or v <= 0 for v in (self.sample_rate, self.total_samples)):
            raise ValueError("Source rate and length must be positive integers")

    def validate(self, span: SampleSpan) -> None:
        if span.end > self.total_samples:
            raise ValueError("Interval exceeds decoded source length")

    def key(self, span: SampleSpan) -> str:
        self.validate(span)
        return f"{self.source_sha256}:{self.sample_rate}:{span.start}:{span.end}"

    def from_seconds(self, start: float, end: float) -> SampleSpan:
        if not all(math.isfinite(v) for v in (start, end)) or not 0 <= start < end:
            raise ValueError("Nonfinite source time")
        span = SampleSpan(round(start * self.sample_rate), round(end * self.sample_rate))
        self.validate(span)
        return span


@dataclass(frozen=True)
class ViewAlignment:
    """Positive delay means a source event appears later in the derived view.

    Delay is measured in view samples; it is not inferred from sample rates.
    Outward rounding preserves coverage, rather than trimming fractional edges.
    """
    source: SourceTimeline
    view_rate: int
    view_samples: int
    delay_samples: int
    verified: bool

    def __post_init__(self) -> None:
        if any(type(v) is not int or v <= 0 for v in (self.view_rate, self.view_samples)):
            raise ValueError("Invalid view geometry")
        if type(self.delay_samples) is not int or type(self.verified) is not bool:
            raise ValueError("An explicit integer delay and verification state are required")

    def to_source(self, span: SampleSpan) -> SampleSpan:
        if not self.verified:
            raise ValueError("Cross-channel alignment has not been verified")
        if span.end > self.view_samples:
            raise ValueError("Interval exceeds view length")
        scale = Fraction(self.source.sample_rate, self.view_rate)
        mapped = SampleSpan(math.floor((span.start - self.delay_samples) * scale),
                            math.ceil((span.end - self.delay_samples) * scale))
        self.source.validate(mapped)
        return mapped

    def from_source(self, span: SampleSpan) -> SampleSpan:
        if not self.verified:
            raise ValueError("Cross-channel alignment has not been verified")
        self.source.validate(span)
        scale = Fraction(self.view_rate, self.source.sample_rate)
        mapped = SampleSpan(math.floor(span.start * scale) + self.delay_samples,
                            math.ceil(span.end * scale) + self.delay_samples)
        if mapped.end > self.view_samples:
            raise ValueError("Mapped interval exceeds available view")
        return mapped


def union_length(spans: list[SampleSpan]) -> int:
    total = 0
    end = 0
    for span in sorted(spans):
        total += max(0, span.end - max(end, span.start))
        end = max(end, span.end)
    return total
