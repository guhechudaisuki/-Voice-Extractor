"""Reference-speaker sentence extraction pipeline."""

from .pipeline import ExtractionPipeline, PipelineOptions
from .types import BatchPipelineResult

__all__ = ["ExtractionPipeline", "PipelineOptions", "BatchPipelineResult"]
