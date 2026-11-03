"""Reference-speaker sentence extraction pipeline."""

__all__ = ["ExtractionPipeline", "PipelineOptions", "BatchPipelineResult"]


def __getattr__(name):
    if name in ("ExtractionPipeline", "PipelineOptions"):
        from . import pipeline
        value = getattr(pipeline, name)
    elif name == "BatchPipelineResult":
        from .types import BatchPipelineResult
        value = BatchPipelineResult
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
