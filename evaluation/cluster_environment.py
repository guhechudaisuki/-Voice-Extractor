"""Match the CPU feature-computation environment used by pipeline VAD."""
from __future__ import annotations

import torch


def configure_pipeline_cpu_threads() -> dict:
    # FunASR AutoModel sets ncpu=4 before pipeline speaker analysis. Different
    # CPU reductions can change embeddings even with identical audio/weights.
    # Offline entry points must use that same environment before reading caches.
    torch.set_num_threads(4)
    return {"cpu_threads": torch.get_num_threads(),
            "source": "FunASR_AutoModel_default_ncpu_in_pipeline"}
