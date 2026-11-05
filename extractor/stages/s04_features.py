"""The lifetime boundary for shared exact-support acoustic frames."""
from contextlib import AbstractContextManager
import torch
from .base import Stage
from ..feature_bus import EpisodeFeatureBus, feature_scope


class FrameFeatureStage(Stage[tuple[torch.Tensor, ...], AbstractContextManager]):
    name = 's04_features'

    def run(self, value):
        return feature_scope(*value)
