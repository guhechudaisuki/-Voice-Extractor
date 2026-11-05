"""Move all FunASR submodels and the runtime device arguments together."""
import torch


class FunASRBackend:
    def __init__(self, **kwargs):
        from funasr import AutoModel
        kwargs.pop('device', None)
        self.engine = AutoModel(device='cpu', **kwargs)
        self.device = 'cpu'

    def to_device(self, device):
        for name, config in (('model', 'kwargs'), ('punc_model', 'punc_kwargs'),
                             ('vad_model', 'vad_kwargs'), ('spk_model', 'spk_kwargs')):
            model = getattr(self.engine, name, None)
            if isinstance(model, torch.nn.Module):
                model.to(device)
            options = getattr(self.engine, config, None)
            if isinstance(options, dict):
                options['device'] = device
        self.device = device

    def generate(self, **kwargs):
        return self.engine.generate(**kwargs)
