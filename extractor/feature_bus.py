"""Task-local frame reuse with sample-exact receptive-field ownership.

Kaldi fbank frames are local 25 ms observations. Reuse only frames whose entire
support is inside the queried waveform; no neural pooled identity is inherited.
Different sample phases have independent grids, so arbitrary trims stay exact.
"""
from contextvars import ContextVar
from contextlib import contextmanager
import torch

_BUSES = ContextVar('episode_feature_buses', default=())


class EpisodeFeatureBus:
    HOP = 160
    SUPPORT = 400
    BLOCK_FRAMES = 512

    def __init__(self, waveform):
        if waveform.ndim != 1 or waveform.dtype != torch.float32 or not waveform.is_contiguous():
            raise ValueError('Feature bus requires contiguous float32 mono PCM at 16 kHz')
        self.waveform = waveform
        self._blocks = {}
        self._prefix = {}
        self.computed_frames = 0
        self.reused_frames = 0

    def bounds(self, wave):
        if (wave.dtype != self.waveform.dtype or wave.device != self.waveform.device
                or not wave.is_contiguous() or wave.ndim != 1
                or wave.untyped_storage().data_ptr() != self.waveform.untyped_storage().data_ptr()):
            return None
        start = wave.storage_offset() - self.waveform.storage_offset()
        end = start + wave.numel()
        return (start, end) if 0 <= start < end <= len(self.waveform) else None

    def _pieces(self, start, end, compute, version):
        if not 0 <= start < end <= len(self.waveform) or end - start < self.SUPPORT:
            raise ValueError('Frame query outside source or shorter than its support')
        count = 1 + (end - start - self.SUPPORT) // self.HOP
        phase = start % self.HOP
        first = start // self.HOP
        while count:
            block = first // self.BLOCK_FRAMES
            key = (version, phase, block)
            block_start = phase + block * self.BLOCK_FRAMES * self.HOP
            if key not in self._blocks:
                available = min(self.BLOCK_FRAMES,
                    1 + (len(self.waveform) - block_start - self.SUPPORT) // self.HOP)
                finish = block_start + (available - 1) * self.HOP + self.SUPPORT
                values = compute(self.waveform[block_start:finish])
                if len(values) != available:
                    raise ValueError('Feature backend does not satisfy the 25/10 ms frame contract')
                self._blocks[key] = values
                self.computed_frames += len(values)
            index = first - block * self.BLOCK_FRAMES
            take = min(count, len(self._blocks[key]) - index)
            yield key, index, take
            self.reused_frames += take
            first += take
            count -= take

    def frames(self, start, end, compute, version):
        pieces = [self._blocks[key][index:index + take]
                  for key, index, take in self._pieces(start, end, compute, version)]
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces)

    def pool(self, start, end, compute, version):
        """Multiscale first/second moments from block prefix sums in O(blocks)."""
        total, squared, count = None, None, 0
        for key, index, take in self._pieces(start, end, compute, version):
            if key not in self._prefix:
                values = self._blocks[key].double()
                zero = values.new_zeros((1, values.shape[1]))
                self._prefix[key] = (torch.cat((zero, values.cumsum(0))),
                                     torch.cat((zero, values.square().cumsum(0))))
            sums, squares = self._prefix[key]
            a, b = sums[index + take] - sums[index], squares[index + take] - squares[index]
            total = a if total is None else total + a
            squared = b if squared is None else squared + b
            count += take
        mean = total / count
        return torch.cat((mean, (squared / count - mean.square()).clamp_min(0).sqrt())).float()

    def clear(self):
        self._blocks.clear()
        self._prefix.clear()


@contextmanager
def feature_scope(*waveforms):
    buses = tuple(EpisodeFeatureBus(wave) for wave in waveforms)
    token = _BUSES.set(buses)
    try:
        yield buses
    finally:
        for bus in buses:
            bus.clear()
        _BUSES.reset(token)


def shared_fbank(wave, compute, version):
    for bus in _BUSES.get():
        bounds = bus.bounds(wave)
        if bounds is not None and len(wave) >= bus.SUPPORT:
            return bus.frames(*bounds, compute, version)
    return compute(wave)


def shared_fbank_centered(wave, compute, version):
    """Exact-interval CMN using prefix moments shared by all window scales."""
    for bus in _BUSES.get():
        bounds = bus.bounds(wave)
        if bounds is not None and len(wave) >= bus.SUPPORT:
            frames = bus.frames(*bounds, compute, version)
            moments = bus.pool(*bounds, compute, version)
            return frames - moments[:frames.shape[-1]]
    frames = compute(wave)
    return frames - frames.mean(dim=0, keepdim=True)
