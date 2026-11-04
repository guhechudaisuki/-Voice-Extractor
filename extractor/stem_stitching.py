"""Bounded-memory overlap-add of UVR chunks, with atomic WAV publication."""
from pathlib import Path
import os
import tempfile

import numpy as np
import soundfile as sf


def write_stitched_stem(chunks, destination: Path, duration: float, starts):
    """Consume separated chunks once, using the original linear blend weights.

    ``starts`` is the separation schedule in seconds. Only the current chunk
    and the unfinished overlap remain in RAM; no source mixture enters here.
    Any failure preserves an existing destination and removes partial output.
    """
    starts = tuple(starts)
    if (not starts or not np.isfinite(duration) or duration <= 0
            or not np.isfinite(starts).all() or starts[0] != 0
            or any(a >= b for a, b in zip(starts, starts[1:]))
            or starts[-1] >= duration):
        raise ValueError('Invalid vocal chunk schedule')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.stem_', suffix='.wav', dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    stream = None
    sample_rate = channels = total_frames = None
    mixed = weights = None
    cursor = previous_finish = count = 0
    try:
        for index, (start, piece, rate) in enumerate(chunks):
            if index >= len(starts) or start != starts[index]:
                raise ValueError('Vocal chunks do not match their schedule')
            piece = np.asarray(piece)
            if (piece.ndim != 2 or not piece.shape[0] or not piece.shape[1]
                    or not np.isfinite(piece).all() or type(rate) is not int or rate <= 0):
                raise ValueError('Invalid separated vocal chunk')
            if stream is None:
                sample_rate, channels = rate, piece.shape[1]
                total_frames = round(duration * rate)
                if total_frames <= 0:
                    raise ValueError('Empty vocal output')
                stream = sf.SoundFile(str(temporary), 'w', samplerate=rate,
                                      channels=channels, subtype='PCM_16', format='WAV')
                mixed = np.empty((0, channels), dtype=np.float64)
                weights = np.empty((0, 1), dtype=np.float64)
            if rate != sample_rate or piece.shape[1] != channels:
                raise ValueError('UVR5 分块采样率或声道数不一致')
            begin = round(start * rate)
            finish = min(total_frames, begin + len(piece))
            next_begin = round(starts[index + 1] * rate) if index + 1 < len(starts) else total_frames
            if begin != cursor or next_begin <= begin or finish < next_begin:
                raise ValueError('UVR5 分块拼接出现未覆盖的音频区间')
            length = finish - begin
            if length < len(mixed):
                raise ValueError('Vocal chunk ends before the previous overlap')
            if length > len(mixed):
                mixed = np.pad(mixed, ((0, length - len(mixed)), (0, 0)))
                weights = np.pad(weights, ((0, length - len(weights)), (0, 0)))
            blend = np.ones((length, 1), dtype=np.float64)
            if index:
                fade = min(length, max(0, previous_finish - begin))
                if fade:
                    blend[:fade, 0] *= np.arange(1, fade + 1, dtype=np.float64) / (fade + 1)
            if index + 1 < len(starts):
                fade = min(length, max(0, finish - next_begin))
                if fade:
                    blend[-fade:, 0] *= np.arange(fade, 0, -1, dtype=np.float64) / (fade + 1)
            mixed += piece[:length] * blend
            weights += blend
            ready = next_begin - cursor
            if np.any(weights[:ready] <= 0):
                raise ValueError('UVR5 分块拼接出现未覆盖的音频区间')
            stream.write((mixed[:ready] / weights[:ready]).astype(np.float32))
            # Copy the small tail so it does not retain a full chunk allocation.
            mixed, weights = mixed[ready:].copy(), weights[ready:].copy()
            cursor, previous_finish, count = next_begin, finish, index + 1
        if count != len(starts) or cursor != total_frames or len(mixed):
            raise ValueError('Incomplete separated vocal chunks')
        stream.close()
        stream = None
        temporary.replace(destination)
    finally:
        if stream is not None:
            stream.close()
        temporary.unlink(missing_ok=True)
    return destination
