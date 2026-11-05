"""Local CTranslate2 Whisper batches with explicit, independent audio fences."""
from collections import defaultdict
import gc
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .config import CACHE_ROOT
from .model_pool import MODEL_POOL, is_memory_error, resource_key


def converted_model(snapshot):
    snapshot = Path(snapshot)
    if (snapshot / 'model.bin').is_file():
        return snapshot
    key = resource_key('ct2-whisper-float16-v1', snapshot)
    digest = hashlib.sha256(repr(key).encode()).hexdigest()
    parent = CACHE_ROOT / 'whisper_ct2'
    output = parent / digest
    if (output / 'conversion.json').is_file() and (output / 'model.bin').is_file():
        return output
    from ctranslate2.converters import TransformersConverter
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='conversion-', dir=parent))
    try:
        TransformersConverter(str(snapshot), copy_files=['tokenizer.json', 'preprocessor_config.json'],
                              load_as_float16=True).convert(str(staging), quantization='float16', force=True)
        model = staging / 'model.bin'
        if not model.is_file() or model.stat().st_size == 0:
            raise RuntimeError('Whisper conversion did not produce weights')
        checksum = hashlib.sha256()
        with model.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                checksum.update(block)
        (staging / 'conversion.json').write_text(json.dumps(
            {'source': digest, 'model_sha256': checksum.hexdigest()}), encoding='utf-8')
        if output.exists():
            # A different worker may have completed the identical conversion.
            if not (output / 'conversion.json').is_file():
                raise RuntimeError('Incomplete Whisper conversion cache: ' + str(output))
        else:
            staging.replace(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output


class FasterWhisperBackend:
    def __init__(self, snapshot, device):
        from faster_whisper import WhisperModel
        self.path = converted_model(snapshot)
        self.device = device
        self.batch_size = 16
        try:
            self.model = WhisperModel(str(self.path), device=device,
                                      compute_type='float16' if device == 'cuda' else 'int8',
                                      local_files_only=True)
        except RuntimeError as error:
            if device != 'cuda' or not is_memory_error(error):
                raise
            self.model = WhisperModel(str(self.path), device='cpu', compute_type='int8',
                                      local_files_only=True)
            self.device = 'cpu'

    def to_device(self, device):
        # CT2 can retain weights on the CPU when unloading GPU buffers. CPU
        # inference needs a CPU runtime, instantiated only on a real OOM.
        if device == 'cpu' and self.device == 'cuda':
            unload = getattr(self.model.model, 'unload_model', None)
            if unload is None:
                return False  # Whisper CT2 builds may not expose offloading.
            unload(to_cpu=True)
            self.device = 'parked'
        elif device == 'cuda' and self.device == 'parked':
            self.model.model.load_model()
            self.device = 'cuda'

    def cpu_fallback(self):
        from faster_whisper import WhisperModel
        self.model = None
        gc.collect()
        self.model = WhisperModel(str(self.path), device='cpu', compute_type='int8',
                                  local_files_only=True)
        self.device = 'cpu'

    def recognize_many(self, waves, language=None, **kwargs):
        """Bucket by language/length; no chunk contains two caller intervals."""
        output = [None] * len(waves)
        groups = defaultdict(list)
        for index, wave in enumerate(waves):
            wave = np.asarray(wave, dtype=np.float32)
            if not len(wave):
                output[index] = {'text': '', 'chunks': [], 'language': language or 'auto'}
                continue
            detected = language
            if detected is None:
                try:
                    detected = self.model.detect_language(audio=wave)[0]
                except RuntimeError as error:
                    if self.device != 'cuda' or not is_memory_error(error):
                        raise
                    detected = None
                if detected is None:
                    self.cpu_fallback()
                    detected = self.model.detect_language(audio=wave)[0]
            groups[detected].append((index, wave))
        for detected, group in groups.items():
            group.sort(key=lambda item: len(item[1]), reverse=True)
            cursor = 0
            while cursor < len(group):
                batch = group[cursor:cursor + self.batch_size]
                try:
                    results = self._batch([wave for _, wave in batch], detected, **kwargs)
                except RuntimeError as error:
                    if not is_memory_error(error) or self.device != 'cuda':
                        raise
                    results = None
                if results is None:
                    if self.batch_size > 1:
                        self.batch_size = max(1, self.batch_size // 2)
                    else:
                        self.cpu_fallback()
                    continue
                for (index, _), result in zip(batch, results):
                    output[index] = result
                cursor += len(batch)
        return output

    def _batch(self, waves, language, **kwargs):
        from faster_whisper import BatchedInferencePipeline
        packed, clips, owners = [], [], []
        offset = 0
        results = [dict(text='', chunks=[], language=language) for _ in waves]
        # 30 seconds is the model's acoustic context. Metadata retains the
        # original window offset even when a caller supplied a longer interval.
        for owner, wave in enumerate(waves):
            for start in range(0, len(wave), 480000):
                part = wave[start:start + 480000]
                packed.append(part)
                clips.append({'start': offset, 'end': offset + len(part)})
                owners.append((owner, offset / 16000., (offset + len(part)) / 16000., start / 16000.))
                offset += len(part)
                # CT2 exposes a floored 10 ms seek. Align starts to that grid
                # so a fractional boundary cannot map to the previous clip.
                pad = (-offset) % 160
                if pad:
                    packed.append(np.zeros(pad, dtype=np.float32))
                    offset += pad
        by_seek = {int(begin * self.model.frames_per_second): row
                   for row in owners for _, begin, _, _ in [row]}
        segments, _ = BatchedInferencePipeline(self.model).transcribe(
            np.concatenate(packed), language=language, task='transcribe',
            word_timestamps=True, vad_filter=False, clip_timestamps=clips,
            batch_size=self.batch_size, condition_on_previous_text=False,
            **kwargs)
        for segment in segments:
            # Match the chunk seek, not a potentially hallucinated word onset.
            owner, begin, end, local = by_seek[segment.seek]
            result = results[owner]
            result['text'] += segment.text
            for word in segment.words or ():
                valid = begin <= word.start < word.end <= end
                start = max(begin, min(end, word.start))
                finish = max(begin, min(end, word.end))
                result['chunks'].append({'text': word.word,
                    'timestamp': (start - begin + local, finish - begin + local)
                                 if finish > start else None,
                    'timing_estimated': not valid})
            if not segment.words and segment.text:
                # Segment timing is not a word alignment. Keep text, explicitly
                # estimated, so partial training exports cannot use it.
                result['chunks'].append({'text': segment.text, 'timestamp': None})
        return results

    def __call__(self, audio, *, generate_kwargs=None, **unused):
        options = dict(generate_kwargs or {})
        options.pop('task', None)
        if 'logprob_threshold' in options:
            options['log_prob_threshold'] = options.pop('logprob_threshold')
        return self.recognize_many([audio['raw']], **options)[0]


def whisper_backend(snapshot, device):
    key = (resource_key('faster-whisper', snapshot), device)
    return MODEL_POOL.get(key, lambda: FasterWhisperBackend(snapshot, device), device=device)
