"""The bundled HP2 separation algorithm with waveform input and stem output."""
import numpy as np
import torch


class MemoryUVR:
    def __init__(self, model_path, device):
        from vr import AudioPre
        pre = AudioPre(agg=10, model_path=str(model_path), device=torch.device('cpu'),
                       is_half=False, tta=False)
        self.model, self.mp, self.data = pre.model, pre.mp, pre.data
        self.device = torch.device('cpu')
        self._pool_gpu_dtype = torch.float16

    def separate(self, samples, sample_rate):
        import librosa
        from lib.lib_v5 import spec_utils
        from lib.utils import inference

        waves, spectra = {}, {}
        bands = self.mp.param['band']
        for band in range(len(bands), 0, -1):
            parameters = bands[band]
            source = samples.T if band == len(bands) else waves[band + 1]
            rate = sample_rate if band == len(bands) else bands[band + 1]['sr']
            waves[band] = (source if rate == parameters['sr'] else librosa.resample(
                source, orig_sr=rate, target_sr=parameters['sr'], res_type=parameters['res_type']))
            if waves[band].shape[0] == 1:
                waves[band] = np.repeat(waves[band], 2, axis=0)
            spectra[band] = spec_utils.wave_to_spectrogram_mt(
                waves[band], parameters['hl'], parameters['n_fft'], self.mp.param['mid_side'],
                self.mp.param['mid_side_b2'], self.mp.param['reverse'])
            if band == len(bands):
                high_size = ((parameters['n_fft'] // 2 - parameters['crop_stop'])
                             + self.mp.param['pre_filter_stop'] - self.mp.param['pre_filter_start'])
                high = spectra[band][:, parameters['n_fft'] // 2 - high_size:parameters['n_fft'] // 2, :]
        spectrum = spec_utils.combine_spectrograms(spectra, self.mp)
        with torch.inference_mode():
            prediction, magnitude, phase = inference(spectrum, self.device, self.model,
                {'value': self.data['agg'] / 100., 'split_bin': bands[1]['crop_stop']}, self.data)
        if self.data['postprocess']:
            prediction = spec_utils.mask_silence(prediction, np.clip(magnitude - prediction, 0, np.inf))
        vocals = spectrum - prediction * phase
        # This is HP2's own separated high-band reconstruction, not post-export
        # mixing of raw audio (hf_restoration remains disabled).
        high_vocals = spec_utils.mirroring(self.data['high_end_process'], vocals, high, self.mp)
        result = spec_utils.cmb_spectrogram_to_wave(vocals, self.mp, high_size, high_vocals)
        result = np.asarray(result)
        if not np.isfinite(result).all():
            raise RuntimeError('UVR5 output contains NaN/Inf')
        # Preserve the old intermediate PCM16 quantization without a file.
        return (result * 32768).astype(np.int16).astype(np.float32) / 32768., self.mp.param['sr']
