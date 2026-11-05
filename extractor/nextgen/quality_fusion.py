"""Small role-independent physical-quality gate (42 universal coefficients)."""
import numpy as np

CHANNELS = ('primary','secondary','anime_char','anime_va','tertiary','raw')
# Inputs: shortness, noise, unvoiced fraction, spectral loss, competition, same-VA.
# Conservative analytic initialization; no claim of a trained quality network.
COEFFICIENTS = np.array([
    [-.7,-.5,-.3,-.8,-.1,-.2], [-1.,-.7,-.4,-1.,-.1,-.3],
    [-.5,-.4,-.3,-.7,.1,.8], [-.3,-.3,-.2,.3,-.1,-.5],
    [.5,-.2,-.1,.3,.1,-.4], [-.7,-1.,-.4,.9,-.2,-.2]])


def quality_weights(quality=None, channels=CHANNELS):
    q = quality or {}
    def finite(name, default):
        value = q.get(name,default)
        return float(value) if value is not None and np.isfinite(value) else default
    duration = max(.02,finite('duration',2.))
    features = np.array([np.clip(1.-duration/2.,0,1),
        np.clip(1.-finite('snr_db',20.)/30.,0,1),
        1.-np.clip(finite('voiced_fraction',1.),0,1),
        np.clip(finite('high_band_loss',0.),0,1),
        np.log1p(max(0.,finite('competitors',0.)))/4.,
        np.clip(finite('same_va_conflict',0.),0,1)])
    alpha = np.logaddexp(0.,1.+COEFFICIENTS@features)+.1
    selected = {c:float(alpha[CHANNELS.index(c)]) for c in channels if c in CHANNELS}
    total = sum(selected.values())
    return {c:v/total for c,v in selected.items()} if total else {}


def measure_quality(waveform, start, end, raw_waveform=None, competitors=0):
    wave = waveform[round(start*16000):round(end*16000)].detach().float().cpu().numpy()
    count = len(wave)//320
    if not count:
        return {'duration':max(0.,end-start),'competitors':competitors}
    frames = wave[:count*320].reshape(count,320)
    energy = np.mean(frames**2,axis=1)
    low,high = np.quantile(energy,[.1,.9])
    snr = 10*np.log10(max(high,1e-12)/max(low,1e-12))
    quality = {'duration':end-start,'snr_db':float(snr),
        'voiced_fraction':float(np.mean(energy>max(1e-8,high*.01))), 'competitors':competitors}
    if raw_waveform is not None:
        raw = raw_waveform[round(start*16000):round(end*16000)].detach().float().cpu().numpy()
        if len(raw) >= count*320:
            spectrum = np.abs(np.fft.rfft(frames,axis=1))**2
            raw_spectrum = np.abs(np.fft.rfft(raw[:count*320].reshape(count,320),axis=1))**2
            ratio = spectrum[:,80:].sum()/max(raw_spectrum[:,80:].sum(),1e-12)
            quality['high_band_loss'] = float(np.clip(1.-ratio,0,1))
    return quality
