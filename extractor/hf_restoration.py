"""Opt-in, coherent high-band detail restoration; never an identity signal."""
import numpy as np


def restore(stem, raw, sample_rate, *, grade, singing=0., overlap=0.):
    from scipy.signal import stft, istft
    from scipy.ndimage import uniform_filter1d
    from .nextgen.physiology import analyze_pair
    a,b=np.asarray(stem,dtype=float),np.asarray(raw,dtype=float)
    audit={'applied':False,'reason':'quality_or_contamination_gate'}
    if grade!='Gold' or singing>=.1 or overlap>=.1 or a.ndim!=1 or b.shape!=a.shape:
        return a,audit
    if sample_rate<12000 or len(a)<sample_rate or not np.isfinite(a).all() or not np.isfinite(b).all():
        return a,{**audit,'reason':'insufficient_aligned_audio'}
    physiology=analyze_pair(a,b,sample_rate)
    if (not physiology.get('available') or physiology.get('reliability',0.)<.8
            or physiology.get('high_band_loss',0.)<.3 or physiology.get('core_coherence',0.)<.95
            or physiology.get('high_band_support',0.)<.8):
        return a,{**audit,'reason':'insufficient_coherence','physiology':physiology}
    lag=int(physiology.get('alignment_lag_samples',0))
    shifted=np.zeros_like(b)
    if lag>0: shifted[lag:]=b[:-lag]
    elif lag<0: shifted[:lag]=b[-lag:]
    else: shifted=b
    nperseg=1024
    f,_,sa=stft(a,fs=sample_rate,nperseg=nperseg,noverlap=768)
    _,_,sb=stft(shifted,fs=sample_rate,nperseg=nperseg,noverlap=768)
    core=(f>=150)&(f<=3500)
    gain=np.sqrt(np.sum(abs(sa[core])**2)/max(1e-20,np.sum(abs(sb[core])**2)))
    sb*=min(gain,4.)
    smooth=lambda v:uniform_filter1d(v,size=9,axis=1,mode='nearest')
    cross=smooth(sa*sb.conj());pa=smooth(abs(sa)**2);pb=smooth(abs(sb)**2)
    coherence=abs(cross)**2/np.maximum(pa*pb,1e-20)
    band=(f>=4000)&(f<=min(10000,sample_rate/2))
    phase_mass=np.sum(abs(cross[band]))
    if phase_mass<=1e-20 or np.sum(cross[band].real)/phase_mass<.95:
        return a,{**audit,'reason':'high_band_phase_conflict'}
    # Positive phase agreement and time-averaged coherence exclude unrelated BGM.
    weight=np.clip((coherence-.95)/.05,0.,1.)*band[:,None]*(cross.real>0)
    missing=np.maximum(0.,abs(sb)-abs(sa))*np.exp(1j*np.angle(sb))
    _,residual=istft(.1*weight*missing,fs=sample_rate,nperseg=nperseg,noverlap=768)
    residual=residual[:len(a)]
    ratio=np.sqrt(np.mean(residual**2))/max(1e-12,np.sqrt(np.mean(a*a)))
    residual*=min(1.,.03/max(ratio,1e-12))
    if not np.any(abs(residual)>1e-10):
        return a,{**audit,'reason':'no_coherent_missing_band'}
    result=a+residual
    if np.max(abs(result))>=.985:
        return a,{**audit,'reason':'clipping_risk'}
    return result,{'applied':True,'reason':'coherent_high_band','band_hz':[4000,min(10000,sample_rate/2)],
                   'maximum_added_rms_ratio':.03,'physiology':physiology}


def restore_export(audio, destination, row, keep, grade):
    """Read only the matching local master slice. Missing masters leave audio intact."""
    from pathlib import Path
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    path=Path((row.get('diagnostics') or {}).get('aligned_raw_audio') or '')
    if not path.is_file(): return {'applied':False,'reason':'raw_master_unavailable'}
    a,rate=sf.read(audio,always_2d=True)
    with sf.SoundFile(path) as stream:
        raw_rate=stream.samplerate
        stream.seek(round((float(row['start'])+keep[0])*raw_rate))
        b=stream.read(round((keep[1]-keep[0])*raw_rate),always_2d=True).mean(axis=1)
    if raw_rate!=rate:
        factor=gcd(raw_rate,rate);b=resample_poly(b,rate//factor,raw_rate//factor)
    if abs(len(b)-len(a))>2: return {'applied':False,'reason':'unaligned_length'}
    b=np.pad(b,(0,max(0,len(a)-len(b))))[:len(a)]
    result,audit=restore(a.mean(axis=1),b,rate,grade=grade,
        singing=float(row.get('singing_score') or 0),overlap=float(row.get('overlap_score') or 0))
    if audit['applied']: sf.write(destination,result,rate,subtype='PCM_16')
    return audit
