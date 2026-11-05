"""Phonetic coverage, five-dimensional acoustics and additive subset manifests."""
from functools import lru_cache
from copy import copy
import json
import re
import numpy as np


@lru_cache(maxsize=4096)
def phonemes(text, language):
    """Local G2P only. Missing dictionaries stay explicitly unavailable."""
    try:
        if language.upper() == 'JA':
            import pyopenjtalk
            tokens = pyopenjtalk.g2p(text).split()
            method = 'pyopenjtalk'
        elif language.upper() == 'ZH':
            from pypinyin import lazy_pinyin, Style
            initials = lazy_pinyin(text,style=Style.INITIALS,strict=False,errors='ignore')
            finals = lazy_pinyin(text,style=Style.FINALS_TONE3,strict=False,errors='ignore')
            tokens = [v for pair in zip(initials,finals) for v in pair if v]
            method = 'pypinyin_initial_final_tone'
        elif language.upper() == 'EN':
            # Read the installed NLTK dictionary without invoking G2p's
            # downloader. Unknown words are reported rather than guessed.
            from nltk.corpus import cmudict
            dictionary = _english_dictionary()
            words = re.findall(r"[a-z]+(?:'[a-z]+)?",text.lower())
            if any(word not in dictionary for word in words):
                return (), 'unavailable_english_oov'
            tokens = [p for word in words for p in dictionary[word][0]]
            method = 'cmudict'
        else:
            return (), 'unavailable_language'
        return tuple(tokens), method
    except (ImportError,LookupError,OSError,RuntimeError,ValueError):
        return (), 'unavailable_local_g2p'


@lru_cache(maxsize=1)
def _english_dictionary():
    from nltk.corpus import cmudict
    return cmudict.dict()


def acoustic_dimensions(audio, text, language):
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    from extractor.acoustic_qc import estimate_voiced_f0_stats
    from launcher.golden_prompts import longest_internal_pause
    samples,rate = sf.read(str(audio),dtype='float32',always_2d=True)
    wave = samples.mean(axis=1)
    if rate != 16000:
        divisor = gcd(rate,16000)
        wave = resample_poly(wave,16000//divisor,rate//divisor)
    pitch = estimate_voiced_f0_stats(wave,16000)
    tokens,method = phonemes(text,language)
    # Vowel nuclei approximate syllables; preserve the explicit measurement basis.
    nuclei = sum(bool(re.search(r'[aeiouAEIOU]',t)) for t in tokens)
    count = len(wave)//400
    slope = None
    if count:
        frames = wave[:count*400].reshape(count,400)*np.hanning(400)
        spectrum = np.mean(np.abs(np.fft.rfft(frames,axis=1))**2,axis=0)
        slope = float(10*np.log10(max(spectrum[75:].sum(),1e-12)/max(spectrum[2:75].sum(),1e-12)))
    f0 = pitch.get('f0_median_hz')
    lo,hi = pitch.get('f0_p15_hz'),pitch.get('f0_p85_hz')
    return {'f0_median_hz':f0,'f0_range_hz':hi-lo if lo is not None and hi is not None else None,
        'syllables_per_second':nuclei/max(len(wave)/16000,1e-6) if tokens else None,
        'rms_db':float(10*np.log10(max(float(np.mean(wave**2)),1e-12))) if len(wave) else None,
        'spectral_tilt_db':slope,'syllable_basis':'g2p_vowel_nuclei',
        'phonemes':list(tokens),'g2p_method':method,
        'longest_internal_pause_seconds':longest_internal_pause(wave),
        'pause_measurement':'20ms_rms_below_relative_minus20db_floor_minus100dbfs'}


def select_subsets(records, core_size=300, extended_size=1000):
    n = len(records)
    from launcher.golden_prompts import select_golden
    if not n:
        return {'core':[],'extended':[],'redundant':[],**select_golden([],np.empty((0,5)))}
    names = ('f0_median_hz','f0_range_hz','syllables_per_second','rms_db','spectral_tilt_db')
    features = np.array([[r.get('curation_features',{}).get(name) for name in names] for r in records],dtype=float)
    for column in range(5):
        values = features[:,column]
        valid = np.isfinite(values)
        center = float(np.median(values[valid])) if valid.any() else 0.
        values[~valid] = center
        features[:,column] = np.clip((values-center)/max(float(np.std(values)),1.),-4,4)
    covers = []
    for r in records:
        tokens = r.get('curation_features',{}).get('phonemes',[])
        prefix = r.get('language','')+':'
        covers.append({prefix+t for t in tokens}|{prefix+a+'>'+b for a,b in zip(tokens,tokens[1:])})
    counts = {}
    for cover in covers:
        for token in cover:
            counts[token] = counts.get(token,0)+1
    token_weight = {t:1./np.sqrt(c) for t,c in counts.items()}
    quality = np.array([{'Gold':1.,'Silver':.8,'Review':.35}.get(r.get('quality_grade'),.2) for r in records])
    # Landmark facility location bounds memory for multi-season corpora.
    landmarks = features[np.linspace(0,n-1,min(n,256)).astype(int)]
    similarities = np.exp(-np.sum((features[:,None]-landmarks[None])**2,axis=-1)/5.)
    covered,represented = set(),np.zeros(len(landmarks))
    order,remaining = [],set(range(n))
    import heapq
    def gain(i):
        phonetic = sum(token_weight[t] for t in covers[i]-covered)
        facility = float(np.maximum(0,similarities[i]-represented).mean())
        return phonetic+3*facility+.05*quality[i]
    heap = [(-gain(i),i) for i in remaining];heapq.heapify(heap)
    while remaining and len(order)<min(n,extended_size):
        _,i = heapq.heappop(heap)
        value = gain(i)
        if heap and value < -heap[0][0]-1e-10:
            heapq.heappush(heap,(-value,i));continue
        order.append(i);remaining.remove(i);covered.update(covers[i])
        represented=np.maximum(represented,similarities[i])
    return {'core':order[:core_size],'extended':order,'redundant':sorted(remaining),
            **select_golden(records,features),
            'method':'phoneme_diphone_coverage_and_5d_landmark_facility_location',
            'acoustic_dimensions':list(names),'phonetic_units_covered':len(covered)}


def write_curation(archive, records, speaker):
    selection = select_subsets(records)
    for name,key in (('Core-300','core'),('Extended-1000','extended'),('Redundant','redundant')):
        rows = [records[i] for i in selection[key]]
        archive.writestr(f'curation/{name}.list',''.join(
            f"{r['path']}|{speaker}|{r['language']}|{r['text']}\n" for r in rows if r.get('valid_text')))
    prompts=[]
    for category,indexes in selection['golden_categories'].items():
        for position,i in enumerate(indexes,1):
            r=records[i];name=f'golden_prompts/{category}/{position:02d}.wav'
            # Python 3.10 on Windows keeps the pre-normalized backslashes in
            # ZipInfo.orig_filename for write(path), while its header uses '/'.
            # Read our own just-written member through an isolated normalized
            # descriptor; leave the archive directory and file bytes untouched.
            member = copy(archive.getinfo(r['path']))
            member.orig_filename = member.filename
            archive.writestr(name,archive.read(member))
            archive.writestr(name[:-4]+'.txt',r['text'])
            prompts.append({'path':name,'source_path':r['path'],'text':r['text'],
                'language':r['language'],'duration_seconds':r['duration_seconds'],
                'category':category,'category_basis':selection['category_basis'],
                'source_audio':r.get('source_audio'),'source_span':r['source_span'],
                'confidence_scope':r['confidence_scope'],'speaker_confidence':r['speaker_confidence'],
                'word_timestamps':r['word_timestamps'],'gate_audit':selection['golden_audit'][i]})
    prompt_manifest = json.dumps({'prompts':prompts,'scope':'recommended_reference_audio',
        'gates':selection['golden_gates'],'category_basis':selection['category_basis'],
        'empty_categories':[k for k,v in selection['golden_categories'].items() if not v],
        'reason_if_empty':'No reviewed reference met every strict quality, timing and confidence gate' if not prompts else ''},
        ensure_ascii=False,allow_nan=False)
    archive.writestr('golden_prompts/manifest.json',prompt_manifest)
    archive.writestr('prompts_manifest.json',prompt_manifest)
    report = {**selection,'subsets':{k:[records[i]['path'] for i in selection[k]]
                                    for k in ('core','extended','redundant')},'golden_prompts':prompts}
    archive.writestr('curation/report.json',json.dumps(report,ensure_ascii=False,allow_nan=False))
    return report
