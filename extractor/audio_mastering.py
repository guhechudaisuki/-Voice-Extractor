"""Optional high-rate export masters, separate from 16 kHz recognition audio."""
from pathlib import Path
import shutil
import soundfile as sf
from .audio import normalize_audio


def prepare_masters(stem, source, output, requested_rate):
    # Never advertise bandwidth manufactured by upsampling a low-rate stem.
    rate = min(int(requested_rate), sf.info(str(stem)).samplerate)
    if requested_rate <= 16000:
        return stem, None, rate
    directory = Path(output) / '_masters'
    directory.mkdir(parents=True, exist_ok=True)
    master = directory / 'target_vocals.wav'
    if Path(stem).resolve() != master.resolve():
        shutil.copyfile(stem, master)
    raw = None
    if source is not None:
        raw = directory / 'target_raw.wav'
        normalize_audio(source, raw, sample_rate=rate, stereo=False)
    return master, raw, rate
