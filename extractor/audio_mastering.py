"""Optional high-rate export masters, separate from 16 kHz recognition audio."""
from pathlib import Path
import os
import shutil
import tempfile
import soundfile as sf


def prepare_masters(stem, source, output, requested_rate):
    # Never advertise bandwidth manufactured by upsampling a low-rate stem.
    rate = min(int(requested_rate), sf.info(str(stem)).samplerate)
    if requested_rate <= 16000:
        return stem, None, rate
    directory = Path(output) / '_masters'
    directory.mkdir(parents=True, exist_ok=True)
    master = directory / 'target_vocals.wav'
    if Path(stem).resolve() != master.resolve():
        # A second name preserves the finalized stem through work cleanup,
        # without copying episode-sized audio on the usual same-volume output.
        descriptor, name = tempfile.mkstemp(prefix='.vocal_master_', suffix='.wav', dir=directory)
        os.close(descriptor)
        temporary = Path(name)
        temporary.unlink()
        try:
            try:
                os.link(stem, temporary)
            except OSError:  # Cross-volume or non-hardlink filesystem.
                shutil.copyfile(stem, temporary)
            temporary.replace(master)
        finally:
            temporary.unlink(missing_ok=True)
    # Original audio is used for analysis only. No redundant raw export master
    # is required now that raw-band restoration is prohibited.
    return master, None, rate
