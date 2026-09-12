"""Read-only audio probe; stores only this experiment's timing report in work."""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extractor.subtitles import SubtitleGuide
from extractor.transcription import FunASRTools
from extractor.types import TimeSpan
from extractor.audio import load_mono
from extractor.pipeline import ExtractionPipeline, PipelineOptions
from extractor.subtitle_assistance import split_at_subtitle_pauses, retry_subtitle_vad


def main():
    stem = ROOT / "versions/VoiceExtractor_boundary_candidate/work/20260825_153507_batch_ba8782_001/stems/target_vocals.wav"
    subtitles = next(path for path in ROOT.glob("*/*/*.sc.ass") if "[01]" in path.name)
    report_path = ROOT / "work/subtitle_timing_probe.json"
    if report_path.exists():
        data = json.loads(report_path.read_text(encoding="utf-8"))
        speech = [TimeSpan(*pair) for pair in data["speech"]]
    else:
        speech = FunASRTools("cuda").vad(stem, progress=lambda v, m: print(m, flush=True))
    guide = SubtitleGuide.load(subtitles)
    guide.calibrate(speech)
    pipeline = ExtractionPipeline.__new__(ExtractionPipeline)
    pipeline.options = PipelineOptions()
    detected = retry_subtitle_vad(guide, speech, 1450, stem, ROOT / "work", FunASRTools("cuda"), lambda _v, m: print(m, flush=True))
    refined = split_at_subtitle_pauses(pipeline, guide, detected, load_mono(stem, 16000), lambda _v, _m: None)
    data = {"speech": [[p.start, p.end] for p in speech], "report": guide.report,
            "refined_speech": [[p.start, p.end] for p in refined],
            "groups": [{"cue": cue.index, "subtitle": [cue.start, cue.end],
                        "spans": [[p.start, p.end] for p in parts]}
                       for cue, parts in guide.groups(refined, .85)]}
    report_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(guide.report, ensure_ascii=False), flush=True)
    print("groups", len(data["groups"]), "retry windows", len(guide.retry_windows(speech, 1450)), flush=True)
    print("self-introduction", [g for g in data["groups"] if 337 < g["subtitle"][0] < 343], flush=True)


if __name__ == "__main__":
    main()
