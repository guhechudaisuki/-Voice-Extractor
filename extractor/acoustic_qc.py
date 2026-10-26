"""工业级声学质检分级（Stage 27-30）。

为每条导出音频自动计算 SNR、削波、有效发声时长等硬核指标并评定质量等级。
"""
from __future__ import annotations

import numpy as np


def evaluate_clip_acoustics(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    speaker_score: float = 0.0,
    window_min: float = 0.0,
) -> dict:
    """计算 4 项声学指标并评定质量等级。"""
    wave = np.asarray(waveform, dtype=np.float32).flatten()
    duration = len(wave) / max(1, sample_rate)

    # 1) SNR：20ms 帧 RMS 能量 → P85(语音) vs P10(底噪) 分贝差
    frame_size = int(0.02 * sample_rate)
    if len(wave) < frame_size * 2:
        return _default_qc(duration)
    n_frames = len(wave) // frame_size
    frames = wave[:n_frames * frame_size].reshape(n_frames, frame_size)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-10))
    p85 = float(np.percentile(rms_db, 85))
    p10 = float(np.percentile(rms_db, 10))
    snr_db = round(p85 - p10, 2)

    # 2) 削波检测
    clipping_ratio = round(float(np.mean(np.abs(wave) >= 0.985)), 6)

    # 3) 有效发声时长（帧 RMS > 静音阈值）
    speech_threshold = max(1e-8, float(np.percentile(rms, 15)))
    voiced_frames = int(np.sum(rms > speech_threshold * 2.0))
    effective_speech_seconds = round(voiced_frames * frame_size / sample_rate, 3)

    # 4) 质量等级
    quality_grade = "Review"
    if (speaker_score >= 0.70 and window_min >= 0.60
            and snr_db >= 18.0 and clipping_ratio < 0.001
            and 1.5 <= duration <= 10.0):
        quality_grade = "Gold"
    elif speaker_score >= 0.55 and snr_db >= 12.0 and clipping_ratio < 0.005:
        quality_grade = "Silver"

    return {
        "snr_db": snr_db,
        "clipping_ratio": clipping_ratio,
        "effective_speech_seconds": effective_speech_seconds,
        "quality_grade": quality_grade,
        "duration_seconds": round(duration, 3),
        "peak_dbfs": round(float(20 * np.log10(max(np.max(np.abs(wave)), 1e-10))), 2),
    }


def _default_qc(duration: float) -> dict:
    return {
        "snr_db": 0.0, "clipping_ratio": 0.0,
        "effective_speech_seconds": 0.0,
        "quality_grade": "Review", "duration_seconds": round(duration, 3),
        "peak_dbfs": 0.0,
    }
