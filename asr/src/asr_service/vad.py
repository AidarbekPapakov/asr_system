"""
pyannote's VAD, window grouping, and the per-channel speech/silence metrics.
"""


import numpy as np
import torch
from gigaam.vad_utils import get_pipeline

from asr_service import config

_pipeline = None


def load_vad_pipeline(device: str):
    """pyannote/segmentation-3.0, loaded once per worker and reused."""
    global _pipeline
    _pipeline = get_pipeline(torch.device(device))
    return _pipeline


def vad_regions(wav: np.ndarray) -> list[tuple[float, float]]:
    """
    Speech regions in one channel, as (start, end) seconds. Fed the
    waveform in memory -- no temp file, no re-decode through ffmpeg.
    """
    if _pipeline is None:
        raise RuntimeError("VAD pipeline not loaded; call load_vad_pipeline() first")
    tensor = torch.as_tensor(wav, dtype=torch.float32)[None, :]
    annotation = _pipeline({"waveform": tensor, "sample_rate": config.SAMPLE_RATE})
    return [(seg.start, seg.end) for seg in annotation.get_timeline().support()]


def group_regions(
    regions: list[tuple[float, float]], duration: float
) -> list[tuple[float, float]]:
    """
    Group speech regions into ASR windows.
    """

    merged: list[list[float]] = []
    for start, end in regions:
        if (
            merged
            and start - merged[-1][1] <= config.MERGE_GAP
            and end - merged[-1][0] <= config.MAX_SEG
        ):
            merged[-1][1] = end          # close enough: same window
        elif end - start >= config.MIN_SEG:
            merged.append([start, end])  # new window (blips never start one)

    windows = [
        [max(0.0, s - config.PAD), min(duration, e + config.PAD)] for s, e in merged
    ]

    # Give very short windows more acoustic context, clamped to the
    # neighbours: unclamped growth would overlap windows and transcribe the
    # same speech twice.
    for i, win in enumerate(windows):
        need = config.MIN_WIN - (win[1] - win[0])
        if need > 0:
            low = windows[i - 1][1] if i else 0.0
            high = windows[i + 1][0] if i + 1 < len(windows) else duration
            win[0] = max(low, win[0] - need / 2)
            win[1] = min(high, win[1] + need / 2)

    # Windows must not overlap, or the same audio's transcribation is duplicated and
    # the duplicated words land in two different segments. Padding might create
    # an overlap on its own: `merged` starts a new window when adding the next
    # region would exceed MAX_SEG, and that split can fall in a gap narrower
    # than 2*PAD.
    result: list[tuple[float, float]] = []
    prev_end = 0.0
    for a, b in windows:
        a, b = round(max(a, prev_end), 3), round(b, 3)
        if b - a <= 0:
            continue                      # entirely swallowed by its predecessor
        result.append((a, b))
        prev_end = b
    return result


def channel_metrics(regions: list[tuple[float, float]], duration: float) -> dict:
    """
    Computed from the RAW pyannote regions, before grouping.
    """
    speech = sum(e - s for s, e in regions)
    silence = max(0.0, duration - speech)
    return {
        "total_speech_time": round(speech, 3),
        "total_silence_time": round(silence, 3),
        "speech_ratio": round(speech / duration, 4) if duration > 0 else 0.0,
        "regions": [[round(s, 3), round(e, 3)] for s, e in regions],
    }
