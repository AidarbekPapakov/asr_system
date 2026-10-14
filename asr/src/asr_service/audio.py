"""
Upload bytes -> per-channel float32 @16 kHz + the file's native sample rate.

soundfile decodes wav/flac/ogg/mp3 (via libsndfile) in-process; ffmpeg is the
fallback for anything it refuses. Both paths keep the file's own sample rate
so `result["sample_rate"]` matches what `split_channels` reports today --
resampling to 16 kHz happens once, after decode, for every model to share.
"""

import io
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import soundfile as sf

from asr_service import config


@dataclass
class AudioBundle:
    channels_16k: List[np.ndarray]   # float32 mono, 16 kHz, one per channel
    native_sr: int                   # the file's own rate, for the wire contract
    duration_sec: float
    is_mono: bool


def decode_upload(raw: bytes) -> AudioBundle:
    data, sr = _read_any(raw)
    if data.ndim == 1:
        data = data[:, None]

    n_channels = data.shape[1]
    is_mono = n_channels == 1
    duration_sec = data.shape[0] / sr if sr else 0.0

    channels_16k: List[np.ndarray] = []
    for i in range(n_channels):
        ch = np.ascontiguousarray(data[:, i], dtype=np.float32)
        if sr != config.SAMPLE_RATE and ch.size:
            ch = librosa.resample(ch, orig_sr=sr, target_sr=config.SAMPLE_RATE).astype(
                np.float32
            )
        channels_16k.append(ch)

    return AudioBundle(
        channels_16k=channels_16k,
        native_sr=sr,
        duration_sec=duration_sec,
        is_mono=is_mono,
    )


def _read_any(raw: bytes) -> Tuple[np.ndarray, int]:
    try:
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        return data, sr
    except Exception:
        return _ffmpeg_fallback(raw)


def _ffmpeg_fallback(raw: bytes) -> Tuple[np.ndarray, int]:
    """Re-encode to PCM WAV at the source's own rate (no forced resample --
    that happens uniformly afterwards) for whatever soundfile would not read
    directly (some mp3 encodings, mostly)."""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp_in:
        tmp_in.write(raw)
        in_path = tmp_in.name
    out_path = in_path + ".wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", in_path, "-c:a", "pcm_f32le", out_path],
            check=True,
            capture_output=True,
        )
        data, sr = sf.read(out_path, dtype="float32", always_2d=True)
        return data, sr
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", "replace") if exc.stderr else ""
        raise ValueError(f"unreadable audio (ffmpeg fallback failed): {stderr[-500:]}") from exc
    finally:
        Path(in_path).unlink(missing_ok=True)
        Path(out_path).unlink(missing_ok=True)


def encode_wav_bytes(wav: np.ndarray, sr: int = config.SAMPLE_RATE) -> bytes:
    """Used only for warmup: a synthetic clip encoded the same way a real
    upload would arrive, so setup() exercises the real decode path too."""
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="FLOAT")
    return buf.getvalue()
