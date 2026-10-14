"""
pyannote/speaker-diarization-community-1, for mono input only.

Stereo input already carries one speaker per channel, so it never reaches
this module. A mono channel is split into one track per detected speaker --
each the full length of the input, silent outside that speaker's segments --
so everything downstream (VAD, ASR, turns, emotion, gender) runs unchanged,
treating each track as a channel.
"""

import logging
from pathlib import Path
from typing import Any, List

import numpy as np
import torch

from asr_service import config

logger = logging.getLogger("asr_service.diarization")

_pipeline = None


def load_diarization_pipeline(device: str, path: str = config.DIARIZATION_DIR):
    """The pipeline is HF-gated, so it is read from a local snapshot (see
    model_weights/README.md) rather than downloaded at run time."""
    global _pipeline
    from pyannote.audio import Pipeline

    if not Path(path).is_dir():
        raise FileNotFoundError(
            f"diarization pipeline not found at {path!r}; see model_weights/README.md "
            "or point DIARIZATION_DIR at a local snapshot"
        )
    try:
        pipeline = Pipeline.from_pretrained(path)
    except Exception:
        # torch>=2.6 defaults torch.load to weights_only=True, which rejects
        # the TorchVersion object the segmentation checkpoint pickles.
        from torch.torch_version import TorchVersion
        with torch.serialization.safe_globals([TorchVersion]):
            pipeline = Pipeline.from_pretrained(path)
    _pipeline = pipeline.to(torch.device(device))
    return _pipeline


def ensure_loaded(device: str):
    """Load on first use -- only mono input needs it, so setup() does not pay
    for it up front."""
    return _pipeline if _pipeline is not None else load_diarization_pipeline(device)


def speaker_tracks(wav: np.ndarray) -> List[np.ndarray]:
    """One 16 kHz mono channel -> one track per detected speaker. Returns []
    if nobody speaks."""
    if _pipeline is None:
        raise RuntimeError("diarization pipeline not loaded; call ensure_loaded() first")
    tensor = torch.as_tensor(wav, dtype=torch.float32)[None, :]
    output = _pipeline(
        {"waveform": tensor, "sample_rate": config.SAMPLE_RATE},
        min_speakers=config.DIARIZATION_MIN_SPEAKERS,
        max_speakers=config.DIARIZATION_MAX_SPEAKERS,
    )
    # At most one speaker per instant, so tracks never overlap and the same
    # speech is never transcribed twice.
    tracks = tracks_from_annotation(output.exclusive_speaker_diarization, wav)
    logger.info("diarization found %d speaker(s)", len(tracks))
    return tracks


def tracks_from_annotation(annotation: Any, wav: np.ndarray) -> List[np.ndarray]:
    """Cut `wav` into per-speaker tracks, ordered by each speaker's first
    onset, so the first to speak lands on track 0."""
    segments_by_speaker: dict = {}
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        segments_by_speaker.setdefault(speaker, []).append(segment)

    order = sorted(segments_by_speaker, key=lambda spk: min(s.start for s in segments_by_speaker[spk]))

    tracks: List[np.ndarray] = []
    for speaker in order:
        track = np.zeros(len(wav), dtype=np.float32)
        for segment in segments_by_speaker[speaker]:
            start = max(0, int(segment.start * config.SAMPLE_RATE))
            end = min(len(wav), int(segment.end * config.SAMPLE_RATE))
            track[start:end] = wav[start:end]
        tracks.append(track)
    return tracks
