"""
The pure pipeline: upload bytes in, the result dict out.

Always produced: per-channel words and speech/silence metrics (VAD runs
regardless -- it is what windows the audio for the encoder) and the merged
turns. Mono input is diarized first, and each detected speaker is processed
as its own channel. Emotion and gender are opt-in per request, and their
models -- like diarization's -- load on first use rather than at start-up.
"""

import logging
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from asr_service import asr
from asr_service import audio as audio_mod
from asr_service import config, diarization, emotion, gender, normalize
from asr_service import turns as turns_mod
from asr_service import vad

logger = logging.getLogger("asr_service.pipeline")


def _turn_wav(wav: np.ndarray, turn: turns_mod.Turn) -> np.ndarray:
    """The turn's own audio, widened to at least config.MIN_WIN seconds and
    clamped to the channel.

    A turn can be a single 0.1 s "ага". Emotion and gender are classified on
    whatever they are handed, and 1600 samples is both too little context to
    mean anything and few enough to degenerate in the encoder; the same
    constant already gives short ASR windows their context in
    vad.group_regions. The reported label is therefore "this turn in its
    immediate surroundings", not "this turn in isolation".
    """
    need = max(0.0, config.MIN_WIN - (turn.end - turn.start))
    start = max(0.0, turn.start - need / 2)
    end = min(len(wav) / config.SAMPLE_RATE, turn.end + need / 2)
    return wav[int(start * config.SAMPLE_RATE) : int(end * config.SAMPLE_RATE)]


def _emotion_per_turn(turns: List[turns_mod.Turn], channels: List[np.ndarray]) -> List[Dict[str, Any]]:
    """One {label, distribution} per turn, index-aligned with `turns`.

    Chunked for the same reason transcribe_windows is: collate pads a batch to
    its longest member, so a 17-minute recording's turns must not enter the encoder
    as one batch.
    """
    emotion.ensure_loaded(asr.get_device())
    out: List[Dict[str, Any]] = []
    for start in range(0, len(turns), config.ASR_WINDOW_BATCH):
        batch = turns[start : start + config.ASR_WINDOW_BATCH]
        wavs = [_turn_wav(channels[t.channel], t) for t in batch]
        for probs4 in emotion.probs_batch(wavs):
            label, distribution = emotion.collapse_emotion(probs4)
            out.append({"label": label, "distribution": distribution})
    return out


def _gender_per_turn(turns: List[turns_mod.Turn], channels: List[np.ndarray]) -> List[Dict[str, Any]]:
    """One {label, distribution} per turn. The CNN scores fixed 5-second
    chunks, so a turn's P(male) is the mean over its chunks."""
    gender.ensure_loaded(asr.get_device(), config.GENDER_WEIGHTS)
    out: List[Dict[str, Any]] = []
    for turn in turns:
        p_male = float(np.mean(gender.p_male_per_chunk(_turn_wav(channels[turn.channel], turn))))
        out.append({
            "label": "male" if p_male >= 0.5 else "female",
            "distribution": {"male": round(p_male, 4), "female": round(1 - p_male, 4)},
        })
    return out


def _diarize(wav: np.ndarray) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Mono -> one channel per speaker. Falls back to the single input
    channel when nobody speaks or the pipeline cannot load or run, reporting
    why in meta.diarization rather than failing the request."""
    try:
        diarization.ensure_loaded(asr.get_device())
        tracks = diarization.speaker_tracks(wav)
    except Exception as exc:                                        # noqa: BLE001
        logger.exception("diarization failed; continuing undiarized")
        return [wav], {"applied": False, "n_speakers": None, "reason": f"failed: {exc}"}
    if not tracks:
        return [wav], {"applied": False, "n_speakers": 0, "reason": "no speech found"}
    return tracks, {"applied": True, "n_speakers": len(tracks), "reason": "mono input"}


def analyze_audio(
    raw: bytes,
    *,
    analyze_emotion: bool = False,
    analyze_gender: bool = False,
    run_llm: Optional[bool] = None,
    speaker_map: Optional[Dict[int, str]] = None,
    hotwords: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    t_start = time.perf_counter()
    timings: Dict[str, float] = {}

    # Applied on every request, defaults are used if no words provided.
    hotwords_meta = asr.use_hotwords(hotwords)

    t0 = time.perf_counter()
    bundle = audio_mod.decode_upload(raw)
    timings["decode"] = round(time.perf_counter() - t0, 3)

    channels = bundle.channels_16k
    if bundle.is_mono and config.DIARIZE_MONO:
        t0 = time.perf_counter()
        channels, diarization_meta = _diarize(channels[0])
        timings["diarization"] = round(time.perf_counter() - t0, 3)
    else:
        reason = "stereo: channels already separate speakers" if not bundle.is_mono else "DIARIZE_MONO is off"
        diarization_meta = {"applied": False, "n_speakers": None, "reason": reason}

    channel_speaker = {i: f"speaker_{i}" for i in range(len(channels))}
    if speaker_map:
        channel_speaker.update({int(k): v for k, v in speaker_map.items()})

    t0 = time.perf_counter()
    channels_out: List[Dict[str, Any]] = []
    channel_words: Dict[int, List[asr.Word]] = {}
    for index, wav in enumerate(channels):
        regions = vad.vad_regions(wav)
        words = asr.transcribe_windows(wav, vad.group_regions(regions, bundle.duration_sec))
        channel_words[index] = words
        channels_out.append({
            "index": index,
            "words": [{"start": w.start, "end": w.end, "text": w.text} for w in words],
            "vad": vad.channel_metrics(regions, bundle.duration_sec),
        })
    timings["vad_asr"] = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    turns = turns_mod.merge_channels(channel_words, channel_speaker)
    timings["turns"] = round(time.perf_counter() - t0, 3)

    run_llm = config.RUN_LLM if run_llm is None else run_llm
    if run_llm:
        t0 = time.perf_counter()
        norm_result = normalize.run_normalization(
            turns, glossary=hotwords_meta["accepted"].keys(),
            allow_external=config.LLM_ALLOW_EXTERNAL, timeout=config.LLM_TIMEOUT,
        )
        turns = norm_result.turns
        timings["llm"] = round(time.perf_counter() - t0, 3)
        llm_meta = {"ran": norm_result.ran, "applied": norm_result.applied, "reason": norm_result.reason}
    else:
        llm_meta = {"ran": False, "applied": False, "reason": "run_llm is not set"}

    turns_out = [asdict(t) for t in turns]
    if analyze_emotion and turns:
        t0 = time.perf_counter()
        for turn_out, data in zip(turns_out, _emotion_per_turn(turns, channels)):
            turn_out["emotion"] = data
        timings["emotion"] = round(time.perf_counter() - t0, 3)
    if analyze_gender and turns:
        t0 = time.perf_counter()
        for turn_out, data in zip(turns_out, _gender_per_turn(turns, channels)):
            turn_out["gender"] = data
        timings["gender"] = round(time.perf_counter() - t0, 3)

    timings["total"] = round(time.perf_counter() - t_start, 3)
    return {
        "turns": turns_out,
        "channels": channels_out,
        "meta": {
            "asr_model": config.ASR_MODEL_NAME,
            "decoder": config.DECODER,
            "device": asr.get_device() or "unknown",
            "duration_sec": round(bundle.duration_sec, 3),
            "n_channels": len(channels),
            "diarization": diarization_meta,
            "llm": llm_meta,
            "hotwords": {
                "applied": config.DECODER == "beam",
                "source": hotwords_meta["source"],
                "count": len(hotwords_meta["accepted"]),
                "rejected": hotwords_meta["rejected"],
            },
            "timings": timings,
            "version": 2,
        },
    }
