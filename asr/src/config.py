"""
Every knob for the service, in one place. Every value has an environment
override; nothing here is a secret -- values for LLM_*, HF_TOKEN and friends
live in deploy/.env (gitignored), this module only names them.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

# pyannote.audio 4.x ships OpenTelemetry tracing that POSTs to
# https://otel.pyannote.ai on every model init and inference. Audio does
# not phone home. Set before anything can import pyannote --
# pyannote/audio/telemetry/metrics.py seeds this variable from its own
# bundled config.yaml at import time, and only if it is unset, so being first
# is what makes the opt-out stick. Export PYANNOTE_METRICS_ENABLED=true to
# turn it back on deliberately.
os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")


def env_bool(name: str, default: bool) -> bool:
    """Parse a boolean env var. "0"/"false"/"no"/"off" all read as False,
    unlike a bare getenv() truthiness check."""
    raw: Optional[str] = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw else default


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


# audio / models

SAMPLE_RATE: int = 16000   # every model in this service runs at 16 kHz

ASR_MODEL_NAME: str = os.getenv("ASR_MODEL", "multilingual_large_ctc")
EMO_MODEL_NAME: str = os.getenv("EMO_MODEL", "emo")
FP16_ENCODER: bool = env_bool("FP16", True)

DEVICE_OVERRIDE: Optional[str] = os.getenv("DEVICE", None)

# beam (hotword-boosted CTC prefix beam search) or greedy, the escape hatch.
DECODER: str = os.getenv("DECODER", "beam").strip().lower()
BEAM_WIDTH: int = env_int("BEAM_WIDTH", 25)

# Per-request hotword overrides (see hotwords.py, asr.use_hotwords).
# Hot word's weight is exponentiated, meaning a weight of 10.0 gives a word ~x22_000 increase.
MAX_REQUEST_HOTWORDS: int = env_int("MAX_REQUEST_HOTWORDS", 200)
MAX_HOTWORD_WEIGHT: float = env_float("MAX_HOTWORD_WEIGHT", 10.0)

# transcribe_windows batches windows through the encoder in chunks of this
# size, so a 17-minute recording does not enter a single forward pass as one
# batch.
ASR_WINDOW_BATCH: int = env_int("ASR_WINDOW_BATCH", 16)

# The CNN's weights are fetched into model_weights/ (see its README), not
# provided within the repo. GENDER_WEIGHTS overrides the path
# the container points it at the /models mount. Loaded lazily, on the first
# request that asks for gender, so a wrong path fails then and not at start-up.
GENDER_WEIGHTS: str = os.getenv("GENDER_WEIGHTS") or str(
    Path(__file__).resolve().parent.parent / "model_weights" / "gender_classifier.pth"
)

# VAD manually tuned constants.
MERGE_GAP: float = env_float("MERGE_GAP", 2.0)
MAX_SEG: float = env_float("MAX_SEG", 20.0)
MIN_SEG: float = env_float("MIN_SEG", 0.20)
PAD: float = env_float("PAD", 0.15)
MIN_WIN: float = env_float("MIN_WIN", 3.0)

# assemble turn
TURN_GAP: float = env_float("TURN_GAP", 0.6)

# Diarization -- mono input only; stereo channels already separate speakers.
# pyannote/speaker-diarization-community-1 is HF-gated, so a local snapshot is
# read from model_weights/ (see its README) rather than downloaded. Loaded
# lazily, on the first mono request. Speaker counts are left to pyannote
# unless bounded here.
DIARIZE_MONO: bool = env_bool("DIARIZE_MONO", True)
DIARIZATION_DIR: str = os.getenv("DIARIZATION_DIR") or str(
    Path(__file__).resolve().parent.parent / "model_weights" / "speaker_diarization" / "community-1"
)
DIARIZATION_MIN_SPEAKERS: Optional[int] = env_int("DIARIZATION_MIN_SPEAKERS", 0) or None
DIARIZATION_MAX_SPEAKERS: Optional[int] = env_int("DIARIZATION_MAX_SPEAKERS", 0) or None

# LLM normalisation
RUN_LLM: bool = env_bool("RUN_LLM", True)
# Note that the server does not perform PII masking, which results
# in senstitive data leak.
LLM_ALLOW_EXTERNAL: bool = env_bool("LLM_ALLOW_EXTERNAL", False)
# Hosts that are not loopback but still never leave the machine -- e.g. the
# `litellm` service on the compose network. Comma-separated.
LLM_INTERNAL_HOSTS: Tuple[str, ...] = tuple(
    h.strip().lower() for h in os.getenv("LLM_INTERNAL_HOSTS", "").split(",") if h.strip()
)
LLM_TIMEOUT: float = env_float("LLM_TIMEOUT", 120.0)

# Names only -- values come from deploy/.env, which is gitignored.
# LLM_BASE_URL, LLM_API_KEY, LLM_MODEL

# server
PORT: int = env_int("PORT", 8400)
WORKERS: int = env_int("WORKERS", 1)
# False = no request timeout (a long recording can legitimately take a while);
# matches deploy/litserve/server.py's own choice.
TIMEOUT = False

MB: int = 1024 ** 2
# Sized for recordings of tens of minutes at telephone-quality bitrates plus
# headroom, not for a worst-case upload.
MAX_PAYLOAD_SIZE: int = env_int("MAX_PAYLOAD_MB", 512) * MB
