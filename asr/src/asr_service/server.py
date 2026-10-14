"""
LitServe server. `POST /analyze` (multipart/form-data) returns the analysis of
a whole recording: per-channel word timings, speech/silence metrics, and the merged
turns -- plus emotion and gender on each turn when asked for. See contract.py
for the guaranteed shape and pipeline.py for the actual work.

    PYTHONPATH=src .venv/bin/python -m asr_service.server
    curl -F 'audio=@audio.mp3' localhost:8400/analyze
    curl -F 'audio=@audio.mp3' -F analyze_emotion=1 -F analyze_gender=1 localhost:8400/analyze
    curl -F 'audio=@audio.mp3' -F 'hotwords={"example": 6.0}' localhost:8400/analyze
    curl -s localhost:8400/health
"""

import json
import logging
import time
from typing import Any

import litserve as ls
import torch
from fastapi import Request

from asr_service import asr, config, vad
from asr_service import audio as audio_mod
from asr_service import pipeline as pipeline_mod

logger = logging.getLogger("asr_service.server")


def _resolve_device() -> str:
    if config.DEVICE_OVERRIDE:
        return config.DEVICE_OVERRIDE
    return "cuda" if torch.cuda.is_available() else "cpu"


def _form_bool(raw: Any, default: bool | None) -> bool | None:
    """Form fields arrive as strings; "0"/"false"/"no"/"off" must read False."""
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


class AnalyzeAPI(ls.LitAPI):
    def __init__(self) -> None:
        # A request is a single whole recording; the useful batching (over ASR windows
        # inside one recording) already happens within pipeline.py, so the LitServe does not
        # need to batch requests together.
        super().__init__(max_batch_size=1, api_path="/analyze")

    def setup(self, device: str) -> None:
        self._device = _resolve_device()
        logger.info("loading models on %s", self._device)

        # ASR and VAD only: they run on every request. The emotion and gender
        # checkpoints are loaded by pipeline.py the first time a request asks
        # for them, so a deployment that never uses them never pays their
        # resident memory.
        asr.load_asr_model(self._device)
        vad.load_vad_pipeline(self._device)

        self._warm_up()
        logger.info("ready: model=%s decoder=%s device=%s", config.ASR_MODEL_NAME, config.DECODER, self._device)

    def _warm_up(self) -> None:
        """
        Real warmup inference before the port is opened, so readiness is actual
        and not just that weights loaded.
        """
        import numpy as np

        # Stereo, so warmup skips diarization: like emotion and gender, it
        # loads on the first request that needs it.
        silence = np.zeros((int(config.SAMPLE_RATE * 1.5), 2), dtype=np.float32)
        try:
            raw = audio_mod.encode_wav_bytes(silence, config.SAMPLE_RATE)
            pipeline_mod.analyze_audio(raw, run_llm=False)
        except Exception:
            logger.exception("warmup failed (non-fatal; the first real request might pay the cost instead)")

    def decode_request(self, request: Request) -> dict[str, Any]:
        """
        `request` MUST stay a fastapi's Request object: LitServe reads this
        annotation to unwrap the body into form data (server.py compares it
        against fastapi.Request). Anything else and every upload 422s.
        """
        raw = None
        options: dict[str, Any] = {"speaker_map": None, "run_llm": None, "hotwords": None}

        if hasattr(request, "get"):
            f = request.get("audio")
            if f is not None:
                raw = f.file.read() if hasattr(f, "file") else f
            for flag in ("analyze_emotion", "analyze_gender"):
                options[flag] = _form_bool(request.get(flag), False)
            options["run_llm"] = _form_bool(request.get("run_llm"), None)
            sm = request.get("speaker_map")
            if sm:
                options["speaker_map"] = json.loads(sm) if isinstance(sm, str) else sm
            hw = request.get("hotwords")
            if hw:
                hw = json.loads(hw) if isinstance(hw, str) else hw
                if not isinstance(hw, dict):
                    raise ValueError("hotwords must be a JSON object {word: weight}")
                options["hotwords"] = hw

        if raw is None:
            raise ValueError("send an 'audio' file part (multipart/form-data)")

        return {"raw": raw, "options": options, "t0": time.perf_counter()}

    def predict(self, item: dict[str, Any]) -> dict[str, Any]:
        try:
            result = pipeline_mod.analyze_audio(item["raw"], **item["options"])
        except Exception as exc:                                    # noqa: BLE001
            logger.exception("analyze_audio failed (options=%s)", item["options"])
            return {"error": str(exc)}
        result["meta"]["timings"]["request_total"] = round(time.perf_counter() - item["t0"], 3)
        return result

    def encode_response(self, output: dict[str, Any]) -> dict[str, Any]:
        return output


def build_server() -> ls.LitServer:
    api = AnalyzeAPI()
    return ls.LitServer(
        api,
        accelerator="cuda" if _resolve_device().startswith("cuda") else "cpu",
        devices=1,
        workers_per_device=config.WORKERS,
        timeout=config.TIMEOUT,
        healthcheck_path="/health",
        max_payload_size=config.MAX_PAYLOAD_SIZE,
        track_requests=True,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print(
        f"ASR service | {config.ASR_MODEL_NAME} | {_resolve_device()} | "
        f"decoder={config.DECODER} | run_llm={config.RUN_LLM} | port {config.PORT}"
    )
    build_server().run(port=config.PORT, generate_client_file=False)
