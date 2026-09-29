---
name: verify
description: Build, launch and drive the audio_analytics stack (ASR + diarization + llamacpp/LiteLLM/Langfuse) to verify a change end to end at POST :8400/analyze.
---

# Verify audio_analytics

## Prerequisites (one-time)
- `asr/deploy/.env` has `HF_TOKEN=` (pyannote/segmentation-3.0 is gated; VAD
  can't load without it and setup dies with "no HF_TOKEN was provided").
- `stack/.env` has `LITELLM_MASTER_KEY=` (else `meta.llm.reason` = missing LLM_API_KEY).
- The old `llm_local_inference_llama` stack is down (same host ports).

## Build / launch
```bash
cd stack
docker compose build asr_service          # slow link: may need retries; steps cache
docker compose up -d
until [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8400/health)" = 200 ]; do sleep 10; done
```
First boot downloads the GigaAM checkpoint (2.34 GB). If the log sits at a
fixed % with the file not growing, it has stalled: gigaam's `urlopen` has no
timeout and treats an existing partial file as complete. Stop the service,
`curl -C -` https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/multilingual_large_ctc.ckpt
until it is 2341654513 bytes, and copy it into volume
`audio_analytics_gigaam-cache` (e.g. via an alpine container).
The "unhealthy" status during that download is just start_period being too short.

## Flows worth driving
- Stereo: `curl -F audio=@../asr/samples/9b2584c9e8334b13aae002353676b171.mp3 -F run_llm=0 localhost:8400/analyze`
  → `meta.diarization.applied=false`, speakers `speaker_0/1`.
- Mono + LLM: `.../57cee54b9cd54132a25884f595649e90.mp3 -F run_llm=1`
  → `diarization.applied=true, n_speakers=2`, `llm.applied=true`.
- Trace: `SELECT name,type,provided_model_name FROM default.events_full` on
  127.0.0.1:8124 (user/pass clickhouse). Arrives ~1 min after the request.
- Edge: mono silence (`ffmpeg -f lavfi -i anullsrc=r=8000:cl=mono -t 5`, run
  inside the container, which has ffmpeg) → `n_speakers=0, "no speech found"`.
