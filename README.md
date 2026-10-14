# audio_analytics

A self-hosted stack that turns audio recordings into speaker-attributed,
LLM-normalised transcripts. Everything runs locally: no audio or text leaves
the machine.

```
audio ─► asr_service :8400
           decode → diarize (mono only) → VAD → GigaAM ASR → turns → [emotion, gender]
                                                                         │ run_llm=1
                                                                         ▼
                                                               litellm :4000 ─► llamacpp :8081 (GPU)
                                                                         │
                                                                         └─ traces ─► langfuse :3000
```

| Component | What it does |
|---|---|
| **asr_service** (`asr/`) | LitServe HTTP service: `POST /analyze` with an audio file returns turns, words, VAD metrics and optional emotion and gender. Details are in [`asr/README.md`](asr/README.md). |
| **Diarization** | `pyannote/speaker-diarization-community-1` splits a **mono** recording into one channel per speaker. Multi-channel recordings are already one speaker per channel and skip it. |
| **llamacpp** | llama.cpp server for any GGUF model (default: Qwen3-8B Q4_K_M), on the GPU. |
| **LiteLLM** | OpenAI-compatible gateway in front of llamacpp. It exposes one model name, `local-llm`. |
| **Langfuse** | Tracing: every LLM call made through LiteLLM shows up here. |

The ASR service is **domain-neutral**. Speakers are `speaker_0`, `speaker_1`, …,
the LLM only formats text (punctuation, casing, numbers, split/merged-word
repair), and the hotword list starts empty. Domain tuning goes on top of it.

## Requirements

- Docker with Compose v2.20+ (for `include`). The GPU needs `nvidia-container-toolkit`.
- An NVIDIA GPU for the LLM (it was sized for 8 GB). ASR runs on CPU by default.
- About 20 GB of disk (the ASR image is ~13 GB, plus the model checkpoints).
- A Hugging Face token that has accepted the terms of
  [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0),
  a gated model the ASR service uses for voice activity detection.
- A GGUF model in `~/.models/`, as named in `stack/llamacpp/model.env`.

## Setup

```bash
# 1. Upstream Langfuse and LiteLLM checkouts (gitignored, pinned to the tested commits)
git clone https://github.com/langfuse/langfuse.git stack/langfuse
git -C stack/langfuse checkout 6e9f8ebdff15456160f52aef02b8583dd8a48743
git clone https://github.com/BerriAI/litellm.git stack/litellm
git -C stack/litellm checkout db37977307853b9d2c2f740ca2cbcaefef06982c

# 2. HF token for the ASR service
echo 'HF_TOKEN=hf_...' > asr/deploy/.env

# 3. Diarization snapshot (gated on HF; accept the terms first, see asr/model_weights/README.md)
HF_TOKEN=hf_... asr/scripts/download_weights.sh

# 4. Stack config and secrets (LiteLLM master key, Langfuse keys and admin password)
cd stack
cp .env.example .env
sh secrets/gen.sh >> .env

# 5. Build and start everything
docker compose up -d --build
docker compose ps
```

The first start of `asr_service` downloads the GigaAM ASR checkpoint (2.3 GB),
so it reports `unhealthy` until that download finishes. See
[Troubleshooting](#troubleshooting) if the download stalls.

This stack uses the same host ports as `~/projects/llm_local_inference_llama`,
so stop that one first if it's running.

## Usage

```bash
# health
curl -s localhost:8400/health

# transcribe a recording (a mono file gets diarized automatically)
curl -F audio=@recording.mp3 localhost:8400/analyze

# with LLM normalisation, emotion, and your own speaker names
curl -F audio=@recording.mp3 -F run_llm=1 -F analyze_emotion=1 \
     -F 'speaker_map={"0":"alice","1":"bob"}' localhost:8400/analyze
```

Example turns from a mono recording with `run_llm=1`:

```
speaker_0  3.44– 4.88  Саат байке салам алейкум.
speaker_1  5.63– 7.31  Алейкум ассалам жакшы рахмат.
```

`meta.diarization` says whether diarization ran and how many speakers it
found. `meta.llm` says whether the LLM output was applied, and if not, why.
The full request/response contract is in [`asr/README.md`](asr/README.md).

| UI / endpoint | Where |
|---|---|
| ASR API + Swagger | http://localhost:8400/docs |
| Langfuse | http://localhost:3000 (login: the email in `stack/overrides/langfuse.override.yml`, `USER_PASSWORD` in `stack/.env`) |
| LiteLLM | http://localhost:4000 (Bearer `LITELLM_MASTER_KEY`) |

Measured on CPU (ASR) + RTX 3070 (LLM) for a 23.5 s mono recording:
diarization ~10–14 s, VAD+ASR ~3 s, LLM ~24 s.

## Configuration

| What | Where |
|---|---|
| Which LLM to serve | `stack/llamacpp/model.env` |
| GPU attachments | `COMPOSE_FILE` in `stack/.env` |
| ASR on GPU | `ASR_DEVICE=cuda` in `stack/.env`, and add `:overrides/asr-gpu.override.yml` to `COMPOSE_FILE`. The LLM will probably need fewer GPU layers or less context to make room. |
| ASR knobs (decoder, VAD, diarization speaker bounds, …) | `asr/deploy/.env`. Every option is listed in `asr/deploy/.env.example` and `asr/src/config.py`. |
| Default hotwords | `asr/src/hotwords.py` (or per request with `hotwords`) |

## Troubleshooting

- **`asr_service` exits with "no HF_TOKEN was provided".** Check that
  `asr/deploy/.env` has a non-empty `HF_TOKEN`, then run
  `docker compose up -d --force-recreate asr_service`.
- **The GigaAM download is stuck at a fixed %.** Its downloader has no
  timeout, and it treats a partial file as complete on restart. Stop the
  service, fetch
  `https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/multilingual_large_ctc.ckpt`
  with `curl -C -` until it is 2,341,654,513 bytes, and copy it into the
  `audio_analytics_gigaam-cache` volume.
- **`meta.llm.reason` says `missing LLM_API_KEY`.** `LITELLM_MASTER_KEY` is
  empty in `stack/.env`.
- **`docker compose` fails with "required variable ... is missing a value".**
  A secret is missing from `stack/.env`. Rerun step 3 of Setup.
- **No traces in Langfuse.** They arrive about a minute after the request. The
  public `/api/public/traces` API returns nothing in Langfuse v4's
  `events_only` mode, so query ClickHouse instead:
  ```bash
  curl -s "http://127.0.0.1:8124/?user=clickhouse&password=clickhouse" \
    --data "SELECT name, type, provided_model_name FROM default.events_full ORDER BY start_time DESC LIMIT 5 FORMAT Vertical"
  ```

## Layout

```
stack/        compose project: langfuse/ and litellm/ (upstream, cloned in Setup), llamacpp/, overrides/, secrets/gen.sh
asr/          the ASR service: src/, tests/, deploy/, model_weights/ (diarization snapshot)
outputs/      llama.cpp server log (gitignored)
```

`asr/samples/` (the recordings `asr/deploy/verify.py` runs against) is kept out
of git because the recordings may contain personal data. Put your own there to
run `verify.py`.

## Status

This is a draft. The end-to-end path is verified: stereo, mono diarization,
the LLM pass, and the Langfuse trace. There is no PII masking yet, so the LLM
pass is limited to local endpoints.
