# deploy/

## Local run (no Docker)

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python --torch-backend=cu129 -r deploy/requirements.txt
cp deploy/.env.example deploy/.env   # fill in HF_TOKEN at least
set -a; source deploy/.env; set +a
PYTHONPATH=src .venv/bin/python -m asr_service.server
```

```bash
curl -s localhost:8400/health
curl -F 'audio=@samples/<file>.mp3' localhost:8400/analyze | python -m json.tool

# with the opt-in analyses:
curl -F 'audio=@<file>.mp3' -F analyze_emotion=1 -F analyze_gender=1 \
     localhost:8400/analyze | python -m json.tool
```

## Verification

```bash
# against a server you started yourself:
PYTHONPATH=src .venv/bin/python deploy/verify.py

# or let it start one:
.venv/bin/python deploy/verify.py --start-server

# include the 1040s stress recording (not shipped in samples/):
.venv/bin/python deploy/verify.py --start-server --stress
```

Functional and system-metrics only -- no WER, no transcript grading. The five
recordings in `samples/` cover degenerate, p25, median and p90 stereo
durations, and one mono case, which exercises diarization.

Check 11 runs first and must: it asks for a flags-off request while emotion and
gender are still unloaded, which is the only moment the lazy loading is
observable. It then reports the cold and warm cost of a flags-on request and the
server's RSS at each step.

Checks 9-10 (the LLM pass) run in-process against `normalize.run_normalization`
directly rather than through the live server's fixed environment; check 9 is
skipped unless `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` are set.

## Docker

`docker-compose.yml` is not run on its own: `../../stack/compose.yml` includes
it, next to the llama.cpp + LiteLLM + Langfuse services it normalises
through.

```bash
cp deploy/.env.example deploy/.env   # fill in HF_TOKEN
cd ../stack && docker compose up -d --build
curl -s localhost:8400/health
```

It publishes 8400 and runs on CPU by default (see the compose file's header
for the GPU switch). `model_weights/` is mounted at `/models`: fetch the
diarization snapshot with `../scripts/download_weights.sh` first, and add the
gender weights yourself (see `model_weights/README.md`) -- unless
`MODEL_WEIGHTS_DIR` points somewhere else.

## CI/CD

There are two pipelines. `../.gitlab-ci.yml` runs the full thing (lint, the
13 GB image build, then `tests/` with torch/gigaam inside that image) on a
self-hosted runner; `../../.github/workflows/ci.yml` runs on GitHub's free
runners instead, so it only lints and runs the tests that don't need
torch/gigaam (those are skipped via the `requires_models` marker on
`test_ctc_beam.py` / `test_diarization.py`). Reproduce the GitLab one locally
with:

```bash
docker run --rm -v "$PWD/..":/src -w /src ghcr.io/astral-sh/ruff:0.16.8 check .
docker build -f Dockerfile -t asr-service:local ..
docker run --rm -v "$PWD/../tests":/app/tests:ro asr-service:local python -m unittest discover -s /app/tests -v
```

or, without Docker, against the `.venv` from "Local run" above:

```bash
PYTHONPATH=../src .venv/bin/python -m unittest discover -s ../tests -v
```

`tests/` covers the pure, CPU-only logic only (`ctc_beam`, `hotwords`,
`contract`, `config`, `diarization`'s track splitting), nothing that needs a model download or `HF_TOKEN`.
End-to-end stays `deploy/verify.py`, run by hand.
