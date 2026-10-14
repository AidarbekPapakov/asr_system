#!/usr/bin/env bash
# Fetches the local pyannote diarization snapshot that used to be vendored in
# model_weights/speaker_diarization/community-1/ (see model_weights/README.md).
# Needs a read-access HF_TOKEN that has accepted the gated terms.
set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
    echo "error: HF_TOKEN is not set." >&2
    echo "  create a read token at https://huggingface.co/settings/tokens" >&2
    echo "  and accept the gated terms at https://huggingface.co/pyannote/speaker-diarization-community-1" >&2
    exit 1
fi

DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/model_weights/speaker_diarization/community-1"

if ! hf download pyannote/speaker-diarization-community-1 --local-dir "$DEST" --token "$HF_TOKEN"; then
    echo "error: download failed -- this usually means HF_TOKEN has not accepted the" >&2
    echo "  gated terms at https://huggingface.co/pyannote/speaker-diarization-community-1" >&2
    exit 1
fi

echo "downloaded to $DEST"
