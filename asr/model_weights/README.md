# model_weights/

Local checkpoints, mounted at `/models` in the container (`MODEL_WEIGHTS_DIR`
points elsewhere, e.g. at a shared volume).

| path | used by | needed when |
|---|---|---|
| `speaker_diarization/community-1/` | `src/diarization.py` (`config.DIARIZATION_DIR`) | a request sends mono audio |
| `gender_classifier.pth` | `src/gender.py` (`config.GENDER_WEIGHTS`) | a request sets `analyze_gender=1` |

`speaker_diarization/community-1/` is a local snapshot of
[pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
(~32 MB). The HF repo is gated and not shipped in this repo (`*.bin` / `*.npz`
are gitignored), so it is read from disk rather than downloaded at run time --
fetch it first with:

```bash
HF_TOKEN=hf_... ../scripts/download_weights.sh
```

which fails with a clear message if `HF_TOKEN` is missing or the gated terms
were not accepted.

`gender_classifier.pth` is **not** shipped (`*.pth` / `*.pt` are gitignored);
drop it in from your model registry.

Both load lazily, on the first request that needs them, so a missing file
fails that request's analysis rather than start-up. Diarization degrades
gracefully: the mono input is then processed as one undiarized channel, with
the reason in `meta.diarization`. VAD, ASR and emotion come from Hugging Face
/ GigaAM's CDN at run time and need nothing here.
