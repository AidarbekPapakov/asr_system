# ASR service

A LitServe service that turns an audio recording into a structured,
speaker-attributed transcript over HTTP: pyannote diarization (mono input),
pyannote VAD windowing, GigaAM CTC ASR with hotword-boosted beam decoding,
turn assembly, and an optional LLM normalisation pass. Emotion (GigaAM) and
gender (a small CNN) are available per request.

It is domain-neutral and reachable by anything that can POST a file. Nothing
in the response assumes a particular consumer, so it is meant to be extended:
add the analyses and the response keys your caller needs on top of a base
that stays generic.

It runs as part of `../stack` (see `../CLAUDE.md`), which provides the
llama.cpp + LiteLLM + Langfuse services the LLM pass goes through.

## Layout

```
src/asr_service/
  config.py          every knob, env override, one place
  audio.py           upload bytes -> per-channel float32 @16 kHz
  diarization.py     mono -> one track per speaker (pyannote community-1)
  vad.py             pyannote regions, group_regions(), speech/silence metrics
  ctc_beam.py        hotword-boosted CTC prefix beam search
  hotwords.py        the default hotword list (empty) + its sanitiser
  asr.py             model load, transcribe_windows() -> List[Word]
  turns.py           Turn, words_to_turns, merge_channels
  emotion.py         batched GigaAM emotion head + 4->3 label collapse
  gender.py          the gender CNN
  normalize.py       LLM normalisation
  pipeline.py        analyze_audio(...) -> result dict   (pure, no HTTP)
  server.py          LitServe LitAPI + build_server()
  client.py          analyze(path) -> dict              (transport only)
  contract.py        assert_contract(res) -- the guaranteed key set
deploy/
  requirements.txt, Dockerfile, docker-compose.yml, .env.example
  verify.py          functional + system-metrics verification
model_weights/       local checkpoints -- see its README
samples/             the recordings verify.py runs against
```

## Speakers: channels and diarization

Every channel in the response is one speaker, labelled `speaker_<i>`.

- **Multi-channel input**: each physical channel is a speaker. That is exact,
  so no diarization runs.
- **Mono input**: `pyannote/speaker-diarization-community-1` splits the
  channel into one track per detected speaker. Each track is the full length
  of the input and silent outside that speaker's segments, and at most one
  speaker is active at any instant. Tracks are ordered by first onset, so
  whoever speaks first is `speaker_0`. From there on each track is handled
  exactly like a physical channel.

`meta.diarization` reports what happened. When it has `applied: true`,
`channels[i]` is a diarized track, not a physical channel. If nobody speaks,
or the pipeline cannot load (`DIARIZE_MONO=0` also turns it off), the mono
input is processed as a single `speaker_0` channel and the reason is given
there.

By default the number of speakers is left to pyannote.
`DIARIZATION_MIN_SPEAKERS` / `DIARIZATION_MAX_SPEAKERS` bound it when you
know the count in advance.

## The wire contract

`POST /analyze`, `multipart/form-data`:

| field | default | effect |
|---|---|---|
| `audio` | — | required file |
| `analyze_emotion` | `0` | adds `turns[].emotion` |
| `analyze_gender` | `0` | adds `turns[].gender` |
| `run_llm` | `RUN_LLM` | per-request override |
| `speaker_map` | — | JSON `{"0":"alice","1":"bob"}`: renames `speaker_<i>`; unlisted channels keep their default |
| `hotwords` | `hotwords.HOTWORDS` | JSON `{"word": weight}`: replaces the default list for this request |

```json
{
  "turns": [
    {"id": 1, "channel": 0, "speaker": "speaker_0", "start": 0.4, "end": 2.1,
     "text": "Алло, добрый день.", "text_raw": "алло добрый день",
     "lang": null, "overlap": false,
     "emotion": {"label": "neutral", "distribution": {"negative": 0.1, "neutral": 0.8, "positive": 0.1}},
     "gender":  {"label": "male",    "distribution": {"male": 0.93, "female": 0.07}}}
  ],
  "channels": [
    {"index": 0,
     "words": [{"start": 0.4, "end": 0.7, "text": "алло"}],
     "vad": {"total_speech_time": 2.1, "total_silence_time": 0.9,
             "speech_ratio": 0.7, "regions": [[0.4, 2.1]]}}
  ],
  "meta": {"duration_sec": 3.0, "n_channels": 2, "asr_model": "...",
           "decoder": "beam", "device": "cpu",
           "diarization": {"applied": true, "n_speakers": 2, "reason": "mono input"},
           "llm": {"ran": true, "applied": true, "reason": "applied"},
           "timings": {...}, "version": 2}
}
```

`emotion` and `gender` are left out when not requested; they are never
`null`. `src/asr_service/contract.py` is the authoritative statement of what is
guaranteed.

## What each part costs you

**VAD always runs.** It splits the audio into windows for the encoder, so its
speech/silence metrics come for free and are always reported.

**Diarization, emotion and gender load on first use.** A service that never
gets mono input, or is never asked for emotion or gender, never pays for
their memory. The first request that needs one pays the checkpoint load
inside that request. Warmup uses stereo silence so that diarization stays
lazy.

Emotion and gender are classified per turn, on that turn's audio **widened to
at least `MIN_WIN` (3 s) within the channel**. A turn can be a single 0.1 s
"ага", which is too little to classify and so few samples that the encoder
degenerates. The label therefore means "this turn in its immediate
surroundings", not "this turn in isolation".

## Hotwords

`hotwords.HOTWORDS` is empty. To tune recognition for a domain, add its
out-of-vocabulary tokens (brands, places, jargon) there, or send them per
request. `hotwords.validate_hotwords` explains what gets rejected and why.

## LLM normalisation: what it does and does not do

It adds punctuation and capitalisation, applies inverse text normalisation,
and repairs CTC word splits and merges. It reaches `turns[].text` only, never
`channels[].words`, because ITN and split/merge repair change token counts by
design. `turns[].text_raw` always keeps the unmodified CTC output. The reply
is applied only if every turn id comes back unchanged; otherwise the raw
transcript is kept and `meta.llm.reason` says why.

In the stack it calls LiteLLM (`http://litellm:4000/v1`, model `local-llm`),
so every request is traced in Langfuse.

PII masking is not in this build. With `run_llm=1` and an `LLM_BASE_URL`
that is neither loopback nor listed in `LLM_INTERNAL_HOSTS`, the raw
transcript would leave the machine unmasked. That is refused unless
`LLM_ALLOW_EXTERNAL=1` is set explicitly.

## Quick start

See `deploy/README.md`.

## Further reading

`ctc_beam.py`'s hotword-boosted decoder builds on standard CTC beam search
ideas, adapted because GigaAM only ships greedy decoding. For background on
CTC and beam search decoding:

- [Introduction to Beam Search](https://www.geeksforgeeks.org/machine-learning/introduction-to-beam-search-algorithm/)
- [CTC architectures (HF Audio Course)](https://huggingface.co/learn/audio-course/chapter3/ctc#ctc-architectures)
- [Beam Search Decoding in CTC-Trained Neural Networks](https://harald-scheidl.medium.com/beam-search-decoding-in-ctc-trained-neural-networks-5a889a3d85a7)
- [Word Beam Search: A CTC Decoding Algorithm](https://harald-scheidl.medium.com/word-beam-search-a-ctc-decoding-algorithm-b051d28f3d2e)
