"""
Model load + `transcribe_windows`, from test_gigaam.ipynb cell 15 (STEP 2),
with the hotword beam decoder from cell 14 (STEP 1) installed at setup().

`AudioDataset.collate` pads the batch and returns the TRUE per-segment sample
lengths, which `_decode` needs: it derives the frame shift from
wav_lens / encoded_len, so passing the padded length would skew every
timestamp.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from gigaam.utils import AudioDataset

from asr_service import config
from asr_service.ctc_beam import CTCBeamDecoding
from asr_service.hotwords import HOTWORDS, sanitize_hotwords, validate_hotwords


@dataclass
class Word:
    """A word on the recording's absolute timeline."""
    start: float
    end: float
    text: str


_model = None
_greedy_decoding = None
_beam_decoding = None
_device: Optional[str] = None
_default_hotwords: Dict[str, float] = {}
_vocab: List[str] = []


def get_device() -> Optional[str]:
    return _device


def load_asr_model(device: str):
    global _model, _greedy_decoding, _beam_decoding, _device, _default_hotwords, _vocab
    import gigaam

    _device = device
    fp16 = config.FP16_ENCODER and device.startswith("cuda")
    _model = gigaam.load_model(config.ASR_MODEL_NAME, fp16_encoder=fp16, device=device)

    _greedy_decoding = _model.decoding
    _vocab = list(_greedy_decoding.tokenizer.vocab)
    _default_hotwords = sanitize_hotwords(HOTWORDS, _vocab, verbose=False)
    _beam_decoding = CTCBeamDecoding(_vocab, beam_width=config.BEAM_WIDTH, hotwords=_default_hotwords)

    set_decoder(config.DECODER)
    return _model


def set_decoder(name: str) -> None:
    if _model is None:
        raise RuntimeError("ASR model not loaded; call load_asr_model() first")
    _model.decoding = _beam_decoding if name == "beam" else _greedy_decoding


def use_hotwords(hotwords: Optional[Dict[str, float]]) -> Dict[str, Any]:
    """Apply this request's hotwords to the shared beam decoder, or restore
    the defaults when `hotwords` is None/empty.

    `_beam_decoding` is one instance shared by every request this worker
    handles (LitServe runs `predict` one request at a time per worker), so
    this must be called on *every* request -- defaults included -- or one
    caller's words would leak into the next request.
    """
    if _beam_decoding is None:
        raise RuntimeError("ASR model not loaded; call load_asr_model() first")

    if not hotwords:
        _beam_decoding.set_hotwords(_default_hotwords)
        return {"source": "default", "accepted": _default_hotwords, "rejected": {}}

    if len(hotwords) > config.MAX_REQUEST_HOTWORDS:
        raise ValueError(
            f"too many hotwords: {len(hotwords)} > {config.MAX_REQUEST_HOTWORDS}"
        )

    accepted, rejected = validate_hotwords(hotwords, _vocab, max_weight=config.MAX_HOTWORD_WEIGHT)
    # The request asked to replace the defaults; if everything it sent was
    # rejected, the request runs with no hotwords rather than silently
    # falling back to HOTWORDS.
    _beam_decoding.set_hotwords(accepted)
    return {"source": "request", "accepted": accepted, "rejected": rejected}


@torch.inference_mode()
def transcribe_windows(
    wav, windows: List[Tuple[float, float]], chunk_size: Optional[int] = None
) -> List[Word]:
    """Transcribe every window; return words on the absolute clock.

    Windows are chunked (chunk_size, default config.ASR_WINDOW_BATCH) so a
    17-minute recording's ~hundreds of windows never enter the encoder as a single
    batch.
    """
    if _model is None:
        raise RuntimeError("ASR model not loaded; call load_asr_model() first")
    if not windows:
        return []
    chunk_size = chunk_size or config.ASR_WINDOW_BATCH

    words: List[Word] = []
    for start in range(0, len(windows), chunk_size):
        batch_windows = windows[start : start + chunk_size]
        segments = [
            torch.as_tensor(wav[int(s * config.SAMPLE_RATE) : int(e * config.SAMPLE_RATE)])
            for s, e in batch_windows
        ]
        batch, lengths = AudioDataset.collate(segments)
        batch = batch.to(_device).to(_model._dtype)
        lengths = lengths.to(_device)

        encoded, encoded_len = _model.forward(batch, lengths)
        for i, (_text, seg_words) in enumerate(
            _model._decode(encoded, encoded_len, lengths, True)
        ):
            offset = batch_windows[i][0]
            for word in seg_words or []:
                words.append(
                    Word(round(word.start + offset, 2), round(word.end + offset, 2), word.text)
                )
    return words
