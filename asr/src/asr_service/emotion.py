"""
Slight modification of GigaAMEmo emotion model, that targets `.get_probs`
(third_party/GigaAM/gigaam/model.py:272), which collapses 4-class into 3-class labels.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
from gigaam.utils import AudioDataset

from asr_service import config

_model = None
_device = None

# GigaAM's emo checkpoint (~/.cache/gigaam/emo.ckpt) ships FOUR labels via
# cfg.id2name: ['angry', 'sad', 'neutral', 'positive']. The wire contract
# reports three -- collapse negative = angry + sad, argmax taken AFTER
# collapsing, so a turn that is half angry and half sad reads as negative
# rather than as whichever of the two edged ahead.
_COLLAPSE = {"angry": "negative", "sad": "negative", "neutral": "neutral", "positive": "positive"}


def load_emotion_model(device: str):
    global _model, _device
    import gigaam

    _device = device
    fp16 = config.FP16_ENCODER and device.startswith("cuda")
    _model = gigaam.load_model(config.EMO_MODEL_NAME, fp16_encoder=fp16, device=device)
    return _model


def ensure_loaded(device: str):
    """Load on first use. The checkpoint is ~1 GB resident and emotion is
    opt-in per request, so setup() deliberately does not pay for it."""
    return _model if _model is not None else load_emotion_model(device)


@torch.inference_mode()
def probs_batch(wavs: List[np.ndarray]) -> List[Dict[str, float]]:
    """Raw 4-class probabilities per input waveform, batched in one forward
    pass. The result is index-aligned with `wavs`, so every waveform must be
    non-empty -- callers slice with a minimum length for exactly that reason."""
    if _model is None:
        raise RuntimeError("emotion model not loaded; call ensure_loaded() first")
    if not wavs:
        return []

    segments = [torch.as_tensor(w) for w in wavs]
    batch, lengths = AudioDataset.collate(segments)
    batch = batch.to(_device).to(_model._dtype)
    lengths = lengths.to(_device)

    encoded, encoded_len = _model.forward(batch, lengths)   # (B, D, T), (B,)
    b, _d, t = encoded.shape
    frame_idx = torch.arange(t, device=encoded.device)[None, :]
    mask = (frame_idx < encoded_len[:, None]).unsqueeze(1).to(encoded.dtype)  # (B, 1, T)
    summed = (encoded * mask).sum(dim=-1)
    counts = mask.sum(dim=-1).clamp(min=1)
    pooled = summed / counts                                 # (B, D)

    logits = _model.head(pooled)
    probs = torch.nn.functional.softmax(logits, dim=-1).float().cpu().tolist()

    id2name = _model.id2name
    return [{id2name[i]: p[i] for i in range(len(id2name))} for p in probs]


def collapse_emotion(probs4: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
    """4-class probs -> (label, 3-class distribution)."""
    dist3 = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
    for name, p in probs4.items():
        dist3[_COLLAPSE.get(name, "neutral")] += float(p)
    dist3 = {k: round(v, 4) for k, v in dist3.items()}
    return max(dist3, key=dist3.get), dist3
