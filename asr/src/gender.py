"""
A small CNN gender classifier over MFCC features, adapted from an earlier
implementation, with two changes:

1. `_load_audio` also accepts a raw ndarray (assumed already resampled to
   16 kHz by audio.py) alongside the original path/BytesIO inputs, so the
   service never round-trips through a WAV file per turn.
2. The convention this returns is P(male) per chunk, always -- not
   "confidence in the predicted class". `GenderPredictor.num2gender` maps
   the sigmoid output's class 1 to "female", so the model's raw output is
   already P(female); `predict_with_probabilities` in the original module
   silently swapped that convention (returning P(female) as if it were
   P(male)) whenever a chunk was classified female, and its caller plugged
   the result straight into the "male" slot uncorrected. That bug is NOT
   reproduced here.

The weights are not within the repo - see model_weights/README.md for where to
retrieve them; config.GENDER_WEIGHTS points elsewhere when they come from a mount
instead.
"""

import io
from typing import List, Union

import librosa
import numpy as np
import torch
import torch.nn as nn

torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dropout_rate=0.33):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate),
        )

    def forward(self, x):
        return self.network(x)


class GenderCNN(nn.Module):

    def __init__(self, dropout_rate=0.33):
        super().__init__()
        self.conv_block1 = ConvBlock(1, 64, 3, padding=1, dropout_rate=dropout_rate)
        self.conv_block2 = ConvBlock(64, 128, 3, padding=1, dropout_rate=dropout_rate)
        self.conv_block3 = ConvBlock(128, 256, 3, padding=1, dropout_rate=dropout_rate)
        self.conv_block4 = ConvBlock(256, 512, 3, padding=1, dropout_rate=dropout_rate)
        self.conv_block5 = ConvBlock(512, 512, 2, padding=0, dropout_rate=dropout_rate)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class GenderPredictor:
    def __init__(self, model_weights_path: str, device: str = None) -> None:
        self.sampling_rate = 16000
        self.secs_to_leave = 5
        self.num_of_frames_to_leave = self.secs_to_leave * self.sampling_rate

        self.frame_length = 2048
        self.hop_length = 512
        self.n_mfcc = 50

        self.device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_weights_path = model_weights_path

        self.model = GenderCNN()
        state_dict = torch.load(model_weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def p_male_per_chunk(self, audio: Union[str, io.BytesIO, bytes, np.ndarray]) -> List[float]:
        """P(male) for each 5-second chunk the clip is split into (the last
        chunk zero-padded if shorter)."""
        mfcc_fragments = self._extract_mfcc(audio)
        out: List[float] = []
        for mfcc in mfcc_fragments:
            feat = torch.FloatTensor(mfcc).unsqueeze(0)          # (1, n_mfcc, T) -- channel dim
            with torch.no_grad():
                x = feat.to(self.device).unsqueeze(0)             # (1, 1, n_mfcc, T) -- batch dim
                p_female = self.model(x).item()                   # sigmoid output = P(class 1) = P(female)
            out.append(1.0 - p_female)
        return out

    def _extract_mfcc(self, audio) -> List[np.ndarray]:
        waveforms = self._load_audio(audio)
        return [
            librosa.feature.mfcc(
                y=w, sr=self.sampling_rate, n_mfcc=self.n_mfcc, norm="ortho",
                lifter=26, n_fft=self.frame_length, hop_length=self.hop_length,
            )
            for w in waveforms
        ]

    def _load_audio(self, audio) -> List[np.ndarray]:
        if isinstance(audio, np.ndarray):
            wv = audio.astype(np.float32)   # already 16 kHz mono -- audio.py's contract
        else:
            if isinstance(audio, io.BytesIO):
                audio.seek(0)
            elif not isinstance(audio, str):
                audio = io.BytesIO(audio)
                audio.seek(0)
            wv, _ = librosa.load(audio, sr=self.sampling_rate)

        audio_length = len(wv)
        if audio_length == 0:
            return [np.zeros(self.num_of_frames_to_leave, dtype=np.float32)]

        if audio_length > self.num_of_frames_to_leave:
            n_fragments = int(np.ceil(audio_length / self.num_of_frames_to_leave))
            fragments = []
            for i in range(n_fragments):
                seg_start = i * self.num_of_frames_to_leave
                seg_end = min(seg_start + self.num_of_frames_to_leave, audio_length)
                seg_len = seg_end - seg_start
                if seg_len < self.num_of_frames_to_leave:
                    fragment = np.concatenate(
                        (wv[seg_start:seg_end], np.zeros(self.num_of_frames_to_leave - seg_len))
                    )
                else:
                    fragment = wv[seg_start:seg_end]
                fragments.append(fragment)
            return fragments

        return [np.concatenate((wv, np.zeros(self.num_of_frames_to_leave - audio_length)))]


_predictor: "GenderPredictor" = None


def load_gender_model(device: str, weights_path: str) -> GenderPredictor:
    global _predictor
    _predictor = GenderPredictor(weights_path, device=device)
    return _predictor


def ensure_loaded(device: str, weights_path: str) -> GenderPredictor:
    """Load on first use -- gender is opt-in per request, so setup() does not
    pay for the weights up front."""
    return _predictor if _predictor is not None else load_gender_model(device, weights_path)


def p_male_per_chunk(wav: np.ndarray) -> List[float]:
    if _predictor is None:
        raise RuntimeError("gender model not loaded; call ensure_loaded() first")
    return _predictor.p_male_per_chunk(wav)
