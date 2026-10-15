import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from asr_service import config

try:
    import numpy as np

    from asr_service.diarization import tracks_from_annotation
    _MODELS_AVAILABLE = True
except ImportError:
    _MODELS_AVAILABLE = False

requires_models = unittest.skipUnless(_MODELS_AVAILABLE, "requires_models: needs numpy/torch")


class FakeAnnotation:
    """Just the one method tracks_from_annotation uses from
    pyannote.core.Annotation."""

    def __init__(self, segments):
        self._segments = segments   # [(start, end, speaker)]

    def itertracks(self, yield_label=False):
        for i, (start, end, speaker) in enumerate(self._segments):
            yield SimpleNamespace(start=start, end=end), i, speaker


SR = config.SAMPLE_RATE


@requires_models
class TestTracksFromAnnotation(unittest.TestCase):
    def setUp(self):
        self.wav = np.arange(4 * SR, dtype=np.float32) + 1.0   # never zero

    def test_nobody_speaks(self):
        self.assertEqual(tracks_from_annotation(FakeAnnotation([]), self.wav), [])

    def test_ordered_by_first_onset_not_label(self):
        # SPEAKER_01 speaks first, so it must land on track 0.
        ann = FakeAnnotation([(2.0, 3.0, "SPEAKER_00"), (0.0, 1.0, "SPEAKER_01")])
        tracks = tracks_from_annotation(ann, self.wav)
        self.assertEqual(len(tracks), 2)
        self.assertTrue(tracks[0][: SR].all())
        self.assertFalse(tracks[0][2 * SR : 3 * SR].any())
        self.assertTrue(tracks[1][2 * SR : 3 * SR].all())

    def test_tracks_are_full_length_and_silent_elsewhere(self):
        ann = FakeAnnotation([(1.0, 2.0, "A"), (3.0, 3.5, "A")])
        (track,) = tracks_from_annotation(ann, self.wav)
        self.assertEqual(len(track), len(self.wav))
        np.testing.assert_array_equal(track[SR : 2 * SR], self.wav[SR : 2 * SR])
        np.testing.assert_array_equal(track[3 * SR : int(3.5 * SR)], self.wav[3 * SR : int(3.5 * SR)])
        self.assertEqual(np.count_nonzero(track), SR + SR // 2)

    def test_segment_past_the_end_is_clamped(self):
        ann = FakeAnnotation([(3.5, 10.0, "A")])
        (track,) = tracks_from_annotation(ann, self.wav)
        self.assertEqual(len(track), len(self.wav))
        self.assertEqual(np.count_nonzero(track), SR // 2)


if __name__ == "__main__":
    unittest.main()
