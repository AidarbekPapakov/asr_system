import sys
import unittest
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from contract import assert_contract


def _valid_response():
    return {
        "turns": [
            {
                "id": 1, "channel": 0, "speaker": "speaker_0", "start": 0.0, "end": 1.0,
                "text": "алло", "text_raw": "алло", "lang": None, "overlap": False,
            },
            {
                "id": 2, "channel": 1, "speaker": "speaker_1", "start": 1.0, "end": 2.0,
                "text": "добрый день", "text_raw": "добрый день", "lang": None, "overlap": False,
                "emotion": {"label": "neutral", "distribution": {"negative": 0.1, "neutral": 0.8, "positive": 0.1}},
                "gender": {"label": "male", "distribution": {"male": 0.9, "female": 0.1}},
            },
        ],
        "channels": [
            {
                "index": 0,
                "words": [{"start": 0.0, "end": 0.4, "text": "алло"}],
                "vad": {"total_speech_time": 1.0, "total_silence_time": 0.0, "speech_ratio": 1.0, "regions": [[0.0, 1.0]]},
            },
            {
                "index": 1,
                "words": [],
                "vad": {"total_speech_time": 1.0, "total_silence_time": 0.0, "speech_ratio": 1.0, "regions": [[1.0, 2.0]]},
            },
        ],
        "meta": {
            "asr_model": "multilingual_large_ctc", "decoder": "beam", "device": "cpu",
            "duration_sec": 2.0, "n_channels": 2,
            "diarization": {"applied": False, "n_speakers": None, "reason": "stereo"}, "llm": {}, "timings": {}, "version": 2,
        },
    }


class TestAssertContract(unittest.TestCase):
    def test_a_valid_response_passes(self):
        assert_contract(_valid_response())  # must not raise

    def test_missing_top_level_key_raises(self):
        res = _valid_response()
        del res["meta"]
        with self.assertRaises(AssertionError):
            assert_contract(res)

    def test_non_contiguous_turn_ids_raises(self):
        res = _valid_response()
        res["turns"][1]["id"] = 3
        with self.assertRaises(AssertionError):
            assert_contract(res)

    def test_turn_start_after_end_raises(self):
        res = _valid_response()
        res["turns"][0]["start"], res["turns"][0]["end"] = 5.0, 1.0
        with self.assertRaises(AssertionError):
            assert_contract(res)

    def test_word_start_after_end_raises(self):
        res = _valid_response()
        res["channels"][0]["words"][0]["start"] = 10.0
        with self.assertRaises(AssertionError):
            assert_contract(res)

    def test_emotion_distribution_not_summing_to_one_raises(self):
        res = deepcopy(_valid_response())
        res["turns"][1]["emotion"]["distribution"]["neutral"] = 0.2  # now sums to ~0.4
        with self.assertRaises(AssertionError):
            assert_contract(res)

    def test_unexpected_emotion_label_raises(self):
        res = deepcopy(_valid_response())
        res["turns"][1]["emotion"]["label"] = "surprised"
        with self.assertRaises(AssertionError):
            assert_contract(res)

    def test_empty_channels_raises(self):
        res = _valid_response()
        res["channels"] = []
        with self.assertRaises(AssertionError):
            assert_contract(res)

    def test_channel_index_mismatch_raises(self):
        res = _valid_response()
        res["channels"][1]["index"] = 5
        with self.assertRaises(AssertionError):
            assert_contract(res)


if __name__ == "__main__":
    unittest.main()
