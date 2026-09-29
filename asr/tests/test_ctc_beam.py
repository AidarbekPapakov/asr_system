import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ctc_beam import CTCBeamDecoding


def _peaked_row(target_idx: int, n_classes: int, target_p: float = 0.9) -> np.ndarray:
    """A near one-hot log-prob row: target_p on target_idx, the rest split evenly."""
    row = np.full(n_classes, np.log((1.0 - target_p) / (n_classes - 1)))
    row[target_idx] = np.log(target_p)
    return row


class TestPeakedDecode(unittest.TestCase):
    """A clearly peaked path decodes to the expected text."""

    def test_decodes_the_obvious_word(self):
        vocab = [" ", "а", "к", "о", "т"]
        dec = CTCBeamDecoding(vocab, beam_width=25, top_k=8, prune_logp=-12.0)
        n_classes = len(vocab) + 1  # + blank
        lp = np.stack([
            _peaked_row(vocab.index("к"), n_classes),
            _peaked_row(vocab.index("о"), n_classes),
            _peaked_row(vocab.index("т"), n_classes),
        ])

        text, ids, frames = dec._decode_one(lp)

        self.assertEqual(text, "кот")
        self.assertEqual(len(ids), len(frames))
        self.assertEqual(list(frames), sorted(frames))


class TestCTCCollapse(unittest.TestCase):
    """Repeated symbol without an intervening blank collapses to one character;
    with a blank between, it is a genuine doubled character."""

    def setUp(self):
        self.vocab = [" ", "к"]  # + blank -> 3 classes: space, к, blank
        self.dec = CTCBeamDecoding(self.vocab, beam_width=25, top_k=8, prune_logp=-12.0)
        self.frame_k = np.log(np.array([1e-9, 0.95, 0.05]))
        self.frame_blank = np.log(np.array([1e-9, 0.05, 0.95]))

    def test_repeat_without_blank_collapses(self):
        lp = np.stack([self.frame_k, self.frame_k])
        text, ids, frames = self.dec._decode_one(lp)
        self.assertEqual(text, "к")
        self.assertEqual(len(ids), 1)

    def test_repeat_with_blank_between_stays_double(self):
        lp = np.stack([self.frame_k, self.frame_blank, self.frame_k])
        text, ids, frames = self.dec._decode_one(lp)
        self.assertEqual(text, "кк")
        self.assertEqual(len(ids), 2)


class TestHotwordBoost(unittest.TestCase):
    """A hotword's partial credit can flip an acoustically close decision
    toward it, and clearing the hotwords removes the effect again."""

    def setUp(self):
        self.vocab = [" ", "б", "к"]  # + blank -> 4 classes
        # б is acoustically favored (0.55 vs 0.45) but "к" is the start of
        # the hotword "кот".
        self.frame = np.log(np.array([1e-9, 0.55, 0.45, 1e-9]))
        self.lp = np.stack([self.frame])

    def test_without_hotwords_the_more_probable_symbol_wins(self):
        dec = CTCBeamDecoding(self.vocab, beam_width=25, top_k=8, prune_logp=-30.0)
        text, _, _ = dec._decode_one(self.lp)
        self.assertEqual(text, "б")

    def test_hotword_flips_the_decision(self):
        dec = CTCBeamDecoding(
            self.vocab, beam_width=25, top_k=8, prune_logp=-30.0,
            hotwords={"кот": 5.0},
        )
        text, _, _ = dec._decode_one(self.lp)
        self.assertEqual(text, "к")

    def test_clearing_hotwords_restores_the_original_decision(self):
        dec = CTCBeamDecoding(
            self.vocab, beam_width=25, top_k=8, prune_logp=-30.0,
            hotwords={"кот": 5.0},
        )
        dec.set_hotwords(None)
        text, _, _ = dec._decode_one(self.lp)
        self.assertEqual(text, "б")
        self.assertEqual(dec.hotwords, {})


if __name__ == "__main__":
    unittest.main()
