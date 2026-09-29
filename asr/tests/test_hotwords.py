import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hotwords import HOTWORDS, MAX_HOTWORD_WEIGHT, MIN_HOTWORD_LEN, validate_hotwords

# A vocabulary big enough to accept every plausibly-valid word in these tests.
VOCAB = list("абвгдежзийклмнопрстуфхцчшщъыьэюя -")


class TestValidateHotwords(unittest.TestCase):
    def test_accepts_a_valid_entry_and_lowercases_it(self):
        ok, rejected = validate_hotwords({"Словарь": 4.0}, VOCAB)
        self.assertEqual(ok, {"словарь": 4.0})
        self.assertEqual(rejected, {})

    def test_rejects_too_short(self):
        ok, rejected = validate_hotwords({"да": 4.0}, VOCAB)
        self.assertEqual(ok, {})
        self.assertIn("да", rejected)
        self.assertIn(str(MIN_HOTWORD_LEN), rejected["да"])

    def test_rejects_space_or_hyphen(self):
        ok, rejected = validate_hotwords({"новый банк": 4.0, "кто-либо": 4.0}, VOCAB)
        self.assertEqual(ok, {})
        self.assertEqual(len(rejected), 2)

    def test_rejects_out_of_vocabulary_characters(self):
        ok, rejected = validate_hotwords({"ўзбек": 4.0}, VOCAB)
        self.assertEqual(ok, {})
        self.assertIn("ў", rejected["ўзбек"])

    def test_rejects_non_finite_or_out_of_range_weight(self):
        cases = {"словарь": math.nan, "термин": math.inf, "пример": 0.0, "проверка": 999.0}
        ok, rejected = validate_hotwords(cases, VOCAB)
        self.assertEqual(ok, {})
        self.assertEqual(set(rejected), set(cases))

    def test_weight_at_the_max_is_accepted(self):
        ok, rejected = validate_hotwords({"словарь": MAX_HOTWORD_WEIGHT}, VOCAB)
        self.assertEqual(ok, {"словарь": MAX_HOTWORD_WEIGHT})
        self.assertEqual(rejected, {})


class TestShippedHotwords(unittest.TestCase):
    """Structural invariants on HOTWORDS itself, so a badly-formed entry
    added later fails CI even without the real model vocab on hand."""

    def test_every_entry_is_long_enough(self):
        for word in HOTWORDS:
            self.assertGreaterEqual(len(word), MIN_HOTWORD_LEN, word)

    def test_no_entry_has_a_space_or_hyphen(self):
        for word in HOTWORDS:
            self.assertNotIn(" ", word)
            self.assertNotIn("-", word)

    def test_every_weight_is_in_range(self):
        for word, weight in HOTWORDS.items():
            self.assertTrue(math.isfinite(weight), word)
            self.assertTrue(0 < weight <= MAX_HOTWORD_WEIGHT, word)

    def test_entries_are_already_lowercase(self):
        for word in HOTWORDS:
            self.assertEqual(word, word.lower(), word)


if __name__ == "__main__":
    unittest.main()
