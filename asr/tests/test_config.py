import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from asr_service.config import env_bool, env_float, env_int


class TestEnvBool(unittest.TestCase):
    def test_unset_returns_default(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertTrue(env_bool("X", True))
            self.assertFalse(env_bool("X", False))

    def test_truthy_values(self):
        for raw in ("1", "true", "True", "yes", "on", "  on  "):
            with mock.patch.dict("os.environ", {"X": raw}):
                self.assertTrue(env_bool("X", False), raw)

    def test_falsy_values(self):
        for raw in ("0", "false", "False", "no", "off"):
            with mock.patch.dict("os.environ", {"X": raw}):
                self.assertFalse(env_bool("X", True), raw)

    def test_garbage_value_reads_as_false(self):
        with mock.patch.dict("os.environ", {"X": "maybe"}):
            self.assertFalse(env_bool("X", True))


class TestEnvInt(unittest.TestCase):
    def test_unset_returns_default(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(env_int("X", 7), 7)

    def test_set_value_is_parsed(self):
        with mock.patch.dict("os.environ", {"X": "42"}):
            self.assertEqual(env_int("X", 7), 42)

    def test_empty_string_falls_back_to_default(self):
        with mock.patch.dict("os.environ", {"X": ""}):
            self.assertEqual(env_int("X", 7), 7)


class TestEnvFloat(unittest.TestCase):
    def test_unset_returns_default(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(env_float("X", 1.5), 1.5)

    def test_set_value_is_parsed(self):
        with mock.patch.dict("os.environ", {"X": "0.6"}):
            self.assertEqual(env_float("X", 1.5), 0.6)


if __name__ == "__main__":
    unittest.main()
