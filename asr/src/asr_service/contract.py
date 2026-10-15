"""
assert_contract(res) -- the exact shape the service guarantees.

Raises AssertionError naming exactly what is missing or mistyped; never
returns a value.

`emotion` and `gender` are checked only where they appear: they are opt-in per
request, so their absence is correct, not a violation. What IS asserted is that
whatever is present is well formed.
"""

from typing import Any

_TURN_KEYS = ("id", "channel", "speaker", "start", "end", "text", "text_raw", "lang", "overlap")
_WORD_KEYS = ("start", "end", "text")
_VAD_KEYS = ("total_speech_time", "total_silence_time", "speech_ratio", "regions")
_META_KEYS = (
    "asr_model", "decoder", "device", "duration_sec", "n_channels", "diarization", "llm", "timings", "version",
)
_EMOTION_LABELS = {"negative", "neutral", "positive"}
_GENDER_LABELS = {"male", "female"}


def _assert_distribution(block: dict[str, Any], labels: set, where: str) -> None:
    assert "label" in block and "distribution" in block, f"{where}: missing label/distribution"
    assert block["label"] in labels, f"{where}: unexpected label {block['label']!r}"
    assert set(block["distribution"]) == labels, f"{where}: distribution keys {set(block['distribution'])}"
    total = sum(block["distribution"].values())
    assert abs(total - 1.0) < 1e-2, f"{where}: distribution sums to {total}, not 1.0"


def assert_contract(res: dict[str, Any]) -> None:
    for key in ("turns", "channels", "meta"):
        assert key in res, f"missing top-level key: {key!r}"

    assert isinstance(res["channels"], list) and res["channels"], "channels must be a non-empty list"
    for expected, channel in enumerate(res["channels"]):
        assert channel.get("index") == expected, f"channels[{expected}]: index is {channel.get('index')!r}"
        for key in ("words", "vad"):
            assert key in channel, f"channel {expected}: missing {key!r}"

        for word in channel["words"]:
            for wkey in _WORD_KEYS:
                assert wkey in word, f"channel {expected}: word missing {wkey!r}"
            assert word["start"] <= word["end"], f"channel {expected}: word start > end: {word}"

        for vkey in _VAD_KEYS:
            assert vkey in channel["vad"], f"channel {expected}.vad missing {vkey!r}"
        assert isinstance(channel["vad"]["total_speech_time"], (int, float))
        assert isinstance(channel["vad"]["total_silence_time"], (int, float))

    ids = [t["id"] for t in res["turns"]]
    assert ids == list(range(1, len(ids) + 1)), f"turn ids are not a contiguous 1..n sequence: {ids}"
    for turn in res["turns"]:
        for key in _TURN_KEYS:
            assert key in turn, f"turn {turn.get('id')}: missing {key!r}"
        assert turn["start"] <= turn["end"], f"turn {turn['id']}: start > end"
        if "emotion" in turn:
            _assert_distribution(turn["emotion"], _EMOTION_LABELS, f"turn {turn['id']}.emotion")
        if "gender" in turn:
            _assert_distribution(turn["gender"], _GENDER_LABELS, f"turn {turn['id']}.gender")

    for mkey in _META_KEYS:
        assert mkey in res["meta"], f"meta missing {mkey!r}"
