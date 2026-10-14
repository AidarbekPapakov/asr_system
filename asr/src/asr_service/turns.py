"""
Turn assembly, from test_gigaam.ipynb cell 16 (STEP 3).

Per-channel word streams -> one time-ordered dialogue. Each channel is one
speaker: either a physical channel of a multi-channel recording (exact, no
clustering) or a track produced by diarization.py from mono input.

`Turn` carries `text_raw` alongside `text` so step 5 (normalize.py) has
somewhere honest to put its rewrite: `text` may be normalised, `text_raw`
never changes after construction.
"""

from dataclasses import dataclass

from asr_service import config
from asr_service.asr import Word


@dataclass
class Turn:
    id: int
    channel: int            # which channel this turn was recorded on
    speaker: str
    start: float
    end: float
    text: str
    text_raw: str = ""
    lang: str | None = None
    overlap: bool = False   # someone else was talking at the same time

    def __post_init__(self) -> None:
        if not self.text_raw:
            self.text_raw = self.text


def words_to_turns(
    words: list[Word], speaker: str, channel: int = -1, turn_gap: float | None = None
) -> list[Turn]:
    """Break one channel's words into turns on any silence wider than turn_gap."""
    turn_gap = config.TURN_GAP if turn_gap is None else turn_gap
    turns: list[Turn] = []
    group: list[Word] = []

    def flush() -> None:
        if group:
            text = " ".join(w.text for w in group)
            turns.append(Turn(id=0, channel=channel, speaker=speaker,
                              start=group[0].start, end=group[-1].end, text=text))

    for word in words:
        if group and word.start - group[-1].end > turn_gap:
            flush()
            group = []
        group.append(word)
    flush()
    return turns


def merge_channels(per_channel: dict[int, list[Word]], channel_speaker: dict[int, str]) -> list[Turn]:
    """Interleave both channels by time, number the turns, flag overlaps."""
    turns: list[Turn] = []
    for ch, words in per_channel.items():
        turns.extend(words_to_turns(words, channel_speaker.get(ch, f"channel_{ch}"), channel=ch))
    turns.sort(key=lambda t: (t.start, t.end))

    for i, turn in enumerate(turns):
        turn.id = i + 1
        turn.overlap = any(
            other.speaker != turn.speaker and other.start < turn.end and turn.start < other.end
            for other in turns
        )
    return turns
