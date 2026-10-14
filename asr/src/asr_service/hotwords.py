"""
The default hotword list, and its sanitiser.

Empty by default: the service is domain-neutral. To adapt it to a domain,
collect the out-of-vocabulary tokens that matter there (brands, places, jargon)
and add them here, or send them per request as the `hotwords` form field.
Weights are in natural log units added to the beam ranking, so 6.0 is worth
~400x from the base.
"""

import math
from collections.abc import Sequence

HOTWORDS: dict[str, float] = {
    # "example": 5.0,
}

MIN_HOTWORD_LEN = 5


MAX_HOTWORD_WEIGHT = 10.0


def validate_hotwords(
    hotwords: dict[str, float], vocab: Sequence[str], max_weight: float = MAX_HOTWORD_WEIGHT
) -> tuple[dict[str, float], dict[str, str]]:
    """Segregate invalid `hotwords` from the valid ones, explicitly stating the reason for
    the latter.

    Four ways an entry is unusable, all silent failures if unchecked:

    * shorter than MIN_HOTWORD_LEN -- the partial-credit term reaches full
      weight after two or three characters, so a short hotword hijacks every
      word that merely starts the same way ("ош" would boost "ошибка").
    * a space or hyphen -- credit is banked at word boundaries, so a
      multi-token entry can never complete. Write "кокжар", not "кок-жар";
      boost a single distinctive token instead of "Айыл Банк".
    * a character outside the vocabulary, e.g. Uzbek ў / ҳ.
    * a non-finite weight number up to `max_weight` --
      relevant only for weights coming from a request; a huge or NaN/inf
      weight would otherwise dominate `_rank` regardless of the acoustics.
    """
    chars, ok, rejected = set(vocab), {}, {}
    for word, weight in hotwords.items():
        w = word.lower()
        if len(w) < MIN_HOTWORD_LEN:
            rejected[word] = f"shorter than {MIN_HOTWORD_LEN} characters"
        elif " " in w or "-" in w:
            rejected[word] = "contains a space or hyphen (not a single token)"
        elif set(w) - chars:
            rejected[word] = f"not in vocabulary: {''.join(sorted(set(w) - chars))}"
        elif not isinstance(weight, (int, float)) or not math.isfinite(weight) or not (0 < weight <= max_weight):
            rejected[word] = f"weight must be a finite number in (0, {max_weight}]"
        else:
            ok[w] = weight
    return ok, rejected


def sanitize_hotwords(
    hotwords: dict[str, float], vocab: Sequence[str], verbose: bool = True
) -> dict[str, float]:
    """Drop hotwords that cannot work, saying why. See `validate_hotwords`."""
    ok, rejected = validate_hotwords(hotwords, vocab)
    if verbose:
        print(f"hotwords: {len(ok)} accepted, {len(rejected)} rejected")
        for word, why in rejected.items():
            print(f"  rejected {word!r}: {why}")
    return ok
