"""
Prefix beam search for GigaAM's CTC head, with hotword boosting.

An improved replacement for `CTCGreedyDecoding` (third_party/GigaAM/gigaam/decoding.py:47) --
same `decode(head, encoded, lengths)` signature, same
`(text, token_ids, token_frames)` return, so `word_timestamps=True` keeps
working:

    import gigaam
    from ctc_beam import CTCBeamDecoding

    m = gigaam.load_model("multilingual_ctc")
    m.decoding = CTCBeamDecoding(
        m.decoding.tokenizer.vocab,
        beam_width=25,
        hotwords={"kcell": 6.0, "билайн": 6.0},
    )
    m.transcribe(path).text

`set_hotwords(...)` swaps the active hotword list on an existing instance --
useful when a caller wants to boost different words per request without
rebuilding the decoder.

Why this exists: CTC assumes frames are conditionally independent given the
audio (P(X)=P(X|Y)), so the head has almost no language modeling capabilities.
Greedy decoding returns the best *alignment* for the current frame;
summing alignments per prefix returns the best *label sequence*,
and -- more usefully -- gives you hypotheses to re-rank with
knowledge the acoustic model cannot have.
"""

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

NEG_INF = -1e30


def _logaddexp(a: float, b: float) -> float:
    if a < b:
        a, b = b, a
    if b <= NEG_INF:
        return a
    return a + np.log1p(np.exp(b - a))


class _Beam:
    """One prefix, with blank/non-blank mass kept apart (Hannun et al., 2014)."""

    __slots__ = ("p_b", "p_nb", "ids", "frames", "word", "hw", "hw_part", "lm", "n_words")

    def __init__(self):
        self.p_b = NEG_INF      # log P(prefix, alignment ends in blank)
        self.p_nb = NEG_INF     # log P(prefix, alignment ends in the last symbol)
        self.ids: tuple[int, ...] = ()
        self.frames: tuple[int, ...] = ()
        self.word = ""          # characters since the last space
        self.hw = 0.0           # bonus banked on completed hotwords
        self.hw_part = 0.0      # provisional credit, refunded if the word does not complete
        self.lm = 0.0           # accumulated log10 LM score
        self.n_words = 0

    def total(self) -> float:
        return _logaddexp(self.p_b, self.p_nb)


class CTCBeamDecoding:
    def __init__(
        self,
        vocabulary: Sequence[str],
        beam_width: int = 25,
        top_k: int = 8,
        prune_logp: float = -12.0,
        hotwords: dict[str, float] | None = None,
        lm=None,                 # kenlm.Model, or None
        alpha: float = 0.5,      # LM weight
        trace=None,              # optional callback(t, beams, self) for debugging
        beta: float = 0.0,       # word insertion bonus; only meaningful with an LM
                                 # (set ~1.5 alongside KenLM to offset its
                                 #  preference for shorter outputs; leave at 0
                                 #  without one, or it just buys word boundaries)
    ):
        self.vocab = list(vocabulary)
        self.blank_id = len(self.vocab)
        self.space_id = self.vocab.index(" ")
        self.beam_width = beam_width
        self.top_k = top_k
        self.prune_logp = prune_logp
        self.set_hotwords(hotwords)
        self.lm, self.alpha, self.beta = lm, alpha, beta
        self.trace = trace
        self.last_beams: dict = {}   # final beam set, kept for inspection

        # Kept so `model.decoding.tokenizer` still resolves for callers that use it.
        from gigaam.decoding import Tokenizer

        self.tokenizer = Tokenizer(self.vocab, None)

    def set_hotwords(self, hotwords: dict[str, float] | None) -> None:
        """
        Specify hotwords. Passing `None` or `{}` clears it, nullifying any boosting
        """
        self.hotwords = {w.lower(): float(v) for w, v in (hotwords or {}).items()}

    def _hotword_partial(self, word: str) -> float:
        """Provisional credit so a half-spelled hotword survives pruning. It is
        proportional to how much of the hotword is matched, and it is NOT
        banked -- `_hotword_finish` refunds it unless the word completes.
        Without the refund the credit leaks to any word merely sharing a
        prefix (boosting one hotword could split another into two words)."""
        if not self.hotwords or not word:
            return 0.0
        best = 0.0
        for hw, weight in self.hotwords.items():
            if hw.startswith(word):
                best = max(best, weight * len(word) / len(hw))
        return best

    def _hotword_finish(self, word: str) -> float:
        """Banked at a word boundary: the full weight if the word is a hotword,
        nothing otherwise."""
        return self.hotwords.get(word, 0.0) if self.hotwords else 0.0

    def _lm_delta(self, history: tuple[str, ...], word: str) -> float:
        if self.lm is None or not word:
            return 0.0
        ctx = " ".join(history[-4:] + (word,))
        prev = " ".join(history[-4:])
        return self.lm.score(ctx, bos=False, eos=False) - (
            self.lm.score(prev, bos=False, eos=False) if prev else 0.0
        )

    def _rank(self, b: _Beam) -> float:
        return (b.total() + self.alpha * b.lm + self.beta * b.n_words
                + b.hw + b.hw_part)

    def _decode_one(self, lp: np.ndarray) -> tuple[str, list[int], list[int]]:
        init = _Beam()
        init.p_b = 0.0
        beams: dict[tuple[int, ...], _Beam] = {(): init}

        for t in range(lp.shape[0]):
            frame = lp[t]
            k = min(self.top_k, frame.shape[0] - 1)
            cand = np.argpartition(-frame, k)[: k]
            cand = [int(c) for c in cand if frame[c] > self.prune_logp]
            if self.blank_id not in cand:
                cand.append(self.blank_id)

            nxt: dict[tuple[int, ...], _Beam] = {}

            def get(ids, parent) -> _Beam:
                b = nxt.get(ids)
                if b is None:
                    b = _Beam()
                    b.ids, b.frames = ids, parent.frames
                    b.word, b.hw, b.hw_part, b.lm, b.n_words = (
                        parent.word, parent.hw, parent.hw_part,
                        parent.lm, parent.n_words)
                    nxt[ids] = b
                return b

            for ids, beam in beams.items():
                last = ids[-1] if ids else None
                for c in cand:
                    p = float(frame[c])

                    if c == self.blank_id:
                        nb = get(ids, beam)
                        nb.p_b = _logaddexp(nb.p_b, beam.total() + p)
                        continue

                    if c == last:
                        # repeat with no blank between -> collapses, same prefix
                        nb = get(ids, beam)
                        nb.p_nb = _logaddexp(nb.p_nb, beam.p_nb + p)
                        # blank between -> genuine doubled character
                        src = beam.p_b
                    else:
                        src = beam.total()
                    if src <= NEG_INF:
                        continue

                    new_ids = ids + (c,)
                    nb = get(new_ids, beam)
                    if not nb.frames or len(nb.frames) < len(new_ids):
                        nb.frames = beam.frames + (t,)
                    nb.p_nb = _logaddexp(nb.p_nb, src + p)

                    if c == self.space_id:
                        if beam.word:
                            nb.lm = beam.lm + self._lm_delta(
                                tuple(self.tokenizer.decode(list(ids)).split()),
                                beam.word)
                            nb.n_words = beam.n_words + 1
                            nb.hw = beam.hw + self._hotword_finish(beam.word)
                        nb.hw_part = 0.0        # refund whatever did not complete
                        nb.word = ""
                    else:
                        nb.word = beam.word + self.vocab[c]
                        nb.hw_part = self._hotword_partial(nb.word)

            beams = dict(sorted(nxt.items(), key=lambda kv: -self._rank(kv[1]))
                         [: self.beam_width])
            if self.trace is not None:
                self.trace(t, beams, self)

        self.last_beams = beams
        best = max(beams.values(), key=self._rank)
        ids = list(best.ids)
        return self.tokenizer.decode(ids), ids, list(best.frames)

    @torch.inference_mode()
    def decode(self, head, encoded: Tensor, lengths: Tensor):
        log_probs = head(encoder_output=encoded).float().cpu().numpy()
        lens = lengths.clamp(min=0, max=log_probs.shape[1]).cpu().tolist()
        return [self._decode_one(log_probs[i, : lens[i]])
                for i in range(log_probs.shape[0])]
