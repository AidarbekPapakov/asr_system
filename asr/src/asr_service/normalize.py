"""
LLM normalisation.

LLM is used because transcript normalisation is not a completely solved NLP
problem even for high-resource languages, and less so for low-resource ones
and code-switched speech. A dedicated model can replace it later; for now a
general-purpose LLM does the job.

One job: punctuation, capitalisation, inverse text normalisation, and repair
of CTC splits/merges VAD windowing could not fix. The model's answer is
applied only if it round-trips the turn ids; otherwise the raw transcript
survives -- see `run_normalization`, which never raises; a refusal or a
failure is reported in `meta.llm`, not thrown.

This reaches `turns[].text`, never the word-level `channels[].words` -- ITN
and split/merge repair change token counts by design, and a caller that
indexes words by position would silently drift. `turns[].text_raw` always
keeps the CTC original.

External-LLM guard: with an LLM_BASE_URL that is neither loopback nor listed
in LLM_INTERNAL_HOSTS, the raw transcript leaves the machine unmasked (PII
masking is not in this build). Refused unless allow_external=True
(LLM_ALLOW_EXTERNAL).
"""

import json
import logging
import re
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import httpx

from asr_service import config
from asr_service.turns import Turn

logger = logging.getLogger("asr_service.normalize")

SYSTEM_PROMPT = """\
You format speech-recognition transcripts. You do not rewrite them.

The input is raw CTC speech-recognition output. It has no punctuation and no
capitalisation, and it may mix languages, sometimes mid-sentence. Each turn is
one uninterrupted stretch of speech from one speaker.

You MUST:
- return every input turn exactly once, with its id unchanged;
- keep the turns in the given order.

You MUST NOT:
- merge, split, drop, reorder, or invent turns;
- move text between turns;
- translate anything, in either direction.

A word you do not recognise is far more likely to be a word from another
language, a personal name, or a brand name than an error. Leave it exactly as
written.

Within a turn, you MAY:
- add punctuation and sentence capitalisation;
- capitalise proper nouns you are confident about;
- write numbers, amounts, dates and phone numbers in figures;
- repair obvious recognition splits and merges, where the intended word is
  unambiguous from context.

If you are not confident, leave the text exactly as it is. Leaving an error in
place is correct behaviour; guessing is not.

Return ONLY JSON, and nothing else - no explanation, no code fence:
{
  "turns": [{"id": <int>, "text": "<formatted>", "lang": "<ISO 639-1 code, or mixed>"}]
}

Every key and every string is in double quotes; no trailing commas.
"""


def build_messages(turns: List[Turn], glossary: Sequence[str]) -> List[dict]:
    payload_turns = [{"id": t.id, "speaker": t.speaker, "text": t.text} for t in turns]
    user = ""
    if glossary:
        user += (
            "Names and terms that may appear -- prefer these spellings and "
            f"capitalise them properly:\n{', '.join(sorted(glossary))}\n\n"
        )
    user += f"Transcript:\n{json.dumps({'turns': payload_turns}, ensure_ascii=False, indent=1)}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def _extract_json(text: str) -> dict:
    """Parse the reply, tolerating what small self-hosted models actually
    emit (unquoted numeric keys, a trailing comma, a code fence, a <think>
    block). Raises json.JSONDecodeError if none of it helps."""
    text = _THINK_RE.sub("", text).strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last > first:
        text = text[first : last + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    repaired = re.sub(r"([{,]\s*)(\d+)(\s*:)", r'\1"\2"\3', text)
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    parsed = json.loads(repaired)
    logger.info("LLM reply was not valid JSON as-is; recovered it")
    return parsed


def validate_llm_turns(original: List[Turn], returned: dict) -> Tuple[List[Turn], bool]:
    """Apply the reply only if the ids round-trip exactly; otherwise keep
    ours. Returns (turns, applied)."""
    by_id: Dict[int, Tuple[str, Optional[str]]] = {}
    for item in returned.get("turns", []):
        try:
            by_id[int(item["id"])] = (str(item["text"]), item.get("lang"))
        except (KeyError, TypeError, ValueError):
            continue

    expected = {t.id for t in original}
    if set(by_id) != expected:
        logger.info(
            "LLM turn ids did not round-trip (missing=%s invented=%s); keeping the raw transcript",
            sorted(expected - set(by_id)) or "none",
            sorted(set(by_id) - expected) or "none",
        )
        return original, False

    updated = [
        replace(t, text=by_id[t.id][0].strip(), lang=by_id[t.id][1]) for t in original
    ]
    return updated, True


def is_external(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").lower()
    return host not in ("127.0.0.1", "localhost", "::1") + config.LLM_INTERNAL_HOSTS


@dataclass
class NormalizeResult:
    turns: List[Turn]
    ran: bool
    applied: bool
    reason: str


def run_normalization(
    turns: List[Turn],
    glossary: Sequence[str] = (),
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    allow_external: bool = False,
    timeout: float = 120.0,
) -> NormalizeResult:
    """Never raises: a broken endpoint, a refused external host, or a bad
    reply all come back as a NormalizeResult with `applied=False` and the raw
    turns intact, reported in `reason`."""
    import os

    base_url = base_url or os.getenv("LLM_BASE_URL")
    api_key = api_key or os.getenv("LLM_API_KEY")
    model_name = model_name or os.getenv("LLM_MODEL")
    missing = [n for n, v in [("LLM_BASE_URL", base_url), ("LLM_API_KEY", api_key), ("LLM_MODEL", model_name)] if not v]
    if missing:
        return NormalizeResult(turns=turns, ran=False, applied=False, reason=f"not run: missing {', '.join(missing)}")

    if is_external(base_url) and not allow_external:
        host = urlparse(base_url).hostname
        reason = (
            f"refused: LLM_BASE_URL is not local ({host}) and LLM_ALLOW_EXTERNAL "
            "is not set -- PII masking is not in this build, so the raw "
            "transcript would leave the machine unmasked"
        )
        logger.warning(reason)
        return NormalizeResult(turns=turns, ran=False, applied=False, reason=reason)

    messages = build_messages(turns, glossary)
    try:
        response = httpx.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model_name, "messages": messages, "temperature": 0,
                "response_format": {"type": "json_object"},
            },
            timeout=timeout,
        )
        response.raise_for_status()
    except Exception as exc:
        reason = f"request to LLM_BASE_URL failed: {exc}"
        logger.warning(reason)
        return NormalizeResult(turns=turns, ran=True, applied=False, reason=reason)

    raw_reply = response.json()["choices"][0]["message"]["content"]
    try:
        reply = _extract_json(raw_reply)
    except json.JSONDecodeError as exc:
        reason = f"reply was not JSON ({exc}); kept the raw transcript"
        logger.warning(reason)
        return NormalizeResult(turns=turns, ran=True, applied=False, reason=reason)

    validated_turns, applied = validate_llm_turns(turns, reply)
    reason = "applied" if applied else "turn ids did not round-trip; kept the raw transcript"
    return NormalizeResult(turns=validated_turns, ran=True, applied=applied, reason=reason)
