#!/usr/bin/env python3
"""
Functional and system-metrics verification against a live service.

No quality checks, no WER, no transcript grading. What this checks is that the
service does not crash, its output satisfies the contract, timestamps and
turn/word bookkeeping are internally consistent, the opt-in analyses really are
opt-in, and how long each recording shape costs.

    # against an already-running server:
    python deploy/verify.py

    # start one first (blocks on /health, stops it at exit):
    python deploy/verify.py --start-server

    # also exercise the LLM pass (needs LLM_BASE_URL/LLM_API_KEY/LLM_MODEL in
    # the environment, pointed at something loopback):
    LLM_BASE_URL=... LLM_API_KEY=... LLM_MODEL=... python deploy/verify.py
"""

import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

import httpx  # noqa: E402

from asr_service import client as service_client  # noqa: E402
from asr_service import config, contract  # noqa: E402

SAMPLES_DIR = REPO_ROOT / "samples"

# (filename, human label, is the mono case)
VERIFICATION_SAMPLES: list[tuple[str, str, bool]] = [
    ("a158dec538524111b13b6731fa6a153e.mp3", "degenerate 0.7s stereo (p_min)", False),
    ("9b2584c9e8334b13aae002353676b171.mp3", "28.3s stereo (p25)", False),
    ("cac891bbacb94dd7a9b901927c11598c.mp3", "73.4s stereo (median)", False),
    ("05489f96451144c3b6857fc70d125868.mp3", "340.6s stereo (p90)", False),
    ("57cee54b9cd54132a25884f595649e90.mp3", "23.5s MONO", True),
]
MEDIAN_SAMPLE = VERIFICATION_SAMPLES[2][0]
STRESS_SAMPLE = ("5f557e8bf78949ca8b6cb1944e388768.mp3", "1040.4s stereo (stress, optional)", False)

# Word timestamps are rounded to 2 dp, so two adjacent words may report the
# same boundary; anything beyond one rounding unit is a real overlap.
ROUND_EPS = 0.011

SERVICE_URL = os.getenv("ASR_SERVICE_URL", "http://127.0.0.1:8400/analyze")
HEALTH_URL = SERVICE_URL.rsplit("/", 1)[0] + "/health"


# small helpers

class Check:
    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []

    def record(self, name: str, ok: bool, detail: str = "") -> None:
        self.results.append({"name": name, "ok": bool(ok), "detail": detail})
        mark = "PASS" if ok else "FAIL"
        suffix = f": {detail}" if detail and not ok else ""
        print(f"    [{mark}] {name}{suffix}")

    def note(self, text: str) -> None:
        print(f"    [NOTE] {text}")

    @property
    def all_ok(self) -> bool:
        return all(r["ok"] for r in self.results)


def wait_for_health(timeout: float = 420.0) -> None:
    t0 = time.time()
    last_err = None
    while time.time() - t0 < timeout:
        try:
            if httpx.get(HEALTH_URL, timeout=5.0).status_code == 200:
                return
        except Exception as exc:
            last_err = exc
        time.sleep(2.0)
    raise TimeoutError(f"{HEALTH_URL} did not become healthy within {timeout}s (last error: {last_err})")


def gpu_mem_used_mb() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5.0, check=True,
        )
        return max(float(x) for x in out.stdout.strip().splitlines())
    except Exception:
        return None


def tree_rss_mb(pid: int | None) -> float | None:
    """Resident memory of a process and every descendant. LitServe runs the
    models in worker processes, so the parent's own RSS says nothing."""
    if pid is None:
        return None
    try:
        children: dict[int, list[int]] = {}
        rss: dict[int, float] = {}
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                stat = (entry / "stat").read_text().rsplit(") ", 1)[1].split()
                children.setdefault(int(stat[1]), []).append(int(entry.name))
                rss[int(entry.name)] = int(stat[21]) * os.sysconf("SC_PAGE_SIZE") / 1024 ** 2
            except Exception:
                continue
        total, stack = 0.0, [pid]
        while stack:
            current = stack.pop()
            total += rss.get(current, 0.0)
            stack.extend(children.get(current, []))
        return round(total, 1)
    except Exception:
        return None


# functional checks

def _max_word_span(res: dict[str, Any]) -> float:
    return max(
        (w["end"] - w["start"] for ch in res["channels"] for w in ch["words"]), default=0.0
    )


def _word_order_problems(res: dict[str, Any]) -> list[str]:
    """Words must be ordered and must not overlap within a channel -- the
    externally visible consequence of vad.group_regions' non-overlap clamp,
    which is what stops the same audio being transcribed into two places."""
    problems = []
    for ch in res["channels"]:
        for a, b in zip(ch["words"], ch["words"][1:], strict=False):  # deliberately uneven: consecutive pairs
            if b["start"] < a["end"] - ROUND_EPS:
                problems.append(f"channel {ch['index']}: {a} then {b}")
    return problems


def _timestamp_problems(res: dict[str, Any], duration: float) -> list[str]:
    problems = []
    for ch in res["channels"]:
        for w in ch["words"]:
            if not (-1e-6 <= w["start"] <= w["end"] <= duration + 0.5):
                problems.append(f"channel {ch['index']}: word outside [0, {duration}]: {w}")
    for turn in res["turns"]:
        if not (-1e-6 <= turn["start"] <= turn["end"] <= duration + 0.5):
            problems.append(f"turn {turn['id']} outside [0, {duration}]: {turn['start']}-{turn['end']}")
    return problems


def _word_silence_problems(res: dict[str, Any]) -> list[str]:
    """The `васвы` guard: GigaAM once welded ~9 s of silence into a single
    "word". A flat span ceiling cannot express this -- the model legitimately
    emits no space token across short pauses -- so the invariant is that no
    word may straddle a silence wider than MERGE_GAP, which is exactly what
    group_regions promises to split on.
    """
    problems = []
    for ch in res["channels"]:
        regions = ch["vad"]["regions"]
        gaps = [
            (a[1], b[0])
            for a, b in zip(regions, regions[1:], strict=False)  # deliberately uneven: consecutive pairs
            if b[0] - a[1] > config.MERGE_GAP
        ]
        for w in ch["words"]:
            for gap_start, gap_end in gaps:
                silent = min(w["end"], gap_end) - max(w["start"], gap_start)
                if silent > config.MERGE_GAP:
                    problems.append(
                        f"channel {ch['index']}: {w['text']!r} [{w['start']}-{w['end']}] "
                        f"covers {silent:.2f}s of silence"
                    )
    return problems


def _word_turn_mismatch(res: dict[str, Any]) -> str | None:
    n_words = sum(len(ch["words"]) for ch in res["channels"])
    n_turn_words = sum(len(t["text_raw"].split()) for t in res["turns"])
    if n_words != n_turn_words:
        return f"{n_words} words across channels vs {n_turn_words} words across turns.text_raw"
    return None


def run_functional_checks(res: dict[str, Any], is_mono: bool, analyses_on: bool) -> Check:
    c = Check()
    try:
        contract.assert_contract(res)
        c.record("1  contract.assert_contract", True)
    except AssertionError as exc:
        c.record("1  contract.assert_contract", False, str(exc))

    duration = res["meta"]["duration_sec"]
    ts_problems = _timestamp_problems(res, duration)
    c.record("2  timestamps within [0, duration]", not ts_problems, "; ".join(ts_problems[:3]))

    silence_problems = _word_silence_problems(res)
    c.record(
        f"3  no word straddles silence > MERGE_GAP ({config.MERGE_GAP}s)",
        not silence_problems, "; ".join(silence_problems[:3]),
    )
    c.note(
        f"max word span {_max_word_span(res):.2f}s "
        "(reported, not asserted -- GigaAM glues words across short pauses)"
    )

    order_problems = _word_order_problems(res)
    c.record("4  words ordered and non-overlapping per channel", not order_problems, "; ".join(order_problems[:3]))

    mismatch = _word_turn_mismatch(res)
    c.record("5  every word appears in exactly one turn", mismatch is None, mismatch or "")
    ids = [t["id"] for t in res["turns"]]
    c.record("5b turn ids are 1..n, no gaps", ids == list(range(1, len(ids) + 1)), str(ids[:6]))

    if analyses_on:
        covered = all("emotion" in t and "gender" in t for t in res["turns"])
        c.record("6  every turn carries emotion and gender when requested", covered)

    if is_mono:
        diar = res["meta"]["diarization"]
        c.record(
            "8  mono input: diarized into >=1 speaker channel",
            diar["applied"] and diar["n_speakers"] >= 1, str(diar),
        )

    return c


def check_gating_and_laziness(path: Path, server_pid: int | None) -> Check:
    """Check 11. Must run FIRST, on a server that has not yet been asked for
    emotion or gender -- that is the only moment the lazy load is observable.
    """
    c = Check()

    mem_start = (tree_rss_mb(server_pid), gpu_mem_used_mb())
    res_off, wall_off = run_one(path)
    mem_after_off = (tree_rss_mb(server_pid), gpu_mem_used_mb())

    stray = [
        f"turn {t['id']}" for t in res_off["turns"] if "emotion" in t or "gender" in t
    ]
    c.record("11 flags off: no emotion/gender on any turn", not stray, "; ".join(stray[:3]))
    timed = [k for k in res_off["meta"]["timings"] if k in ("emotion", "gender")]
    c.record("11 flags off: neither analysis was timed (i.e. neither ran)", not timed, str(timed))

    res_on, wall_on_cold = run_one(path, analyze_emotion=True, analyze_gender=True)
    mem_after_on = (tree_rss_mb(server_pid), gpu_mem_used_mb())
    _res_on2, wall_on_warm = run_one(path, analyze_emotion=True, analyze_gender=True)

    same_words = [ch["words"] for ch in res_off["channels"]] == [ch["words"] for ch in res_on["channels"]]
    same_text = [t["text_raw"] for t in res_off["turns"]] == [t["text_raw"] for t in res_on["turns"]]
    c.record("11 the analyses are additive: identical words and text_raw either way", same_words and same_text)

    c.note(
        f"wall: flags off {wall_off:.2f}s | first flags-on {wall_on_cold:.2f}s "
        f"(pays the model load) | second {wall_on_warm:.2f}s"
    )
    # On a GPU the emotion checkpoint lands in VRAM, so host RSS barely moves
    # and the second number is the one that shows the lazy load; on CPU it is
    # the other way round.
    c.note(
        "memory (host RSS MB, device MB): "
        f"start {mem_start} -> after flags-off {mem_after_off} -> after flags-on {mem_after_on}"
    )
    return c


# check 7: emotion batching equivalence

def check_emotion_equivalence(sample_wav: Path) -> Check:
    """probs_batch re-implements GigaAMEmo.get_probs so a batch of segments can
    share one forward pass: masked mean-pooling over encoded_len instead of
    avg_pool1d over the whole clip. Both must agree on a single clip.

    Both sides are given the waveform GigaAM's own loader produced and the same
    model instance -- get_probs decodes through ffmpeg while the service decodes
    through soundfile/librosa, and comparing across decoders measures mp3
    decoding, not pooling.

    Runs on CPU: it needs its own copy of the model while the server holds one,
    and the pooling arithmetic it checks is the same on either device.
    """
    c = Check()
    try:
        from gigaam.preprocess import load_audio

        from asr_service import emotion

        model = emotion.ensure_loaded("cpu")
        ref_probs = model.get_probs(str(sample_wav))
        batch_probs = emotion.probs_batch([load_audio(str(sample_wav)).numpy()])[0]

        max_diff = max(abs(ref_probs[k] - batch_probs.get(k, 0.0)) for k in ref_probs)
        c.record(
            "7  emotion.probs_batch vs GigaAMEmo.get_probs, max abs diff < 1e-3",
            max_diff < 1e-3, f"max diff {max_diff:.5f}",
        )
    except Exception as exc:                                          # noqa: BLE001
        c.record("7  emotion.probs_batch vs GigaAMEmo.get_probs", False, f"check itself failed: {exc}")
    return c


# checks 9-10: the LLM pass

def check_external_guard() -> Check:
    """In-process, no network: the server's own env is fixed at startup, so
    this exercises normalize.run_normalization directly instead."""
    from asr_service import normalize
    from asr_service.turns import Turn

    c = Check()
    fake_turns = [Turn(id=1, channel=0, speaker="speaker_0", start=0.0, end=1.0, text="алло")]
    result = normalize.run_normalization(
        fake_turns,
        base_url="https://example.com/v1", api_key="x", model_name="x",
        allow_external=False,
    )
    c.record(
        "10 external-LLM guard refuses without LLM_ALLOW_EXTERNAL",
        (not result.ran) and "refused" in result.reason, result.reason,
    )
    return c


def check_llm_normalization() -> Check | None:
    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY")
    model_name = os.getenv("LLM_MODEL")
    if not (base_url and api_key and model_name):
        print("    [SKIP] check 9 (LLM normalisation): LLM_BASE_URL/LLM_API_KEY/LLM_MODEL not set")
        return None

    from asr_service import normalize
    from asr_service.turns import Turn

    turns_in = [
        Turn(id=1, channel=0, speaker="speaker_0", start=0.0, end=1.0, text="алло добрый день"),
        Turn(id=2, channel=1, speaker="speaker_1", start=1.2, end=2.4, text="здравствуйте слушаю вас"),
    ]
    result = normalize.run_normalization(
        turns_in, base_url=base_url, api_key=api_key,
        model_name=model_name, allow_external=True, timeout=60.0,
    )
    c = Check()
    c.record("9  LLM ran", result.ran, result.reason)
    ids_in = {t.id for t in turns_in}
    ids_out = {t.id for t in result.turns}
    c.record("9  returned id set equals input id set", ids_in == ids_out, f"in={ids_in} out={ids_out}")
    c.record("9  validate_llm_turns did not fall back", result.applied, result.reason)
    raw_preserved = all(out.text_raw == orig.text for out, orig in zip(result.turns, turns_in, strict=True))
    c.record("9  text_raw preserved", raw_preserved)
    return c


# driver

def run_one(path: Path, **options: Any) -> tuple[dict[str, Any], float]:
    t0 = time.perf_counter()
    res = service_client.analyze(str(path), endpoint_url=SERVICE_URL, timeout=900.0, **options)
    return res, time.perf_counter() - t0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-server", action="store_true", help="launch asr_service.server and stop it at exit")
    parser.add_argument("--stress", action="store_true", help="also run the 1040s stress recording")
    args = parser.parse_args()

    server_proc = None
    if args.start_server:
        print(f"starting server: python -m asr_service.server (cwd={REPO_ROOT})")
        # Own process group: LitServe's inference workers do not die with the
        # parent, and an orphan holding port 8400 silently makes the next run
        # verify the previous build.
        server_proc = subprocess.Popen(
            [sys.executable, "-m", "asr_service.server"],
            cwd=str(REPO_ROOT),
            env={**os.environ, "PYTHONPATH": str(SRC)},
            start_new_session=True,
        )

    try:
        print(f"waiting for {HEALTH_URL} ...")
        wait_for_health()
        print("server is healthy\n")

        samples = list(VERIFICATION_SAMPLES)
        if args.stress:
            samples.append(STRESS_SAMPLE)

        missing = [f for f, _, _ in samples if not (SAMPLES_DIR / f).is_file()]
        if missing:
            print(f"ERROR: missing files under {SAMPLES_DIR}: {missing}")
            return 2

        # First, while emotion and gender are still unloaded.
        print("=== check 11: opt-in gating and lazy model loading ===")
        c11 = check_gating_and_laziness(SAMPLES_DIR / MEDIAN_SAMPLE, server_proc.pid if server_proc else None)
        all_ok = c11.all_ok
        print()

        rows: list[dict[str, Any]] = []
        for filename, label, is_mono in samples:
            print(f"=== {filename}  ({label}) ===")
            res, wall = run_one(SAMPLES_DIR / filename, analyze_emotion=True, analyze_gender=True)

            duration = res["meta"]["duration_sec"]
            xrt = (duration / wall) if wall > 0 else 0.0
            print(
                f"    wall={wall:.2f}s  duration={duration:.1f}s  xRT={xrt:.1f}x  "
                f"timings={json.dumps(res['meta']['timings'])}"
            )

            c = run_functional_checks(res, is_mono, analyses_on=True)
            all_ok &= c.all_ok
            rows.append({
                "file": filename, "wall_s": round(wall, 2), "duration_s": round(duration, 1),
                "xrt": round(xrt, 1), "gpu_used_mb": gpu_mem_used_mb(),
            })
            print()

        print("=== check 7: emotion batching equivalence ===")
        c7 = check_emotion_equivalence(SAMPLES_DIR / VERIFICATION_SAMPLES[0][0])
        all_ok &= c7.all_ok
        print()

        print("=== check 10: external-LLM guard ===")
        c10 = check_external_guard()
        all_ok &= c10.all_ok
        print()

        print("=== check 9: LLM normalisation (skipped if not configured) ===")
        c9 = check_llm_normalization()
        if c9 is not None:
            all_ok &= c9.all_ok
        print()

        print("=== concurrency: 4x concurrent requests against the median sample ===")
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(run_one, SAMPLES_DIR / MEDIAN_SAMPLE) for _ in range(4)]
            walls = [f.result()[1] for f in futures]
        per_request = [f"{w:.2f}" for w in walls]
        print(f"    4 concurrent requests: total wall {time.perf_counter() - t0:.2f}s, per-request {per_request}")
        print(f"    GPU memory used (whole device, best-effort): {gpu_mem_used_mb()} MB")
        print()

        print("=== system metrics summary (flags on) ===")
        for row in rows:
            print(f"  {row['file']:<38} wall={row['wall_s']:>7.2f}s  xRT={row['xrt']:>6.1f}x  gpu={row['gpu_used_mb']}")

        print()
        print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED -- see [FAIL] lines above")
        return 0 if all_ok else 1

    finally:
        if server_proc is not None:
            print("stopping server")
            group = os.getpgid(server_proc.pid)
            os.killpg(group, signal.SIGTERM)
            try:
                server_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(group, signal.SIGKILL)


if __name__ == "__main__":
    raise SystemExit(main())
