"""
analyze(path) -> dict. Transport only -- no business logic, no contract
checking (see contract.py for that). Used by deploy/verify.py, and by any
service that would rather import this than write the multipart POST itself.
"""

import json
import os
from typing import Any

import httpx

DEFAULT_URL = os.getenv("ASR_SERVICE_URL", "http://127.0.0.1:8400/analyze")


def analyze(
    audio_path: str,
    *,
    analyze_emotion: bool = False,
    analyze_gender: bool = False,
    run_llm: bool | None = None,
    speaker_map: dict[int, str] | None = None,
    hotwords: dict[str, float] | None = None,
    endpoint_url: str | None = None,
    timeout: float = 600.0,
) -> dict[str, Any]:
    url = endpoint_url or DEFAULT_URL
    data: dict[str, str] = {
        "analyze_emotion": "1" if analyze_emotion else "0",
        "analyze_gender": "1" if analyze_gender else "0",
    }
    if speaker_map is not None:
        data["speaker_map"] = json.dumps({str(k): v for k, v in speaker_map.items()})
    if run_llm is not None:
        data["run_llm"] = "1" if run_llm else "0"
    if hotwords is not None:
        data["hotwords"] = json.dumps(hotwords, ensure_ascii=False)

    with open(audio_path, "rb") as fh:
        files = {"audio": (os.path.basename(audio_path), fh, "application/octet-stream")}
        response = httpx.post(url, data=data, files=files, timeout=timeout)
    response.raise_for_status()

    result = response.json()
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(f"asr_service returned an error for {audio_path}: {result['error']}")
    return result

# Example usage after a successful deploy:
#   PYTHONPATH=src python -m asr_service.client samples/<file>.mp3 [...]
if __name__ == "__main__":
    import sys

    for path in sys.argv[1:]:
        print(json.dumps(analyze(path, analyze_emotion=True, run_llm=True), indent=2, ensure_ascii=False))
