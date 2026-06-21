"""
RunPod Serverless TTS client — F5-TTS voice cloning.

First call: pass ref_audio_path to upload reference to volume.
Later calls: omit ref_audio_path, uses cached WAV on volume.

ref_text: transcript of the reference audio (optional but improves quality).
          Leave empty to let F5-TTS auto-transcribe on Linux.
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path

import httpx

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
TTS_ENDPOINT_ID = os.environ.get("RUNPOD_TTS_ENDPOINT_ID", "")

_BASE = "https://api.runpod.ai/v2"
_POLL_INTERVAL = 2
_TIMEOUT = 300


def _headers() -> dict:
    return {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}


def _submit(payload: dict) -> str:
    url = f"{_BASE}/{TTS_ENDPOINT_ID}/run"
    r = httpx.post(url, json={"input": payload}, headers=_headers(), timeout=30)
    r.raise_for_status()
    return r.json()["id"]


def _poll(job_id: str) -> dict:
    url = f"{_BASE}/{TTS_ENDPOINT_ID}/status/{job_id}"
    t0 = time.time()
    while time.time() - t0 < _TIMEOUT:
        r = httpx.get(url, headers=_headers(), timeout=30)
        r.raise_for_status()
        data = r.json()
        status = data.get("status")
        if status == "COMPLETED":
            return data.get("output", {})
        if status == "FAILED":
            raise RuntimeError(f"TTS job failed: {data.get('error')}")
        time.sleep(_POLL_INTERVAL)
    raise TimeoutError(f"TTS job {job_id} timed out after {_TIMEOUT}s")


def save_ref(voice_id: str, ref_audio_path: str | Path) -> None:
    """Upload reference audio to RunPod volume (call once per voice)."""
    ref_bytes = Path(ref_audio_path).read_bytes()
    job_id = _submit({
        "mode": "save_ref",
        "voice_id": voice_id,
        "ref_audio_base64": base64.b64encode(ref_bytes).decode(),
    })
    result = _poll(job_id)
    if "error" in result:
        raise RuntimeError(f"save_ref failed: {result['error']}")
    print(f"[TTS] Reference saved on volume: {result.get('ref_path')}")


def clone_voice(
    text: str,
    voice_id: str,
    ref_audio_path: str | Path | None = None,
    ref_text: str = "",
    speed: float = 1.0,
) -> bytes:
    """
    Generate audio cloning voice_id. Returns MP3 bytes.
    Pass ref_audio_path on the very first call to upload reference.
    Subsequent calls omit it — uses cached WAV on volume.
    """
    payload: dict = {
        "mode": "clone",
        "voice_id": voice_id,
        "text": text,
        "ref_text": ref_text,
        "speed": speed,
    }
    if ref_audio_path:
        payload["ref_audio_base64"] = base64.b64encode(
            Path(ref_audio_path).read_bytes()
        ).decode()

    job_id = _submit(payload)
    result = _poll(job_id)

    if "error" in result:
        raise RuntimeError(f"TTS clone failed: {result['error']}")

    audio_b64 = result.get("audio_base64")
    if not audio_b64:
        raise RuntimeError("No audio_base64 in TTS response")

    return base64.b64decode(audio_b64)
