"""
RunPod Serverless handler — F5-TTS voice cloning.

Two modes:
  - "clone": generate speech cloning voice_id. Pass ref_audio_base64 on first call;
             subsequent calls load cached WAV from volume.
  - "save_ref": save reference audio to volume for voice_id (no generation).

Model lazy-loaded on first job. Reference WAV cached on volume per voice_id.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import subprocess
import time
from pathlib import Path

import runpod

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("tts_handler")
logger.info("=== TTS worker starting ===")
print("=== TTS worker starting ===", flush=True)

_vol = Path(os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume"))
VOLUME_PATH = _vol if _vol.exists() else Path("/tmp")
REF_DIR = VOLUME_PATH / "tts_refs"
MODEL_CACHE = {}


def _load_model():
    if "tts" in MODEL_CACHE:
        return MODEL_CACHE["tts"]
    logger.info("Loading F5-TTS Vietnamese model...")
    t0 = time.time()
    from f5_tts.api import F5TTS
    from huggingface_hub import hf_hub_download

    # Download Vietnamese fine-tuned checkpoint (hynt/F5-TTS-Vietnamese-ViVoice)
    ckpt_path = hf_hub_download(
        repo_id="hynt/F5-TTS-Vietnamese-ViVoice",
        filename="model_last.pt",
        cache_dir=str(VOLUME_PATH / "hf-cache"),
    )
    logger.info("Vietnamese checkpoint: %s", ckpt_path)
    # No custom vocab — use default F5-TTS vocab
    tts = F5TTS(ckpt_file=ckpt_path)
    MODEL_CACHE["tts"] = tts
    logger.info("F5-TTS Vietnamese loaded in %.1fs", time.time() - t0)
    return tts


def _get_ref_wav(voice_id: str, ref_audio_bytes: bytes | None) -> Path:
    """Return path to cached reference WAV on volume, saving if provided."""
    REF_DIR.mkdir(parents=True, exist_ok=True)
    ref_wav = REF_DIR / f"{voice_id}.wav"

    if ref_audio_bytes is not None:
        # Save incoming bytes (mp3/wav/any) as WAV 24kHz mono via ffmpeg
        tmp_in = Path(f"/tmp/ref_in_{voice_id}")
        tmp_in.write_bytes(ref_audio_bytes)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(tmp_in), "-ar", "24000", "-ac", "1", str(ref_wav)],
            check=True, capture_output=True,
        )
        tmp_in.unlink(missing_ok=True)
        logger.info("Saved ref WAV: %s (%d KB)", ref_wav, ref_wav.stat().st_size // 1024)

    if not ref_wav.exists():
        raise ValueError(
            f"No cached reference audio for voice_id='{voice_id}'. "
            "Send ref_audio_base64 on first call."
        )
    return ref_wav


def handler(job: dict) -> dict:
    job_input = job.get("input", {})
    t_start = time.time()

    mode = job_input.get("mode", "clone")
    voice_id = job_input.get("voice_id", "default")
    text = job_input.get("text", "")
    ref_text = job_input.get("ref_text", "")   # transcript of reference audio
    speed = float(job_input.get("speed", 1.0))
    ref_audio_b64 = job_input.get("ref_audio_base64")

    if mode not in ("clone", "save_ref"):
        return {"error": f"Invalid mode '{mode}'. Use 'clone' or 'save_ref'."}

    ref_audio_bytes = base64.b64decode(ref_audio_b64) if ref_audio_b64 else None

    try:
        ref_wav = _get_ref_wav(voice_id, ref_audio_bytes)

        if mode == "save_ref":
            return {
                "voice_id": voice_id,
                "mode": "save_ref",
                "status": "ok",
                "ref_path": str(ref_wav),
                "duration_seconds": round(time.time() - t_start, 2),
            }

        # mode == clone
        if not text:
            return {"error": "text is required for clone mode"}

        tts = _load_model()

        logger.info("Generating: voice_id=%s chars=%d speed=%.2f", voice_id, len(text), speed)
        t_gen = time.time()

        wav, sr, _ = tts.infer(
            ref_file=str(ref_wav),
            ref_text=ref_text,
            gen_text=text,
            speed=speed,
        )

        # wav → MP3 bytes
        import numpy as np, soundfile as sf
        wav_np = wav.squeeze().cpu().numpy() if hasattr(wav, "cpu") else np.array(wav).squeeze()
        buf = io.BytesIO()
        sf.write(buf, wav_np, sr, format="MP3")
        audio_bytes = buf.getvalue()

        gen_seconds = round(time.time() - t_gen, 2)
        logger.info("Done: %d chars -> %d bytes in %.2fs", len(text), len(audio_bytes), gen_seconds)

        return {
            "voice_id": voice_id,
            "mode": "clone",
            "audio_base64": base64.b64encode(audio_bytes).decode(),
            "audio_mime": "audio/mp3",
            "char_count": len(text),
            "generation_seconds": gen_seconds,
            "duration_seconds": round(time.time() - t_start, 2),
        }

    except Exception as e:
        logger.error("Handler error: %s", e, exc_info=True)
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
