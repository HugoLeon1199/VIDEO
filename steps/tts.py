"""Step 2: Text-to-Speech using Kokoro TTS (local), fallback to edge-tts."""

import asyncio
import subprocess
import sys
import tempfile
from pathlib import Path

from loguru import logger

import config


def _get_audio_duration(audio_path: Path) -> float:
    """Return duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
        )
        return float(result.stdout.strip())
    except (ValueError, FileNotFoundError):
        return 0.0


def _kokoro_tts(text: str, output_path: Path) -> None:
    from kokoro import KPipeline
    import numpy as np
    import soundfile as sf

    logger.info("Using Kokoro TTS (voice: {})", config.TTS_VOICE)
    pipeline = KPipeline(lang_code="a")  # American English

    audio_chunks = []
    for _, _, audio in pipeline(text, voice=config.TTS_VOICE, speed=config.TTS_SPEED):
        audio_chunks.append(audio)

    if not audio_chunks:
        raise RuntimeError("Kokoro produced no audio output")

    combined = np.concatenate(audio_chunks)
    sample_rate = 24000  # Kokoro default

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    sf.write(str(tmp_path), combined, sample_rate)

    # Convert WAV → MP3 via ffmpeg
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(tmp_path), "-codec:a", "libmp3lame", "-qscale:a", "2", str(output_path)],
        check=True,
        capture_output=True,
    )
    tmp_path.unlink(missing_ok=True)
    logger.info("Kokoro TTS complete → {}", output_path)


async def _edge_tts_async(text: str, output_path: Path) -> None:
    import edge_tts

    logger.info("Using edge-tts fallback (voice: en-US-GuyNeural)")
    communicate = edge_tts.Communicate(text, voice="en-US-GuyNeural", rate="-5%")
    await communicate.save(str(output_path))
    logger.info("edge-tts complete → {}", output_path)


def _edge_tts(text: str, output_path: Path) -> None:
    asyncio.run(_edge_tts_async(text, output_path))


def run(video_id: str) -> None:
    video_dir = Path(config.OUTPUT_DIR) / video_id
    script_path = video_dir / "script.txt"
    output_path = video_dir / "audio.mp3"

    if not script_path.exists():
        logger.error("script.txt not found: {}", script_path)
        sys.exit(1)

    text = script_path.read_text(encoding="utf-8").strip()
    if not text:
        logger.error("script.txt is empty")
        sys.exit(1)

    logger.info("Generating TTS for {} words...", len(text.split()))

    try:
        _kokoro_tts(text, output_path)
    except (ImportError, Exception) as e:
        logger.warning("Kokoro TTS failed ({}), falling back to edge-tts", e)
        try:
            _edge_tts(text, output_path)
        except Exception as e2:
            logger.error("edge-tts also failed: {}", e2)
            sys.exit(1)

    if output_path.exists():
        duration = _get_audio_duration(output_path)
        minutes = duration / 60
        logger.info("Audio duration: {:.1f} minutes ({:.0f}s)", minutes, duration)
        if minutes < 7:
            logger.warning("Audio is short ({:.1f} min). Target: 8–10 min.", minutes)
        elif minutes > 11:
            logger.warning("Audio is long ({:.1f} min). Target: 8–10 min.", minutes)
    else:
        logger.error("TTS output file not created: {}", output_path)
        sys.exit(1)
