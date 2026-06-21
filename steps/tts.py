"""Step 2: Text-to-Speech using Kokoro TTS (local), fallback to edge-tts, or RunPod F5-TTS clone."""

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
        ["ffmpeg", "-y", "-i", str(tmp_path), "-codec:a", "libmp3lame", "-qscale:a", "0", str(output_path)],
        check=True,
        capture_output=True,
    )
    tmp_path.unlink(missing_ok=True)
    logger.info("Kokoro TTS complete → {}", output_path)


async def _edge_tts_async(text: str, output_path: Path, voice: str, rate: str = "-5%") -> None:
    import edge_tts

    logger.info("Using edge-tts (voice: {})", voice)
    communicate = edge_tts.Communicate(text, voice=voice, rate=rate)
    await communicate.save(str(output_path))
    logger.info("edge-tts complete → {}", output_path)


def _edge_tts(text: str, output_path: Path, voice: str = "en-US-GuyNeural", rate: str = "-5%") -> None:
    asyncio.run(_edge_tts_async(text, output_path, voice=voice, rate=rate))


def run(video_id: str) -> None:
    video_dir = Path(config.OUTPUT_DIR) / video_id
    script_path = video_dir / "script.txt"
    output_path = video_dir / "audio.mp3"

    # Per-video TTS config override via tts_config.json
    tts_config_path = video_dir / "tts_config.json"
    tts_engine = "kokoro"
    edge_voice = "en-US-GuyNeural"
    edge_rate = "-5%"
    clone_voice_id = "default"
    clone_ref_audio = None  # path to reference audio for first-time embed
    clone_speed = 1.0
    if tts_config_path.exists():
        import json
        tts_cfg = json.loads(tts_config_path.read_text(encoding="utf-8"))
        tts_engine = tts_cfg.get("engine", tts_engine)
        edge_voice = tts_cfg.get("voice", edge_voice)
        edge_rate = tts_cfg.get("rate", edge_rate)
        clone_voice_id = tts_cfg.get("voice_id", clone_voice_id)
        clone_ref_audio = tts_cfg.get("ref_audio")  # absolute path or relative to video_dir
        clone_speed = tts_cfg.get("speed", clone_speed)
        clone_ref_text = tts_cfg.get("ref_text", "")
        logger.info("TTS config: engine={} voice={} rate={}", tts_engine, edge_voice, edge_rate)

    if not script_path.exists():
        logger.error("script.txt not found: {}", script_path)
        sys.exit(1)

    text = script_path.read_text(encoding="utf-8").strip()
    if not text:
        logger.error("script.txt is empty")
        sys.exit(1)

    logger.info("Generating TTS for {} words...", len(text.split()))

    if tts_engine == "clone":
        try:
            from tts_generation.runpod_tts_client import clone_voice
            ref_path = None
            if clone_ref_audio:
                p = Path(clone_ref_audio)
                ref_path = p if p.is_absolute() else video_dir / p
            logger.info("Using F5-TTS clone (voice_id={}, ref={})", clone_voice_id, ref_path)
            audio_bytes = clone_voice(text, voice_id=clone_voice_id, ref_audio_path=ref_path, ref_text=clone_ref_text, speed=clone_speed)
            output_path.write_bytes(audio_bytes)
            logger.info("F5-TTS clone complete → {}", output_path)
        except Exception as e:
            logger.error("F5-TTS clone failed: {}", e)
            sys.exit(1)
    elif tts_engine == "edge":
        try:
            _edge_tts(text, output_path, voice=edge_voice, rate=edge_rate)
        except Exception as e:
            logger.error("edge-tts failed: {}", e)
            sys.exit(1)
    else:
        try:
            _kokoro_tts(text, output_path)
        except (ImportError, Exception) as e:
            logger.warning("Kokoro TTS failed ({}), falling back to edge-tts", e)
            try:
                _edge_tts(text, output_path, voice=edge_voice, rate=edge_rate)
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
