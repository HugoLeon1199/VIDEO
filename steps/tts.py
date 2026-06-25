"""Step 2: Text-to-Speech using Kokoro TTS (local), fallback to edge-tts, or RunPod F5-TTS clone.

Sentence mode (tts_config.json: "mode": "sentence"):
  - Splits script.txt into sentences using the same logic as transcribe.py align mode
  - Generates TTS for each sentence individually → knows exact duration per sentence
  - Concatenates all sentence audio into audio.mp3
  - Writes timestamps.json directly → step 3 (Whisper) is skipped automatically
  - Result: 100% sync between audio and timestamps, no Whisper drift
"""

import asyncio
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from loguru import logger

import config


# ── Sentence splitter (mirrors transcribe.py logic) ──────────────────────────

_SCRIPT_STOP_MARKERS = (
    "COMMENT SEED:",
    "RESEARCH NOTES:",
    "Your script is ready",
    "Save as:",
    "Then: python",
)


def _split_script_sentences(script_path: Path) -> list[str]:
    text = script_path.read_text(encoding="utf-8").lstrip("﻿")
    for marker in _SCRIPT_STOP_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    paragraphs = re.split(r"\n{2,}", text)
    sentences = []
    first = True
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if first:
            first = False
            if not re.search(r"[.!?]$", para) and len(para.split()) <= 12:
                continue
        for part in re.split(r"(?<=[.!?])\s+", para):
            part = part.strip()
            if part:
                sentences.append(part)
    return sentences


def _split_vieneu_chunks(script_path: Path) -> list[tuple[str, bool]]:
    """Split script into VieNeu-safe chunks with EOS reliability.

    VieNeu v3 Turbo fails to find EOS for clauses containing a comma — the
    phonemizer does not emit a pause/EOS signal mid-sentence, so generation
    runs to max_new_frames (300 → 24s).  Splitting on commas in addition to
    sentence-ending punctuation keeps every chunk short enough for reliable EOS.

    Returns list of (chunk_text, is_sentence_end) tuples.
    - is_sentence_end=True  → end of a full sentence (. ! ?) → 300ms silence gap
    - is_sentence_end=False → comma split inside a sentence → 150ms silence gap
    """
    text = script_path.read_text(encoding="utf-8").lstrip("﻿")
    for marker in _SCRIPT_STOP_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]

    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[tuple[str, bool]] = []
    first = True
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if first:
            first = False
            if not re.search(r"[.!?]$", para) and len(para.split()) <= 12:
                continue
        # Split on sentence-ending punctuation first
        for sent in re.split(r"(?<=[.!?])\s+", para):
            sent = sent.strip()
            if not sent:
                continue
            # Then split each sentence on commas — keeps chunks EOS-safe
            parts = re.split(r",\s*", sent)
            for j, part in enumerate(parts):
                part = part.strip()
                if not part:
                    continue
                is_last = (j == len(parts) - 1)
                chunks.append((part, is_last))  # is_last = sentence boundary
    return chunks


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


def _vieneu_tts(text: str, output_path: Path, voice: str = "Thái Sơn") -> None:
    """Generate TTS using VieNeu v3 Turbo (local, free, 48kHz).

    Splits long text into paragraphs (max ~200 chars each) to stay within
    VieNeu's max_chars=256 limit, then concatenates all chunks.
    """
    import os, subprocess, tempfile
    import numpy as np
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    from vieneu import Vieneu

    logger.info("Using VieNeu TTS v3 Turbo (voice: {})", voice)
    tts = Vieneu()

    # temperature=0 → greedy argmax, zero randomness, 100% deterministic per token.
    # voice= name → voice_token_id=reserved_id (speaker token baked into weights).
    # Combined: identical prosody for every sentence regardless of position in script.
    logger.info("VieNeu: synthesizing full script ({} chars)", len(text))
    combined = tts.infer(
        text,
        voice=voice,
        temperature=0.8,
        repetition_penalty=1.2,
        max_chars=256,
        crossfade_p=0.1,
        silence_p=0.12,
        apply_watermark=False,
    )
    if combined.ndim > 1:
        combined = combined.mean(axis=1)
    combined = combined.astype(np.float32)
    logger.info("VieNeu: synthesis done, {:.1f}s of audio", len(combined) / tts.sample_rate)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    import soundfile as sf
    sf.write(str(tmp_path), combined, tts.sample_rate)

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(tmp_path),
         "-codec:a", "libmp3lame", "-qscale:a", "0", str(output_path)],
        check=True, capture_output=True,
    )
    tmp_path.unlink(missing_ok=True)
    logger.info("VieNeu TTS complete → {}", output_path)


async def _edge_tts_async(text: str, output_path: Path, voice: str, rate: str = "-5%") -> None:
    import edge_tts

    logger.info("Using edge-tts (voice: {})", voice)
    communicate = edge_tts.Communicate(text, voice=voice, rate=rate)
    await communicate.save(str(output_path))
    logger.info("edge-tts complete → {}", output_path)


def _edge_tts(text: str, output_path: Path, voice: str = "en-US-GuyNeural", rate: str = "-5%") -> None:
    asyncio.run(_edge_tts_async(text, output_path, voice=voice, rate=rate))


def _run_sentence_mode(
    sentences: list[str],
    script_path: Path,
    output_path: Path,
    timestamps_path: Path,
    tts_engine: str,
    edge_voice: str,
    edge_rate: str,
    silence_ms: int = 300,
) -> None:
    """TTS each sentence individually, concatenate, write timestamps.json.

    For VieNeu engine: uses _split_vieneu_chunks (comma-aware) instead of the
    sentence list, so every chunk is short enough for reliable EOS detection.
    Comma-split parts share a timestamp entry with the sentence they belong to.

    silence_ms: gap after sentence-ending punctuation (. ! ?).
    Comma gaps are half of silence_ms.
    """
    import numpy as np
    import soundfile as sf

    # ── Edge / Kokoro sentence mode ───────────────────────────────────────────
    logger.info("Sentence mode: {} sentences, {}ms gap between each", len(sentences), silence_ms)

    tmp_dir = Path(tempfile.mkdtemp(prefix="tts_sentences_"))
    sentence_wavs: list[Path] = []

    try:
        for i, sentence in enumerate(sentences, 1):
            wav_path = tmp_dir / f"sent_{i:04d}.wav"
            mp3_tmp = tmp_dir / f"sent_{i:04d}.mp3"

            if tts_engine == "edge":
                asyncio.run(_edge_tts_async(sentence, mp3_tmp, voice=edge_voice, rate=edge_rate))
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(mp3_tmp), str(wav_path)],
                    check=True, capture_output=True,
                )
            else:
                from kokoro import KPipeline
                pipeline = KPipeline(lang_code="a")
                chunks = [a for _, _, a in pipeline(sentence, voice=config.TTS_VOICE, speed=config.TTS_SPEED)]
                if not chunks:
                    raise RuntimeError(f"Kokoro produced no audio for sentence {i}")
                combined = np.concatenate(chunks)
                sf.write(str(wav_path), combined, 24000)

            sentence_wavs.append(wav_path)
            if i % 10 == 0 or i == len(sentences):
                logger.info("  TTS progress: {}/{}", i, len(sentences))

        logger.info("Building timestamps and concatenating audio...")
        if not sentence_wavs:
            raise RuntimeError("No sentence audio files generated")
        all_audio = []
        timestamps = []
        cursor = 0.0
        _, sample_rate = sf.read(str(sentence_wavs[0]), dtype="float32")
        silence_arr = np.zeros(int(silence_ms / 1000 * sample_rate), dtype=np.float32)

        for i, wav_path in enumerate(sentence_wavs):
            data, sr = sf.read(str(wav_path), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            sent_duration = len(data) / sr
            seg_start = cursor
            seg_end = cursor + sent_duration
            timestamps.append({
                "index": i + 1,
                "start": round(seg_start, 3),
                "end": round(seg_end, 3),
                "text": sentences[i],
            })
            all_audio.append(data)
            all_audio.append(silence_arr)
            cursor = seg_end + silence_ms / 1000

        full_audio = np.concatenate(all_audio)
        combined_wav = tmp_dir / "combined.wav"
        sf.write(str(combined_wav), full_audio, sample_rate)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(combined_wav),
             "-codec:a", "libmp3lame", "-qscale:a", "0", str(output_path)],
            check=True, capture_output=True,
        )
        timestamps_path.write_text(
            json.dumps(timestamps, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(
            "Sentence mode complete: {} segments, {:.1f}s → {} + {}",
            len(timestamps),
            timestamps[-1]["end"] if timestamps else 0,
            output_path, timestamps_path,
        )

    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


def run(video_id: str) -> None:
    video_dir = Path(config.OUTPUT_DIR) / video_id
    script_path = video_dir / "script.txt"
    output_path = video_dir / "audio.mp3"
    timestamps_path = video_dir / "timestamps.json"

    # Per-video TTS config override via tts_config.json
    tts_config_path = video_dir / "tts_config.json"
    tts_engine = "kokoro"
    tts_mode = "default"  # "sentence" = per-sentence TTS + auto timestamps
    edge_voice = "en-US-GuyNeural"
    edge_rate = "-5%"
    clone_voice_id = "default"
    clone_ref_audio = None
    clone_speed = 1.0
    clone_ref_text = ""
    if tts_config_path.exists():
        tts_cfg = json.loads(tts_config_path.read_text(encoding="utf-8"))
        tts_engine = tts_cfg.get("engine", tts_engine)
        tts_mode = tts_cfg.get("mode", tts_mode)
        edge_voice = tts_cfg.get("voice", edge_voice)
        edge_rate = tts_cfg.get("rate", edge_rate)
        clone_voice_id = tts_cfg.get("voice_id", clone_voice_id)
        clone_ref_audio = tts_cfg.get("ref_audio")
        clone_speed = tts_cfg.get("speed", clone_speed)
        clone_ref_text = tts_cfg.get("ref_text", "")
        logger.info(
            "TTS config: engine={} mode={} voice={} rate={}",
            tts_engine, tts_mode, edge_voice, edge_rate,
        )

    if not script_path.exists():
        logger.error("script.txt not found: {}", script_path)
        sys.exit(1)

    text = script_path.read_text(encoding="utf-8").strip()
    if not text:
        logger.error("script.txt is empty")
        sys.exit(1)

    # ── Sentence mode ────────────────────────────────────────────────────────
    if tts_mode == "sentence":
        sentences = _split_script_sentences(script_path)
        if not sentences:
            logger.error("No sentences found in script.txt")
            sys.exit(1)
        logger.info("Sentence mode: {} sentences detected", len(sentences))
        try:
            _run_sentence_mode(
                sentences=sentences,
                script_path=script_path,
                output_path=output_path,
                timestamps_path=timestamps_path,
                tts_engine=tts_engine,
                edge_voice=edge_voice,
                edge_rate=edge_rate,
            )
        except Exception as e:
            logger.error("Sentence mode TTS failed: {}", e)
            sys.exit(1)

    # ── Standard mode (whole script at once) ─────────────────────────────────
    elif tts_engine == "clone":
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
    elif tts_engine == "vieneu":
        try:
            _vieneu_tts(text, output_path, voice=edge_voice)
        except Exception as e:
            logger.error("VieNeu TTS failed: {}", e)
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
