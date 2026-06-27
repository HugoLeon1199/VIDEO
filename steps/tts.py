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


def _split_script_paragraphs(script_path: Path) -> list[str]:
    text = script_path.read_text(encoding="utf-8").lstrip("\ufeff")
    for marker in _SCRIPT_STOP_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    paragraphs = re.split(r"\n{2,}", text)
    cleaned = []
    first = True
    for para in paragraphs:
        para = re.sub(r"\s+", " ", para.strip())
        if not para:
            continue
        if first:
            first = False
            if not re.search(r"[.!?]$", para) and len(para.split()) <= 12:
                continue
        cleaned.append(para)
    return cleaned


def _split_vieneu_chunks(script_path: Path, legacy: bool = False) -> list[dict]:
    """Split script into VieNeu-safe chunks with EOS reliability.

    VieNeu v3 Turbo fails to find EOS for clauses containing a comma — the
    phonemizer does not emit a pause/EOS signal mid-sentence, so generation
    runs to max_new_frames (300 → 24s).  Splitting on commas in addition to
    sentence-ending punctuation keeps every chunk short enough for reliable EOS.

    Returns a list of chunk dicts.
    - ``text`` is the original spoken clause used for timestamps/diagnostics
    - ``synth_text`` is the text actually sent to VieNeu
    - ``is_sentence_end=True``  → end of a full sentence (. ! ?) → 300ms silence gap
    - ``is_sentence_end=False`` → comma split inside a sentence → shorter comma gap

    Default mode adds a terminal period to intermediate comma chunks so VieNeu sees
    a clean EOS marker instead of an abruptly truncated fragment. Set ``legacy=True``
    to keep the old behavior for before/after comparison.
    """
    text = script_path.read_text(encoding="utf-8").lstrip("﻿")
    for marker in _SCRIPT_STOP_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]

    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[dict] = []
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
                synth_text = part
                if not is_last and not legacy and not re.search(r"[.!?…]$", synth_text):
                    synth_text = f"{synth_text}."
                chunks.append(
                    {
                        "text": part,
                        "synth_text": synth_text,
                        "is_sentence_end": is_last,
                    }
                )
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
    """LEGACY whole-script VieNeu path — NOT used by run() anymore.

    Kept for reference/manual use only.  Feeding the whole script at once makes
    VieNeu v3 Turbo run to max_new_frames (~20s of silence) after every clause
    that contains a comma.  Production VieNeu now goes through
    _run_vieneu_sentence_mode() (comma-aware chunking + synced timestamps).
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
        temperature=0.0,
        repetition_penalty=1.2,
        max_chars=1024,
        crossfade_p=0.0,
        silence_p=0.15,
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


async def _edge_tts_async(
    text: str, output_path: Path, voice: str, rate: str = "-5%",
    retries: int = 4, log: bool = True,
) -> None:
    import edge_tts

    if log:
        logger.info("Using edge-tts (voice: {})", voice)
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            communicate = edge_tts.Communicate(text, voice=voice, rate=rate)
            await communicate.save(str(output_path))
            if output_path.exists() and output_path.stat().st_size > 0:
                if log:
                    logger.info("edge-tts complete → {}", output_path)
                return
            raise RuntimeError("empty audio file")
        except Exception as e:  # transient "No audio received" / network throttling
            last_err = e
            if attempt < retries:
                wait = attempt * 1.5
                logger.warning(
                    "edge-tts attempt {}/{} failed ({}); retrying in {:.1f}s",
                    attempt, retries, str(e)[:60], wait,
                )
                await asyncio.sleep(wait)
    raise RuntimeError(f"edge-tts failed after {retries} attempts: {last_err}")


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
                asyncio.run(_edge_tts_async(sentence, mp3_tmp, voice=edge_voice, rate=edge_rate, log=False))
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


def _trim_silence(audio, sample_rate: int, threshold: float = 0.01, keep_ms: int = 40):
    """Trim leading/trailing near-silence from a mono float32 chunk.

    VieNeu v3 Turbo sometimes fails to emit EOS and runs to max_new_frames,
    appending ~20s of trailing silence to a chunk.  We control the gap between
    chunks ourselves, so we strip whatever silence VieNeu produced (front+back)
    and keep a tiny `keep_ms` cushion.  Robust regardless of why EOS missed.
    """
    import numpy as np

    if audio.size == 0:
        return audio
    loud = np.abs(audio) > threshold
    if not loud.any():
        return audio[:0]  # entirely silent → drop it
    first = int(np.argmax(loud))
    last = len(loud) - int(np.argmax(loud[::-1]))
    pad = int(keep_ms / 1000 * sample_rate)
    start = max(0, first - pad)
    end = min(len(audio), last + pad)
    return audio[start:end]


def _run_vieneu_sentence_mode(
    script_path: Path,
    output_path: Path,
    timestamps_path: Path,
    voice: str,
    silence_ms: int = 300,
    diagnostics_path: Path | None = None,
    infer_overrides: dict | None = None,
    trim_chunks: bool = True,
    legacy_chunking: bool = False,
    comma_gap_ms: int = 90,
    trim_keep_ms: int = 90,
) -> None:
    """VieNeu sentence mode: comma-aware chunking for reliable EOS, synced timestamps.

    VieNeu v3 Turbo fails to find EOS for clauses containing a comma — reading the
    whole script at once makes generation run to max_new_frames (≈24s of silence)
    after every such clause.  We split each sentence on commas (`_split_vieneu_chunks`)
    so every chunk is short enough for reliable EOS, synthesize each chunk, then
    concatenate.  Timestamps are grouped per SENTENCE (comma-split parts share the
    timestamp of the sentence they belong to), so timestamps.json matches the audio
    exactly and step 3 (transcribe/align) is not needed.

    silence_ms: gap after a sentence end (. ! ?).  Comma gaps are half of silence_ms.
    """
    import numpy as np
    import soundfile as sf

    chunks = _split_vieneu_chunks(script_path, legacy=legacy_chunking)
    if not chunks:
        raise RuntimeError("No chunks found in script.txt")
    logger.info(
        "VieNeu sentence mode: {} chunks (comma-aware), voice={}", len(chunks), voice
    )

    import os
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    from vieneu import Vieneu

    tts = Vieneu()  # heavy model — instantiate ONCE and reuse for every chunk
    sample_rate = tts.sample_rate  # v3 Turbo = 48 kHz

    sentence_gap = np.zeros(int(silence_ms / 1000 * sample_rate), dtype=np.float32)
    comma_gap = np.zeros(int(comma_gap_ms / 1000 * sample_rate), dtype=np.float32)

    all_audio: list[np.ndarray] = []
    timestamps: list[dict] = []
    diagnostics_chunks: list[dict] = []
    cursor = 0.0          # seconds written so far
    sent_start = 0.0      # start of the current sentence
    sent_text_parts: list[str] = []
    sent_index = 0
    infer_kwargs = {
        "voice": voice,
        "temperature": 0.0,
        "repetition_penalty": 1.2,
        "silence_p": 0.15,
        "crossfade_p": 0.0,
        "apply_watermark": False,
    }
    if infer_overrides:
        infer_kwargs.update(infer_overrides)

    for i, chunk in enumerate(chunks):
        chunk_text = chunk["text"]
        synth_text = chunk["synth_text"]
        is_sentence_end = chunk["is_sentence_end"]
        audio = tts.infer(synth_text, **infer_kwargs)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        # Strip VieNeu's EOS-miss trailing silence; we add our own gaps below.
        if trim_chunks:
            audio = _trim_silence(audio, sample_rate, keep_ms=trim_keep_ms)

        if not sent_text_parts:
            sent_start = cursor
        sent_text_parts.append(chunk_text)

        chunk_start = cursor
        chunk_dur = len(audio) / sample_rate
        all_audio.append(audio)
        cursor += chunk_dur

        if is_sentence_end:
            gap_ms = silence_ms
            sent_index += 1
            timestamps.append({
                "index": sent_index,
                "start": round(sent_start, 3),
                "end": round(cursor, 3),
                "text": " ".join(p.strip() for p in sent_text_parts).strip(),
            })
            sent_text_parts = []
            all_audio.append(sentence_gap)
            cursor += silence_ms / 1000
        else:
            gap_ms = comma_gap_ms
            all_audio.append(comma_gap)
            cursor += comma_gap_ms / 1000

        diagnostics_chunks.append({
            "chunk_index": i + 1,
            "sentence_index": sent_index + (0 if is_sentence_end else 1),
            "text": chunk_text,
            "synth_text": synth_text,
            "is_sentence_end": is_sentence_end,
            "start": round(chunk_start, 3),
            "speech_end": round(chunk_start + chunk_dur, 3),
            "gap_ms": gap_ms,
            "end_with_gap": round(cursor, 3),
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            logger.info("  VieNeu progress: {}/{} chunks", i + 1, len(chunks))

    # Flush a trailing sentence that never hit an end marker (defensive)
    if sent_text_parts:
        sent_index += 1
        timestamps.append({
            "index": sent_index,
            "start": round(sent_start, 3),
            "end": round(cursor, 3),
            "text": " ".join(p.strip() for p in sent_text_parts).strip(),
        })

    full_audio = np.concatenate(all_audio)
    tmp_dir = Path(tempfile.mkdtemp(prefix="tts_vieneu_"))
    try:
        combined_wav = tmp_dir / "combined.wav"
        sf.write(str(combined_wav), full_audio, sample_rate)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(combined_wav),
             "-codec:a", "libmp3lame", "-qscale:a", "0", str(output_path)],
            check=True, capture_output=True,
        )
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    timestamps_path.write_text(
        json.dumps(timestamps, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if diagnostics_path is not None:
        diagnostics = {
            "engine": "vieneu",
            "mode": "chunked_sentence",
            "voice": voice,
            "sample_rate": sample_rate,
            "silence_ms": silence_ms,
            "comma_gap_ms": comma_gap_ms,
            "trim_chunks": trim_chunks,
            "trim_keep_ms": trim_keep_ms,
            "legacy_chunking": legacy_chunking,
            "infer_kwargs": infer_kwargs,
            "chunk_count": len(chunks),
            "sentence_count": len(timestamps),
            "chunks": diagnostics_chunks,
        }
        diagnostics_path.write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    logger.info(
        "VieNeu sentence mode complete: {} sentences, {:.1f}s → {} + {}",
        len(timestamps),
        timestamps[-1]["end"] if timestamps else 0,
        output_path, timestamps_path,
    )


def _run_vieneu_paragraph_mode(
    script_path: Path,
    output_path: Path,
    voice: str,
    diagnostics_path: Path | None = None,
    silence_ms: int = 450,
    infer_overrides: dict | None = None,
    trim_paragraphs: bool = False,
    trim_keep_ms: int = 90,
) -> None:
    """Audit/helper mode: synthesize one whole paragraph per VieNeu call."""
    import numpy as np
    import soundfile as sf

    paragraphs = _split_script_paragraphs(script_path)
    if not paragraphs:
        raise RuntimeError("No paragraphs found in script.txt")
    logger.info("VieNeu paragraph mode: {} paragraphs, voice={}", len(paragraphs), voice)

    import os
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    from vieneu import Vieneu

    tts = Vieneu()
    sample_rate = tts.sample_rate
    paragraph_gap = np.zeros(int(silence_ms / 1000 * sample_rate), dtype=np.float32)

    infer_kwargs = {
        "voice": voice,
        "temperature": 0.0,
        "repetition_penalty": 1.2,
        "silence_p": 0.15,
        "crossfade_p": 0.0,
        "apply_watermark": False,
    }
    if infer_overrides:
        infer_kwargs.update(infer_overrides)

    all_audio: list[np.ndarray] = []
    diagnostics_chunks: list[dict] = []
    cursor = 0.0

    for i, paragraph in enumerate(paragraphs, 1):
        audio = tts.infer(paragraph, **infer_kwargs)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        if trim_paragraphs:
            audio = _trim_silence(audio, sample_rate, keep_ms=trim_keep_ms)

        para_duration = len(audio) / sample_rate
        para_start = cursor
        all_audio.append(audio)
        cursor += para_duration
        if i != len(paragraphs):
            all_audio.append(paragraph_gap)
            cursor += silence_ms / 1000

        diagnostics_chunks.append({
            "paragraph_index": i,
            "text": paragraph,
            "start": round(para_start, 3),
            "speech_end": round(para_start + para_duration, 3),
            "end_with_gap": round(cursor, 3),
        })
        logger.info("  VieNeu paragraph progress: {}/{}", i, len(paragraphs))

    full_audio = np.concatenate(all_audio)
    tmp_dir = Path(tempfile.mkdtemp(prefix="tts_vieneu_paragraph_"))
    try:
        combined_wav = tmp_dir / "combined.wav"
        sf.write(str(combined_wav), full_audio, sample_rate)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(combined_wav),
             "-codec:a", "libmp3lame", "-qscale:a", "0", str(output_path)],
            check=True, capture_output=True,
        )
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if diagnostics_path is not None:
        diagnostics = {
            "engine": "vieneu",
            "mode": "paragraph_whole_blocks",
            "voice": voice,
            "sample_rate": sample_rate,
            "silence_ms": silence_ms,
            "trim_paragraphs": trim_paragraphs,
            "trim_keep_ms": trim_keep_ms,
            "infer_kwargs": infer_kwargs,
            "paragraph_count": len(paragraphs),
            "paragraphs": diagnostics_chunks,
        }
        diagnostics_path.write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    logger.info(
        "VieNeu paragraph mode complete: {} paragraphs, {:.1f}s → {}",
        len(paragraphs),
        cursor,
        output_path,
    )


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

    # ── VieNeu: always sentence mode (comma-aware) to avoid EOS-fail silence ──
    # Reading the whole script at once makes VieNeu insert ~20s of silence after
    # every comma clause (EOS never fires mid-clause → runs to max_new_frames),
    # so VieNeu always routes through the per-chunk path below.  The legacy
    # whole-script _vieneu_tts() is kept only for reference/manual use.
    if tts_engine == "vieneu":
        try:
            _run_vieneu_sentence_mode(
                script_path=script_path,
                output_path=output_path,
                timestamps_path=timestamps_path,
                voice=edge_voice,
            )
        except Exception as e:
            logger.error("VieNeu sentence mode TTS failed: {}", e)
            sys.exit(1)

    # ── Sentence mode (edge / kokoro) ─────────────────────────────────────────
    elif tts_mode == "sentence":
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
