"""Step 3: Transcribe audio to sentence-level timestamps.

Two engines (set via transcribe_config.json):

  stable_ts (default for VI):
    Uses stable-ts forced alignment against script.txt canonical sentences.
    Gives exact script text with accurate per-sentence timing.
    Config: {"engine": "stable_ts", "model": "medium", "language": "vi", "mode": "align"}

  faster_whisper (legacy / EN fallback):
    Two modes:
      default — Whisper auto-segments (sentence boundaries + max 4.5s)
      align   — greedy word-count matching against script.txt sentences
    Config: {"model": "medium.en", "language": "en"}
"""

import json
import re
import sys
import unicodedata
from pathlib import Path

from loguru import logger

import config


def _normalize(text: str) -> str:
    """Lowercase, strip diacritics for fuzzy word matching."""
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


_SCRIPT_STOP_MARKERS = (
    "COMMENT SEED:",
    "RESEARCH NOTES:",
    "Your script is ready",
    "Save as:",
    "Then: python",
)


def _split_script_sentences(script_path: Path) -> list[str]:
    """Split script.txt into non-empty spoken sentences.

    Splits only on sentence-ending punctuation (. ! ?) or blank lines.
    Commas are NOT split points — they stay inside the same sentence.
    Strips trailing metadata sections. Skips a leading title line.
    """
    text = script_path.read_text(encoding="utf-8")
    text = text.lstrip("﻿")  # Strip BOM

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
        parts = re.split(r"(?<=[.!?])\s+", para)
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)
    return sentences


# ── stable-ts engine ──────────────────────────────────────────────────────────

def _run_stable_ts(
    audio_path: Path,
    script_path: Path,
    model_name: str,
    language: str,
    device: str = "cpu",
) -> list[dict]:
    """
    Align audio to canonical script sentences using stable-ts.

    Strategy:
    1. Load model and run align() against the full canonical script text.
    2. Map aligned word spans back to script sentence boundaries.
    3. Sentence start = first aligned word; end = last aligned word.
    4. Gaps/failures → interpolate from surrounding sentences.
    5. Enforce monotonicity and cap final end to audio duration.
    """
    import stable_whisper
    import mutagen.mp3

    sentences = _split_script_sentences(script_path)
    if not sentences:
        logger.error("No sentences found in script.txt")
        sys.exit(1)

    logger.info("Script: {} sentences", len(sentences))

    # Build canonical text for alignment (join with spaces)
    canonical_text = " ".join(sentences)

    logger.info("Loading stable-ts model ({})...", model_name)
    model = stable_whisper.load_model(model_name, device=device)

    logger.info("Aligning {} to script text ({} chars)...", audio_path.name, len(canonical_text))
    result = model.align(str(audio_path), canonical_text, language=language)

    # Get audio duration for capping
    try:
        audio_info = mutagen.mp3.MP3(str(audio_path))
        audio_duration = audio_info.info.length
    except Exception:
        audio_duration = None

    # Extract word-level timestamps from stable-ts result
    words: list[dict] = []
    for seg in result.segments:
        for w in seg.words:
            if w.start is not None and w.end is not None:
                words.append({"word": w.word, "start": w.start, "end": w.end})

    logger.info("stable-ts produced {} aligned word timestamps", len(words))

    if not words:
        logger.error("stable-ts alignment produced no word timestamps — check audio/script match")
        sys.exit(1)

    # Map words to sentences by word count
    timestamps = _assign_words_to_sentences(sentences, words, audio_duration)
    return timestamps


def _assign_words_to_sentences(
    sentences: list[str],
    words: list[dict],
    audio_duration: float | None,
) -> list[dict]:
    """
    Assign aligned word timestamps to canonical sentences.

    Consumes words greedily by word count per sentence.
    Interpolates spans for sentences with no aligned words.
    Enforces monotonicity and caps to audio_duration.
    """
    def word_count(s: str) -> int:
        return len([w for w in _normalize(s).split() if w])

    # First pass: assign word ranges
    w_idx = 0
    n_words = len(words)
    spans: list[tuple[float, float] | None] = []

    for sentence in sentences:
        n = word_count(sentence)
        if n == 0:
            spans.append(None)
            continue

        if w_idx >= n_words:
            spans.append(None)
            continue

        end_idx = min(w_idx + n, n_words) - 1
        start = words[w_idx]["start"]
        end = words[end_idx]["end"]
        spans.append((start, end))
        w_idx = end_idx + 1

    # Second pass: interpolate None spans
    def _find_prev(i: int) -> float | None:
        for j in range(i - 1, -1, -1):
            if spans[j] is not None:
                return spans[j][1]
        return None

    def _find_next(i: int) -> float | None:
        for j in range(i + 1, len(spans)):
            if spans[j] is not None:
                return spans[j][0]
        return audio_duration

    resolved: list[tuple[float, float]] = []
    for i, span in enumerate(spans):
        if span is not None:
            resolved.append(span)
        else:
            prev = _find_prev(i) or 0.0
            nxt = _find_next(i)
            # Count consecutive Nones to divide interval
            block_start = i
            while block_start > 0 and spans[block_start - 1] is None:
                block_start -= 1
            block_end = i
            while block_end < len(spans) - 1 and spans[block_end + 1] is None:
                block_end += 1
            block_size = block_end - block_start + 1
            block_pos = i - block_start
            if nxt is not None:
                step = (nxt - prev) / block_size
                s = round(prev + block_pos * step, 3)
                e = round(prev + (block_pos + 1) * step, 3)
            else:
                s = round(prev + block_pos * 1.0, 3)
                e = round(prev + (block_pos + 1) * 1.0, 3)
            resolved.append((s, e))

    # Third pass: enforce monotonicity + cap to audio_duration
    result = []
    prev_end = 0.0
    for i, (sentence, (start, end)) in enumerate(zip(sentences, resolved)):
        start = max(round(start, 3), prev_end)
        end = max(round(end, 3), start + 0.1)
        if audio_duration is not None and i == len(sentences) - 1:
            end = min(end, round(audio_duration, 3))
        result.append({
            "index": i + 1,
            "start": start,
            "end": end,
            "text": sentence,
        })
        prev_end = end

    return result


# ── faster-whisper engine (legacy) ────────────────────────────────────────────

def _align_sentences_to_words(
    sentences: list[str],
    whisper_words: list[dict],
) -> list[dict]:
    """Greedy word-count matching (faster-whisper align mode)."""
    def tokenize(s: str) -> list[str]:
        return [w for w in _normalize(s).split() if w]

    result = []
    w_idx = 0
    n_words = len(whisper_words)

    for i, sentence in enumerate(sentences):
        s_words = tokenize(sentence)
        if not s_words:
            continue
        n_needed = len(s_words)

        if w_idx >= n_words:
            prev_end = result[-1]["end"] if result else 0.0
            result.append({
                "index": len(result) + 1,
                "start": round(prev_end, 3),
                "end": round(prev_end + 1.0, 3),
                "text": sentence,
            })
            continue

        seg_start = whisper_words[w_idx]["start"]
        end_idx = min(w_idx + n_needed, n_words) - 1
        seg_end = whisper_words[end_idx]["end"]
        w_idx = end_idx + 1

        result.append({
            "index": len(result) + 1,
            "start": round(seg_start, 3),
            "end": round(seg_end, 3),
            "text": sentence,
        })

    return result


def _run_faster_whisper(
    audio_path: Path,
    script_path: Path | None,
    model_name: str,
    language: str | None,
    align_mode: bool,
) -> list[dict]:
    from faster_whisper import WhisperModel

    logger.info("Loading faster-whisper model ({}, CPU, int8)...", model_name)
    model = WhisperModel(model_name, device="cpu", compute_type="int8")

    logger.info("Transcribing {}...", audio_path)
    kwargs = {"word_timestamps": True}
    if language:
        kwargs["language"] = language
    segments_gen, info = model.transcribe(str(audio_path), **kwargs)
    segments = list(segments_gen)

    logger.info(
        "Detected language: {} (probability {:.2f})",
        info.language, info.language_probability,
    )

    if align_mode and script_path and script_path.exists():
        whisper_words = []
        for seg in segments:
            if seg.words:
                for w in seg.words:
                    whisper_words.append({"word": w.word, "start": w.start, "end": w.end})
        logger.info("Whisper produced {} word timestamps", len(whisper_words))
        sentences = _split_script_sentences(script_path)
        logger.info("Script has {} sentences", len(sentences))
        return _align_sentences_to_words(sentences, whisper_words)

    # Default auto-segmentation mode
    result = []
    index = 1
    current_words: list[str] = []
    current_start = None
    current_end = None

    for segment in segments:
        for word in segment.words:
            if current_start is None:
                current_start = word.start
            current_words.append(word.word)
            current_end = word.end
            text_so_far = "".join(current_words).strip()
            ends_sentence = text_so_far.endswith((".", "!", "?", "..."))
            long_enough = (current_end - current_start) >= 4.5
            if ends_sentence or long_enough:
                result.append({
                    "index": index,
                    "start": round(current_start, 3),
                    "end": round(current_end, 3),
                    "text": text_so_far,
                })
                index += 1
                current_words = []
                current_start = None
                current_end = None

    if current_words and current_start is not None and current_end is not None:
        result.append({
            "index": index,
            "start": round(current_start, 3),
            "end": round(current_end, 3),
            "text": "".join(current_words).strip(),
        })

    return result


# ── main entry point ──────────────────────────────────────────────────────────

def run(video_id: str) -> None:
    video_dir = Path(config.OUTPUT_DIR) / video_id
    audio_path = video_dir / "audio.mp3"
    output_path = video_dir / "timestamps.json"
    script_path = video_dir / "script.txt"

    if not audio_path.exists():
        logger.error("audio.mp3 not found: {}", audio_path)
        sys.exit(1)

    # Load per-video config
    engine = "faster_whisper"
    whisper_model = "medium.en"
    whisper_language: str | None = None
    align_mode = False
    device = "cpu"

    cfg_path = video_dir / "transcribe_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        engine = cfg.get("engine", engine)
        whisper_model = cfg.get("model", whisper_model)
        whisper_language = cfg.get("language", whisper_language)
        align_mode = cfg.get("mode", "") == "align"
        device = cfg.get("device", device)
        logger.info(
            "Transcribe config: engine={} model={} language={} mode={} device={}",
            engine, whisper_model, whisper_language,
            "align" if align_mode else "default", device,
        )

    if engine == "stable_ts":
        if not script_path.exists():
            logger.error("stable_ts engine requires script.txt: {}", script_path)
            sys.exit(1)
        result = _run_stable_ts(
            audio_path, script_path,
            model_name=whisper_model,
            language=whisper_language or "vi",
            device=device,
        )
    else:
        result = _run_faster_whisper(
            audio_path, script_path,
            model_name=whisper_model,
            language=whisper_language,
            align_mode=align_mode,
        )

    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    total_duration = result[-1]["end"] if result else 0
    logger.info(
        "Transcription complete: {} segments, {:.1f}s total → {}",
        len(result), total_duration, output_path,
    )
