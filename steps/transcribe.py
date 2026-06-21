"""Step 3: Transcribe audio to sentence-level timestamps using faster-whisper."""

import json
import sys
from pathlib import Path

from loguru import logger

import config


def run(video_id: str) -> None:
    import json as _json
    from faster_whisper import WhisperModel

    video_dir = Path(config.OUTPUT_DIR) / video_id
    audio_path = video_dir / "audio.mp3"
    output_path = video_dir / "timestamps.json"

    if not audio_path.exists():
        logger.error("audio.mp3 not found: {}", audio_path)
        sys.exit(1)

    # Per-video transcription config override via transcribe_config.json
    whisper_model = "medium.en"
    whisper_language = None  # None = auto-detect
    transcribe_cfg_path = video_dir / "transcribe_config.json"
    if transcribe_cfg_path.exists():
        cfg = _json.loads(transcribe_cfg_path.read_text(encoding="utf-8"))
        whisper_model = cfg.get("model", whisper_model)
        whisper_language = cfg.get("language", whisper_language)
        logger.info("Transcribe config: model={} language={}", whisper_model, whisper_language)

    logger.info("Loading faster-whisper model ({}, CPU, int8)...", whisper_model)
    model = WhisperModel(whisper_model, device="cpu", compute_type="int8")

    logger.info("Transcribing {}...", audio_path)
    transcribe_kwargs = {"word_timestamps": True}
    if whisper_language:
        transcribe_kwargs["language"] = whisper_language
    segments, info = model.transcribe(str(audio_path), **transcribe_kwargs)

    logger.info(
        "Detected language: {} (probability {:.2f})",
        info.language,
        info.language_probability,
    )

    # Accumulate word-level data into sentence-level segments
    result = []
    index = 1
    current_words = []
    current_start = None
    current_end = None

    for segment in segments:
        for word in segment.words:
            if current_start is None:
                current_start = word.start

            current_words.append(word.word)
            current_end = word.end

            # Break into a new segment at sentence boundaries or ~5s
            text_so_far = "".join(current_words).strip()
            ends_sentence = text_so_far.endswith((".", "!", "?", "..."))
            long_enough = (current_end - current_start) >= 4.5

            if ends_sentence or long_enough:
                result.append(
                    {
                        "index": index,
                        "start": round(current_start, 3),
                        "end": round(current_end, 3),
                        "text": text_so_far,
                    }
                )
                index += 1
                current_words = []
                current_start = None
                current_end = None

    # Flush any remaining words
    if current_words and current_start is not None and current_end is not None:
        result.append(
            {
                "index": index,
                "start": round(current_start, 3),
                "end": round(current_end, 3),
                "text": "".join(current_words).strip(),
            }
        )

    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    total_duration = result[-1]["end"] if result else 0
    logger.info(
        "Transcription complete: {} segments, {:.1f}s total → {}",
        len(result),
        total_duration,
        output_path,
    )
