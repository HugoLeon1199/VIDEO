# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the pipeline

```powershell
# Set API keys (required before any run)
$env:GEMINI_API_KEY = "AIza..."          # required for steps 4 and 5
$env:ANTHROPIC_API_KEY = "sk-ant-..."   # required for step 7 only

# Install dependencies (first time)
pip install -r requirements.txt

# Full pipeline (steps 2–7) — requires script.txt already placed manually
python main.py --video-id "my-video-slug"

# Single step
python main.py --video-id "my-video-slug" --step 4

# From a specific step to the end
python main.py --video-id "my-video-slug" --from-step 5

# Resume after interruption (step 5 takes ~100 min and is the most likely to be interrupted)
python main.py --video-id "my-video-slug" --resume

# Demo mode: generate only N images (~30s video) for quick testing
python main.py --video-id "my-video-slug" --from-step 4 --demo 10

# Step 6 with subtitle burn-in (off by default)
python main.py --video-id "my-video-slug" --step 6 --subtitles
```

FFmpeg must be installed separately and available in `PATH` (`winget install Gyan.FFmpeg`). All AI inference (TTS, transcription) runs locally on CPU — no GPU required.

## Before first run

1. Create the video directory and place the script:
   ```
   output\{video-id}\script.txt
   ```
2. Script is written manually on Claude Web using the system prompt in `AGENTS.md`.

## Architecture

The pipeline is a linear 7-step sequence. Each step is a standalone module in `steps/` with a single `run(video_id: str)` entrypoint. `main.py` is a thin orchestrator that dispatches to these modules by step number and handles `--resume` logic.

**Step flow and I/O:**

| Step | Module | Input | Output |
|------|--------|-------|--------|
| 1 | `generate_script.py` | — | validates `script.txt` exists |
| 2 | `tts.py` | `script.txt` | `audio.mp3` |
| 3 | `transcribe.py` | `audio.mp3` | `timestamps.json` |
| 4 | `image_prompts.py` | `timestamps.json` + `script.txt` | `image_prompts.json` |
| 5 | `generate_images.py` | `image_prompts.json` | `images/img_001.png`…`img_200.png` |
| 6 | `render_video.py` | `images/` + `audio.mp3` + `image_prompts.json` | `final.mp4` |
| 7 | `metadata.py` | `script.txt` | `metadata.json` |

Each video lives in `output/{video_id}/`. Step 1 (writing the script) is always done manually; the automated pipeline starts at step 2.

**Resume logic** (`detect_resume_step` in `main.py`): checks existence of each output file in reverse order to find the last completed step, then continues from the next one. Step 5 additionally checks `images/progress.json` for partial image completion.

**Demo mode** (`--demo N`): limits image count to N (default 10) and trims timestamps to cover only the first `N × 3` seconds. Useful for testing the full pipeline end-to-end without waiting 100 minutes.

## Key implementation details

**TTS (step 2):** Tries Kokoro TTS first (`KPipeline(lang_code="a")`), falls back to `edge-tts` automatically on any failure. Kokoro outputs numpy chunks at 24kHz → written to a temp WAV via `soundfile` → converted to MP3 via `ffmpeg -codec:a libmp3lame`.

**Transcription (step 3):** Uses `faster_whisper` model `"base"` with `compute_type="int8"` for CPU efficiency. Word-level timestamps are accumulated into sentence-level segments (break at punctuation or after 4.5s).

**Image prompts (step 4):** Prefers Gemini Flash text (`gemini-2.5-flash`) if `GEMINI_API_KEY` is set, falls back to Claude Haiku if only `ANTHROPIC_API_KEY` is set. Makes a single API call with the full transcript. If fewer than N prompts are returned, duplicates the last entry to pad. Always appends the fixed style suffix: `"ancient art style, prehistoric, minimalist, warm earth tones, no text"`.

**Image generation (step 5):** Uses `google.genai` SDK (not `google.generativeai`). Model: `gemini-2.5-flash-image` (requires billing — no free tier for image generation). Rate-limited to 2 req/min with 31s sleep between requests. On 429/RESOURCE_EXHAUSTED sleeps 60s and retries up to 3 times. Progress is persisted to `images/progress.json` after every successful image so the step can be safely interrupted and resumed.

**Video render (step 6):** Builds a single FFmpeg `filter_complex` string: each image gets `scale → pad → zoompan` (Ken Burns 5% zoom) then all clips are concatenated. With `--subtitles`, `timestamps.json` is converted to a temp `.srt` file and burned in via the `subtitles=` filter. FFmpeg receives all image inputs plus audio as the last input (`-map {n}:a`).

**API calls (steps 4 and 7):** JSON is extracted from responses by stripping markdown fences and finding the first `[`…`]` or `{`…`}` bounds. Retry up to `CLAUDE_MAX_RETRIES` (2) times with `CLAUDE_RETRY_SLEEP` (5s) on failure.

## Config

All tunable values are in `config.py` (read directly as constants, not via env vars except API keys).

Key values:
- `IMAGES_PER_VIDEO = 200` — changing this also affects the prompt and FFmpeg input count
- `GEMINI_IMAGE_MODEL = "gemini-2.5-flash-image"` — correct model name (old name `gemini-2.5-flash-preview-image-generation` no longer works)
- `GEMINI_RATE_LIMIT_SLEEP = 31` — keep ≥31s to stay under 2 req/min
- `KEN_BURNS_ZOOM = 0.05` — 5% zoom applied over each clip's duration via `zoompan`

## Prompts

`prompts/system_prompt.txt` — used in step 4. Contains `{N}` placeholder replaced at runtime with image count.

`prompts/metadata_prompt.txt` — used in step 7. Both prompts instruct the model to return only raw JSON with no markdown.

## Logging

`loguru` configured once in `main.py:setup_logging()`. Console shows INFO; `pipeline.log` captures DEBUG with 10 MB rotation. All step modules use `from loguru import logger` directly.
