# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the pipeline

```powershell
# Set API keys (required before any run)
$env:ANTHROPIC_API_KEY = "sk-ant-..."
$env:GEMINI_API_KEY = "AIza..."

# Install dependencies
pip install -r requirements.txt

# Full pipeline (steps 2–7) — requires script.txt already placed manually
python main.py --video-id "my-video-slug"

# Single step
python main.py --video-id "my-video-slug" --step 4

# From a specific step to the end
python main.py --video-id "my-video-slug" --from-step 5

# Resume after interruption (step 5 takes ~100 min and is the most likely to be interrupted)
python main.py --video-id "my-video-slug" --resume

# Step 6 with subtitle burn-in (off by default)
python main.py --video-id "my-video-slug" --step 6 --subtitles
```

FFmpeg must be installed separately and available in `PATH`. All AI inference (TTS, transcription) runs locally on CPU — no GPU required.

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

Each video lives in `output/{video_id}/`. Step 1 (writing the script) is always done manually on Claude Web; the pipeline starts at step 2.

**Resume logic** (`detect_resume_step` in `main.py`): walks backward through output files to find the highest completed step, then starts from the next one. Step 5 also checks `images/progress.json` for partial image completion.

## Key implementation details

**TTS (step 2):** Tries Kokoro TTS first (`KPipeline(lang_code="a")`), falls back to `edge-tts` on any failure. Kokoro outputs numpy chunks at 24kHz → written to a temp WAV via `soundfile` → converted to MP3 via `ffmpeg -codec:a libmp3lame`.

**Transcription (step 3):** Uses `faster_whisper` model `"base"` with `compute_type="int8"` for CPU efficiency. Word-level timestamps are accumulated into sentence-level segments (break at punctuation or after 4.5s).

**Image prompts (step 4):** Makes a single Claude Haiku API call with the full transcript. If the response returns fewer than 200 prompts, duplicates the last entry to pad. Trims if over 200. Always appends the fixed style suffix: `"ancient art style, prehistoric, minimalist, warm earth tones, no text"`.

**Image generation (step 5):** Uses `google.genai` (the new SDK, not `google.generativeai`). Rate-limited to 2 req/min with a 31s sleep between requests. On 429/RESOURCE_EXHAUSTED sleeps 60s and retries up to 3 times. Progress is persisted to `images/progress.json` after every successful image so the step can be safely interrupted and resumed.

**Video render (step 6):** Builds a single large FFmpeg `filter_complex` string: each image gets `scale → pad → zoompan` (Ken Burns 5% zoom) then all clips are concatenated. With `--subtitles`, `timestamps.json` is converted to a temp `.srt` file and burned in via the `subtitles=` filter. FFmpeg receives 200 image inputs plus the audio as the last input.

**API calls (steps 4 and 7):** Both use `anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)`. Retry up to `CLAUDE_MAX_RETRIES` (2) times with `CLAUDE_RETRY_SLEEP` (5s) between attempts. JSON is extracted from responses by stripping markdown fences and finding the first `[`…`]` or `{`…`}` bounds.

## Config

All tunable values are in `config.py` (imported directly, not via env vars except for API keys). Key values to know:

- `IMAGES_PER_VIDEO = 200` — changing this also changes the Claude prompt and the FFmpeg input count
- `GEMINI_RATE_LIMIT_SLEEP = 31` — must stay ≥31s to stay under 2 req/min free-tier limit
- `KEN_BURNS_ZOOM = 0.05` — fraction of zoom applied over each clip's duration via `zoompan`

## Prompts

`prompts/system_prompt.txt` — system prompt for step 4 (image prompts). Contains `{N}` placeholder replaced at runtime with `IMAGES_PER_VIDEO`.

`prompts/metadata_prompt.txt` — system prompt for step 7 (metadata). Both prompts instruct Claude to return only raw JSON with no markdown.

## Logging

`loguru` is configured once in `main.py:setup_logging()`. Console shows INFO; `pipeline.log` captures DEBUG with rotation at 10 MB. All step modules use `from loguru import logger` directly — no additional setup needed per module.
