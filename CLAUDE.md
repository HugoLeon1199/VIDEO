# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the pipeline

```powershell
$python = "C:\Users\LEON_RM\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

# Full pipeline (steps 2–7) — requires script.txt already placed manually
& $python main.py --video-id "my-video-slug"

# Single step
& $python main.py --video-id "my-video-slug" --step 4

# From a specific step to the end
& $python main.py --video-id "my-video-slug" --from-step 5

# Resume after interruption (checks timestamps vs prompts count; step 5 most likely to be interrupted)
& $python main.py --video-id "my-video-slug" --resume

# Demo mode: runs steps 4–6 in output/{slug}_demo{N}/ — never touches production folder
& $python main.py --video-id "my-video-slug" --demo 30
```

`GEMINI_API_KEY` is hardcoded in `config.py` (used for step 4 text API only). `ANTHROPIC_API_KEY` (step 7 only) and `RUNPOD_API_KEY` (step 5) must be set as env vars before running those steps. `RUNPOD_TEMPLATE_ID` must also be filled in `config.py` after creating a pod once in RunPod console.

FFmpeg must be installed via `winget install Gyan.FFmpeg`. The path is hardcoded in `render_video.py:_ensure_ffmpeg_path()` as fallback if not in PATH.

## Architecture

The pipeline is a linear 7-step sequence. Each step is a standalone module in `steps/` with a single `run(video_id: str)` entrypoint. `main.py` is a thin orchestrator.

**Step flow and I/O:**

| Step | Module | Input | Output |
|------|--------|-------|--------|
| 1 | `generate_script.py` | — | validates `script.txt` exists |
| 2 | `tts.py` | `script.txt` | `audio.mp3` |
| 3 | `transcribe.py` | `audio.mp3` | `timestamps.json` |
| 4 | `image_prompts.py` | `timestamps.json` + `script.txt` | `image_prompts.json` |
| 5 | `generate_images.py` | `image_prompts.json` | `images/img_001.png` … |
| 6 | `render_video.py` | `images/` + `audio.mp3` + `image_prompts.json` | `final.mp4` + `subtitles.srt` |
| 7 | `metadata.py` | `script.txt` | `metadata.json` |

**Core invariant:** `N images = N sentences` (from `timestamps.json`). No padding or trimming — 100% audio-image sync.

Each video lives in `output/{video_id}/`. Step 1 is always done manually; the pipeline starts at step 2.

**Demo mode** (`--demo N`): creates `output/{video_id}_demo{N}/`, copies `script.txt` + `timestamps.json`, trims `audio.mp3` to the last demo sentence end, runs steps 4–6 in isolation. Never overwrites production files.

**Resume logic** (`detect_resume_step`): reads `image_prompts.json` and `timestamps.json` to get real counts. If counts differ (stale demo overwrite), resumes from step 4. Otherwise resumes from the first incomplete step.

## Key implementation details

**TTS (step 2):** Tries Kokoro TTS first (`KPipeline(lang_code="a")`, voice `am_fenrir`), falls back to `edge-tts` on any failure. Kokoro numpy chunks → temp WAV via `soundfile` → MP3 via `ffmpeg -codec:a libmp3lame`.

**Transcription (step 3):** `faster_whisper` model `"base"`, `compute_type="int8"`. Word-level timestamps → sentence-level segments (break at punctuation or after 4.5s).

**Image prompts (step 4):** Single **Gemini Flash text** API call (`gemini-2.5-flash`) with full transcript. Returns exactly N prompts (1 per sentence). `_enforce_timings()` overwrites AI-generated timestamps with exact values from `timestamps.json`. Style suffix appended to every prompt.

**Image generation (step 5):** RunPod GPU cloud + ComfyUI + FLUX.2 Klein 4B Distilled FP8. Pipeline auto spin-up pod (`steps/runpod_manager.py`) → queues all prompts to ComfyUI API (`steps/comfyui_client.py`) → downloads PNG → terminates pod. 4 parallel workers, 1344×768 native 16:9. Progress saved to `images/progress.json`; safe to interrupt and `--resume` (pod will spin up again for remaining images).

**Video render (step 6):** Two-pass FFmpeg:
1. Each image → individual clip at exact duration (`next_prompt.start - current.start`; last image → `audio_duration - last.start`)
2. Concat all clips + mux audio → `final.mp4`

Scale: Lanczos to 1920×1080. Audio: AAC stereo 192k 48kHz. Bitrate: 8 Mbps. Also writes `subtitles.srt` sidecar for YouTube caption upload.

**API calls (step 7):** Claude Haiku via `anthropic.Anthropic`. Retry 2× with 5s sleep.

## Config

All tunable values in `config.py`. Key values:

- `GEMINI_API_KEY` — hardcoded default; used for step 4 text API only; never ask user to set it
- `RUNPOD_API_KEY` — set as env var before step 5; `RUNPOD_TEMPLATE_ID` must be filled in config.py
- `COMFYUI_MODEL = "flux2-klein-4b-distilled-fp8.safetensors"` — must exist in pod's ComfyUI models folder
- `VIDEO_BITRATE = "8M"`, `VIDEO_FPS = 30`, `VIDEO_WIDTH/HEIGHT = 1920/1080`
- `TTS_VOICE = "am_fenrir"` — Kokoro dramatic male voice

## Prompts

`prompts/system_prompt.txt` — step 4. Contains `{N}` placeholder (replaced at runtime). Includes VISUAL VARIETY RULES and forces horizontal 16:9 cave painting style.

`prompts/metadata_prompt.txt` — step 7. Returns raw JSON only.

## Logging

`loguru` configured once in `main.py:setup_logging()`. Console INFO, `pipeline.log` DEBUG with 10 MB rotation.
