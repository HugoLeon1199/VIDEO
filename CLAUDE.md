# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Running The Pipeline

```powershell
$python = "C:\Users\LEON_RM\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

# Full pipeline, requires output/<video-id>/script.txt
& $python main.py --video-id "my-video-slug"

# Single step
& $python main.py --video-id "my-video-slug" --step 4

# From a specific step to the end
& $python main.py --video-id "my-video-slug" --from-step 5

# Resume after interruption
& $python main.py --video-id "my-video-slug" --resume

# Demo mode, isolated under output/<video-id>_demo<N>/
& $python main.py --video-id "my-video-slug" --demo 30

# Generate images directly (preferred for step 5 — supports --workers and --candidates)
& $python scripts/generate_images.py --video-id "my-video-slug" --candidates 1 --workers 10
```

`GEMINI_API_KEY` is hardcoded in `config.py` and used only for step 4 text prompts.
`RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID` are required for step 5 and are loaded from `.env`.
`ANTHROPIC_API_KEY` is required only for step 7.

FFmpeg must be installed via `winget install Gyan.FFmpeg`; `steps/render_video.py` has a local WinGet path fallback.

## Architecture

Each step is a standalone module in `steps/`; `main.py` dispatches the step sequence.

| Step | Module | Input | Output |
|------|--------|-------|--------|
| 1 | `generate_script.py` | manual `script.txt` | validates format |
| 2 | `tts.py` | `script.txt` | `audio.mp3` |
| 3 | `transcribe.py` | `audio.mp3` | `timestamps.json` |
| 4 | `image_prompts.py` | `timestamps.json` + `script.txt` | `image_prompts.json` |
| 5 | `generate_images.py` | `image_prompts.json` | `images/img_001.png` … |
| 6 | `render_video.py` | images + audio + prompts | `final.mp4` + `subtitles.srt` |
| 7 | `metadata.py` | `script.txt` | `metadata.json` |

Core invariant: one prompt/image per timestamp entry. Demo mode is isolated and never overwrites production files.

## Key Details

### TTS
- English: Kokoro first (`am_fenrir`), fallback to `edge-tts`
- Vietnamese: always `edge-tts` with per-video override `output/{video-id}/tts_config.json`:
  ```json
  {"engine": "edge", "voice": "vi-VN-NamMinhNeural", "rate": "-8%"}
  ```
- **Vietnamese script must not contain**: em dash `—` (ord 8212), leading `!`, `/` — these cause edge-tts "No audio received"

### Transcription
- English: `faster_whisper` model `base`, CPU/int8, sentence-level segments
- Vietnamese: `stable_ts` forced alignment against `script.txt` — requires per-video override `output/{video-id}/transcribe_config.json`:
  ```json
  {"engine": "stable_ts", "model": "medium", "language": "vi", "mode": "align", "device": "cpu"}
  ```
- `stable_ts` aligns audio to canonical script sentences (exact text, accurate timing, no ASR hallucination)
- Fallback: omit `engine` field to use legacy `faster_whisper` greedy mode

### Image Prompts (Step 4)
- Gemini text model `gemini-2.5-flash` generates one prompt per timestamp
- `_enforce_timings()` overwrites model-provided timings with exact transcript timings
- For Vietnamese videos, prompts are often written manually and saved directly to `image_prompts.json`
- English style: `Cinematic wide shot, [scene], photorealistic, natural lighting, shallow depth of field, 16:9, no text`
- Vietnamese style: rural/village people (người quê), earthy, photorealistic cinematic, NOT doodle/stick-figure

### Image Generation (Step 5)
- **Backend:** RunPod Serverless endpoint `vej2dld6x9p0gh` (FLUX.2 Klein 9B, RTX 6000 Ada 48GB)
- Run via `scripts/generate_images.py` with `--candidates 1 --workers 10` for production
- Each scene: submit job → poll → save webp candidates under `images/scene_XXX/` → promote first candidate to `images/img_XXX.png`
- Resume: skips any scene where `img_XXX.png` already exists
- Progress stored in `generation_log.json`
- Speed: ~4s/image, ~167s for 153 images at 5 workers

### RunPod Serverless Worker
- Source: `serverless_worker/` (Dockerfile, handler.py, model_loader.py)
- RunPod build: context = `serverless_worker`, Dockerfile = `serverless_worker/Dockerfile`
- **ENTRYPOINT not CMD** — RunPod's "Container Start Command" overrides CMD but not ENTRYPOINT
- Model lazy-loads inside `handler()` so `runpod.serverless.start()` runs immediately
- Network Volume `jqscreri1e` (US-IL-1, 50GB) mounted at `/runpod-volume` caches 13GB model
- `HF_TOKEN` injected via RunPod Secrets (never in Dockerfile)

### Vast.ai backend (cost rules — bandwidth is the #1 cost, not GPU)
- Worker: `vast_worker/` (Dockerfile, server.py). Image `leon1199/vast-flux:latest`.
- **Bandwidth (download) dominates cost**: each rental pulls the model; a gouging host ($0.012/GB) cost $0.73 just for the pull (7× the GPU charge). `find_offer` ranks by TRUE cost (gpu-hours + download×$/GB + upload + storage), caps `inet_down_cost` at 0.005, and `server.py` skips the 23.8GB redundant `flux1-dev.safetensors` (download ~58GB→~34GB).
- **ONE rental draws the WHOLE batch** — download is paid once per rental, so amortize it. Production: `--vast-instances 1 --workers 1`, draw all scenes in one go. NEVER rent-per-scene while debugging (each rental re-downloads the model). To reuse a running box across debug runs, set `VAST_INSTANCE_HOST` + `VAST_INSTANCE_PORT` (skips rent + teardown).
- **Don't raise `--vast-instances`** unless truly needed: each instance downloads its own copy of the model = multiplied bandwidth cost.
- **Orphan safety**: every rented id is logged to `image_generation/rented_instances.log`. Run `python scripts/vast_reaper.py` on a 5-min schedule (Windows Task Scheduler) to destroy any box older than `MAX_LEASE_MINUTES` — a crash/power-loss won't leave a machine billing.

### Render (Step 6)
- Two-pass FFmpeg: each canonical PNG → clip → concatenate + mux audio
- Clip duration = `current.start` to `next.start`; last clip to audio end
- Output: H.264 1920×1080, `VIDEO_BITRATE`, AAC stereo 192k 48kHz

## Config

Key settings in `config.py`:

- `IMAGE_BACKEND = "runpod_serverless"`
- `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID` from `.env`
- `IMAGE_WIDTH = 1024`, `IMAGE_HEIGHT = 576`
- `IMAGE_CANDIDATE_SEEDS = [11001, 11002, 11003]`
- `IMAGE_OUTPUT_FORMAT = "WEBP"` for candidates; canonical render files are PNG
- `VIDEO_BITRATE = "8M"`, `VIDEO_FPS = 30`, `VIDEO_WIDTH/HEIGHT = 1920/1080`

## Vietnamese Video Workflow

```powershell
$python = "C:\Users\LEON_RM\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$vid = "my-topic-vi"

# 1. Write Vietnamese script (no em dash, no leading !, no /)
#    Save to output/$vid/script.txt

# 2. Create per-video config files
'{"engine": "edge", "voice": "vi-VN-NamMinhNeural", "rate": "-8%"}' | Out-File "output/$vid/tts_config.json" -Encoding utf8
'{"engine": "stable_ts", "model": "medium", "language": "vi", "mode": "align", "device": "cpu"}' | Out-File "output/$vid/transcribe_config.json" -Encoding utf8

# 3. TTS + Transcribe
& $python main.py --video-id $vid --step 2
& $python main.py --video-id $vid --step 3

# 4. Write image_prompts.json manually (one entry per timestamps.json entry)
#    Style: rural/village people, photorealistic cinematic

# 5. Generate images
& $python scripts/generate_images.py --video-id $vid --candidates 1 --workers 5

# 6. Render
& $python main.py --video-id $vid --step 6
```

## Validation

```powershell
& $python -m pytest tests -q
& $python scripts/validate_runpod_serverless.py --video-id what-ancient-humans-did-all-day
& $python scripts/generate_images.py --video-id what-ancient-humans-did-all-day --dry-run --to-scene 3
```

Use `scripts/test_one_scene.py` for one real RunPod job, then purge the queue if it gets stuck.
