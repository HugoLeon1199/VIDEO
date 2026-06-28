# AGENTS.md - YouTube Autopilot Pipeline

Read this file before changing production workflow.

## Primary UX

Leon should be able to paste one narration script into Cursor and let Cursor drive the repo to a finished video.

Preferred repo entrypoint:

```powershell
$python = "C:\Users\LEON_RM\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $python main.py --autopilot --video-id "<safe-id>" --script-file "C:\path\to\script.txt"
```

Cursor responsibilities in this production flow:
- create a safe `video-id`
- save narration-only text to `output/<video-id>/script.txt`
- normalize script text
- detect `vi|en`
- create or reuse `creative_package.json`
- run the full pipeline
- verify real output files before reporting success

Leon should not need to:
- create `script.txt`
- hand-write configs
- edit JSON
- move images between folders
- burn subtitles manually

## Production Defaults

### Vietnamese
- engine: `vieneu`
- voice: `Thái Sơn`
- mode: `block`

### English
- engine: `kokoro`
- voice: `am_fenrir`
- speed: `0.95`
- mode: `block`

### Timing ownership
- step 2 creates audio and block artifacts
- step 3 is the only writer of:
  - `timestamps.json`
  - `word_timestamps.json`
  - `word_timestamps_diagnostics.json`

## Visual Planning Contract

- `1 canonical sentence = 1 sentence timestamp`
- `1 canonical sentence = 1..3 visual beats`
- `1 visual beat = 1 image`

When exact word timing exists:
- step 4 may split a sentence into multiple semantic beats
- beat timing must come from exact canonical word boundaries only

When exact word timing does not exist:
- step 4 falls back to `1 sentence = 1 image`
- subtitles are still blocked until exact word timing is available

Canonical scene images for render must land in:

```text
output/<video-id>/images/img_001.png
output/<video-id>/images/img_002.png
...
```

## Image Backend Contract

Autopilot production uses `Vast.ai` only.

- no silent RunPod fallback in autopilot
- scene images and thumbnails must reuse one real shared Vast lifecycle when GPU work is pending
- autoplay production must not shell out to the image CLI
- owned rented instances must be destroy-verified before reporting teardown success
- predictable/default worker tokens are not allowed for rented public workers

Manual tools may still keep RunPod support for debugging or standby use.

## Effects Contract

- step 6 now designs:
  - `soundscape.json`
  - `effects_plan.json`
  - `effects_diagnostics.json`
- `image_prompts.json` remains the semantic timing source of truth
- `effects_plan.json` carries render-only display timing:
  - `source_start`
  - `source_end`
  - `display_start`
  - `display_end`
- render must preserve pause coverage:
  - first scene starts visually at `0`
  - non-final scenes hold until the next canonical scene starts
  - final scene holds until audio end
- effects must stay restrained:
  - no shake
  - no bounce
  - no rotation
  - no glitch
  - no aggressive zoom
- effects-disabled mode must keep the same timing and visual boundaries without motion, transitions, or look filters

## Subtitle Contract

- subtitle text must reconstruct the canonical script exactly
- no missing words
- no repeated words
- no reordered words
- no fabricated or interpolated word timing
- no overlap

Render order:
1. create clean `final.mp4`
2. only then create `final_subbed.mp4` from current exact subtitle assets

Production completion requires both files.

## Creative Package Contract

- `script.txt` contains narration only
- `creative_package.json` contains titles, description, keywords, chapter plan, thumbnail concepts
- validated copy lives under `publishing/creative_package.validated.json`
- stale package reuse must fail unless explicitly bypassed

## Manual Commands Still Supported

```powershell
& $python main.py --video-id "<id>" --step 4
& $python main.py --video-id "<id>" --from-step 5
& $python main.py --video-id "<id>" --resume
& $python scripts/generate_images.py --video-id "<id>" --backend vast_instance --qa --workers 1
& $python scripts/preview_effects.py --video-id "<id>" --seconds 45
```

## Validation Commands

```powershell
& $python -m pytest tests/test_autopilot.py -q
& $python -m pytest tests/test_visual_beats.py -q
& $python -m pytest tests/test_creative_package.py -q
& $python -m pytest tests/test_thumbnails.py -q
& $python -m pytest tests/test_subtitles.py -q
& $python -m pytest tests -q
```

## Do Not Commit

- anything under `output/`
- generated audio
- generated images
- generated videos
