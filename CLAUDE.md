# CLAUDE.md

## Current Production UX

Cursor-first production is now the preferred path.

```powershell
$python = "C:\Users\LEON_RM\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

# Full production autopilot from a saved narration script
& $python main.py --autopilot --video-id "my-video-slug" --script-file "C:\path\to\script.txt"
```

Autopilot responsibilities:
- saves normalized narration to `output/<video-id>/script.txt`
- creates `tts_config.json` and `transcribe_config.json`
- creates or reuses `creative_package.json`
- runs TTS -> exact alignment -> visual beats -> scene images -> thumbnails -> soundscape -> effects plan -> clean render -> subtitle burn -> publishing package
- writes `autopilot_state.json` and `autopilot_summary.json`

Locked defaults:
- Vietnamese: VieNeu `Thái Sơn`, `mode=block`
- English: Kokoro `am_fenrir`, `speed=0.95`, `mode=block`
- production image backend in autopilot: `Vast.ai` only
- production render uses one final FFmpeg encode with timeline-safe display timing
- production completion requires both `final.mp4` and `final_subbed.mp4`

## Manual Workflow Still Supported

```powershell
# Single step
& $python main.py --video-id "my-video-slug" --step 4

# From a specific step to the end
& $python main.py --video-id "my-video-slug" --from-step 5

# Resume legacy/manual step flow
& $python main.py --video-id "my-video-slug" --resume

# Demo mode
& $python main.py --video-id "my-video-slug" --demo 30

# Standalone image generation CLI
& $python scripts/generate_images.py --video-id "my-video-slug" --backend vast_instance --qa --workers 1

# Effects preview using the production renderer
& $python scripts/preview_effects.py --video-id "my-video-slug" --seconds 45
```

## Step Contracts

| Step | Module | Notes |
|------|--------|-------|
| 1 | `generate_script.py` | validate existing `script.txt` only |
| 2 | `tts.py` | block-mode production audio |
| 3 | `transcribe.py` | sole writer of `timestamps.json`, `word_timestamps.json`, `word_timestamps_diagnostics.json` |
| 4 | `image_prompts.py` | sentence -> `1..3` visual beats when exact words exist; fallback `1 sentence = 1 image` when they do not |
| 5 | `generate_images.py` | writes canonical `images/img_XXX.png`; now still runs thumbnails even if scenes are already complete |
| 6 | `post_production_design.py` | writes `soundscape.json`, `effects_plan.json`, `effects_diagnostics.json` |
| 7 | `render_video.py` | renders clean `final.mp4` from semantic prompts + effects plan, then optional `final_subbed.mp4` |
| 8 | `metadata.py` | writes publishing package under `publishing/` |

## Important Invariants

- Step 3 owns production timing artifacts.
- `image_prompts.json` owns semantic timing; `effects_plan.json` owns display timing only.
- Subtitle generation never fabricates word timing.
- `final.mp4` is never overwritten by subtitle burn.
- Canonical render images must live in `output/<video-id>/images/`.
- Autopilot scene images and thumbnails must reuse one real Vast backend/session when GPU work is pending.
- `creative_package.json` is strategy only; `script.txt` is narration only.

## Validation

```powershell
& $python -m pytest tests/test_autopilot.py -q
& $python -m pytest tests/test_visual_beats.py -q
& $python -m pytest tests/test_creative_package.py -q
& $python -m pytest tests/test_thumbnails.py -q
& $python -m pytest tests/test_subtitles.py -q
& $python -m pytest tests -q
```
