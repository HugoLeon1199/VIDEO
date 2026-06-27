# CURSOR_WORKLOG - Repo worklog

## Session 2026-06-27 - Final TTS continuity audit for `to-tien-ban-lam-gi-ca-ngay-vi`

### Goal
Run a final continuity audit on the current Vietnamese block-TTS output and only regenerate blocks if a real continuity defect is found.

### Files touched
| File | Action |
|------|--------|
| `.ai/CURSOR_WORKLOG.md` | MODIFIED - recorded audit evidence and decision |

### Audit inputs
- `output/to-tien-ban-lam-gi-ca-ngay-vi/audio_master.wav`
- `output/to-tien-ban-lam-gi-ca-ngay-vi/tts_blocks/blocks.json`
- `output/to-tien-ban-lam-gi-ca-ngay-vi/tts_blocks/diagnostics.json`
- `output/to-tien-ban-lam-gi-ca-ngay-vi/tts_blocks/alignment_diagnostics.json`

### Findings
- No block in the current manifest used fallback sentence mode.
- No block used non-default retry params in the manifest.
- `reused_block_count = 52`, `regenerated_block_count = 0`, `fallback_block_count = 0`.
- `voice` stayed constant across the run and the manifest does not show any per-block voice drift.
- Block-to-block gap in the audio master is the configured `300ms` everywhere.
- The main continuity outliers are short-block transitions and a few amplitude-contrast-heavy boundaries, but none of them forced regeneration by the current deterministic checks.

### Boundary clips exported for manual listening
Created under:

`output/to-tien-ban-lam-gi-ca-ngay-vi/continuity_audit/`

Selected clips:
- `boundary_012_b012_to_b013.mp3`
- `boundary_013_b013_to_b014.mp3`
- `boundary_016_b016_to_b017.mp3`
- `boundary_021_b021_to_b022.mp3`
- `boundary_023_b023_to_b024.mp3`
- `boundary_028_b028_to_b029.mp3`
- `boundary_037_b037_to_b038.mp3`
- `boundary_043_b043_to_b044.mp3`
- `boundary_046_b046_to_b047.mp3`
- `boundary_049_b049_to_b050.mp3`

### Quantitative boundary notes
- Loudness contrast was highest around boundaries `37` and `46`.
- The shortest blocks were `7` (`1.022s`), `52` (`1.704s`), and `17` (`2.169s`), so those were included in the audit set indirectly through their neighboring boundaries.
- The current block builder cache stayed intact; no block was regenerated during this audit.

### Decision
- No production block was changed.
- Cache was preserved exactly as-is because there was no clear audible defect proven by the available evidence.
- If Leon hears a problem in one of the exported clips, the next step should be to regenerate only that specific block and re-run the adjacent boundary clips.

## Session 2026-06-27 - Block-TTS production hardening

### Goal
Patch commit `56eea65` so the block-TTS architecture is production-ready without expanding scope.

### Files changed
| File | Action |
|------|--------|
| `requirements.txt` | MODIFIED - pinned tested `stable-ts` and `vieneu` versions |
| `steps/tts.py` | MODIFIED - block hash reuse, VieNeu sentence legacy routing, Kokoro per-video voice/speed, BOM-safe config load |
| `steps/transcribe.py` | MODIFIED - always prefer block mode when block artifacts exist, restart-safe fallback, BOM-safe config load |
| `tests/test_tts_block_mode.py` | MODIFIED - added coverage for rerun, fallback offset, block mode re-entry, VieNeu sentence legacy, Kokoro caps, per-video voice/speed |
| `output/to-tien-ban-lam-gi-ca-ngay-vi/transcribe_config.json` | MODIFIED - switched stable-ts model to `medium` |

### Runtime verification before pinning
Command run:
```powershell
$python = "C:\Users\LEON_RM\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $python -c "import importlib.metadata as m; print('vieneu', m.version('vieneu')); print('stable-ts', m.version('stable-ts'))"
```

Observed:
- `vieneu 3.0.5`
- `stable-ts 2.19.1`

API/runtime verification also run:
```powershell
import stable_whisper
from vieneu import Vieneu
tts = Vieneu()
print(hasattr(stable_whisper, "load_model"))
print(type(tts).__name__)
print(tts.sample_rate)
print(inspect.signature(tts.infer))
```

Observed:
- `stable_whisper.load_model` available
- VieNeu instance type: `V3TurboVieNeuTTS`
- VieNeu sample rate: `48000`
- current `infer(...)` signature supports the params used by this repo

Reason for pins:
- `vieneu==3.0.5` and `stable-ts==2.19.1` are the exact versions verified in this runtime and exercised by the end-to-end tests below.

### Implemented hardening
- Added deterministic block hash reuse contract:
  - cache schema version
  - engine
  - voice
  - speed
  - full `block_config`
  - full effective `infer_params`
  - block text
  - library version (`vieneu` or `kokoro`)
- Reuse only happens when the hash matches and the WAV:
  - exists
  - is non-empty
  - is readable
  - has the expected sample rate
  - has duration `> 0`
- Step 2 no longer deletes `tts_blocks/` unconditionally; unchanged blocks are reused.
- Step 3 now chooses block-aware mode from artifact shape, not `mtime`:
  - `blocks.json` with `mode=block`
  - non-empty blocks list
  - `audio_master.wav` exists
- After any fallback rebuild in Step 3:
  - manifest is reloaded
  - the whole pass restarts
  - the already-loaded model is reused
  - maximum restart guard is `2`
  - if a block is already `fallback_level=2` and still invalid, the run fails clearly
- `sentence_legacy` now routes `engine=vieneu` into real VieNeu sentence synthesis, not Kokoro.
- Kokoro block builder now:
  - fails only when a single sentence exceeds the hard cap
  - closes the current block and starts a new one when the candidate overflows but the sentence itself is still valid
  - uses per-video `voice` and `speed`
- JSON config loaders for `tts_config.json` and `transcribe_config.json` now accept UTF-8 BOM via `utf-8-sig`.

### Tests run
```powershell
& $python -m py_compile steps/text_units.py steps/tts.py steps/transcribe.py tests/test_tts_block_mode.py
& $python -m pytest tests/test_tts_block_mode.py -q
& $python -m pytest tests -q
& $python main.py --video-id "to-tien-ban-lam-gi-ca-ngay-vi" --step 2
& $python main.py --video-id "to-tien-ban-lam-gi-ca-ngay-vi" --step 2
& $python main.py --video-id "to-tien-ban-lam-gi-ca-ngay-vi" --step 3
& $python main.py --video-id "to-tien-ban-lam-gi-ca-ngay-vi" --step 3
```

### Real test results
- `pytest tests/test_tts_block_mode.py -q`
  - `11 passed`
- `pytest tests -q`
  - `125 passed`
- VI sample `to-tien-ban-lam-gi-ca-ngay-vi`, Step 2 rerun without input change:
  - `reused_block_count = 52`
  - `regenerated_block_count = 0`
  - `fallback_block_count = 0`
- VI sample `to-tien-ban-lam-gi-ca-ngay-vi`, Step 3 run twice:
  - both runs used block-aware stable-ts mode
  - `restart_count = 0`
  - timestamp SHA256 stayed identical across both runs:
    - `7CCDAA9895BB27CCF94745DD5F46E301827366697A4BF70FE9EBAA89E7D55AA0`
- Small VI smoke case with one sentence edited:
  - first run: `reused=0 regenerated=4 fallback=0`
  - second run after editing exactly one sentence: `reused=3 regenerated=1 fallback=0`

### Known notes
- Existing tracked output files under `output/to-tien-ban-lam-gi-ca-ngay-vi/` are still noisy in git status from earlier repo history; they were not part of the code patch scope.
- Real fallback-restart behavior is covered by unit tests; the VI end-to-end sample completed with `restart_count=0`, so no natural fallback occurred in that run.

## Session 2026-06-27 - Block-TTS architecture patch

### Goal
Patch step 2 / step 3 so both VieNeu and Kokoro can run in block-TTS mode while preserving the downstream invariant:
- `1 sentence = 1 timestamp = 1 image`
- Step 2 owns block audio + manifest only
- Step 3 owns `timestamps.json`

### Files changed
| File | Action |
|------|--------|
| `steps/text_units.py` | CREATED - shared sentence source of truth for step 2 / 3 |
| `steps/tts.py` | REWRITTEN - block builders, manifest, audio master rebuild, per-block fallback |
| `steps/transcribe.py` | REWRITTEN - block-aware stable-ts / faster-whisper alignment |
| `tests/test_tts_block_mode.py` | CREATED - splitter / builder smoke tests |

### Implemented architecture
- Shared sentence splitter:
  - strips metadata markers
  - splits only on `. ! ?`
  - keeps `sentence_index` and `paragraph_index`
- Step 2 block mode:
  - writes `tts_blocks/block_XXX.wav`
  - writes `tts_blocks/blocks.json`
  - writes `tts_blocks/diagnostics.json`
  - writes `audio_master.wav` + `audio.mp3`
  - renames stale `timestamps.json` to `timestamps.stale.json`
- VieNeu rules:
  - normalize with `PuncNormalizer().normalize(text, punc_norm=True)`
  - soft / hard caps `240 / 280 normalized chars`
  - target around `10-16s`, hard ceiling around `20s`
  - trim trailing silence only
- Kokoro rules:
  - measure with `pipeline.g2p(candidate_text, preprocess=True)`
  - soft / hard caps `420 / 500 phoneme chars`
  - reuse one `KPipeline` per run
- Step 3 block-aware mode:
  - prefers `tts_blocks/blocks.json` when newer than `timestamps.json`
  - aligns per block instead of full file
  - sequence-matches canonical vs aligned words with `SequenceMatcher`
  - can replace a single low-coverage block with sentence fallback audio only for that block

### Default configs
- VieNeu block config:
  - `block_soft_max_normalized_chars = 240`
  - `block_hard_max_normalized_chars = 280`
  - `block_target_max_seconds = 16`
  - `block_hard_max_seconds = 20`
  - `max_chars = 384`
  - `max_new_frames = 300`
- VieNeu infer defaults:
  - `temperature = 0.45`
  - `top_k = 25`
  - `top_p = 0.92`
  - `repetition_penalty = 1.18`
  - `crossfade_p = 0.0`
  - `silence_p = 0.15`
- Kokoro block config:
  - `block_soft_max_phoneme_chars = 420`
  - `block_hard_max_phoneme_chars = 500`
  - `official_phoneme_cap = 510`

### Tests run
```powershell
$python = "C:\Users\LEON_RM\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $python -m py_compile steps/text_units.py steps/tts.py steps/transcribe.py
& $python -m pytest tests/test_tts_block_mode.py -q
& $python -m pytest tests -q
& $python main.py --video-id "to-tien-ban-lam-gi-ca-ngay-vi" --step 2
& $python main.py --video-id "to-tien-ban-lam-gi-ca-ngay-vi" --step 3
$env:PYTHONIOENCODING = "utf-8"
& $python scripts/retime_prompts.py --video-id "to-tien-ban-lam-gi-ca-ngay-vi" --dry-run
```

### Output / A-B evidence
- VI production sample:
  - `output/to-tien-ban-lam-gi-ca-ngay-vi/tts_blocks/`
  - `audio_master.wav`
  - refreshed `audio.mp3`
  - refreshed `timestamps.json`
  - `tts_blocks/alignment_diagnostics.json`
- Existing compare artifacts still relevant for listening:
  - `output/to-tien-ban-lam-gi-ca-ngay-vi/compare_vieneu/`
  - `output/to-tien-ban-lam-gi-ca-ngay-vi/compare_vieneu_sentence_test/`

### Known issues
- Repo docs still contain older sentence-mode wording in `CLAUDE.md` / `AGENTS.md`.
- `ancient-child-surgery-31000-years` currently fails Kokoro block validation because one sentence is over the new `500` phoneme-char hard cap; script needs sentence splitting before block mode can pass.
- `retime_prompts.py --dry-run` on VI text needs UTF-8 console output in this shell.

### Leon needs to listen
- New `to-tien-ban-lam-gi-ca-ngay-vi/audio.mp3` from VieNeu block mode
- Compare it against:
  - `compare_vieneu/current_pipeline_fixed__thai_son.mp3`
  - `compare_vieneu/current_pipeline_fixed__binh_an.mp3`
  - `compare_vieneu_sentence_test/sentence_mode__bình_an.mp3`
- Decision points:
  - block mode has fewer unnatural resets than sentence legacy or not
  - `Thái Sơn` vs `Bình An`
  - any remaining repetition / voice drift worth tuning before wider rollout

## Session 2026-06-22 - Dual-track image generation

### Goal
Implement dual-track image generation:
- VN track (`--track vi`): VIDEO12B, FLUX.1-dev 12B, 20 steps, cinematic paleo art
- EN track (`--track en`): VIDEO9B, FLUX.2-klein 9B, 4 steps, ink sketch parchment

### Key findings
- `serverless_worker_flux32b/handler.py` 12B path does pass `negative_prompt` to `FluxPipeline`
- `serverless_worker/handler.py` 9B path accepts `negative_prompt` in schema but does not pass it into the distilled pipeline
- `scripts/generate_images.py` already forwards `negative_prompt` from `image_prompts.json`
- `config.py` was extended from one endpoint setting to track-aware routing

### Files changed
| File | Action |
|------|--------|
| `config.py` | MODIFIED - added track-aware endpoint and style config |
| `prompts/image_prompt_vi.txt` | CREATED |
| `prompts/image_prompt_en.txt` | CREATED |
| `scripts/generate_images.py` | MODIFIED - added `--track vi|en` |
| `tests/test_track_routing.py` | CREATED |

### Verification
- `tests/test_track_routing.py`: passed
- full `tests/`: passed in that session
