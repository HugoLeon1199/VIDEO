# handoff.md — Ghi chú kỹ thuật nội bộ

File này chứa thông tin kỹ thuật chi tiết, lịch sử debug, và quyết định thiết kế.
Để quy trình làm video, xem **AGENTS.md**. Để lệnh chạy, xem **CLAUDE.md**.

---

## VieNeu voice swap + subtitle burn - 2026-06-28

- Goal:
  - switch the attached Vietnamese video from edge-tts to VieNeu
  - keep the existing images untouched
  - produce a subtitle-burned video

### What changed

- `output/to-tien-cua-ban-chi-lam-viec-15-tieng-mot-tuan-vi/tts_config.json`
  - changed from `edge` to `vieneu`
  - voice set to `Thái Sơn`
  - mode set to `sentence`

### Commands run

- `python main.py --video-id to-tien-cua-ban-chi-lam-viec-15-tieng-mot-tuan-vi --step 2`
- `python main.py --video-id to-tien-cua-ban-chi-lam-viec-15-tieng-mot-tuan-vi --step 7`
- manual FFmpeg subtitle burn from the existing `subtitles.srt` to `final_subbed.mp4`

### Results

- Step 2 completed with VieNeu `Thái Sơn`
  - audio duration: about `565s`
- Step 7 completed without rerendering images
  - `final.mp4` rendered successfully
- Subtitle burn succeeded
  - `final_subbed.mp4` created successfully

## Vietnamese end-to-end rerun - 2026-06-28

- Goal in this session:
  - run the attached Vietnamese script from TTS through render end-to-end
  - verify what still works in the current checkout without comparing against other output folders

### What I ran

- Target video:
  - `output/to-tien-cua-ban-chi-lam-viec-15-tieng-mot-tuan-vi`
- Commands run:
  - `python main.py --video-id to-tien-cua-ban-chi-lam-viec-15-tieng-mot-tuan-vi --step 2`
  - `python main.py --video-id to-tien-cua-ban-chi-lam-viec-15-tieng-mot-tuan-vi --step 3`
  - `python scripts/generate_images.py --video-id to-tien-cua-ban-chi-lam-viec-15-tieng-mot-tuan-vi --backend vast_instance --no-qa --force --candidates 1 --workers 5`
  - `python scripts/generate_images.py --video-id to-tien-cua-ban-chi-lam-viec-15-tieng-mot-tuan-vi --backend runpod_serverless --no-qa --force --candidates 1 --workers 5`
  - `python main.py --video-id to-tien-cua-ban-chi-lam-viec-15-tieng-mot-tuan-vi --step 6`
  - `python main.py --video-id to-tien-cua-ban-chi-lam-viec-15-tieng-mot-tuan-vi --step 7`

### Results

- Step 2:
  - initially failed because `tts_config.json` had `rate=0%`
  - after changing it to `rate=-8%`, step 2 completed successfully
  - audio duration: about `657s`
- Step 3:
  - initially failed under `faster_whisper` because timestamps count did not match the script
  - after switching `transcribe_config.json` to `stable_ts` align, step 3 completed successfully
  - output count: `146` timestamps
  - `word_timestamps_diagnostics.json` reported subtitles not ready, so no approximate word export was fabricated
- Vast.ai image generation:
  - attempted first, but Vast could not find an offer that matched the repo's current filters
  - that path was blocked by offer availability, not by the video assets
- RunPod image generation:
  - succeeded and regenerated all `70` prompt scenes into `images/`
- Step 6:
  - passed
  - wrote `soundscape.json` with rule-based SFX
- Step 7:
  - passed
  - rendered `final.mp4`
  - final size: about `82.5 MB`

### Important notes

- I did not compare against other output folders; this run used the attached Vietnamese script and the current folder only.
- The current folder still has a grouped `image_prompts.json` with `70` scenes, so the render is scene-grouped rather than strict 1-sentence-1-image.
- No generated `output/` files were staged for commit.

## Vietnamese pipeline rerun audit - 2026-06-27

- Goal in this session:
  - rerun a real Vietnamese pipeline path end-to-end
  - include image-generation step behavior
  - identify where the current repo state still breaks

### Runs executed

- First VI candidate tested:
  - `output/nao-ban-nho-hon-nao-to-tien-vi`
- Commands run:
  - `python main.py --video-id nao-ban-nho-hon-nao-to-tien-vi --step 2`
  - `python main.py --video-id nao-ban-nho-hon-nao-to-tien-vi --step 3`
- Result:
  - step 2 completed with `edge-tts`
  - step 3 failed with:
    - `timestamps.json count 124 does not match script sentence count 114`
- Interpretation:
  - this video folder is not clean for the current strict `1 sentence = 1 timestamp = 1 image` pipeline
  - it should not be used as the primary regression sample for VI end-to-end checks until its script/timestamps/prompt set is rebuilt

### Clean VI regression sample used

- Switched to:
  - `output/buc-tranh-co-nhat-the-gioi-khong-nam-o-chau-au-vi`
- Reason:
  - current repo state still shows this folder as a clean VI case with:
    - `75` parsed script sentences
    - `75` timestamps

### Commands run on the clean VI sample

- `python main.py --video-id buc-tranh-co-nhat-the-gioi-khong-nam-o-chau-au-vi --step 2`
- `python main.py --video-id buc-tranh-co-nhat-the-gioi-khong-nam-o-chau-au-vi --step 3`
- `python main.py --video-id buc-tranh-co-nhat-the-gioi-khong-nam-o-chau-au-vi --step 4`
- `python scripts/generate_images.py --video-id buc-tranh-co-nhat-the-gioi-khong-nam-o-chau-au-vi --workers 5 --candidates 1`
- `python main.py --video-id buc-tranh-co-nhat-the-gioi-khong-nam-o-chau-au-vi --step 6`
- `python main.py --video-id buc-tranh-co-nhat-the-gioi-khong-nam-o-chau-au-vi --step 7`

### Observed results

- Step 2:
  - passed
  - `edge-tts` regenerated `audio.mp3`
  - audio duration about `480.89s`
- Step 3:
  - passed
  - produced `75` timestamp segments
- Step 4:
  - failed immediately because this shell had no text-model key available
  - exact symptom:
    - `Neither ANTHROPIC_API_KEY nor GEMINI_API_KEY set. Add to .env file.`
  - practical fallback used:
    - continue with the existing checked-in `image_prompts.json`
- Step 5:
  - passed in resume mode
  - all `155` scenes were already done and skipped cleanly
- Step 6:
  - passed
  - wrote `soundscape.json` with rule-based SFX only
- Step 7:
  - initial run failed in `_mix_sfx_audio(...)` with:
    - `FileNotFoundError: [WinError 206] The filename or extension is too long`
  - root cause:
    - the FFmpeg command line for the SFX mix becomes too long on Windows when `soundscape.json` has many events

### Practical completion workaround used

- To finish the video without changing repo code:
  - temporarily renamed `soundscape.json`
  - reran step 7 so render used voice-only audio
  - restored `soundscape.json` afterward
- Result:
  - `final.mp4` rendered successfully
  - final size about `75.9 MB`

### Important conclusions from this rerun

- Current VI pipeline status is mixed:
  - text-to-speech works
  - transcription works on clean folders
  - image generation resume works
  - render works without SFX mix explosion
- Two real blockers remain for a fully healthy VI A-to-Z path:
  - missing text-model key in this environment prevents step 4 regeneration
  - step 7 SFX mixing can still hit Windows command-line length limits
- Best current VI regression sample:
  - `buc-tranh-co-nhat-the-gioi-khong-nam-o-chau-au-vi`
- Not safe as a regression sample right now:
  - `nao-ban-nho-hon-nao-to-tien-vi`

## Kokoro Voice Lab simplification - 2026-06-27

- Refactored `scripts/kokoro_voice_lab.py` into a strict round-gated workflow:
  - `base`
  - `topic`
  - `blend`
  - `speed`
  - `final`
  - `report`
- Backward-compatible aliases kept:
  - `topics -> topic`
  - `blends -> blend`
  - `longform -> final`
- Production TTS was **not** modified.

### Workflow changes

- Artifacts now carry:
  - `round`
  - `round_order`
  - `lineage`
- Manifest now carries:
  - `active_round`
  - `round_counts`
  - `round_configs`
- Review/report storage switched from `scores.csv` to `decisions.csv`.
- All commands accept `--decisions <path>`; default is `output/voice_lab/decisions.csv`.
- Finalists are now round-gated strictly from the immediately previous round only:
  - `topic` reads `base` decisions
  - `blend` reads `topic` decisions
  - `speed` reads `blend` decisions
  - `final` reads `speed` decisions

### Audio simplification rules now implemented

- Base round:
  - generates all English base voices
  - one richer calibration sample only
  - target `20-25s`, acceptable `18-30s`
- Topic round:
  - max `6` finalists
  - exactly one `topic_reel` per finalist
  - no per-topic audio fanout
- Blend round:
  - top `3` topic finalists only
  - includes `3` base comparison artifacts in the blend round itself
  - includes up to `6` real blend artifacts
  - primary ratios are `70/30`, with at most one `80/20` and one `50/50`
- Speed round:
  - top `2` blend-round picks only
  - accepts source kind `base` or `blend`
  - only `0.95` and `0.98`
  - target `25-30s`, acceptable `22-35s`
- Final round:
  - top `2` speed artifacts only
  - preserves lineage end-to-end:
    - base
    - blend if present
    - speed
    - final
  - uses the existing block-mode style longform render path inside the lab
  - exports:
    - main audio
    - `blocks.json`
    - up to `5` suspicious boundary clips
  - hard-required duration `90-120s`

### Review UI changes

- Blind review now uses simple decisions only:
  - `Keep`
  - `Maybe`
  - `Reject`
- Voice / family / source stay hidden before choice.
- Reveal button stays disabled until a decision is chosen.
- Browser `localStorage` is used only to restore UI state on reload.
- Export from the HTML review page is always `decisions.csv`.
- `report` ranks only candidates from `manifest.active_round`.

### Tests completed

- `python -m py_compile scripts/kokoro_voice_lab.py tests/test_kokoro_voice_lab.py`
- `python -m pytest tests/test_kokoro_voice_lab.py -q`
  - result: `3 passed`
- `python -m pytest tests -q`
  - result: `128 passed`

### Files changed

- `scripts/kokoro_voice_lab.py`
- `tests/test_kokoro_voice_lab.py`
- `.ai/CURSOR_WORKLOG.md`
- `handoff.md`

### Important repo note

- Existing dirty files under `output/` were left untouched and should stay out of commits for this feature patch.

## Kokoro Voice Lab bootstrap - 2026-06-27

- Added `scripts/kokoro_voice_lab.py` as an independent blind-test lab for Kokoro voices.
- The lab does not modify production TTS.
- It supports:
  - `base`
  - `topics`
  - `blends`
  - `speed`
  - `longform`
  - `report`
- Lab outputs are ignored under `output/voice_lab/`.
- Smoke test results:
  - `base --limit 2`
  - `topics`
  - `blends --limit 1`
  - `speed --limit 1`
  - `longform --limit 1`
  - `report`
  - follow-up full base pass: `base`
  - final manifest counts: `base=28`, `blend=1`, `speed=4`, `longform=1`
- Mapping repair:
  - longform initially failed on a missing blend mapping entry
  - lab now reconstructs missing mapping entries from `manifest.json` before longform resolution
- Longform smoke exported suspicious boundary clips under `output/voice_lab/longform/<blind_id>/`

## Step 2/3 block-TTS architecture patch - 2026-06-27

- Implemented a shared sentence source of truth in `steps/text_units.py`.
  - strips trailing metadata blocks
  - splits only on `. ! ?`
  - preserves `sentence_index` and `paragraph_index`
- Reworked `steps/tts.py` production defaults for VieNeu and Kokoro to `mode=block`.
  - Step 2 now writes:
    - `tts_blocks/block_XXX.wav`
    - `tts_blocks/blocks.json`
    - `tts_blocks/diagnostics.json`
    - `tts_blocks/needs_alignment.json`
    - `audio_master.wav`
    - `audio.mp3`
  - Step 2 renames stale `timestamps.json` to `timestamps.stale.json` before step 3 regenerates timing.
  - `mode=sentence_legacy` and `mode=paragraph_audit` were kept for debug / compare runs.
- VieNeu block builder details now implemented:
  - normalized-char measurement via `PuncNormalizer().normalize(text, punc_norm=True)`
  - default soft / hard block caps: `240 / 280 normalized chars`
  - target duration heuristic: about `10-16s`, hard ceiling about `20s`
  - no automatic comma splitting in production block mode
  - trailing silence is trimmed only at the tail, not the head
- Kokoro block builder details now implemented:
  - phoneme measurement via `pipeline.g2p(candidate_text, preprocess=True)`
  - default soft / hard block caps: `420 / 500 phoneme chars`
  - production path rejects a single sentence that already exceeds the hard cap
  - block synth reuses a single `KPipeline` instance for the whole video run
- Reworked `steps/transcribe.py` to prefer `tts_blocks/blocks.json` whenever it is newer than `timestamps.json`.
  - `stable_ts` path aligns one block WAV at a time against that block's canonical sentences
  - `faster_whisper` path also runs block-aware matching per block instead of whole-file greedy matching
  - canonical words vs aligned words are matched with `SequenceMatcher`, not raw word-count slicing
  - if a block falls below coverage / sentence-match acceptance, step 3 can regenerate that block into sentence fallback audio only for that block
- Added block-level sentence fallback plumbing:
  - `steps.tts.materialize_sentence_fallback_for_block(video_dir, block_index)`
  - fallback writes `block_XXX_sentence_YYY.wav`
  - manifest keeps `fallback_level=2` and `fallback_segments`
  - `audio_master.wav` / `audio.mp3` are rebuilt from the manifest after fallback replacement

### Files changed

- `steps/text_units.py`
- `steps/tts.py`
- `steps/transcribe.py`
- `tests/test_tts_block_mode.py`

### Tests and runs completed

- `python -m py_compile steps/text_units.py steps/tts.py steps/transcribe.py`
- `python -m pytest tests/test_tts_block_mode.py -q`
- `python -m pytest tests -q`
- Real VI run:
  - `python main.py --video-id to-tien-ban-lam-gi-ca-ngay-vi --step 2`
  - `python main.py --video-id to-tien-ban-lam-gi-ca-ngay-vi --step 3`
- Retiming compatibility check:
  - `PYTHONIOENCODING=utf-8 python scripts/retime_prompts.py --video-id to-tien-ban-lam-gi-ca-ngay-vi --dry-run`

### Verified outputs

- `output/to-tien-ban-lam-gi-ca-ngay-vi/tts_blocks/`
- `output/to-tien-ban-lam-gi-ca-ngay-vi/audio_master.wav`
- `output/to-tien-ban-lam-gi-ca-ngay-vi/audio.mp3`
- `output/to-tien-ban-lam-gi-ca-ngay-vi/timestamps.json`
- `output/to-tien-ban-lam-gi-ca-ngay-vi/tts_blocks/alignment_diagnostics.json`

### Result snapshot

- VI sample `to-tien-ban-lam-gi-ca-ngay-vi`
  - `154` sentences -> `52` VieNeu blocks
  - `audio_master.wav` duration: about `531.599s`
  - `timestamps.json` count: `154`
  - block alignment diagnostics reported full coverage for all `52` blocks in this run
- EN smoke result on `ancient-child-surgery-31000-years`
  - Kokoro block mode correctly failed validation because at least one single sentence exceeded the `500` phoneme-char hard cap
  - this is expected under the new invariant; that script needs sentence splitting before it can pass production block mode

### Known issues / follow-up

- `CLAUDE.md` and `AGENTS.md` still contain older wording about sentence-mode TTS and should be refreshed in a follow-up docs pass.
- `scripts/retime_prompts.py --dry-run` on Vietnamese text still needs `PYTHONIOENCODING=utf-8` in this Windows console or it can crash on `cp1252` printing.
- Existing `image_prompts.json` for `to-tien-ban-lam-gi-ca-ngay-vi` is still an older grouped-scene artifact, so retime dry-run shows long holds; that is expected until prompts are regenerated under strict `1 sentence = 1 image`.

### Leon still needs to listen / decide

- Compare new VieNeu block-mode audio against earlier sentence-legacy / compare artifacts for:
  - repetition
  - pitch reset across blocks
  - voice drift
  - whether `Thái Sơn` is still acceptable or `Bình An` should become the preferred audit candidate

## Step 2/3 production-hardening follow-up - 2026-06-27

- Hardened commit `56eea65` without expanding feature scope.
- Verified runtime versions before pinning:
  - `vieneu 3.0.5`
  - `stable-ts 2.19.1`
- Verified runtime APIs in this environment:
  - `import stable_whisper` succeeded
  - `from vieneu import Vieneu; Vieneu()` succeeded
  - active VieNeu runtime reported `sample_rate = 48000`
  - current `tts.infer(...)` signature matches the params used by repo code
- `requirements.txt` now pins the exact versions that were actually verified and exercised end-to-end:
  - `vieneu==3.0.5`
  - `stable-ts==2.19.1`

### Production hardening shipped

- `steps/tts.py`
  - Added deterministic block cache hashing over:
    - cache schema version
    - engine
    - voice
    - speed
    - full `block_config`
    - full effective `infer_params`
    - block text
    - engine package version
  - Cached WAV reuse now requires:
    - file exists
    - non-empty
    - readable
    - correct sample rate
    - duration `> 0`
  - Step 2 no longer deletes `tts_blocks/` unconditionally; unchanged blocks can be reused.
  - Added `reused_block_count`, `regenerated_block_count`, and `fallback_block_count` to manifest and diagnostics.
  - Fixed `mode=sentence_legacy` so `engine=vieneu` uses real VieNeu sentence synthesis instead of falling into Kokoro.
  - Fixed Kokoro block builder so candidate overflow closes the current block and starts a new block; only a single over-limit sentence is a hard fail.
  - Kokoro runtime now respects per-video `voice` and `speed`.
  - TTS config loader now accepts UTF-8 BOM via `utf-8-sig`.
- `steps/transcribe.py`
  - Block-aware mode is selected from artifact shape, not manifest-vs-timestamp `mtime`.
  - If `blocks.json` says `mode=block` and `audio_master.wav` exists, step 3 always aligns by block.
  - After fallback rebuild:
    - reload manifest
    - restart full pass
    - reuse loaded model
    - cap restart guard at `2`
    - fail clearly if a `fallback_level=2` block is still invalid
  - Transcribe config loader now also accepts UTF-8 BOM.
- `output/to-tien-ban-lam-gi-ca-ngay-vi/transcribe_config.json`
  - Updated to:
    - `engine=stable_ts`
    - `mode=align`
    - `model=medium`
    - `language=vi`
    - `device=cpu`

### Tests and real runs completed

- Static + unit + suite:
  - `python -m py_compile steps/text_units.py steps/tts.py steps/transcribe.py tests/test_tts_block_mode.py`
  - `python -m pytest tests/test_tts_block_mode.py -q` -> `11 passed`
  - `python -m pytest tests -q` -> `125 passed`
- Real VI sample `output/to-tien-ban-lam-gi-ca-ngay-vi`
  - Step 2 rerun with unchanged input:
    - `reused_block_count = 52`
    - `regenerated_block_count = 0`
    - `fallback_block_count = 0`
  - Step 3 run twice:
    - both runs stayed in block-aware `stable_ts` mode
    - `restart_count = 0`
    - `timestamps.json` SHA256 stayed identical across both runs
- Minimal block-cache smoke case:
  - first run regenerated all blocks
  - after editing exactly one sentence:
    - `reused_block_count = 3`
    - `regenerated_block_count = 1`
    - `fallback_block_count = 0`

### Remaining notes

- `.ai/CURSOR_WORKLOG.md` was updated with the same verification commands and test outcomes for review continuity.
- Existing tracked output churn under `output/to-tien-ban-lam-gi-ca-ngay-vi/` was intentionally left out of this patch scope except `transcribe_config.json`.
- The real VI sample did not trigger a natural fallback, so fallback-restart behavior is primarily proven by unit coverage in this patch.

## Final TTS continuity audit - 2026-06-27

- Audited `output/to-tien-ban-lam-gi-ca-ngay-vi` for continuity after the production hardening patch.
- Input artifacts checked:
  - `audio_master.wav`
  - `tts_blocks/blocks.json`
  - `tts_blocks/diagnostics.json`
  - `tts_blocks/alignment_diagnostics.json`
- Findings:
  - no fallback sentence mode in the manifest
  - no retry params different from default in the manifest
  - `reused_block_count = 52`
  - `regenerated_block_count = 0`
  - `fallback_block_count = 0`
  - voice stayed constant across the run
  - block gap stayed at the configured `300ms`
- Exported manual listening clips under:
  - `output/to-tien-ban-lam-gi-ca-ngay-vi/continuity_audit/`
- Selected suspicious boundaries:
  - `12`, `13`, `16`, `21`, `23`, `28`, `37`, `43`, `46`, `49`
- Quantitative note:
  - highest loudness-contrast boundaries were `37` and `46`
  - shortest blocks in the run were `7`, `52`, and `17`
- Decision:
  - no block was regenerated
  - cache was preserved as-is
  - if Leon hears a defect in one of the exported clips, only that specific block should be regenerated next

## Step 4 one-sentence-one-image update - 2026-06-27

- `steps/image_prompts.py` now treats `1 sentence = 1 scene = 1 image` as a hard invariant.
- Updated both `VI_SYSTEM_PROMPT` and `EN_SYSTEM_PROMPT` to require exactly one sentence index per scene, exact `scene_text`, full `1..N` coverage, and no grouping/splitting.
- Added language/profile selection in step 4:
  - `-vi` video ids or Vietnamese-looking scripts use `VI_SYSTEM_PROMPT` + `VI_NEGATIVE_PROMPT`
  - all others use `EN_SYSTEM_PROMPT` + `EN_NEGATIVE_PROMPT`
- Fixed a real bug in `run()` where step 4 previously always used `VI_SYSTEM_PROMPT`, even for English videos.
- Added client-side validation before `_map_scene_times()`:
  - `sentences` must exist and contain exactly one integer
  - indices must be in `1..N`
  - all indices must be covered exactly once
  - `scene_text` must be non-empty
  - invalid model output now fails fast with explicit logs
- Added an early hard error when `timestamps.json` count and parsed script sentence count do not match in normal 1:1 mode.
- `CLAUDE.md` updated to document:
  - new invariant `1 sentence = 1 scene = 1 image`
  - sentence writing guideline `3-7s`, roughly `10-25` Vietnamese words
  - step 4 should fail early if script/timestamps are not 1:1 compatible
- Smoke-test method used because API keys were unavailable in the environment:
  - monkeypatched `_call_claude()` to return exact 1:1 scenes
  - verified step 4 language selection, prompt mapping, output count, and negative prompt selection
- Smoke-test results:
  - `output/smoke-ancient-en/` → `92` scenes, `92` timestamps, first `[1]`, last `[92]`, EN prompt selected, EN negative prompt selected
  - `output/smoke-ancient-vi/` → `154` scenes, `154` timestamps, first `[1]`, last `[154]`, VI prompt selected, VI negative prompt selected
- `scripts/retime_prompts.py --dry-run` results:
  - EN smoke case: `strictly-increasing: True`, `non-positive durations: 0`
  - VI smoke case: `strictly-increasing: True`, `non-positive durations: 0`
- Important repo data note discovered during testing:
  - several existing output folders are not yet compatible with strict 1:1 mode because `script.txt` sentence count and `timestamps.json` count already differ
  - confirmed clean EN case: `ancient-child-surgery-31000-years` (`92/92`)
  - confirmed clean VI cases: `to-tien-ban-lam-gi-ca-ngay-vi` (`154/154`), `buc-tranh-co-nhat-the-gioi-khong-nam-o-chau-au-vi` (`75/75`)
  - example mismatch case: `brain-smaller-than-ancestors-en` (`119` parsed sentences vs `109` timestamps)

## VieNeu compare audit - 2026-06-27

- Added `scripts/audit_vieneu_compare.py` to compare three VieNeu render modes on the same `script.txt` without touching production `audio.mp3` or render outputs.
- Extended `steps/tts.py` with optional VieNeu diagnostics output for audit runs only. Default production behavior is unchanged.
- Audit target: `output/to-tien-ban-lam-gi-ca-ngay-vi/`.
- Artifacts written under `output/to-tien-ban-lam-gi-ca-ngay-vi/compare_vieneu/`:
  - `full_script.mp3`
  - `current_pipeline.mp3`
  - `chunked_helper_params.mp3`
  - `*_diagnostics.json`
  - `*_timestamps.json`
  - `compare_report.json`
  - `compare_report.md`
  - `excerpts/` clips for manual listening
- Confirmed active production VieNeu path in `steps/tts.py` is `_run_vieneu_sentence_mode()`, not the legacy whole-script `_vieneu_tts()` path.
- Helper-script mismatch confirmed:
  - `scripts/_gen_voice_compare.py` still uses full-script VieNeu with `temperature=0.5`, `top_k=20`, `top_p=0.9`, `max_chars=256`, `crossfade_p=0.1`, `silence_p=0.12`.
  - `scripts/_duc_tri_gen.py` uses a different ref-code workflow and is not comparable to the production `voice=` path.
- Silence metrics from `compare_report.json`:
  - `full_script`: `589.35s`, silence ratio `0.3097`, longest silence `23.65s`, `5` blocks over `1.5s`
  - `current_pipeline`: `553.95s`, silence ratio `0.1880`, longest silence `1.95s`, `2` blocks over `1.5s`
  - `chunked_helper_params`: `579.60s`, silence ratio `0.2114`, longest silence `19.15s`, `1` block over `1.5s`
- Timing checks for current pipeline:
  - `154` timestamp entries
  - `154` script sentences
  - final timestamp end `553.621s`
  - current audio duration `553.921s`
  - delta `0.3s`
- Current evidence says:
  - the chunked production VieNeu path is materially better than full-script VieNeu for long-silence control
  - helper decode settings (`temperature/top_k/top_p/max_chars/crossfade`) still produce long silences even when using the same chunking logic
  - this points more strongly to inference-parameter behavior than to chunking alone for the silence bug
- Still not auto-resolved:
  - repeated-word behavior
  - odd prosody / pitch resets
  - audible join artifacts
  - voice identity drift
- Those need manual listening against the generated excerpt clips in `compare_vieneu/excerpts/`.
- Follow-up tuning in the same session:
  - Updated `steps/tts.py` so non-final VieNeu comma chunks are synthesized with an explicit terminal period in `synth_text` while preserving original `text` for timestamps/diagnostics.
  - Reduced comma-gap control from implicit `silence_ms/2` to explicit `comma_gap_ms` and increased trim keep cushion for the fixed path.
  - Added a legacy/fixed A/B mode to `scripts/audit_vieneu_compare.py`.
- New A/B artifacts for direct listening:
  - `compare_vieneu/legacy_chunked.mp3`
  - `compare_vieneu/current_pipeline_fixed.mp3`
  - paired clips under `compare_vieneu/excerpts/legacy_chunked/` and `compare_vieneu/excerpts/current_pipeline_fixed/`
- New A/B metrics:
  - `legacy_chunked`: `553.95s`, silence ratio `0.1880`, longest silence `1.95s`
  - `current_pipeline_fixed`: `570.15s`, silence ratio `0.2092`, longest silence `1.95s`
- Interpretation of the follow-up:
  - the fix did not improve silence metrics; that was already controlled in the legacy chunked path
  - the fix intentionally changes chunk prosody/EOS behavior, so the real decision point is manual listening for repetition, joins, and unnatural intonation
  - if `current_pipeline_fixed` sounds worse, revert to legacy chunking and try a smaller prosody change instead of keeping the punctuation-injection path blindly
- Added a new audit-only VieNeu helper mode in `steps/tts.py`:
  - `_split_script_paragraphs()` strips metadata and keeps blank-line-separated paragraph blocks
  - `_run_vieneu_paragraph_mode()` synthesizes one whole paragraph per VieNeu call, with no sentence split and no comma split
  - this is for listening comparison only, not production routing
- `scripts/audit_vieneu_compare.py` now renders `paragraph_whole_blocks.mp3` by default alongside the chunked variants.
- Fresh paragraph-mode run on `output/to-tien-ban-lam-gi-ca-ngay-vi/compare_vieneu/`:
  - `paragraph_whole_blocks.mp3`
  - `paragraph_whole_blocks_diagnostics.json`
  - excerpt clips under `compare_vieneu/excerpts/paragraph_whole_blocks/`
- Paragraph whole-block metrics:
  - duration `777.90s`
  - silence ratio `0.4974`
  - longest silence `23.6s`
  - silence blocks over `1.5s`: `13`
- Current interpretation:
  - paragraph-level whole-block synthesis keeps more context, so it is worth subjective listening for voice consistency
  - but on this script it clearly reintroduces the old EOS/silence blow-up behavior, so it is not a safe production replacement as-is
  - next decision should be based on listening tradeoff: whether the paragraph version sounds materially more natural despite the long pauses
- Voice preset comparison follow-up:
  - Added voice-slugged outputs under `compare_vieneu/` so voices are easier to compare side-by-side.
  - Current `Bình An` run produced:
    - `current_pipeline_fixed__binh_an.mp3`
    - `paragraph_whole_blocks__binh_an.mp3`
  - After copying the earlier `Thái Sơn` files to explicit names, the easy comparison set is:
    - `current_pipeline_fixed__thai_son.mp3`
    - `current_pipeline_fixed__binh_an.mp3`
    - `paragraph_whole_blocks__thai_son.mp3`
    - `paragraph_whole_blocks__binh_an.mp3`
  - Metric snapshot:
    - `current_pipeline_fixed__binh_an`: duration `582.05s`, silence ratio `0.2279`, longest silence `3.25s`
    - `paragraph_whole_blocks__binh_an`: duration `799.35s`, silence ratio `0.5507`, longest silence `23.65s`
  - Interpretation:
    - `Bình An` does not look like a magic fix for the repetition/silence issue, but it is still worth subjective listening because the voice color may be calmer.
    - The main production blocker remains the chunking/EOS behavior, not just the voice preset.

## Kokoro English compare render - 2026-06-27

- Added `scripts/render_kokoro_compare.py` to render compare-only English Kokoro audio without touching production `audio.mp3`.
- Script interface:
  - `python scripts/render_kokoro_compare.py --video-id ancient-child-surgery-31000-years --video-id what-ancient-humans-did-all-day`
- Behavior:
  - uses the current production Kokoro path in `steps/tts.py` (`am_fenrir`, speed `0.95`)
  - reads full `script.txt` in one pass, with no sentence/comma chunking
  - writes artifacts under `output/<video-id>/compare_kokoro/`
  - writes `tts_diagnostics.json` and four excerpt clips: `first_30s`, `middle_30s`, `near_end_30s`, `final_30s`
  - also writes aggregate summaries:
    - `output/_kokoro_compare_summary.json`
    - `output/_kokoro_compare_summary.md`
- Fresh renders completed successfully:
  - `ancient-child-surgery-31000-years`
    - output: `compare_kokoro/full_script_kokoro.mp3`
    - duration: `385.975s`
    - script sentence count: `92`
    - fallback used: `false`
  - `what-ancient-humans-did-all-day`
    - output: `compare_kokoro/full_script_kokoro.mp3`
    - duration: `465.700s`
    - script sentence count: `154`
    - fallback used: `false`
- Current conclusion from architecture + artifacts:
  - English/Kokoro is structurally different from VieNeu in this repo because it reads the full script in one pass instead of stitching many TTS chunks together.
  - That means it should be judged mainly on listening continuity and voice consistency, not on chunk-boundary artifact behavior, because those boundaries do not exist in the same way here.

## Image anatomy review - 2026-06-22

- Reviewed malformed human anatomy in FLUX.2 Klein output (extra arms, legs, and merged bodies).
- Root cause is generation quality, not RunPod GPU capacity, aspect ratio, or FFmpeg rendering.
- Current production settings use the distilled 4-step model at guidance 1.0 and only one candidate (seed 11001).
- The worker ignores negative prompts for FLUX.2 Klein, and the pipeline always promotes candidate 1 without visual quality scoring.
- Crowded prompts are the highest-risk cases: many people, overlapping poses, dancing, carrying, repairing, and close-up hands.
- Recommended next improvement: simplify high-risk compositions to one or two clearly separated people, generate 2-3 seeds only for risky scenes, then select or validate the clean candidate before render.
- Do not assume that increasing inference steps will fix anatomy; this distilled model is designed around its fast-step workflow.

## Model recommendation - 2026-06-22

- Repo is currently pinned to `black-forest-labs/FLUX.2-klein-4B` in `serverless_worker/handler.py` and `serverless_worker/model_loader.py`.
- Production config is still optimized around this model: `IMAGE_STEPS = 4`, `IMAGE_GUIDANCE_SCALE = 1.0`, one candidate, serverless queue workflow.
- Official FLUX model cards indicate `FLUX.2-klein-4B` fits in about 13GB VRAM and is intended for consumer GPUs, while `FLUX.2-klein-9B` needs about 29GB VRAM and is non-commercial.
- Current RunPod target class in this project is 24GB serverless GPUs (A5000/L4/3090/4090). Therefore 9B is not the practical default for this pipeline.
- Cost-optimized recommendation for this repo: keep `FLUX.2-klein-4B` and prefer the cheapest 24GB option available, which is currently RTX A5000 when pricing is favorable.
- If higher visual quality is needed, first spend budget on selective multi-seed regeneration for risky scenes rather than switching the whole pipeline to a larger model.
- Only consider moving to 9B if licensing is acceptable and the deployment changes to a larger GPU tier with enough VRAM and tolerance for slower cold starts.
- Clarification: `FLUX.2-klein-9B` is described by BFL as the "flagship small model" with excellent quality and prompt adherence, so it should improve visual quality over 4B in many scenes, but it is not evidence that it is the strongest overall model across all hosted/proprietary BFL offerings.

## Trạng thái pipeline (2026-06-21)

Pipeline hoàn chỉnh và đã sản xuất 4 video:

| Video | Ngôn ngữ | Scenes | File size | Ghi chú |
|-------|----------|--------|-----------|---------|
| `ancient-child-surgery-31000-years` | EN | 131 | - | Bản gốc tiếng Anh |
| `ancient-child-surgery-31000-years-vi` | VI | 131 | 111.8 MB | Cinematic NatGeo style |
| `what-ancient-humans-did-all-day` | EN | ~144 | - | Bản gốc tiếng Anh |
| `what-ancient-humans-did-all-day-vi` | VI | 153 | 144.7 MB | Người quê aesthetic |
| `buc-tranh-co-nhat-the-gioi-khong-nam-o-chau-au-vi` | VI | 155 | 93.2 MB | Cartoon người quê, cave-art story |

---

## Latest session update - 2026-06-21

**Video completed:** `buc-tranh-co-nhat-the-gioi-khong-nam-o-chau-au-vi`

- Input script copied from `C:\Users\LEON_RM\Downloads\script.txt`.
- Title/topic: `Bức Tranh Cổ Nhất Thế Giới Không Nằm Ở Châu Âu` (Sulawesi cave art / oldest painting not in Europe).
- No runtime code changes were made.
- TTS: `edge-tts vi-VN-NamMinhNeural`, rate `-8%`, output `audio.mp3`, duration about `605.21s`.
- Transcription: faster-whisper `medium`, language `vi`, output `timestamps.json`, `155` segments.
- Image prompts: generated locally, `155` prompts matched to `155` timestamps.
- Visual decision: original simple cartoon style with one recurring rural Vietnamese villager narrator, not copied from reference screenshots; 16:9, no text.
- Image generation: RunPod endpoint `9hs6ppcsssn990`, `candidates=1`, `workers=5`, output `155/155` images, `0` failed, total about `134.7s`.
- Render: `final.mp4`, 1920x1080, 30fps, H.264 + AAC stereo, duration `605.208s`, size `93.2 MB`.
- Verification artifact: `review_contact.jpg` generated for quick visual QA.
- Subtitle note: current `steps/render_video.py` creates `subtitles.srt` sidecar, but does not burn subtitles into `final.mp4`; this was left unchanged because the user requested not to edit code unless needed.
- Prompt update: `prompts/script_prompt.txt` was rewritten as a voice-first script prompt. It now requires one blank-line-separated paragraph per production chunk so future TTS, subtitles, image prompts, and render timing can share the same script-derived timeline.
- Prompt correction: removed outdated wording that said the pipeline feeds each sentence to TTS individually. The intended unit is now clearly paragraph/chunk first, with clean sentences inside each chunk.
- Prompt review follow-up: addressed Claude feedback by replacing Whisper-specific wording, adding Vietnamese rural visual examples, adding Vietnamese pacing guidance, merging duplicate audit checks, and standardizing sentence length to 35 words.
- Prompt merge review: Claude merged brand/psychology/stage workflow back into `prompts/script_prompt.txt`. Final wording was lightly adjusted so blank-line chunking is described as a production workflow convention, and Research Notes stay optional instead of appearing after every script.

**Remaining optional work:**

- If hard captions are desired, either add real subtitle burn-in support to `steps/render_video.py` or render a separate subtitled copy with FFmpeg.
- Human should still watch the final video once for subjective art quality before upload.

---

## RunPod Serverless — Cấu hình hoạt động

**Endpoint:** `9hs6ppcsssn990` (emberlore-flux2-klein)
**GitHub:** `HugoLeon1199/VIDEO` branch `main`
**Build context:** `serverless_worker`
**Dockerfile path:** `serverless_worker/Dockerfile`
**Network Volume:** `jqscreri1e` (50GB, US-IL-1) → `/runpod-volume`
**GPU:** 24GB (L4/A5000/RTX3090/RTX4090), locked US-IL-1 do volume
**idleTimeout:** 120s, **execTimeout:** 900s, **FlashBoot:** on

### Quyết định thiết kế quan trọng

**ENTRYPOINT thay vì CMD:**
RunPod endpoint có "Container Start Command" = `Endpoint` (cấu hình sai còn sót). Field này ghi đè `CMD` của Dockerfile nhưng không ghi đè `ENTRYPOINT`. Nếu dùng `CMD`, container crash với `exec Endpoint: no such file`. Workaround: dùng `ENTRYPOINT ["python", "-u", "/handler.py"]`.

Fix gốc (tùy chọn, không bắt buộc): vào RunPod UI → Edit endpoint → xóa "Container Start Command" → Save.

**Lazy-load model:**
Model phải load bên trong `handler()`, không được load ở import-time. Nếu load lúc import, `runpod.serverless.start()` chưa kịp chạy → serverless loop không attach queue → job kẹt IN_QUEUE mãi.

**python:3.11.1-slim thay vì runpod/pytorch hay runpod/base:**
- `runpod/pytorch`: dùng cho Pods (SSH/Jupyter), entrypoint không phù hợp serverless
- `runpod/base`: cũng gây vấn đề job không được nhặt
- `python:slim`: sạch nhất, kiểm soát hoàn toàn, không có entrypoint conflict

**HF_TOKEN:**
Inject qua RunPod Secrets. Model FLUX.2 Klein là Apache-2.0, không bị gated, nhưng vẫn cần token để truy cập HuggingFace Hub.

---

## TTS tiếng Việt

**Đã thử và bỏ — F5-TTS voice cloning:**
- Triển khai lên RunPod endpoint `syo26j5rexxrbl`
- Thử checkpoint: `hynt/F5-TTS-Vietnamese-ViVoice` (vocab mismatch), `giahy2507/f5-tts-vietnamese`
- Kết quả: âm thanh không ra tiếng Việt, nghe như tiếng Trung hoặc ngôn ngữ khác
- Kết luận: không có checkpoint Vietnamese F5-TTS chất lượng tốt ở public
- **Quyết định:** dùng `edge-tts vi-VN-NamMinhNeural` — đáng tin cậy, miễn phí

**edge-tts gotchas:**
- Em dash `—` (U+2014, ord=8212) → "No audio received" (không báo lỗi, chỉ silent fail)
- Ký tự `!` đầu từ (vd `!Kung`) → lỗi
- Ký tự `/` trong từ (vd `Ju/'hoansi`) → lỗi
- Fix: replace `—` → `, `, `!Kung` → `Kung`, `Ju/'hoansi` → `Ju hoansi`

---

## Whisper tiếng Việt

- Model `base` không đủ tốt cho tiếng Việt → phải dùng `medium`
- Phải set `language="vi"` explicit, không để auto-detect
- Config qua `transcribe_config.json` trong thư mục video

---

## Image prompts — style guidelines

### English videos (NatGeo cinematic)
```
Cinematic wide shot of [scene], photorealistic, natural lighting, shallow depth of field, 16:9, no text
```
Ví dụ: `Cinematic wide shot of ancient surgical tools made of bone and obsidian arranged on a flat stone, photorealistic, natural lighting, shallow depth of field, 16:9, no text`

### Vietnamese videos — người quê aesthetic
Style: earthy, grounded, human moments, rural/village ancient people.
```
[Scene with rural/village ancient people doing natural activity], photorealistic, cinematic, 16:9
```
Tránh: doodle style, stick figures, EMBERLORE style, cartoon, abstract.
Dùng: firelight, dappled shade, dramatic but simple compositions, wide shots, close-ups of hands/faces.

---

## Vấn đề kỹ thuật đã giải quyết (lịch sử)

### FFmpeg WinError 206 — command line quá dài
Video nhiều ảnh → command line FFmpeg vượt giới hạn Windows.
Fix: two-pass render (mỗi ảnh → clip riêng → concat).

### Image gen exit 255 với PowerShell
`2>&1` redirect với tty rất nhiều output → buffer overflow PowerShell.
Fix: dùng `Tee-Object -FilePath log.txt` thay vì capture trực tiếp.

### Ảnh 16:9 từ FLUX
FLUX.2 Klein output theo `IMAGE_WIDTH × IMAGE_HEIGHT` trong config (`1024×576`).
FFmpeg upscale lên 1920×1080 bằng Lanczos khi render.

### RunPod job stuck IN_QUEUE (lịch sử debug)
Trình tự lỗi đã qua: build context sai → COPY path sai → HF_TOKEN thiếu → load_model() import-time treo → idle timeout 5s kill worker → base image sai (runpod/pytorch/base) → SDK version → **ROOT CAUSE: "Container Start Command"="Endpoint" ghi đè CMD** → fix bằng ENTRYPOINT.

---

## Quản lý RunPod endpoint

```powershell
# Kiểm tra health
$headers = @{Authorization = "Bearer $env:RUNPOD_API_KEY"}
Invoke-RestMethod "https://api.runpod.ai/v2/9hs6ppcsssn990/health" -Headers $headers

# Purge queue khi job kẹt
Invoke-RestMethod "https://api.runpod.ai/v2/9hs6ppcsssn990/purge-queue" -Method Post -Headers $headers -Body "{}"

# Force rollout sau build mới
# 1. Set workersMax = 0 qua RunPod UI → đợi workers drain về 0
# 2. Set workersMax = 3 → workers mới pull image mới
```

Mỗi commit push `main` → RunPod tự build lại. Không push dồn (nhiều build song song gây lẫn lộn).
## Final Kokoro production audit - 2026-06-27

- Audited `output/ancient-child-surgery-31000-years` under Kokoro block mode.
- Cap validation:
  - largest sentence measured `278 phoneme chars`
  - script is valid for the `500 phoneme chars` hard cap
- Step 2 / Step 3 results:
  - `engine = kokoro`
  - `mode = block`
  - `voice = am_fenrir`
  - `speed = 0.95`
  - `reused_block_count = 20`
  - `regenerated_block_count = 0`
  - `fallback_block_count = 0`
  - `sentence_count = 92`
  - `block_count = 20`
  - `timestamps.json` count matched sentence count
  - `restart_count = 0`
  - `max_phoneme_chars = 418`
  - `max_actual_seconds = 21.625`
- Boundary clips exported under:
  - `output/ancient-child-surgery-31000-years/kokoro_continuity_audit/`
- Selected listening boundaries:
  - `1`, `3`, `5`, `6`, `8`, `9`, `10`, `12`, `14`, `16`
- Decision:
  - Kokoro production config is `FROZEN`
  - no voice, speed, or block-limit tuning was applied
  - cache was preserved as-is

## VieNeu Voice Lab bootstrap - 2026-06-28

- Added a new independent lab CLI at `scripts/vieneu_voice_lab.py`.
- Production TTS in `steps/tts.py` was left unchanged.

### New workflow

- Supported commands:
  - `base`
  - `topic`
  - `style`
  - `final`
  - `report`
- Output root is isolated at:
  - `output/vieneu_voice_lab/`
- Added `.gitignore` coverage for:
  - `output/vieneu_voice_lab/`

### Round behavior implemented

- Base round:
  - discovers VieNeu voices from the runtime via `Vieneu().list_preset_voices()`
  - creates exactly one blind sample per discovered voice
  - stores:
    - `round`
    - `round_order`
    - `source_voice`
    - `preset`
    - `effective_infer_params`
    - `duration_seconds`
  - enforces target `20-25s`, acceptable `18-30s`
  - warns outside target, fails outside acceptable
- Topic round:
  - reads finalists only from `base` decisions
  - caps finalists at `5`
  - creates exactly one `topic_reel` per finalist
  - hard-enforces `45-60s`
- Style round:
  - reads finalists only from `topic` decisions
  - caps finalists at `3`
  - tests only:
    - `production_default`
    - `natural_calm`
  - stores full effective infer params per artifact
  - enforces target `25-30s`, acceptable `22-35s`
- Final round:
  - reads finalists only from `style` decisions
  - caps finalists at `2`
  - calls the real production VieNeu block-mode path inside the lab only
  - exports:
    - main audio
    - `blocks.json`
    - `diagnostics.json`
    - up to `5` suspicious boundary clips
  - boundary ranking uses:
    - RMS delta
    - peak delta
    - short blocks
    - trailing silence
    - gap anomaly
  - hard-enforces `90-120s`

### Review and report behavior

- `report` reads `decisions.csv` and ranks only the `active_round`.
- Ranking order:
  - `Keep`
  - `Maybe`
  - `Reject`
  - tie-break by `round_order`
- Review HTML now:
  - restores local state from `localStorage`
  - keeps Reveal disabled until a decision is selected
  - exports `decisions.csv`
  - avoids rendering raw voice names in the pre-reveal HTML payload

### Follow-up after first real base render

- The first real `base` run showed that one longer calibration draft could push slower voices past the `30s` acceptable ceiling.
- `scripts/vieneu_voice_lab.py` was tightened again so `BASE_SAMPLE` still contains all required calibration tokens but lands safely in the target/acceptable window across the current discovered VieNeu voices.
- Real base run result after the trim:
  - `10` samples generated successfully under `output/vieneu_voice_lab/audio/base/`
  - `manifest.active_round = base`
  - one sample (`V434`) remained slightly above target at `25.70s`, but still inside the acceptable range, so this is warning-only and valid

### Tests completed

- `python -m py_compile scripts/vieneu_voice_lab.py tests/test_vieneu_voice_lab.py`
- `python -m pytest tests/test_vieneu_voice_lab.py -q`
  - result: `9 passed`
- `python -m pytest tests -q`
  - result: `137 passed`

### Files changed

- `.gitignore`
- `scripts/vieneu_voice_lab.py`
- `tests/test_vieneu_voice_lab.py`
- `.ai/CURSOR_WORKLOG.md`
- `handoff.md`

### Commit scope reminder

- Keep generated `output/` audio, review exports, and other dirty repo output artifacts out of commits.

## Subtitle patch - 2026-06-28

- Added a subtitle pipeline that is independent from production TTS and sentence-level image timing.
- `timestamps.json` remains the image-timing source of truth and keeps its prior shape/semantics.

### What changed

- `steps/transcribe.py`
  - still writes `timestamps.json` exactly as before
  - now also writes:
    - `word_timestamps.json` only when exact canonical word timing is available
    - `word_timestamps_diagnostics.json` on every run
  - does not run a second alignment/model pass
  - does not fabricate per-word timing when exact canonical matches are missing
  - marks subtitle readiness false for fallback blocks or partial canonical coverage
- `steps/subtitles.py`
  - new subtitle builder from:
    - `script.txt`
    - `word_timestamps.json`
    - `audio_master.wav` or `audio.mp3`
  - writes atomically:
    - `subtitle_cues.json`
    - `subtitles.srt`
    - `subtitles.ass`
    - `subtitle_diagnostics.json`
  - supports:
    - `cinematic_clean`
    - `cinematic_accent`
  - preview rendering now rebases cue times to preview-local time before ASS serialization
  - Windows libass burn path now uses a temp working directory and a safe local `subtitles.ass` filename
- `scripts/generate_subtitles.py`
  - new standalone CLI for subtitle generation / validation / preview
- `steps/render_video.py`
  - still renders clean `final.mp4` first
  - with `--subtitles`, burns from `final.mp4` to `final_subbed.mp4`
  - never overwrites `final.mp4`
- `main.py`
  - fixed help text so `--subtitles` points to step 7, not step 6
- `config.py`
  - added subtitle style/layout defaults and configurability for font family and rendering constants

### Tests completed

- `python -m py_compile steps/transcribe.py steps/subtitles.py steps/render_video.py scripts/generate_subtitles.py tests/test_subtitles.py`
- `python -m pytest tests/test_subtitles.py -q`
  - result: `9 passed`
- `python -m pytest tests -q`
  - result: `146 passed`

### Real commands run

- Production-sample subtitle readiness checks:
  - `python main.py --video-id buc-tranh-co-nhat-the-gioi-khong-nam-o-chau-au-vi --step 3`
  - `python main.py --video-id ancient-child-surgery-31000-years --step 3`
  - `python scripts/generate_subtitles.py --video-id buc-tranh-co-nhat-the-gioi-khong-nam-o-chau-au-vi --style cinematic_clean --validate-only`
  - `python scripts/generate_subtitles.py --video-id ancient-child-surgery-31000-years --style cinematic_clean --validate-only`
- Synthetic smoke for preview and Windows burn path:
  - generated local fixture under `output/subtitle-smoke-local/`
  - `python scripts/generate_subtitles.py --video-id subtitle-smoke-local --style cinematic_clean --preview-seconds 45`
  - `python scripts/generate_subtitles.py --video-id subtitle-smoke-local --style cinematic_accent --preview-seconds 45`
  - `python main.py --video-id subtitle-smoke-local --step 7 --subtitles`

### Real smoke outcomes

- `buc-tranh-co-nhat-the-gioi-khong-nam-o-chau-au-vi`
  - step 3 succeeded
  - subtitle diagnostics reported:
    - `subtitle_ready = false`
    - `reason = exact_canonical_word_timing_unavailable`
  - `generate_subtitles.py --validate-only` hard-failed as designed with rerun guidance
- `ancient-child-surgery-31000-years`
  - step 3 succeeded
  - subtitle diagnostics reported:
    - `subtitle_ready = false`
    - `reason = exact_canonical_word_timing_unavailable`
  - `generate_subtitles.py --validate-only` hard-failed as designed with rerun guidance
- `subtitle-smoke-local`
  - subtitle generation succeeded
  - preview outputs succeeded:
    - `subtitle_preview_clean.mp4`
    - `subtitle_preview_accent.mp4`
  - step 7 subtitle burn succeeded:
    - clean `final.mp4`
    - burned `final_subbed.mp4`

### Important behavior note

- This patch intentionally prefers fail-safe blocking over approximate subtitle timing.
- If exact canonical word timing is unavailable, operators will get:
  - `word_timestamps_diagnostics.json`
  - a clear failure from subtitle generation / `--subtitles`
  - no fabricated `word_timestamps.json`

## Creative package and thumbnail workflow - 2026-06-28

- Added a separate creative-package path that stays isolated from TTS, subtitle alignment, sentence timing, and clean video render behavior.
- New authoring contract:
  - save narration only to `output/<video-id>/script.txt`
  - save strategy only to `output/<video-id>/creative_package.json`

### What changed

- `prompts/script_prompt.txt`
  - Stage 2 now outputs two explicit sections:
    - `SCRIPT`
    - `CREATIVE_PACKAGE_JSON`
  - SCRIPT is narration-only
  - metadata / thumbnail strategy must not be appended into `script.txt`
- `steps/creative_package.py`
  - new reusable loader/validator
  - validates schema, title ids, concept distribution, word limits, paired title ids
  - computes the real `script_sha256`
  - stores a validated copy at:
    - `publishing/creative_package.validated.json`
  - detects stale package after `script.txt` changes
- `steps/image_prompts.py`
  - keeps `image_prompts.json` behavior unchanged for scenes
  - if `creative_package.json` exists and validates, generates separate:
    - `publishing/thumbnail_prompts.json`
    - `publishing/thumbnail_prompt_diagnostics.json`
  - thumbnail prompt failure no longer destroys a valid `image_prompts.json`
- `steps/thumbnails.py`
  - new thumbnail background + Pillow overlay pipeline
  - reuses the existing image backend contract
  - outputs:
    - `publishing/thumbnails/thumbnail_XX_background.png`
    - `publishing/thumbnails/thumbnail_XX.jpg`
    - `publishing/thumbnail_contact_sheet.jpg`
    - `publishing/thumbnail_generation_log.json`
    - `publishing/thumbnail_generation_diagnostics.json`
- `scripts/generate_thumbnails.py`
  - new standalone selective thumbnail CLI
  - supports:
    - `--video-id`
    - `--regenerate <concept_id>`
    - `--allow-stale-package`
- `steps/generate_images.py`
  - after normal scene images finish, it now also generates publishing thumbnails when `publishing/thumbnail_prompts.json` exists
- `steps/metadata.py`
  - Step 8 now prefers `creative_package.json`
  - writes:
    - `publishing/package.json`
    - `publishing/title_options.txt`
    - `publishing/description.txt`
    - `publishing/chapters.txt`
    - `publishing/tags.txt`
    - `publishing/publishing_diagnostics.json`
  - still falls back to the legacy AI metadata path only when `creative_package.json` is absent
  - stale/invalid creative packages hard-fail Step 8 instead of silently falling back
- docs updated:
  - `CLAUDE.md`
  - `AGENTS.md`

### Tests completed

- `python -m py_compile steps/creative_package.py steps/thumbnails.py steps/image_prompts.py steps/metadata.py scripts/generate_thumbnails.py tests/test_creative_package.py tests/test_thumbnails.py tests/test_publishing.py`
- `python -m pytest tests/test_creative_package.py -q`
  - result: `5 passed`
- `python -m pytest tests/test_thumbnails.py -q`
  - result: `3 passed`
- `python -m pytest tests/test_publishing.py -q`
  - result: `3 passed`
- `python -m pytest tests -q`
  - result: `157 passed`

### Smoke commands run

- Synthetic VI + EN smoke for the full new workflow using local fake model responses / fake image backend:
  - created `output/creative-smoke-vi/`
  - created `output/creative-smoke-en/`
  - executed:
    - `steps.image_prompts.run(...)`
    - `steps.thumbnails.generate_thumbnail_assets(...)`
    - `steps.metadata.run(...)`
- Outcome:
  - both VI and EN smoke folders produced:
    - `image_prompts.json`
    - `publishing/thumbnail_prompts.json`
    - `publishing/thumbnails/thumbnail_XX_background.png`
    - `publishing/thumbnails/thumbnail_XX.jpg`
    - `publishing/thumbnail_contact_sheet.jpg`
    - `publishing/package.json`

### Changed files

- `prompts/script_prompt.txt`
- `steps/creative_package.py`
- `steps/thumbnails.py`
- `steps/image_prompts.py`
- `steps/generate_images.py`
- `steps/metadata.py`
- `scripts/generate_thumbnails.py`
- `tests/test_creative_package.py`
- `tests/test_thumbnails.py`
- `tests/test_publishing.py`
- `CLAUDE.md`
- `AGENTS.md`
- `.ai/CURSOR_WORKLOG.md`
- `handoff.md`

### Important note

- Generated files under `output/` were used only for smoke validation and must stay out of commits.

## Vietnamese rerun smoke - 2026-06-28

- Chosen video:
  - `creative-smoke-vi`
- Why this one:
  - it is the only Vietnamese folder in the current checkout with `script.txt` and `timestamps.json` still aligned 5/5, so it was safe to rerun from the source script without inventing a new sample.

### Real commands run

- `python main.py --video-id creative-smoke-vi --step 2`
- `python main.py --video-id creative-smoke-vi --step 3`
- `python scripts/generate_images.py --video-id creative-smoke-vi --force --candidates 1 --workers 5`
- `python main.py --video-id creative-smoke-vi --step 6`
- `python main.py --video-id creative-smoke-vi --step 7`

### Observed results

- Step 2:
  - Kokoro block TTS reran successfully
  - `audio.mp3` regenerated
  - duration about `13.6s`
- Step 3:
  - faster-whisper block alignment ran
  - sentence fallback triggered for the single block
  - fresh `timestamps.json` was written
  - `word_timestamps_diagnostics.json` reported subtitles not ready
- Step 5:
  - forced regeneration of all 5 images succeeded through RunPod
- Step 6:
  - soundscape produced no usable SFX for this smoke case
- Step 7:
  - clean `final.mp4` rendered successfully
  - size about `2.5 MB`

### Important note

- `step 4` was not rerun in this environment because `GEMINI_API_KEY` is not set in the checkout, so I reused the existing `image_prompts.json` for the rerun.
- This rerun is a short smoke fixture, not a production-length Vietnamese video.

## Vast image rerun - 2026-06-28

- Target video:
  - `creative-smoke-vi`
- What I ran:
  - `scripts/generate_images.py --video-id creative-smoke-vi --track vi --force --candidates 1 --workers 5` with `IMAGE_BACKEND=vast_instance`
  - then copied `images_vi/img_*.png` into canonical `images/`
  - then reran `main.py --video-id creative-smoke-vi --step 7`

### Result

- Vast.ai image generation completed successfully for all 5 scenes.
- No scene failures or QA failures were reported.
- Final render completed successfully after copying the new Vast images into canonical `images/`.

### Issue found

- The `track vi` path in `scripts/generate_images.py` saves promoted files into `images_vi/`.
- `steps/render_video.py` still prefers canonical `images/` when that folder already exists.
- Because of that, a pure `--track vi` rerun does not automatically become the render source unless we sync or swap the directories.

## Production autopilot patch - 2026-06-28

- Added a reusable production entrypoint:
  - `python main.py --autopilot --video-id <id> --script-file <path>`
- Goal:
  - let Cursor take a pasted narration script and drive the repo to a finished production package without Leon manually creating files, configs, or subtitle burns.

### Shipped behavior

- New `steps/autopilot.py`
  - normalizes narration text
  - detects `vi|en`
  - writes:
    - `script.txt`
    - `tts_config.json`
    - `transcribe_config.json`
    - `autopilot_state.json`
    - `autopilot_summary.json`
  - creates a default `creative_package.json` when none exists
  - respects an existing manual creative package
  - blocks unsafe overwrite and validates resume by script hash
- `main.py`
  - now supports:
    - `--autopilot`
    - `--script-file`
- `steps/image_prompts.py`
  - no longer uses duplicated inline prompt policy as the primary source of truth
  - reads:
    - `prompts/image_prompt_vi.txt`
    - `prompts/image_prompt_en.txt`
  - supports sentence -> `1..3` visual beats when exact word timing exists
  - falls back to `1 sentence = 1 image` when exact words are unavailable
  - emits round-trippable beat fields including:
    - `source_sentence_index`
    - `beat_index`
    - `word_start`
    - `word_end`
    - `visual_intent`
  - writes template/model/style diagnostics into scene entries
- New `steps/visual_beats.py`
  - exact word-based beat timing
  - fallback sentence-beat planning
  - beat normalization / validation
- `steps/generate_images.py`
  - keeps canonical output in `images/`
  - no longer exits early before thumbnail generation when scenes are already complete
  - reuses the same backend for scene images and thumbnails inside the step wrapper
- `steps/thumbnails.py`
  - accepts backend reuse via override hooks so the same rented lifecycle can be shared
- Prompt templates refreshed
  - production-safe clothing and anatomy language is now explicit
  - test-locked legacy keywords for VI/EN prompt tracks were preserved

### Files changed

- `main.py`
- `steps/autopilot.py`
- `steps/image_prompts.py`
- `steps/visual_beats.py`
- `steps/generate_images.py`
- `steps/thumbnails.py`
- `prompts/image_prompt_vi.txt`
- `prompts/image_prompt_en.txt`
- `tests/test_autopilot.py`
- `tests/test_visual_beats.py`
- `tests/test_thumbnails.py`
- `CLAUDE.md`
- `AGENTS.md`
- `.ai/CURSOR_WORKLOG.md`
- `handoff.md`

### Tests run

- `C:\Users\LEON_RM\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m pytest tests/test_autopilot.py -q`
  - `5 passed`
- `...python.exe -m pytest tests/test_visual_beats.py -q`
  - `5 passed`
- `...python.exe -m pytest tests/test_thumbnails.py -q`
  - `4 passed`
- `...python.exe -m pytest tests/test_creative_package.py -q`
  - `5 passed`
- `...python.exe -m pytest tests/test_subtitles.py -q`
  - `9 passed`
- `...python.exe -m pytest tests/test_publishing.py -q`
  - `3 passed`
- `...python.exe -m pytest tests -q`
  - `168 passed`

### Notes

- This patch implements the repo entrypoint and data contracts for Cursor-first production.
- I did not run a real paid Vast autopilot smoke in this patch turn.
- Generated `output/` artifacts were not added to commit scope.

## Subtitle recovery - 2026-06-28

- Goal:
  - finish the fresh Vietnamese production run with exact subtitles burned into `final_subbed.mp4`

### What changed

- `steps/transcribe.py`
  - accepts exact zero-duration aligned words instead of failing subtitle readiness
- `steps/subtitles.py`
  - sentence-bound fallback for zero-duration spans
  - dynamic wrap width using the configured subtitle line budget
  - clamp cue starts so sequential cues do not overlap
- `config.py`
  - subtitle default line budget increased to `46` chars
- `tests/test_subtitles.py`
  - added regression coverage for zero-duration word timings and fallback sentence spans

### Commands run

- `python -m pytest tests/test_subtitles.py -q`
- `python -m pytest tests/test_autopilot.py tests/test_thumbnails.py -q`
- `python -m pytest tests -q`
- `python main.py --video-id to-tien-cua-ban-chi-lam-viec-15-tieng-mot-tuan-vi-fresh-full-20260628-1320 --step 3`
- `python main.py --video-id to-tien-cua-ban-chi-lam-viec-15-tieng-mot-tuan-vi-fresh-full-20260628-1320 --step 7 --subtitles`

### Results

- `word_timestamps.json` regenerated successfully
- `subtitle_diagnostics.json` now passes with:
  - `missing_word_count = 0`
  - `repeated_word_count = 0`
  - `overlap_count = 0`
  - `max_lines = 2`
- `final.mp4` exists
- `final_subbed.mp4` exists
- full test suite passed: `170 passed`

### Notes

- Fresh run folder:
  - `output/to-tien-cua-ban-chi-lam-viec-15-tieng-mot-tuan-vi-fresh-full-20260628-1320`
- No `output/` files were committed.
