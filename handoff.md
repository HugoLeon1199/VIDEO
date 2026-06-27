# handoff.md — Ghi chú kỹ thuật nội bộ

File này chứa thông tin kỹ thuật chi tiết, lịch sử debug, và quyết định thiết kế.
Để quy trình làm video, xem **AGENTS.md**. Để lệnh chạy, xem **CLAUDE.md**.

---

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
