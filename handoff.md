# handoff.md — Ghi chú kỹ thuật nội bộ

File này chứa thông tin kỹ thuật chi tiết, lịch sử debug, và quyết định thiết kế.
Để quy trình làm video, xem **AGENTS.md**. Để lệnh chạy, xem **CLAUDE.md**.

---

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
