# AGENTS.md — YouTube Autopilot Pipeline

Đọc file này trước khi làm bất cứ việc gì trong project.

---

## MỤC TIÊU DỰ ÁN

Tự động hoá toàn bộ quy trình sản xuất video YouTube niche **"Ancient Humans / Prehistoric Life"** bằng tiếng Anh **và tiếng Việt**.

**Style kênh:** History storytelling cinematic, educational, mystery/civilization discovery.

**Target:** Mỗi video ~8–10 phút, 1500–2400 từ, phong cách dramatic + authoritative, dùng "you" (tiếng Anh) hoặc "bạn" (tiếng Việt) để kéo viewer vào.

---

## CẤU TRÚC THƯ MỤC

```
youtube-autopilot/
├── AGENTS.md                    ← File này
├── CLAUDE.md                    ← Hướng dẫn cho Claude Code CLI
├── handoff.md                   ← Ghi chú kỹ thuật nội bộ (không cần đọc khi làm video mới)
├── main.py                      ← Orchestrator chính
├── config.py                    ← API keys, settings, constants
│
├── steps/
│   ├── generate_script.py       ← Bước 1: Validate script.txt
│   ├── tts.py                   ← Bước 2: Text-to-Speech
│   ├── transcribe.py            ← Bước 3: Tạo timestamps.json
│   ├── image_prompts.py         ← Bước 4: Tạo image_prompts.json
│   ├── generate_images.py       ← Bước 5: Generate ảnh qua RunPod
│   ├── render_video.py          ← Bước 6: Dựng video FFmpeg
│   └── metadata.py              ← Bước 7: Tạo title/description/tags
│
├── image_generation/
│   ├── runpod_client.py         ← Submit job + poll status
│   └── runpod_serverless_backend.py  ← Save candidates, promote canonical
│
├── serverless_worker/           ← Code deploy lên RunPod
│   ├── handler.py               ← RunPod handler (FLUX.2 Klein)
│   ├── model_loader.py          ← Lazy-load model vào global cache
│   └── Dockerfile               ← FROM python:3.11.1-slim, ENTRYPOINT
│
├── scripts/
│   ├── generate_images.py       ← Chạy step 5 độc lập (có --dry-run, --from-scene, --to-scene)
│   ├── test_one_scene.py        ← Submit 1 job RunPod thật để test
│   └── validate_runpod_serverless.py  ← Validate env + dry-run
│
└── output/
    └── {video-id}/              ← Mỗi video có thư mục riêng
        ├── script.txt           ← Kịch bản narration-only (input thủ công)
        ├── creative_package.json ← Title/thumbnail/chapter strategy lưu riêng
        ├── audio.mp3            ← Output bước 2
        ├── timestamps.json      ← Output bước 3
        ├── image_prompts.json   ← Output bước 4 (hoặc viết tay)
        ├── images/
        │   ├── scene_001/       ← Candidates (webp)
        │   │   └── candidate_01_seed_11001.webp
        │   └── img_001.png      ← Canonical PNG → input cho render
        ├── generation_log.json  ← Progress image gen (resume)
        ├── final.mp4            ← Output bước 6
        ├── subtitles.srt        ← Output bước 6
        ├── metadata.json        ← Legacy output bước 8
        └── publishing/          ← Creative package + thumbnails + publishing package
```

---

## PIPELINE CHI TIẾT

### Bước 1 — Script (thủ công)
- Human viết script với Claude → lưu narration vào `output/{video-id}/script.txt`
- Lưu creative strategy riêng vào `output/{video-id}/creative_package.json`
- Dùng system prompt ở `prompts/script_prompt.txt` khi yêu cầu Claude viết
- Step 1 chỉ validate format, không sửa nội dung

**Quy tắc câu bắt buộc** (pipeline dùng sentence mode — TTS từng câu):
- Mỗi câu phải kết thúc bằng `.` `!` `?`
- Mỗi câu là một ý hoàn chỉnh, không phụ thuộc câu trước/sau
- Không câu nào dài quá 40 từ
- Không dùng `—` (em dash), `!` đầu từ, `/` trong từ — edge-tts lỗi
- Không bullet points, không header trong phần script chính

Không append metadata/thumbnails vào cuối `script.txt`. Các phần đó phải nằm trong `creative_package.json`.

### Bước 2 — TTS (`steps/tts.py`)

**Tiếng Anh:**
- Engine: Kokoro TTS (`am_fenrir`), fallback → edge-tts `en-US-GuyNeural`
- Config mặc định trong `config.py`

**Tiếng Việt (khuyến nghị: sentence mode):**
- Tạo file `output/{video-id}/tts_config.json`:
```json
{"engine": "edge", "voice": "vi-VN-NamMinhNeural", "rate": "-8%", "mode": "sentence"}
```
- `"mode": "sentence"` = TTS từng câu riêng → ghép → tự tạo `timestamps.json` chính xác 100%
- Step 3 (Whisper) sẽ tự **skip** vì `timestamps.json` đã có — không cần `transcribe_config.json`
- Kết quả: ảnh và voice khớp hoàn toàn, không có drift

**Không dùng Kokoro cho tiếng Việt** — không hỗ trợ.
**Các ký tự phá edge-tts** (phải loại khỏi script): `—` (em dash, ord=8212), `!` đầu từ, `/`

### Bước 3 — Transcribe (`steps/transcribe.py`)

**Khi dùng sentence mode (TTS tạo timestamps):** Step 3 tự skip — không cần làm gì.

**Khi không dùng sentence mode:**

Tiếng Anh:
- faster-whisper model `base`, CPU/int8, không cần config thêm

Tiếng Việt (fallback, ít chính xác hơn sentence mode):
- Tạo file `output/{video-id}/transcribe_config.json`:
```json
{"model": "medium", "language": "vi", "mode": "align"}
```
- `"mode": "align"` dùng script.txt làm text chuẩn, Whisper chỉ lấy timing từng từ
- Tốt hơn Whisper auto, nhưng vẫn kém sentence mode vì timing vẫn là ước lượng

### Bước 4 — Image Prompts (`steps/image_prompts.py`)

Tạo 1 prompt JSON cho mỗi timestamp entry. Có 2 cách:

**Cách A — Tự động (Gemini):**
- `GEMINI_API_KEY` đã hardcode trong `config.py`, không cần set env
- Model: `gemini-2.5-flash`
- Output: `image_prompts.json` với `{index, start, end, prompt}`

**Cách B — Viết tay (khuyến nghị cho video tiếng Việt):**
- Viết 1 prompt mỗi timestamp, lưu thẳng vào `output/{video-id}/image_prompts.json`
- Format bắt buộc:
```json
[
  {"index": 1, "start": 0.0, "end": 3.5, "prompt": "Cinematic wide shot..."}
]
```
- `index` phải trùng số với timestamps.json
- Số lượng entries phải bằng số entries trong timestamps.json

**Style prompts tiếng Anh (English videos):**
- Cinematic photorealistic, National Geographic style
- `Cinematic wide shot, [scene], photorealistic, natural lighting, shallow depth of field, 16:9, no text`

**Style prompts tiếng Việt (Vietnamese videos - người quê aesthetic):**
- Rural/village ancient people, high visual impact, simple but compelling
- `[Scene with rural/village people], photorealistic, cinematic, 16:9`
- Tránh: doodle style, EMBERLORE style, stick figures
- Dùng: firelight, earthy tones, dramatic but simple compositions, human moments

### Bước 5 — Generate Images

```powershell
$python = "C:\Users\LEON_RM\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $python scripts/generate_images.py --video-id {video-id} --candidates 1 --workers 5
```

- **Backend:** RunPod Serverless endpoint `9hs6ppcsssn990` (FLUX.2 Klein 4B)
- **Mỗi scene:** submit 1 job → poll → save webp candidates → promote `img_XXX.png`
- **Resume:** tự động skip scene đã có `img_XXX.png`
- **Workers:** 5 parallel là optimal (không flood endpoint)
- **Candidates:** dùng `--candidates 1` cho production (tiết kiệm chi phí)
- **Progress:** lưu vào `generation_log.json`

Thời gian: ~4s/ảnh, ~167s cho 153 ảnh (5 workers song song).

### Bước 6 — Render Video (`steps/render_video.py`)

```powershell
& $python main.py --video-id {video-id} --step 6
```

- Two-pass FFmpeg: mỗi PNG → clip → concat + audio mux
- Mỗi clip = từ `current.start` đến `next.start`; clip cuối đến hết audio
- Output: H.264 1920×1080 8Mbps, AAC 192k 48kHz
- Cũng tạo `subtitles.srt`

### Bước 7 — Metadata (`steps/metadata.py`)

```powershell
& $python main.py --video-id {video-id} --step 7
```

- Dùng Claude Haiku (`claude-haiku-4-5-20251001`)
- Cần `ANTHROPIC_API_KEY` trong env
- Output: `metadata.json` với title, description, tags

---

## CÁCH CHẠY

### Setup lần đầu

```powershell
# API keys (đã có trong .env, không cần set thủ công trừ ANTHROPIC)
# GEMINI_API_KEY: hardcode trong config.py
# RUNPOD_API_KEY + RUNPOD_ENDPOINT_ID: trong .env
# ANTHROPIC_API_KEY: cần set nếu chạy step 7
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

### Quy trình làm 1 video tiếng Anh mới

```powershell
$python = "C:\Users\LEON_RM\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$vid = "my-video-slug"

# 1. Lưu script vào output/$vid/script.txt (thủ công)

# 2-3. TTS + Transcribe
& $python main.py --video-id $vid --step 2
& $python main.py --video-id $vid --step 3

# 4. Image prompts (tự động hoặc viết tay)
& $python main.py --video-id $vid --step 4

# 5. Generate images
& $python scripts/generate_images.py --video-id $vid --candidates 1 --workers 5

# 6. Render
& $python main.py --video-id $vid --step 6
```

### Quy trình làm 1 video tiếng Việt mới

```powershell
$vid = "my-video-vi"

# 1. Dịch script sang tiếng Việt, loại bỏ —, !, /
#    Lưu vào output/$vid/script.txt

# 2. Tạo TTS config
'{"engine": "edge", "voice": "vi-VN-NamMinhNeural", "rate": "-8%"}' | Out-File output/$vid/tts_config.json -Encoding utf8

# 3. Tạo transcribe config  
'{"model": "medium", "language": "vi"}' | Out-File output/$vid/transcribe_config.json -Encoding utf8

# 4. TTS + Transcribe
& $python main.py --video-id $vid --step 2
& $python main.py --video-id $vid --step 3

# 5. Viết 153 prompts (hoặc số tương ứng) theo style người quê
#    Lưu vào output/$vid/image_prompts.json

# 6. Generate images
& $python scripts/generate_images.py --video-id $vid --candidates 1 --workers 5

# 7. Render
& $python main.py --video-id $vid --step 6
```

### Resume sau khi bị dừng

```powershell
# Image gen (tự động skip scene đã có)
& $python scripts/generate_images.py --video-id $vid --candidates 1 --workers 5

# Hoặc toàn bộ pipeline từ step nào đó
& $python main.py --video-id $vid --from-step 5
& $python main.py --video-id $vid --resume
```

### Demo trước khi chạy full

```powershell
& $python main.py --video-id $vid --demo 30
# Tạo output/{vid}_demo30/, không ghi đè production
```

---

## RUNPOD SERVERLESS — CẤU HÌNH HOẠT ĐỘNG

**Endpoint:** `9hs6ppcsssn990` (emberlore-flux2-klein)
**GitHub:** `HugoLeon1199/VIDEO` branch `main`
**Dockerfile path:** `serverless_worker/Dockerfile`
**Build context:** `serverless_worker` (KHÔNG phải `.`)
**Network Volume:** `jqscreri1e` (50GB, US-IL-1) → `/runpod-volume` (cache model 13GB)
**GPU:** 24GB (L4/A5000/RTX3090/RTX4090)
**idleTimeout:** 120s, **execTimeout:** 900s, **FlashBoot:** on

**Lưu ý quan trọng về Dockerfile:**
```dockerfile
FROM python:3.11.1-slim
ENTRYPOINT ["python", "-u", "/handler.py"]  # PHẢI dùng ENTRYPOINT, không dùng CMD
# Lý do: RunPod có "Container Start Command" = "Endpoint" ghi đè CMD
# ENTRYPOINT không bị ghi đè → worker khởi động được
```

**Model:** FLUX.2 Klein 4B, lazy-load trong `handler()` (không load lúc import).
**HF_TOKEN:** inject qua RunPod Secrets, không bake vào image.
**Chi phí:** ~$0.0005/ảnh, ~$0.07–0.09/video (153 ảnh × 1 candidate).

---

## CONFIG (`config.py`)

| Key | Giá trị | Ghi chú |
|-----|---------|---------|
| `GEMINI_API_KEY` | hardcoded | Dùng cho step 4 text prompts |
| `RUNPOD_API_KEY` | `.env` | Bắt buộc cho step 5 |
| `RUNPOD_ENDPOINT_ID` | `.env` = `9hs6ppcsssn990` | Endpoint FLUX |
| `IMAGE_BACKEND` | `runpod_serverless` | Backend hiện tại |
| `IMAGE_WIDTH/HEIGHT` | 1024×576 | Output của worker |
| `IMAGE_CANDIDATE_SEEDS` | `[11001, 11002, 11003]` | Seeds cho candidates |
| `IMAGE_OUTPUT_FORMAT` | `WEBP` | Candidates; canonical PNG |
| `VIDEO_BITRATE` | `8M` | H.264 bitrate |
| `VIDEO_FPS` | `30` | |
| `VIDEO_WIDTH/HEIGHT` | 1920×1080 | YouTube standard |

---

## LỖI THƯỜNG GẶP

| Lỗi | Nguyên nhân | Fix |
|-----|-------------|-----|
| edge-tts "No audio received" | Script có `—` (ord=8212), `!Kung`, `Ju/'hoansi` | Thay `—` → `, `, bỏ `!`, `/` |
| Whisper nhận sai ngôn ngữ | Không set `language=vi` | Tạo `transcribe_config.json` |
| Job RunPod stuck IN_QUEUE | Worker crash-loop | Xem RunPod UI → Workers → Logs |
| Image gen exit 255 | PowerShell buffer overflow khi redirect 2>&1 | Dùng `Tee-Object` + log file |
| Số ảnh < số prompts | Một số job fail | Chạy lại — resume tự skip scene đã có |

---

## VIDEO ĐÃ HOÀN THÀNH

| Video ID | Ngôn ngữ | Scenes | Thời lượng | Style prompts |
|----------|----------|--------|------------|---------------|
| `ancient-child-surgery-31000-years` | EN | 131 | 8.2 min | Cinematic photorealistic NatGeo |
| `ancient-child-surgery-31000-years-vi` | VI | 131 | 8.2 min | Cinematic photorealistic NatGeo |
| `what-ancient-humans-did-all-day` | EN | ~144 | ~9 min | Cinematic photorealistic NatGeo |
| `what-ancient-humans-did-all-day-vi` | VI | 153 | 8.9 min | Rural/village người quê aesthetic |

---

## CHI PHÍ ƯỚC TÍNH

| Hạng mục | Tool | Chi phí/video |
|----------|------|--------------|
| TTS | Kokoro local / edge-tts free | $0 |
| Timestamp | faster-whisper local | $0 |
| Image prompts | Gemini text (step 4) hoặc viết tay | ~$0.01 hoặc $0 |
| ~150 ảnh × 1 candidate | RunPod FLUX | ~$0.08 |
| Render | FFmpeg local | $0 |
| Metadata | Claude Haiku (step 7, optional) | ~$0.02 |
| **TỔNG** | | **~$0.10/video** |
