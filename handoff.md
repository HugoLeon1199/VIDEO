# Handoff — YouTube Autopilot Pipeline

## Trạng thái hiện tại (2026-06-19)

Pipeline hoàn chỉnh, đã test demo 1/5 nội dung. Video full-HD 1920×1080 xuất ra đúng chuẩn YouTube.

---

## Vấn đề đã giải quyết

### 1. Ảnh Gemini xuất 16:9 đúng cách
`gemini-2.5-flash-image` **có hỗ trợ aspect ratio** qua `ImageGenerationConfig(aspect_ratio="16:9")`.

**Giải pháp:** Truyền config vào `GenerateContentConfig` trong `_generate_image()`:
```python
config=types.GenerateContentConfig(
    response_modalities=["IMAGE", "TEXT"],
    image_config=types.ImageGenerationConfig(aspect_ratio="16:9"),
)
```
API trả về ảnh native 16:9 (thường ~1232×688 hoặc tương đương). FFmpeg render chỉ cần `scale=1920:1080`.

**Lưu ý:** Model `gemini-2.5-flash-image-preview` đã bị tắt từ 15/01/2026. Chỉ dùng `gemini-2.5-flash-image` (không có `-preview`).

### 2. Ảnh cùng style quá giống nhau
Fixed bằng cách viết lại `prompts/system_prompt.txt` với VISUAL VARIETY RULES rõ ràng (scale, time of day, environment, subjects, actions).

### 3. Video chỉ hiện 1 ảnh đầu tiên
Fixed bằng cách thêm `-t {duration:.3f}` cho mỗi `-loop 1 -i img.png` input trong FFmpeg.

### 4. WinError 206 — command line quá dài
Fixed bằng 2-step render: mỗi ảnh → clip riêng (ultrafast), sau đó concat + audio mux.

---

## Kiến trúc pipeline

```
script.txt (viết tay)
  → Step 2: Kokoro TTS → audio.mp3
  → Step 3: faster-whisper → timestamps.json  (1 entry = 1 câu)
  → Step 4: Gemini Flash text → image_prompts.json  (N prompts = N câu)
  → Step 5: Gemini Flash Image × N parallel → images/img_001.png … (1920×1080 sau Pillow)
  → Step 6: FFmpeg per-clip + concat → final.mp4
  → Step 7: Claude Haiku → metadata.json
```

**Quy tắc bất biến:** `N images = N sentences`. Không padding, không cắt. Image-voice sync 100%.

---

## Cách chạy

```powershell
$python = "C:\Users\LEON_RM\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

# Demo 1/5 nội dung (để kiểm tra trước)
& $python main.py --video-id <slug> --demo 30

# Full video
& $python main.py --video-id <slug>

# Resume sau khi bị gián đoạn
& $python main.py --video-id <slug> --resume

# Chạy từng bước
& $python main.py --video-id <slug> --step 5
& $python main.py --video-id <slug> --from-step 6
```

---

## Cấu hình quan trọng (`config.py`)

| Key | Giá trị | Ghi chú |
|-----|---------|---------|
| `GEMINI_API_KEY` | hardcoded | AIzaSyDm60NB3t... — đã set sẵn, không cần set env |
| `ANTHROPIC_API_KEY` | env var | Cần set trước step 7 |
| `GEMINI_IMAGE_MODEL` | `gemini-2.5-flash-image` | Model image gen |
| `CLAUDE_MODEL` | `claude-haiku-4-5-20251001` | Step 4 dùng Gemini text, step 7 dùng Claude |
| `TTS_VOICE` | `am_fenrir` | Kokoro voice — dramatic male |
| `VIDEO_WIDTH/HEIGHT` | 1920×1080 | YouTube standard |

---

## FFmpeg path

```
C:\Users\LEON_RM\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin
```
`_ensure_ffmpeg_path()` trong `render_video.py` tự động thêm vào PATH.

---

## Rate limits

- **Gemini image gen (step 5):** 10 RPM, 10 workers parallel, sliding-window rate limiter
- **Gemini text (step 4):** 1 call duy nhất, retry 2 lần
- **Claude (step 7):** retry 2 lần, sleep 5s

---

## Step 5 có thể resume

Progress lưu vào `images/progress.json` sau mỗi ảnh thành công. Nếu bị interrupt, chạy `--resume` hoặc `--step 5` sẽ bỏ qua các ảnh đã có.

---

## Demo mode (isolated)

`--demo N` tự tạo subfolder `output/{video_id}_demo{N}/`, copy `script.txt`, `audio.mp3`, `timestamps.json` từ folder gốc, chạy step 4–6 trong đó. **Không bao giờ ghi đè file production.**

```powershell
# Demo 30 ảnh → output/what-ancient-humans-did-all-day_demo30/
& $python main.py --video-id what-ancient-humans-did-all-day --demo 30
```

## Sync ảnh-voice

Mỗi ảnh chiếm từ `current.start` đến `next.start` (bao gồm khoảng nghỉ giữa câu). Ảnh cuối kéo đến hết `audio_duration`. Ảnh đầu luôn bắt đầu từ `0`. Logic trong `_compute_clip_durations()` ở `render_video.py`.

## YouTube export spec

| Thông số | Giá trị |
|----------|---------|
| Video codec | H.264 (libx264), preset medium |
| Bitrate | 8 Mbps |
| Resolution | 1920×1080 (upscale từ 1344×768 bằng Lanczos) |
| FPS | 30 |
| Audio codec | AAC stereo, 192k, 48kHz |

## Việc chưa làm

- [ ] Step 7 (metadata) chưa test thực tế — cần `$env:ANTHROPIC_API_KEY = "sk-ant-..."`
- [ ] Chạy full video (144 ảnh) — chỉ mới test demo 30 ảnh
- [ ] Upload YouTube — làm thủ công sau khi có `metadata.json`
