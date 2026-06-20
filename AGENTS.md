# AGENTS.md — YouTube Autopilot Pipeline
> Đọc file này trước khi làm bất cứ việc gì trong project.

---

## 🎯 MỤC TIÊU DỰ ÁN

Tự động hoá toàn bộ quy trình sản xuất video YouTube niche **"Ancient Humans / Prehistoric Life"** (người cổ đại thời tiền sử) bằng tiếng Anh.

**Style kênh:** Hỗn hợp 3 dạng:
1. History storytelling (kể chuyện lịch sử cinematic)
2. Educational (giải thích cuộc sống người cổ đại)
3. Mystery / Civilization discovery (bí ẩn nền văn minh cổ)

**Target:** Mỗi video ~8–10 phút, 1500–2400 từ tiếng Anh, phong cách dramatic + authoritative, dùng "you" để kéo viewer vào.

---

## 📁 CẤU TRÚC THƯ MỤC

```
youtube-autopilot/
├── AGENTS.md                  ← File này (đọc trước tiên)
├── main.py                    ← Orchestrator chính, chạy toàn bộ pipeline
├── config.py                  ← API keys, settings, constants
├── requirements.txt           ← Các thư viện cần cài
│
├── steps/
│   ├── 01_generate_script.py  ← Bước 1: Validate script file (do human viết trên Claude Web)
│   ├── 02_tts.py              ← Bước 2: Text-to-Speech bằng Kokoro TTS (local, free)
│   ├── 03_transcribe.py       ← Bước 3: Tạo timestamp bằng faster-whisper (local, free)
│   ├── 04_image_prompts.py    ← Bước 4: Tạo 200 image prompts bằng Claude Haiku API
│   ├── 05_generate_images.py  ← Bước 5: Tạo ảnh bằng Gemini 2.5 Flash Image API (free)
│   ├── 06_render_video.py     ← Bước 6: Dựng video bằng FFmpeg (local, free)
│   └── 07_metadata.py         ← Bước 7: Tạo title/description/tags bằng Claude Haiku API
│
├── prompts/
│   ├── system_prompt.txt      ← System prompt cho bước tạo image prompts
│   └── metadata_prompt.txt    ← System prompt cho bước tạo metadata
│
├── output/
│   └── {video_id}/            ← Mỗi video có thư mục riêng, đặt tên theo slug tiêu đề
│       ├── script.txt         ← Kịch bản do human duyệt (input thủ công)
│       ├── audio.mp3          ← Output bước 2
│       ├── timestamps.json    ← Output bước 3
│       ├── image_prompts.json ← Output bước 4
│       ├── images/            ← Output bước 5 (200 ảnh PNG)
│       ├── final.mp4          ← Output bước 6
│       └── metadata.json      ← Output bước 7
│
└── pipeline.log               ← Log toàn bộ quá trình chạy
```

---

## 🔄 PIPELINE CHI TIẾT

### BƯỚC 1 — Kịch bản (THỦ CÔNG - Human làm)
- Human chat với **Claude Web (claude.ai)** để tạo kịch bản
- Dùng system prompt chuẩn (xem phần PROMPTS bên dưới)
- Sau khi duyệt xong, lưu file vào: `output/{video_id}/script.txt`
- Đây là bước DUY NHẤT human can thiệp trước khi chạy pipeline

### BƯỚC 2 — Text-to-Speech (`steps/02_tts.py`)
- **Tool:** Kokoro TTS (chạy local, hoàn toàn free)
- **Input:** `output/{video_id}/script.txt`
- **Output:** `output/{video_id}/audio.mp3`
- Giọng đọc: tiếng Anh, tone dramatic, ấm
- Fallback: nếu Kokoro lỗi → dùng edge-tts (Microsoft, cũng free)
- Ghi log thời lượng audio (mục tiêu: 8–10 phút)

### BƯỚC 3 — Transcribe + Timestamp (`steps/03_transcribe.py`)
- **Tool:** faster-whisper (chạy local, free, dùng CPU — Ryzen 9 9950X3D handle tốt)
- **Input:** `output/{video_id}/audio.mp3`
- **Output:** `output/{video_id}/timestamps.json`
- Format output:
```json
[
  {"index": 1, "start": 0.0, "end": 4.2, "text": "Sixty thousand years ago..."},
  {"index": 2, "start": 4.2, "end": 8.7, "text": "..."}
]
```
- Mục tiêu: word-level hoặc sentence-level timestamps

### BƯỚC 4 — Image Prompts (`steps/04_image_prompts.py`)
- **Tool:** Claude Haiku API (`claude-haiku-4-5-20251001`)
- **Chi phí:** ~$0.09/video (200 prompts)
- **Input:** `output/{video_id}/timestamps.json` + `output/{video_id}/script.txt`
- **Output:** `output/{video_id}/image_prompts.json`
- Logic: gom các timestamp thành ~200 scene (mỗi scene ~2.5–3 giây)
- Format output:
```json
[
  {
    "index": 1,
    "start": 0.0,
    "end": 3.0,
    "prompt": "Ancient human silhouette standing on rocky cliff at sunset, stick figure art style, cave painting aesthetic, warm orange tones, prehistoric landscape",
    "duration": 3.0
  }
]
```
- Style ảnh bắt buộc trong mọi prompt: **stick figure / cave painting / ancient art style** để đảm bảo consistency
- Thêm vào cuối mỗi prompt: `"ancient art style, prehistoric, minimalist, warm earth tones"`

### BƯỚC 5 — Generate Images (`steps/05_generate_images.py`)
- **Tool:** Gemini 2.5 Flash Image API (`gemini-2.5-flash-preview-image-generation`)
- **Chi phí:** FREE (500 ảnh/ngày, reset midnight UTC)
- **Input:** `output/{video_id}/image_prompts.json`
- **Output:** `output/{video_id}/images/img_001.png` đến `img_200.png`
- **Rate limit:** 2 requests/phút → sleep 31 giây giữa mỗi ảnh
- Tổng thời gian: ~100 phút cho 200 ảnh → chạy background/qua đêm
- Retry logic: nếu lỗi 429 → sleep 60s rồi retry, tối đa 3 lần
- Lưu progress vào `output/{video_id}/images/progress.json` để resume nếu bị dừng giữa chừng
- Resolution: 1024x1024 (landscape crop trong FFmpeg)

### BƯỚC 6 — Render Video (`steps/06_render_video.py`)
- **Tool:** FFmpeg (local, free)
- **Input:** audio.mp3 + images/ + image_prompts.json (có timestamp)
- **Output:** `output/{video_id}/final.mp4`
- Logic render:
  - Mỗi ảnh show đúng theo `start` → `end` từ image_prompts.json
  - Ken Burns effect nhẹ (zoom in 5%) để ảnh không tĩnh
  - Fade transition 0.3 giây giữa các ảnh
  - Audio track từ audio.mp3
  - Subtitle burn-in từ timestamps.json (font Arial, size 36, màu trắng, shadow đen)
  - Output: 1920x1080, H.264, AAC audio, bitrate 4Mbps
- Subtitle mặc định TẮT, bật bằng flag `--subtitles` trong main.py

### BƯỚC 7 — Metadata (`steps/07_metadata.py`)
- **Tool:** Claude Haiku API
- **Chi phí:** ~$0.02/video
- **Input:** `output/{video_id}/script.txt`
- **Output:** `output/{video_id}/metadata.json`
- Format output:
```json
{
  "title": "What Did Ancient Humans Actually Do All Day?",
  "description": "...(500 từ, SEO optimized)...",
  "tags": ["ancient humans", "prehistoric life", "..."],
  "thumbnail_prompts": [
    "dramatic ancient human face close-up, cave painting style...",
    "prehistoric hunter silhouette at dawn...",
    "ancient fire ceremony at night..."
  ]
}
```
- Tạo 3 thumbnail prompt options để human chọn 1

---

## 🚀 CÁCH CHẠY

### Setup lần đầu:
```bash
pip install -r requirements.txt
```

### Set API keys (Windows PowerShell):
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
$env:GEMINI_API_KEY = "AIza..."
```

### Workflow hàng ngày:
```bash
# 1. Human tạo script trên Claude Web → lưu vào:
# output/what-ancient-humans-did-all-day/script.txt

# 2. Chạy toàn bộ pipeline:
python main.py --video-id "what-ancient-humans-did-all-day"

# 3. Hoặc chạy từng bước riêng:
python main.py --video-id "what-ancient-humans-did-all-day" --step 4
python main.py --video-id "what-ancient-humans-did-all-day" --step 5
python main.py --video-id "what-ancient-humans-did-all-day" --from-step 5

# 4. Resume nếu bị dừng giữa chừng (bước 5 hay bị):
python main.py --video-id "what-ancient-humans-did-all-day" --resume

# 5. Bật subtitle burn-in:
python main.py --video-id "what-ancient-humans-did-all-day" --step 6 --subtitles
```

---

## ⚙️ CONFIG (`config.py`)

API keys đọc từ environment variables. Tất cả settings khác trong `config.py`.

---

## 💰 CHI PHÍ ƯỚC TÍNH

| Hạng mục | Tool | Chi phí/video | 30 video/tháng |
|----------|------|--------------|----------------|
| Kịch bản | Claude Web (Pro $20/tháng) | $0 thêm | $0 |
| TTS | Kokoro local | $0 | $0 |
| Timestamp | faster-whisper local | $0 | $0 |
| Image prompts | Claude Haiku API | ~$0.09 | ~$2.70 |
| 200 ảnh | Gemini 2.5 Flash Image (free) | $0 | $0 |
| Render | FFmpeg local | $0 | $0 |
| Metadata | Claude Haiku API | ~$0.02 | ~$0.60 |
| **TỔNG** | | **~$0.11** | **~$3.30** |

---

## ⚠️ LƯU Ý QUAN TRỌNG

1. **Gemini free tier:** 500 ảnh/ngày, reset midnight UTC. Đủ cho 2 video/ngày.

2. **Bước 5 chạy ~100 phút:** Thiết kế để chạy background. Luôn có resume logic.

3. **Image consistency:** Style "stick figure / cave painting" giúp 200 ảnh trông đồng nhất.

4. **PC specs:** Ryzen 9 9950X3D, 32GB RAM, không có GPU. Kokoro TTS và faster-whisper chạy trên CPU — hoàn toàn ổn.

5. **Không upload tự động:** Human review video + chọn thumbnail → upload thủ công.

---

## 🐛 XỬ LÝ LỖI THƯỜNG GẶP

- `429 Too Many Requests` (Gemini) → Sleep 60s, retry tối đa 3 lần
- `Kokoro TTS fail` → Fallback sang edge-tts tự động
- `FFmpeg error` → Log lỗi cụ thể, không crash toàn pipeline
- `Claude API timeout` → Retry 2 lần với sleep 5s
- Mọi lỗi đều ghi vào `pipeline.log` với timestamp

---

## 📝 PROMPTS CHUẨN

### System prompt cho Claude Web (Bước 1 - Human dùng):
```
You are a scriptwriter for an ancient history YouTube channel.
Niche: prehistoric humans / ancient civilizations / human survival.
Style: cinematic, mysterious, authoritative. Use "you" to pull viewer in.
Structure: Hook (20s) → Context → 3 Acts → Emotional conclusion
Length: 1500–2400 words English
Avoid: dry academic tone, bullet points, lists
Always start with a dramatic scene that throws viewer into the action.
```
