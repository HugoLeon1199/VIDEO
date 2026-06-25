# Style Guide — VI Track (Vietnamese Videos)

## Image Prompt Style

**Bắt buộc bắt đầu bằng:**
```
cinematic 2D painted documentary illustration, semi-realistic prehistoric humans, warm ochre earth tones, firelight atmosphere
```

**Cụ thể cảnh:**
- Người: semi-realistic, không quá manga, không quá photo
- Bối cảnh: savanna, Kalahari, camp fire, acacia trees, baobab, rocky landscape
- Ánh sáng: warm golden hour, firelight, dappled shade — không cold/blue
- Góc: wide shot, close-up face, overhead camp view — mix để tránh monotonous

**Negative prompt chuẩn:**
```
nudity, bare chest, naked, nsfw, western cartoon style, anime, 3D render, CGI, watermark, text, logo, signature, doodle, stick figure, flat 2D vector, children, old people only, modern clothing, modern buildings, technology
```

## Scene Grouping Rules

- Câu ngắn (<6 từ) cùng chủ đề → gom 1 scene (VD: "Họ chơi. Họ hát. Họ nhảy." → 1 scene múa)
- Câu dài ý nghĩa độc lập → 1 scene riêng
- Câu số liệu + câu giải thích liền sau → gom 1 scene
- Target: 65-80 scenes cho script 8-10 phút (avg ~7s/scene)

## Icon Overlays

Chỉ dùng khi câu liệt kê vật thể hiện đại đối lập với lối sống tiền sử:
```json
{"icon": "email", "position": "center", "label": "Không có email chưa đọc"}
{"icon": "calendar", "position": "center", "label": "Không có ca tiếp theo"}
```
Icons: email, calendar, clock, phone, laptop, checkmark, x-mark, fire, leaf, wheat, skull, heart, star, sun, moon, mountain, river, tree, person, group

## Text Overlays (subtitle only — không burn vào video)

Subtitle .srt xử lý tất cả text. `text_overlays` trong image_prompts.json chỉ là metadata cho reference, không dùng trong render hiện tại.

## Timing

Proportional theo character count:
```
dur(sentence) = total_audio_duration × len(sentence_chars) / total_chars
```

## RunPod Backend

- Endpoint: FLUX.1-dev 12B (tsq8xb64xj3c57)
- Steps: 22, guidance_scale: 3.5
- Resolution: 1024×576
- Candidates: 1 (production), 3 (nếu muốn chọn)
- Workers: 5-10
