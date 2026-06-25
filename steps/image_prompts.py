"""Step 4: Generate image prompts using Claude — smart scene segmentation.

Claude reads the full script and groups sentences into ~100-120 scenes semantically,
then writes one image prompt per scene. Timestamps are computed proportionally
by character count across total audio duration.
"""

import json
import re
import subprocess
import sys
import time
from pathlib import Path

from loguru import logger

import config

# Negative prompt used for all VI scenes
VI_NEGATIVE_PROMPT = (
    "nudity, bare chest, naked, nsfw, western cartoon style, anime, 3D render, "
    "CGI, watermark, text, logo, signature, doodle, stick figure, flat 2D vector, "
    "children, old people only, modern clothing, modern buildings, technology"
)

# Negative prompt for EN scenes
EN_NEGATIVE_PROMPT = (
    "nudity, bare chest, naked, nsfw, anime, 3D render, CGI, watermark, text, logo, signature"
)

# ── System prompt sent to Claude ─────────────────────────────────────────────

VI_SYSTEM_PROMPT = """Bạn là chuyên gia tạo storyboard cho video YouTube lịch sử tiền sử.

Tôi sẽ gửi cho bạn một script tiếng Việt và danh sách các câu đã được đánh số (1-based).
Nhiệm vụ của bạn: chia toàn bộ script thành các SCENES — mỗi scene = 1 ảnh.

NGUYÊN TẮC GOM CÂU:
- Câu ngắn (<6 từ) cùng chủ đề liệt kê → gom lại 1 scene (VD: "Họ chơi. Họ hát. Họ nhảy." → 1 scene)
- Câu dài có ý nghĩa độc lập → 1 scene riêng
- Câu trích dẫn số liệu + câu giải thích liền sau → gom 1 scene
- Câu chuyển đoạn (dấu hỏi rhetorical) → 1 scene riêng
- Target: ~100-120 scenes cho script ~8-10 phút

CÁCH VIẾT PROMPT ẢNH:
Style bắt buộc: "cinematic 2D painted documentary illustration, semi-realistic prehistoric humans, warm ochre earth tones, firelight atmosphere"
- Mô tả cảnh cụ thể: người, hành động, bối cảnh
- KHÔNG dùng text/chữ trong ảnh
- Câu về số liệu (1960, mười hai tiếng...): thêm chi tiết môi trường, đừng cố render số
- Câu về địa danh (Kalahari, Tanzania): thêm cảnh thiên nhiên đặc trưng

ICON OVERLAYS (chỉ dùng khi thực sự cần):
- Câu liệt kê hành động hiện đại đối lập (email, ca làm việc, deadline): dùng icon
- Format: [{"icon": "email", "position": "center", "label": "Không có email chưa đọc"}]
- Icon hợp lệ: email, calendar, clock, phone, laptop, checkmark, x-mark, fire, leaf, wheat, skull, heart, star, sun, moon, mountain, river, tree, person, group

TEXT OVERLAYS: để trống [] — subtitle .srt xử lý text

OUTPUT: JSON array thuần, không có markdown, không có giải thích.
Mỗi phần tử:
{
  "index": 1,
  "sentences": [1, 2],
  "scene_text": "nội dung gộp",
  "prompt": "cinematic 2D painted...",
  "icon_overlays": [],
  "text_overlays": []
}

QUAN TRỌNG: Phủ kín tất cả câu từ 1 đến N. Không bỏ sót câu nào."""

EN_SYSTEM_PROMPT = """You are a storyboard expert for a prehistoric history YouTube channel.

I will send you an English script and a numbered list of sentences (1-based).
Your task: group the script into SCENES — one image per scene.

GROUPING RULES:
- Short sentences (<6 words) on the same theme → group into 1 scene
- Long independent sentences → 1 scene each
- Stat/quote + following explanation → group together
- Target: ~100-120 scenes for an ~8-10 minute script

IMAGE PROMPT STYLE:
Required style: "cinematic wide shot, photorealistic, natural lighting, shallow depth of field, 16:9, no text"
- Describe specific scene: people, action, environment
- No text/numbers rendered in the image
- For stat sentences: describe the environment context instead

ICON OVERLAYS: only for modern-life-contrast listing sentences
TEXT OVERLAYS: leave as [] — subtitle .srt handles all text

OUTPUT: pure JSON array only, no markdown, no explanation.
Each element:
{
  "index": 1,
  "sentences": [1, 2],
  "scene_text": "combined text",
  "prompt": "cinematic wide shot...",
  "icon_overlays": [],
  "text_overlays": []
}

IMPORTANT: Cover all sentences from 1 to N. Miss none."""


# ── Script parsing ────────────────────────────────────────────────────────────

_SCRIPT_STOP_MARKERS = (
    "COMMENT SEED:",
    "RESEARCH NOTES:",
    "Your script is ready",
    "Save as:",
    "Then: python",
)


def _load_script_text(script_path: Path) -> str:
    text = script_path.read_text(encoding="utf-8").lstrip("﻿")
    for marker in _SCRIPT_STOP_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    return text.strip()


def _split_sentences(script_text: str) -> list[str]:
    """Split script into sentences for proportional timestamp mapping."""
    paragraphs = re.split(r"\n{2,}", script_text)
    sentences = []
    first = True
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if first:
            first = False
            if not re.search(r"[.!?]$", para) and len(para.split()) <= 12:
                continue
        for part in re.split(r"(?<=[.!?])\s+", para):
            part = part.strip()
            if part:
                sentences.append(part)
    return sentences


# ── Timestamp computation ─────────────────────────────────────────────────────

def _get_audio_duration(audio_path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def _compute_sentence_times(sentences: list[str], total_dur: float) -> list[dict]:
    """Assign start/end to each sentence proportional to character count."""
    total_chars = sum(len(s) for s in sentences)
    if total_chars == 0:
        return []
    cursor = 0.0
    result = []
    for i, s in enumerate(sentences, 1):
        dur = total_dur * len(s) / total_chars
        result.append({
            "index": i,
            "start": round(cursor, 3),
            "end": round(cursor + dur, 3),
            "text": s,
        })
        cursor += dur
    return result


# ── Claude API ────────────────────────────────────────────────────────────────

def _call_claude(script_text: str, sentences: list[str], system_prompt: str) -> list[dict]:
    import anthropic

    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences, 1))
    user_message = (
        f"Đây là toàn bộ script ({len(sentences)} câu):\n\n"
        f"SCRIPT ĐẦY ĐỦ:\n{script_text}\n\n"
        f"DANH SÁCH CÂU ĐÁNH SỐ:\n{numbered}\n\n"
        f"Hãy tách thành scenes. Trả về JSON array thuần (không markdown)."
    )

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=16000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    logger.info(
        "Claude tokens: {} in / {} out",
        response.usage.input_tokens, response.usage.output_tokens,
    )
    return _parse_json_response(response.content[0].text)


def _call_gemini(script_text: str, sentences: list[str], system_prompt: str) -> list[dict]:
    from google import genai

    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences, 1))
    full_prompt = (
        f"{system_prompt}\n\n"
        f"SCRIPT ĐẦY ĐỦ:\n{script_text}\n\n"
        f"DANH SÁCH CÂU ĐÁNH SỐ:\n{numbered}\n\n"
        f"Hãy tách thành scenes. Trả về JSON array thuần (không markdown)."
    )
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=full_prompt,
    )
    return _parse_json_response(response.text)


def _parse_json_response(text: str) -> list[dict]:
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON array found in response")
    return json.loads(text[start:end])


# ── Scene → timestamp mapping ─────────────────────────────────────────────────

def _map_scene_times(scenes: list[dict], sentence_times: list[dict]) -> list[dict]:
    """Map each scene's sentences[] to start/end timestamps."""
    st_map = {s["index"]: s for s in sentence_times}
    result = []
    for i, scene in enumerate(scenes, 1):
        sent_indices = scene.get("sentences", [])
        if not sent_indices:
            logger.warning("Scene {} has no sentences", i)
            continue
        first_idx = min(sent_indices)
        last_idx = max(sent_indices)
        start = st_map.get(first_idx, {}).get("start", 0.0)
        end = st_map.get(last_idx, {}).get("end", 0.0)
        result.append({
            "index": i,
            "sentences": sent_indices,
            "scene_text": scene.get("scene_text", ""),
            "start": round(start, 3),
            "end": round(end, 3),
            "prompt": scene.get("prompt", ""),
            "negative_prompt": VI_NEGATIVE_PROMPT,
            "icon_overlays": scene.get("icon_overlays", []),
            "text_overlays": scene.get("text_overlays", []),
        })
    return result


def _validate_scene_times(prompts: list[dict], audio_duration: float) -> None:
    """Validate scene timings before render. Exits on any critical error."""
    errors = []
    warnings = []

    prev_end = 0.0
    for p in prompts:
        idx = p["index"]
        s, e = p["start"], p["end"]

        if e <= s:
            errors.append(f"Scene {idx}: end ({e}s) <= start ({s}s)")
        if s < prev_end - 0.01:
            errors.append(
                f"Scene {idx}: start ({s}s) goes back before previous end ({prev_end}s)"
            )
        prev_end = e

    # Last scene should end near audio duration
    if prompts:
        last_end = prompts[-1]["end"]
        drift = abs(last_end - audio_duration)
        if drift > 5.0:
            errors.append(
                f"Last scene ends at {last_end}s but audio is {audio_duration}s "
                f"(drift {drift:.1f}s > 5s threshold)"
            )
        elif drift > 1.0:
            warnings.append(
                f"Last scene ends at {last_end}s vs audio {audio_duration}s (drift {drift:.1f}s)"
            )

    for w in warnings:
        logger.warning("Timing warning: {}", w)
    if errors:
        for e in errors:
            logger.error("Timing error: {}", e)
        logger.error(
            "{} timing error(s) found — fix timestamps.json and re-run step 3 before rendering",
            len(errors),
        )
        sys.exit(1)

    logger.info(
        "Timing OK: {} scenes, {:.1f}s — {:.1f}s (audio {:.1f}s)",
        len(prompts),
        prompts[0]["start"] if prompts else 0,
        prompts[-1]["end"] if prompts else 0,
        audio_duration,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def run(video_id: str, n_override: int | None = None) -> None:
    video_dir = Path(config.OUTPUT_DIR) / video_id
    script_path = video_dir / "script.txt"
    audio_path = video_dir / "audio.mp3"
    output_path = video_dir / "image_prompts.json"

    if not script_path.exists():
        logger.error("script.txt not found: {}", script_path)
        sys.exit(1)
    if not audio_path.exists():
        logger.error("audio.mp3 not found: {}", audio_path)
        sys.exit(1)

    use_claude = bool(config.ANTHROPIC_API_KEY)
    use_gemini = bool(config.GEMINI_API_KEY)
    if not use_claude and not use_gemini:
        logger.error("Neither ANTHROPIC_API_KEY nor GEMINI_API_KEY set. Add to .env file.")
        sys.exit(1)
    if use_claude:
        logger.info("Using Claude claude-sonnet-4-6 for scene segmentation")
    else:
        logger.info("Falling back to Gemini (ANTHROPIC_API_KEY not set)")

    script_text = _load_script_text(script_path)
    sentences = _split_sentences(script_text)

    if n_override is not None:
        sentences = sentences[:n_override]
        logger.info("Demo mode: using first {} sentences", len(sentences))

    logger.info("Script: {} sentences detected", len(sentences))

    total_dur = _get_audio_duration(audio_path)
    if total_dur <= 0:
        logger.error("Could not read audio duration from {}", audio_path)
        sys.exit(1)
    logger.info("Audio duration: {:.1f}s", total_dur)

    # Prefer timestamps.json (from sentence-mode TTS or Whisper align) — exact timing
    # Fallback: proportional by character count (less accurate)
    timestamps_path = video_dir / "timestamps.json"
    if timestamps_path.exists():
        sentence_times = json.loads(timestamps_path.read_text(encoding="utf-8"))
        logger.info("Using sentence timestamps from timestamps.json ({} entries)", len(sentence_times))
    else:
        sentence_times = _compute_sentence_times(sentences, total_dur)
        logger.info("Using proportional timestamps (no timestamps.json found)")

    logger.info("Calling Claude claude-sonnet-4-6 to generate scenes...")

    scenes = None
    last_error = None
    for attempt in range(1, config.CLAUDE_MAX_RETRIES + 2):
        try:
            if use_claude:
                scenes = _call_claude(script_text, sentences, VI_SYSTEM_PROMPT)
            else:
                scenes = _call_gemini(script_text, sentences, VI_SYSTEM_PROMPT)
            break
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            logger.warning("Attempt {}: JSON parse failed — {}", attempt, e)
            if attempt <= config.CLAUDE_MAX_RETRIES:
                time.sleep(config.CLAUDE_RETRY_SLEEP)
        except Exception as e:
            last_error = e
            logger.warning("Attempt {}: API error — {}", attempt, e)
            if attempt <= config.CLAUDE_MAX_RETRIES:
                time.sleep(config.CLAUDE_RETRY_SLEEP)

    if scenes is None:
        logger.error("All Claude attempts failed: {}", last_error)
        sys.exit(1)

    logger.info("Claude returned {} scenes (from {} sentences)", len(scenes), len(sentences))

    prompts = _map_scene_times(scenes, sentence_times)

    # Fix last entry end time to exactly match audio duration
    if prompts:
        prompts[-1]["end"] = round(total_dur, 3)

    _validate_scene_times(prompts, total_dur)

    output_path.write_text(json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved {} scenes → {}", len(prompts), output_path)
