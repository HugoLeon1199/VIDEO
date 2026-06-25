"""Step 6: Design soundscape — rule-based SFX selection per scene.

Reads image_prompts.json + assets/sfx/library.json → matches keywords in
scene_text/prompt → writes soundscape.json. No API calls required.

Falls back to Claude/Gemini if ANTHROPIC_API_KEY or GEMINI_API_KEY is set
(for richer selection), but works fully offline with keyword rules.

soundscape.json format:
[
  {
    "scene_index": 1,
    "events": [
      {
        "type": "ambience",
        "tag": "birds",
        "offset": 0.0,
        "duration_mode": "scene",
        "volume": 0.18,
        "fade_in": 0.5,
        "fade_out": 0.8
      }
    ]
  }
]

duration_mode:
  "scene"   — play/trim to scene duration (scene["end"] - scene["start"])
  "oneshot" — play file once from start time, no trim
  "loop"    — loop/trim file to scene duration
"""

import json
import re
import sys
import time
from pathlib import Path

from loguru import logger

import config

SFX_LIBRARY_PATH = Path("assets/sfx/library.json")

# ── Rule-based keyword → tag mapping ─────────────────────────────────────────
# Each rule: (tag, [keywords], volume_override or None, fade_in, fade_out)
# Keywords matched against scene_text + prompt (case-insensitive)
# First matching rule per scene wins for ambience; oneshots can stack

AMBIENCE_RULES: list[tuple[str, list[str], float, float, float]] = [
    # (tag, keywords, volume, fade_in, fade_out)
    ("fire",          ["lửa", "bếp", "lửa trại", "fire", "campfire", "hearth", "flame"], 0.18, 0.8, 1.0),
    ("water",         ["sông", "suối", "nước", "river", "stream", "water", "lake", "hồ"], 0.16, 0.8, 1.0),
    ("night_insects", ["đêm", "ban đêm", "tối", "night", "dusk", "midnight", "stars", "sao"], 0.14, 1.0, 1.0),
    ("crowd_murmur",  ["cộng đồng", "bộ lạc", "tribe", "group", "crowd", "gather", "village", "làng", "nhóm", "người"], 0.13, 0.6, 0.8),
    ("wind",          ["gió", "đồng cỏ", "savanna", "wind", "breeze", "plain", "open", "thảo nguyên"], 0.13, 0.8, 1.0),
    ("birds",         ["rừng", "cây", "buổi sáng", "forest", "tree", "morning", "birds", "jungle", "thiên nhiên", "nature"], 0.16, 0.8, 1.0),
    ("drone_tension", ["nguy hiểm", "săn", "hunt", "danger", "predator", "tension", "căng thẳng", "chạy", "flee"], 0.12, 1.0, 1.2),
]

ONESHOT_RULES: list[tuple[str, list[str], float, float]] = [
    # (tag, keywords, volume, offset_from_scene_start)
    ("footsteps",   ["đi bộ", "di chuyển", "walk", "move", "travel", "trek", "footstep"], 0.20, 0.3),
    ("impact_hard", ["va chạm", "tấn công", "attack", "strike", "hit", "kill", "impact"], 0.25, 0.5),
    ("impact_soft", ["thả", "đặt xuống", "drop", "place", "lay down", "sit", "ngồi"], 0.18, 0.4),
    ("whoosh",      ["chuyển", "transition", "shift", "change", "fast", "nhanh", "vụt"], 0.18, 0.0),
    ("ui_alert",    ["tiếng", "giờ", "tuần", "hours", "percent", "số liệu", "%", "tiếng một", "15 tiếng", "12 tiếng", "nghiên cứu cho thấy", "studies show", "research"], 0.28, 0.2),
]

# Scenes that are clearly modern contrast (email, deadline, calendar) → ui_alert
MODERN_CONTRAST_KEYWORDS = ["email", "deadline", "calendar", "laptop", "phone", "lịch", "ca làm việc", "overtime"]


def _match_text(text: str, keywords: list[str]) -> bool:
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def _rule_based_soundscape(prompts: list[dict], available_tags: set[str]) -> list[dict]:
    """Generate soundscape using keyword matching — no API needed."""
    result = []

    for scene in prompts:
        text = (scene.get("scene_text", "") + " " + scene.get("prompt", "")).strip()
        events = []

        # Modern contrast scenes → ui_alert oneshot
        if _match_text(text, MODERN_CONTRAST_KEYWORDS) and "ui_alert" in available_tags:
            events.append({
                "type": "oneshot",
                "tag": "ui_alert",
                "offset": 0.2,
                "duration_mode": "oneshot",
                "volume": 0.30,
            })

        # Ambience: first matching rule
        for tag, keywords, volume, fade_in, fade_out in AMBIENCE_RULES:
            if tag not in available_tags:
                continue
            if _match_text(text, keywords):
                events.append({
                    "type": "ambience",
                    "tag": tag,
                    "offset": 0.0,
                    "duration_mode": "scene",
                    "volume": volume,
                    "fade_in": fade_in,
                    "fade_out": fade_out,
                })
                break  # max 1 ambience per scene

        # Oneshots (skip if already added ui_alert for this scene)
        has_ui = any(e["tag"] == "ui_alert" for e in events)
        if not has_ui:
            for tag, keywords, volume, offset in ONESHOT_RULES:
                if tag not in available_tags:
                    continue
                if tag in ("ui_alert",):
                    continue
                if _match_text(text, keywords):
                    events.append({
                        "type": "oneshot",
                        "tag": tag,
                        "offset": offset,
                        "duration_mode": "oneshot",
                        "volume": volume,
                    })
                    break  # max 1 oneshot per scene

        if events:
            result.append({"scene_index": scene["index"], "events": events})

    return result


# ── Optional: Claude/Gemini API path ─────────────────────────────────────────

SYSTEM_PROMPT = """Bạn là sound designer cho video lịch sử YouTube về người tiền sử.

Nhiệm vụ: Với mỗi scene trong danh sách, chọn sound effects phù hợp từ TAG CATALOG cho sẵn.

QUY TẮC BẮT BUỘC:
1. Chỉ dùng tags có trong catalog — KHÔNG tự đặt tag hoặc filename mới
2. Tối đa 1 ambience event per scene
3. One-shot chỉ khi hành động trong cảnh THỰC SỰ cần âm thanh đó
4. Không SFX mọi cảnh — nếu cảnh không cần âm thanh thì bỏ qua hoàn toàn
5. Volume: ambience 0.12–0.22, oneshot 0.20–0.30, ui 0.25–0.35
6. fade_in/fade_out: ambience thường 0.5–1.0s, oneshot không cần fade
7. Cảnh tiền sử không dùng modern SFX trừ cảnh đối lập cố tình (email/deadline)
8. Cảnh đêm → night_insects. Cảnh lửa → fire. Cảnh sông → water. Cảnh đông người → crowd_murmur

OUTPUT: JSON array thuần, không markdown, không giải thích.
Chỉ include scenes có events (bỏ qua scenes rỗng).
[{"scene_index": 1, "events": [{"type": "ambience", "tag": "birds", "offset": 0.0, "duration_mode": "scene", "volume": 0.18, "fade_in": 0.6, "fade_out": 0.8}]}]"""


def _build_catalog_text(library: list[dict]) -> str:
    lines = ["TAG CATALOG:"]
    for item in library:
        lines.append(f'  "{item["tag"]}" — {item["category"]}, default_volume={item["default_volume"]}')
    return "\n".join(lines)


def _parse_json_response(text: str) -> list[dict]:
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON array found in response")
    return json.loads(text[start:end])


def _call_claude(prompts: list[dict], library: list[dict]) -> list[dict]:
    import anthropic
    catalog = _build_catalog_text(library)
    scene_lines = [f'[{p["index"]}] {p.get("scene_text","")[:80]}' for p in prompts]
    user_msg = f"{catalog}\n\nSCENES:\n" + "\n".join(scene_lines)
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=config.CLAUDE_MODEL, max_tokens=8000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    logger.info("Claude tokens: {} in / {} out", resp.usage.input_tokens, resp.usage.output_tokens)
    return _parse_json_response(resp.content[0].text)


def _call_gemini(prompts: list[dict], library: list[dict]) -> list[dict]:
    from google import genai
    catalog = _build_catalog_text(library)
    scene_lines = [f'[{p["index"]}] {p.get("scene_text","")[:80]}' for p in prompts]
    user_msg = f"{SYSTEM_PROMPT}\n\n{catalog}\n\nSCENES:\n" + "\n".join(scene_lines)
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    resp = client.models.generate_content(model="gemini-2.5-flash", contents=user_msg)
    return _parse_json_response(resp.text)


def _validate(soundscape: list[dict], valid_tags: set[str]) -> list[dict]:
    cleaned = []
    for entry in soundscape:
        valid_events = [ev for ev in entry.get("events", []) if ev.get("tag") in valid_tags]
        if valid_events:
            cleaned.append({**entry, "events": valid_events})
    return cleaned


# ── Main ──────────────────────────────────────────────────────────────────────

def run(video_id: str) -> None:
    video_dir = Path(config.OUTPUT_DIR) / video_id
    prompts_path = video_dir / "image_prompts.json"
    output_path = video_dir / "soundscape.json"

    if not prompts_path.exists():
        logger.error("image_prompts.json not found: {}", prompts_path)
        sys.exit(1)
    if not SFX_LIBRARY_PATH.exists():
        logger.error("SFX library not found: {}", SFX_LIBRARY_PATH)
        sys.exit(1)

    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    library = json.loads(SFX_LIBRARY_PATH.read_text(encoding="utf-8"))
    valid_tags = {item["tag"] for item in library}

    # Check which tags have actual WAV files available
    sfx_dir = Path("assets/sfx")
    available_tags = {item["tag"] for item in library if (sfx_dir / item["file"]).exists()}
    missing_tags = valid_tags - available_tags
    if missing_tags:
        logger.warning("SFX files missing for tags: {} — will skip these", sorted(missing_tags))

    use_claude = bool(config.ANTHROPIC_API_KEY)
    use_gemini = bool(config.GEMINI_API_KEY)

    if use_claude or use_gemini:
        # API path: richer selection
        logger.info(
            "Designing soundscape via {} for {} scenes...",
            "Claude" if use_claude else "Gemini", len(prompts),
        )
        soundscape = None
        last_error = None
        max_retries = getattr(config, "CLAUDE_MAX_RETRIES", 2)
        retry_sleep = getattr(config, "CLAUDE_RETRY_SLEEP", 3)
        for attempt in range(1, max_retries + 2):
            try:
                soundscape = _call_claude(prompts, library) if use_claude else _call_gemini(prompts, library)
                break
            except Exception as e:
                last_error = e
                logger.warning("Attempt {}: {} — retrying", attempt, e)
                if attempt <= max_retries:
                    time.sleep(retry_sleep)
        if soundscape is None:
            logger.warning("API failed ({}), falling back to rule-based", last_error)
            soundscape = _rule_based_soundscape(prompts, available_tags)
        else:
            soundscape = _validate(soundscape, available_tags)
    else:
        # No API keys — use rule-based matching
        logger.info("No API keys — using rule-based SFX matching for {} scenes", len(prompts))
        soundscape = _rule_based_soundscape(prompts, available_tags)

    output_path.write_text(json.dumps(soundscape, ensure_ascii=False, indent=2), encoding="utf-8")
    total_events = sum(len(s.get("events", [])) for s in soundscape)
    logger.info(
        "Soundscape saved: {} scenes with SFX, {} total events -> {}",
        len(soundscape), total_events, output_path,
    )
