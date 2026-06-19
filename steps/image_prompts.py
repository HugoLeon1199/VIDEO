"""Step 4: Generate image prompts using Gemini Flash (text) or Claude Haiku API."""

import json
import re
import sys
import time
from pathlib import Path

from loguru import logger

import config

STYLE_SUFFIX = "ancient art style, prehistoric, minimalist, warm earth tones, no text"

GEMINI_TEXT_MODEL = "gemini-2.5-flash"


def _build_prompt(timestamps: list[dict], script: str, n: int, system_prompt: str) -> str:
    transcript_lines = "\n".join(
        f"[{seg['start']:.1f}s–{seg['end']:.1f}s] {seg['text']}"
        for seg in timestamps
    )
    total_duration = timestamps[-1]["end"] if timestamps else 600.0
    scene_duration = total_duration / n

    return (
        f"{system_prompt.replace('{N}', str(n))}\n\n"
        f"Create exactly {n} image prompts for a video that is {total_duration:.1f} seconds long.\n"
        f"Each scene is approximately {scene_duration:.1f} seconds.\n\n"
        f"TRANSCRIPT WITH TIMESTAMPS:\n{transcript_lines}\n\n"
        f"FULL SCRIPT (for additional context):\n{script[:3000]}\n\n"
        f"Return exactly {n} JSON objects covering the full duration from 0.0 to {total_duration:.1f}."
    )


def _parse_json_response(text: str) -> list[dict]:
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    text = text.rstrip("`").strip()
    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON array found in response")
    return json.loads(text[start:end])


def _add_style_suffix(prompts: list[dict]) -> list[dict]:
    for item in prompts:
        p = item.get("prompt", "")
        if STYLE_SUFFIX not in p:
            item["prompt"] = f"{p.rstrip(', ')}, {STYLE_SUFFIX}"
    return prompts


def _run_gemini(full_prompt: str) -> list[dict]:
    from google import genai

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_TEXT_MODEL,
        contents=full_prompt,
    )
    return _parse_json_response(response.text)


def _run_claude(timestamps: list[dict], script: str, n: int, system_prompt: str) -> tuple[list[dict], object]:
    import anthropic

    user_message = (
        f"Create exactly {n} image prompts for a video that is "
        f"{timestamps[-1]['end']:.1f} seconds long.\n"
        + "\n".join(f"[{s['start']:.1f}s–{s['end']:.1f}s] {s['text']}" for s in timestamps)
        + f"\n\nReturn exactly {n} JSON objects."
    )
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=8192,
        system=system_prompt.replace("{N}", str(n)),
        messages=[{"role": "user", "content": user_message}],
    )
    return _parse_json_response(response.content[0].text), response


def run(video_id: str, n_override: int | None = None) -> None:
    video_dir = Path(config.OUTPUT_DIR) / video_id
    timestamps_path = video_dir / "timestamps.json"
    script_path = video_dir / "script.txt"
    output_path = video_dir / "image_prompts.json"

    if not timestamps_path.exists():
        logger.error("timestamps.json not found: {}", timestamps_path)
        sys.exit(1)
    if not script_path.exists():
        logger.error("script.txt not found: {}", script_path)
        sys.exit(1)

    timestamps = json.loads(timestamps_path.read_text(encoding="utf-8"))
    script = script_path.read_text(encoding="utf-8")

    system_prompt_path = Path(config.PROMPTS_DIR) / "system_prompt.txt"
    system_prompt = system_prompt_path.read_text(encoding="utf-8")

    n = n_override if n_override is not None else config.IMAGES_PER_VIDEO

    # If demo mode, trim timestamps to first N*3 seconds
    if n_override is not None:
        target_duration = n * 3.0
        timestamps = [t for t in timestamps if t["start"] < target_duration]
        if timestamps:
            timestamps[-1]["end"] = min(timestamps[-1]["end"], target_duration)

    # Prefer Gemini (free), fall back to Claude if GEMINI_API_KEY not set
    use_gemini = bool(config.GEMINI_API_KEY)
    use_claude = bool(config.ANTHROPIC_API_KEY)

    if not use_gemini and not use_claude:
        logger.error("Neither GEMINI_API_KEY nor ANTHROPIC_API_KEY is set")
        sys.exit(1)

    prompts = None
    last_error = None

    for attempt in range(1, config.CLAUDE_MAX_RETRIES + 2):
        try:
            if use_gemini:
                logger.info("Requesting {} image prompts from Gemini Flash...", n)
                full_prompt = _build_prompt(timestamps, script, n, system_prompt)
                prompts = _run_gemini(full_prompt)
            else:
                logger.info("Requesting {} image prompts from Claude Haiku...", n)
                prompts, response = _run_claude(timestamps, script, n, system_prompt)
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                cost = (input_tokens / 1_000_000) * 0.25 + (output_tokens / 1_000_000) * 1.25
                logger.info("Claude tokens: {} in / {} out — ${:.4f}", input_tokens, output_tokens, cost)
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

    if prompts is None:
        logger.error("All attempts failed: {}", last_error)
        sys.exit(1)

    # Normalize count
    if len(prompts) > n:
        prompts = prompts[:n]
    while len(prompts) < n:
        last = dict(prompts[-1])
        last["index"] = len(prompts) + 1
        prompts.append(last)

    for i, item in enumerate(prompts, 1):
        item["index"] = i
    prompts = _add_style_suffix(prompts)

    output_path.write_text(json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Image prompts saved ({} items) → {}", len(prompts), output_path)
