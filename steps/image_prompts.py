"""Step 4: Generate image prompts — 1 image per transcript sentence, exact timing match."""

import json
import re
import sys
import time
from pathlib import Path

from loguru import logger

import config

STYLE_SUFFIX = "cave painting style, ochre and charcoal on stone, no text, 16:9"
GEMINI_TEXT_MODEL = "gemini-2.5-flash"


def _build_prompt(timestamps: list[dict], script: str, system_prompt: str) -> str:
    n = len(timestamps)
    transcript_lines = "\n".join(
        f"[{seg['index']}] [{seg['start']:.2f}s–{seg['end']:.2f}s] {seg['text']}"
        for seg in timestamps
    )
    total_duration = timestamps[-1]["end"] if timestamps else 0.0

    return (
        f"{system_prompt.replace('{N}', str(n))}\n\n"
        f"The video has {n} sentences and is {total_duration:.1f} seconds long.\n"
        f"You must return EXACTLY {n} JSON objects — one per sentence, using the EXACT index, start, and end from below.\n\n"
        f"TRANSCRIPT (index, timing, text):\n{transcript_lines}\n\n"
        f"FULL SCRIPT (for style context):\n{script[:3000]}\n\n"
        f"Return a JSON array of exactly {n} objects:\n"
        f'[{{"index": 1, "start": <exact>, "end": <exact>, "prompt": "..."}}, ...]\n'
        f"Use the exact index/start/end values from the transcript above. Do not invent timings."
    )


def _parse_json_response(text: str) -> list[dict]:
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    text = text.rstrip("`").strip()
    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON array found in response")
    return json.loads(text[start:end])


_STRIP_PATTERNS = [
    "ancient art style", "prehistoric", "minimalist", "warm earth tones",
    "no text", "wide 16:9 landscape composition", "horizontal orientation",
    "cave painting style", "ochre and charcoal on stone", "prehistoric rock art",
    "16:9",
]

def _add_style_suffix(prompts: list[dict]) -> list[dict]:
    for item in prompts:
        p = item.get("prompt", "").rstrip(", ")
        # Strip any existing style keywords to avoid duplication
        for pattern in _STRIP_PATTERNS:
            # Remove trailing occurrences only (AI tends to append them at end)
            p = re.sub(r",?\s*" + re.escape(pattern) + r"\s*(?=,|$)", "", p, flags=re.IGNORECASE).strip().rstrip(",").strip()
        item["prompt"] = f"{p}, {STYLE_SUFFIX}"
    return prompts


def _enforce_timings(prompts: list[dict], timestamps: list[dict]) -> list[dict]:
    """Force each prompt's index/start/end to match the source timestamp exactly."""
    ts_map = {t["index"]: t for t in timestamps}
    result = []
    for item in prompts:
        idx = item.get("index")
        if idx in ts_map:
            item["index"] = ts_map[idx]["index"]
            item["start"] = ts_map[idx]["start"]
            item["end"] = ts_map[idx]["end"]
        result.append(item)
    return result


def _run_gemini(full_prompt: str) -> list[dict]:
    from google import genai

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_TEXT_MODEL,
        contents=full_prompt,
    )
    return _parse_json_response(response.text)


def _run_claude(timestamps: list[dict], script: str, system_prompt: str) -> tuple[list[dict], object]:
    import anthropic

    n = len(timestamps)
    transcript_lines = "\n".join(
        f"[{s['index']}] [{s['start']:.2f}s–{s['end']:.2f}s] {s['text']}"
        for s in timestamps
    )
    user_message = (
        f"Return exactly {n} image prompts, one per sentence. "
        f"Use the exact index/start/end from each line.\n\n"
        f"{transcript_lines}"
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
    system_prompt = (Path(config.PROMPTS_DIR) / "system_prompt.txt").read_text(encoding="utf-8")

    # Demo mode: trim to first n_override sentences
    if n_override is not None:
        timestamps = timestamps[:n_override]
        logger.info("Demo mode: using first {} sentences", len(timestamps))

    n = len(timestamps)
    logger.info("Generating {} image prompts (1 per sentence)...", n)

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
                full_prompt = _build_prompt(timestamps, script, system_prompt)
                prompts = _run_gemini(full_prompt)
            else:
                prompts, response = _run_claude(timestamps, script, system_prompt)
                logger.info(
                    "Claude tokens: {} in / {} out",
                    response.usage.input_tokens, response.usage.output_tokens,
                )
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

    # Trim if AI returned too many
    prompts = prompts[:n]

    # Pad if AI returned too few (duplicate last)
    while len(prompts) < n:
        last = dict(prompts[-1])
        prompts.append(last)

    # Force exact timings from timestamps (AI must not invent timings)
    prompts = _enforce_timings(prompts, timestamps)

    # Re-index sequentially
    for i, item in enumerate(prompts, 1):
        item["index"] = i

    prompts = _add_style_suffix(prompts)

    output_path.write_text(json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved {} prompts → {}", len(prompts), output_path)
