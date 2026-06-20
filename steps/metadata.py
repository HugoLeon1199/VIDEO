"""Step 7: Generate video metadata (title, description, tags, thumbnail prompts) using Claude Haiku."""

import json
import re
import sys
import time
from pathlib import Path

import anthropic
from loguru import logger

import config


def _parse_json_response(text: str) -> dict:
    """Extract JSON object from Claude's response."""
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    text = text.rstrip("`").strip()

    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in response")

    return json.loads(text[start:end])


def run(video_id: str) -> None:
    video_dir = Path(config.OUTPUT_DIR) / video_id
    script_path = video_dir / "script.txt"
    output_path = video_dir / "metadata.json"

    if not script_path.exists():
        logger.error("script.txt not found: {}", script_path)
        sys.exit(1)
    if not config.ANTHROPIC_API_KEY:
        logger.error("ANTHROPIC_API_KEY is not set")
        sys.exit(1)

    script = script_path.read_text(encoding="utf-8")

    metadata_prompt_path = Path(config.PROMPTS_DIR) / "metadata_prompt.txt"
    system_prompt = metadata_prompt_path.read_text(encoding="utf-8")

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    logger.info("Generating metadata with Claude Haiku...")

    last_error = None
    metadata = None
    for attempt in range(1, config.CLAUDE_MAX_RETRIES + 2):
        try:
            response = client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=2048,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": f"Generate metadata for this video script:\n\n{script}",
                    }
                ],
            )
            raw_text = response.content[0].text
            metadata = _parse_json_response(raw_text)
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = (input_tokens / 1_000_000) * 0.80 + (output_tokens / 1_000_000) * 4.00
            logger.info(
                "Tokens: {} in / {} out — estimated cost: ${:.4f}",
                input_tokens, output_tokens, cost,
            )
            break
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            logger.warning("Attempt {}: JSON parse failed — {}", attempt, e)
            if attempt <= config.CLAUDE_MAX_RETRIES:
                time.sleep(config.CLAUDE_RETRY_SLEEP)
        except anthropic.APIError as e:
            last_error = e
            logger.warning("Attempt {}: API error — {}", attempt, e)
            if attempt <= config.CLAUDE_MAX_RETRIES:
                time.sleep(config.CLAUDE_RETRY_SLEEP)

    if metadata is None:
        logger.error("All attempts failed: {}", last_error)
        sys.exit(1)

    output_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Metadata saved → {}", output_path)
    logger.info("Title: {}", metadata.get("title", "N/A"))
    logger.info("Tags: {} tags", len(metadata.get("tags", [])))
    logger.info("Chapters: {}", metadata.get("chapters", []))
    logger.info("Thumbnail options: {}", len(metadata.get("thumbnail_prompts", [])))
