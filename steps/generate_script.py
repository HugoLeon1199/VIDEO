"""Step 1: Validate that script.txt exists and is non-empty."""

import sys
from pathlib import Path

from loguru import logger

import config


def run(video_id: str) -> None:
    script_path = Path(config.OUTPUT_DIR) / video_id / "script.txt"

    if not script_path.exists():
        logger.error("script.txt not found: {}", script_path)
        logger.info(
            "Write your script on Claude Web (claude.ai) using the system prompt in AGENTS.md, "
            "then save it to: {}", script_path
        )
        sys.exit(1)

    text = script_path.read_text(encoding="utf-8").strip()
    if not text:
        logger.error("script.txt is empty: {}", script_path)
        sys.exit(1)

    word_count = len(text.split())
    logger.info("Script validated — {} words ({:.1f} min estimated)", word_count, word_count / 150)

    if word_count < 1000:
        logger.warning("Script is short ({} words). Target is 1500–2400 words.", word_count)
    elif word_count > 2800:
        logger.warning("Script is long ({} words). Target is 1500–2400 words.", word_count)
