"""Step 5: Generate 200 images using Gemini 2.5 Flash Image API with resume support."""

import base64
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from tqdm import tqdm

import config


def _load_progress(progress_path: Path) -> set[int]:
    """Return set of already-completed image indices."""
    if not progress_path.exists():
        return set()
    try:
        data = json.loads(progress_path.read_text(encoding="utf-8"))
        return set(data.get("completed", []))
    except (json.JSONDecodeError, KeyError):
        return set()


def _save_progress(progress_path: Path, completed: set[int]) -> None:
    data = {
        "completed": sorted(completed),
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }
    progress_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _generate_image(client, prompt: str) -> bytes:
    """Call Gemini image generation API and return raw PNG bytes."""
    from google.genai import types

    response = client.models.generate_content(
        model=config.GEMINI_IMAGE_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )

    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            return part.inline_data.data  # raw bytes

    raise RuntimeError("Gemini returned no image data")


def run(video_id: str, n_override: int | None = None) -> None:
    from google import genai

    video_dir = Path(config.OUTPUT_DIR) / video_id
    prompts_path = video_dir / "image_prompts.json"
    images_dir = video_dir / "images"
    progress_path = images_dir / "progress.json"

    if not prompts_path.exists():
        logger.error("image_prompts.json not found: {}", prompts_path)
        sys.exit(1)
    if not config.GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY is not set")
        sys.exit(1)

    images_dir.mkdir(parents=True, exist_ok=True)

    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    completed = _load_progress(progress_path)

    # Also check for existing image files (handles manual resume without progress.json)
    for img_file in images_dir.glob("img_*.png"):
        try:
            idx = int(img_file.stem.split("_")[1])
            completed.add(idx)
        except (ValueError, IndexError):
            pass

    remaining = [p for p in prompts if p["index"] not in completed]
    logger.info(
        "Images: {}/{} already done, {} remaining",
        len(completed), len(prompts), len(remaining),
    )

    if not remaining:
        logger.info("All images already generated.")
        return

    client = genai.Client(api_key=config.GEMINI_API_KEY)

    with tqdm(total=len(remaining), desc="Generating images", unit="img") as pbar:
        for item in remaining:
            idx = item["index"]
            img_path = images_dir / f"img_{idx:03d}.png"
            prompt_text = item["prompt"]

            success = False
            for attempt in range(1, config.GEMINI_MAX_RETRIES + 1):
                try:
                    image_bytes = _generate_image(client, prompt_text)
                    img_path.write_bytes(image_bytes)
                    completed.add(idx)
                    _save_progress(progress_path, completed)
                    success = True
                    break
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                        logger.warning(
                            "Rate limit (img {}), attempt {}/{}. Sleeping {}s...",
                            idx, attempt, config.GEMINI_MAX_RETRIES, config.GEMINI_RETRY_SLEEP,
                        )
                        time.sleep(config.GEMINI_RETRY_SLEEP)
                    else:
                        logger.warning(
                            "Error generating img {} (attempt {}): {}",
                            idx, attempt, e,
                        )
                        if attempt < config.GEMINI_MAX_RETRIES:
                            time.sleep(5)

            if not success:
                logger.error("Failed to generate img {} after {} attempts. Skipping.", idx, config.GEMINI_MAX_RETRIES)
            else:
                pbar.update(1)
                pbar.set_postfix({"last": f"img_{idx:03d}"})

            # Always sleep between requests to respect rate limit
            if remaining.index(item) < len(remaining) - 1:
                time.sleep(config.GEMINI_RATE_LIMIT_SLEEP)

    final_count = len(list(images_dir.glob("img_*.png")))
    logger.info(
        "Image generation complete: {}/{} images in {}",
        final_count, len(prompts), images_dir,
    )

    if final_count < len(prompts):
        logger.warning(
            "{} images missing. Run with --resume to retry.",
            len(prompts) - final_count,
        )
