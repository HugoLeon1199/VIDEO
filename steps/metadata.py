"""Step 8: Build publishing metadata, preferring creative_package.json."""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import anthropic
from loguru import logger

import config
from steps.creative_package import CreativePackageError, _atomic_write_json, load_validated_package


def _parse_json_response(text: str) -> dict:
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    text = text.rstrip("`").strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in response")
    return json.loads(text[start:end])


def _publishing_dir(video_dir: Path) -> Path:
    return video_dir / config.PUBLISHING_DIRNAME


def _format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _build_chapters(chapter_plan: list[dict[str, Any]], timestamps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(timestamps) < 3:
        raise ValueError("timestamps.json must contain at least 3 sentences for chapter mapping")
    chapter_candidates: list[dict[str, Any]] = []
    seen_sentence_indices: set[int] = set()
    for idx, item in enumerate(chapter_plan):
        sentence_index = int(item["sentence_index"])
        if sentence_index in seen_sentence_indices or sentence_index > len(timestamps):
            continue
        seen_sentence_indices.add(sentence_index)
        start = float(timestamps[sentence_index - 1]["start"])
        if idx == 0:
            start = 0.0
        chapter_candidates.append({"sentence_index": sentence_index, "label": item["label"], "start": start})
    if not chapter_candidates:
        raise ValueError("chapter_plan did not map to any timestamp entries")
    chapter_candidates.sort(key=lambda item: item["start"])
    chapter_candidates[0]["start"] = 0.0
    merged: list[dict[str, Any]] = []
    for item in chapter_candidates:
        if not merged:
            merged.append(item)
            continue
        if item["start"] - merged[-1]["start"] < 10.0:
            merged[-1]["label"] = f"{merged[-1]['label']} / {item['label']}"
            merged[-1]["sentence_index"] = min(merged[-1]["sentence_index"], item["sentence_index"])
        else:
            merged.append(item)
    while len(merged) < 3 and len(merged) < len(timestamps):
        next_sentence = min(len(timestamps), merged[-1]["sentence_index"] + 1)
        candidate_start = max(float(timestamps[next_sentence - 1]["start"]), merged[-1]["start"] + 10.0)
        if candidate_start >= float(timestamps[-1]["end"]):
            break
        merged.append({"sentence_index": next_sentence, "label": f"Chapter {len(merged) + 1}", "start": candidate_start})
    if len(merged) < 3:
        raise ValueError("Could not build at least 3 valid chapters from chapter_plan")
    return merged


def _write_creative_package_outputs(video_dir: Path, validated_package: dict[str, Any]) -> dict[str, Any]:
    timestamps_path = video_dir / "timestamps.json"
    if not timestamps_path.exists():
        raise FileNotFoundError(f"Missing timestamps.json: {timestamps_path}")
    timestamps = json.loads(timestamps_path.read_text(encoding="utf-8"))
    chapters = _build_chapters(validated_package["chapter_plan"], timestamps)
    publishing_dir = _publishing_dir(video_dir)
    publishing_dir.mkdir(parents=True, exist_ok=True)

    title_options = [item["text"] for item in validated_package["title_options"]]
    tags = [str(keyword).strip() for keyword in validated_package.get("search_keywords", []) if str(keyword).strip()]
    chapter_lines = [f"{_format_timestamp(item['start'])} {item['label']}" for item in chapters]
    package = {
        "package_version": validated_package["package_version"],
        "script_sha256": validated_package["script_sha256"],
        "language": validated_package["language"],
        "core_promise": validated_package["core_promise"],
        "target_viewer": validated_package["target_viewer"],
        "primary_hook": validated_package["primary_hook"],
        "title_options": validated_package["title_options"],
        "description_draft": validated_package["description_draft"],
        "search_keywords": tags,
        "chapters": chapters,
        "thumbnail_prompts_path": str(publishing_dir / "thumbnail_prompts.json"),
    }
    _atomic_write_json(publishing_dir / "package.json", package)
    (publishing_dir / "title_options.txt").write_text("\n".join(title_options), encoding="utf-8")
    (publishing_dir / "description.txt").write_text(validated_package["description_draft"].strip() + "\n", encoding="utf-8")
    (publishing_dir / "chapters.txt").write_text("\n".join(chapter_lines) + "\n", encoding="utf-8")
    (publishing_dir / "tags.txt").write_text(", ".join(tags), encoding="utf-8")

    thumbnail_contact_sheet = publishing_dir / "thumbnail_contact_sheet.jpg"
    generated_thumbnail_count = len([path for path in (publishing_dir / "thumbnails").glob("thumbnail_*.jpg") if path.stem[10:12].isdigit()]) if (publishing_dir / "thumbnails").exists() else 0
    diagnostics = {
        "package_version": validated_package["package_version"],
        "script_sha256": validated_package["script_sha256"],
        "language": validated_package["language"],
        "title_count": len(validated_package["title_options"]),
        "concept_count": len(validated_package["thumbnail_concepts"]),
        "thumbnail_prompt_count": len(json.loads((publishing_dir / "thumbnail_prompts.json").read_text(encoding="utf-8"))) if (publishing_dir / "thumbnail_prompts.json").exists() else 0,
        "thumbnail_generated_count": generated_thumbnail_count,
        "thumbnail_failed_ids": json.loads((publishing_dir / "thumbnail_generation_diagnostics.json").read_text(encoding="utf-8")).get("thumbnail_failed_ids", []) if (publishing_dir / "thumbnail_generation_diagnostics.json").exists() else [],
        "chapter_count": len(chapters),
        "validation_passed": True,
        "warnings": validated_package.get("validation_warnings", []),
        "thumbnail_contact_sheet": str(thumbnail_contact_sheet) if thumbnail_contact_sheet.exists() else None,
    }
    _atomic_write_json(publishing_dir / "publishing_diagnostics.json", diagnostics)

    legacy_metadata = {
        "title": title_options[0],
        "title_options": title_options,
        "description": validated_package["description_draft"],
        "tags": tags,
        "chapters": chapter_lines,
    }
    _atomic_write_json(video_dir / "metadata.json", legacy_metadata)
    return package


def _generate_legacy_metadata(script: str) -> dict[str, Any]:
    metadata_prompt_path = Path(config.PROMPTS_DIR) / "metadata_prompt.txt"
    system_prompt = metadata_prompt_path.read_text(encoding="utf-8")
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    last_error = None
    metadata = None
    for attempt in range(1, config.CLAUDE_MAX_RETRIES + 2):
        try:
            response = client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=2048,
                system=system_prompt,
                messages=[{"role": "user", "content": f"Generate metadata for this video script:\n\n{script}"}],
            )
            metadata = _parse_json_response(response.content[0].text)
            break
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            logger.warning("Attempt {}: JSON parse failed — {}", attempt, exc)
            if attempt <= config.CLAUDE_MAX_RETRIES:
                time.sleep(config.CLAUDE_RETRY_SLEEP)
        except anthropic.APIError as exc:
            last_error = exc
            logger.warning("Attempt {}: API error — {}", attempt, exc)
            if attempt <= config.CLAUDE_MAX_RETRIES:
                time.sleep(config.CLAUDE_RETRY_SLEEP)
    if metadata is None:
        raise RuntimeError(f"All metadata attempts failed: {last_error}")
    return metadata


def run(video_id: str, allow_stale_package: bool = False) -> None:
    video_dir = Path(config.OUTPUT_DIR) / video_id
    script_path = video_dir / "script.txt"
    if not script_path.exists():
        logger.error("script.txt not found: {}", script_path)
        sys.exit(1)

    creative_package_path = video_dir / "creative_package.json"
    if creative_package_path.exists():
        try:
            validated_package = load_validated_package(video_dir, allow_stale_package=allow_stale_package)
        except CreativePackageError as exc:
            logger.error("Creative package validation failed: {}", exc)
            sys.exit(1)
        package = _write_creative_package_outputs(video_dir, validated_package)
        logger.info("Publishing package saved → {}", _publishing_dir(video_dir) / "package.json")
        logger.info("Primary title: {}", package["title_options"][0]["text"])
        return

    if not config.ANTHROPIC_API_KEY:
        logger.error("ANTHROPIC_API_KEY is not set")
        sys.exit(1)
    script = script_path.read_text(encoding="utf-8")
    metadata = _generate_legacy_metadata(script)
    output_path = video_dir / "metadata.json"
    _atomic_write_json(output_path, metadata)
    logger.info("Metadata saved → {}", output_path)
