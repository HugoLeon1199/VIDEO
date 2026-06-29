from __future__ import annotations

import json
import math
import shutil
import string
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from loguru import logger

import config
from image_generation.schemas import SceneRequest
from steps.creative_package import _atomic_write_json, load_validated_package


class ThumbnailGenerationError(RuntimeError):
    pass


def _publishing_dir(video_dir: Path) -> Path:
    return video_dir / config.PUBLISHING_DIRNAME


def _thumbnails_dir(video_dir: Path) -> Path:
    return _publishing_dir(video_dir) / "thumbnails"


def _thumbnail_prompts_path(video_dir: Path) -> Path:
    return _publishing_dir(video_dir) / "thumbnail_prompts.json"


def _thumbnail_log_path(video_dir: Path) -> Path:
    return _publishing_dir(video_dir) / "thumbnail_generation_log.json"


def _thumbnail_diagnostics_path(video_dir: Path) -> Path:
    return _publishing_dir(video_dir) / "thumbnail_generation_diagnostics.json"


def _thumbnail_contact_sheet_path(video_dir: Path) -> Path:
    return _publishing_dir(video_dir) / "thumbnail_contact_sheet.jpg"


def _background_path(video_dir: Path, concept_id: int) -> Path:
    return _thumbnails_dir(video_dir) / f"thumbnail_{concept_id:02d}_background.png"


def _thumbnail_path(video_dir: Path, concept_id: int) -> Path:
    return _thumbnails_dir(video_dir) / f"thumbnail_{concept_id:02d}.jpg"


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _image_is_readable(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            width, height = img.size
        return width > 0 and height > 0
    except (OSError, ValueError):
        return False


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(config.THUMBNAIL_FONT_FAMILY, size=size)
    except OSError:
        logger.warning("Could not load configured thumbnail font '{}'; falling back to default", config.THUMBNAIL_FONT_FAMILY)
        return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width or not current:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    if len(lines) > 2:
        raise ThumbnailGenerationError(f"Thumbnail text cannot fit within 2 lines: {text}")
    return lines


def _draw_text_block(image: Image.Image, text: str, text_side: str) -> Image.Image:
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _load_font(config.THUMBNAIL_FONT_SIZE)
    shadow_color = tuple(config.THUMBNAIL_SHADOW_COLOR)
    stroke_color = tuple(config.THUMBNAIL_STROKE_COLOR)
    text_color = tuple(config.THUMBNAIL_TEXT_COLOR)
    width, height = image.size
    safe_margin = config.THUMBNAIL_SAFE_MARGIN
    panel_width = int(width * config.THUMBNAIL_TEXT_PANEL_RATIO)
    x0 = safe_margin if text_side == "left" else width - panel_width - safe_margin
    x1 = width // 2 - safe_margin if text_side == "left" else width - safe_margin
    max_width = max(40, x1 - x0)
    lines = _wrap_text(draw, text, font, max_width)
    line_heights = [draw.textbbox((0, 0), line, font=font, stroke_width=config.THUMBNAIL_STROKE_WIDTH)[3] for line in lines]
    total_height = sum(line_heights) + config.THUMBNAIL_LINE_SPACING * (len(lines) - 1)
    y = height - safe_margin - total_height
    for line, line_height in zip(lines, line_heights):
        bbox = draw.textbbox((0, 0), line, font=font, stroke_width=config.THUMBNAIL_STROKE_WIDTH)
        line_width = bbox[2] - bbox[0]
        x = x0 if text_side == "left" else x1 - line_width
        draw.text(
            (x + config.THUMBNAIL_SHADOW_OFFSET[0], y + config.THUMBNAIL_SHADOW_OFFSET[1]),
            line,
            font=font,
            fill=shadow_color,
            stroke_width=config.THUMBNAIL_STROKE_WIDTH,
            stroke_fill=shadow_color,
        )
        draw.text(
            (x, y),
            line,
            font=font,
            fill=text_color,
            stroke_width=config.THUMBNAIL_STROKE_WIDTH,
            stroke_fill=stroke_color,
        )
        y += line_height + config.THUMBNAIL_LINE_SPACING
    return image


def render_thumbnail_overlay(video_dir: Path, prompt_entry: dict[str, Any]) -> Path:
    concept_id = int(prompt_entry["concept_id"])
    background_path = _background_path(video_dir, concept_id)
    if not background_path.exists():
        raise FileNotFoundError(f"Missing thumbnail background: {background_path}")
    with Image.open(background_path) as background:
        composited = _draw_text_block(background.copy(), prompt_entry["thumbnail_text"], prompt_entry["text_side"])
        output_path = _thumbnail_path(video_dir, concept_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        composited.save(output_path, format="JPEG", quality=config.THUMBNAIL_JPEG_QUALITY, optimize=True)
        return output_path


def build_contact_sheet(video_dir: Path, prompt_entries: list[dict[str, Any]]) -> Path:
    thumbs: list[tuple[str, Image.Image]] = []
    for idx, entry in enumerate(prompt_entries):
        thumb_path = _thumbnail_path(video_dir, int(entry["concept_id"]))
        if _image_is_readable(thumb_path):
            thumbs.append((string.ascii_uppercase[idx], Image.open(thumb_path).convert("RGB")))
    if not thumbs:
        raise ThumbnailGenerationError("No finished thumbnails available for contact sheet")
    tile_width = 320
    tile_height = 180
    label_height = 32
    gutter = 16
    cols = min(len(thumbs), 3)
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new(
        "RGB",
        (cols * tile_width + (cols + 1) * gutter, rows * (tile_height + label_height) + (rows + 1) * gutter),
        tuple(config.THUMBNAIL_CONTACT_SHEET_BG),
    )
    draw = ImageDraw.Draw(sheet)
    font = _load_font(24)
    for idx, (label, image) in enumerate(thumbs):
        row = idx // cols
        col = idx % cols
        x = gutter + col * (tile_width + gutter)
        y = gutter + row * (tile_height + label_height + gutter)
        draw.text((x, y), label, font=font, fill=tuple(config.THUMBNAIL_CONTACT_SHEET_LABEL_COLOR))
        resized = image.resize((tile_width, tile_height), Image.LANCZOS)
        sheet.paste(resized, (x, y + label_height))
    out_path = _thumbnail_contact_sheet_path(video_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, format="JPEG", quality=95, optimize=True)
    return out_path


def _build_backend():
    if config.IMAGE_BACKEND == "vast_instance":
        from scripts.generate_images import _build_vast_backend

        return _build_vast_backend(n_images=config.CREATIVE_PACKAGE_DEFAULT_CONCEPT_COUNT)
    from steps.generate_images import _build_runpod_backend

    return _build_runpod_backend()


def _copy_candidate_to_background(candidate_path: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(candidate_path) as img:
        img.convert("RGB").save(destination, format="PNG", optimize=True)


def _cleanup_candidate_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _expected_thumbnail_count(package: dict[str, Any]) -> int:
    return len(package.get("thumbnail_concepts", []))


def _gpu_pending_entries(
    video_dir: Path,
    prompt_entries: list[dict[str, Any]],
    regenerate_set: set[int],
) -> list[dict[str, Any]]:
    pending: list[dict[str, Any]] = []
    for entry in prompt_entries:
        concept_id = int(entry["concept_id"])
        if regenerate_set:
            if concept_id in regenerate_set:
                pending.append(entry)
            continue
        if not _image_is_readable(_background_path(video_dir, concept_id)):
            pending.append(entry)
    return pending


def _overlay_pending_entries(
    video_dir: Path,
    prompt_entries: list[dict[str, Any]],
    regenerate_set: set[int],
) -> list[dict[str, Any]]:
    pending: list[dict[str, Any]] = []
    for entry in prompt_entries:
        concept_id = int(entry["concept_id"])
        if regenerate_set:
            if concept_id not in regenerate_set:
                continue
        if _image_is_readable(_background_path(video_dir, concept_id)) and not _image_is_readable(_thumbnail_path(video_dir, concept_id)):
            pending.append(entry)
    return pending


def _existing_thumbnail_entries(video_dir: Path, prompt_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [entry for entry in prompt_entries if _image_is_readable(_thumbnail_path(video_dir, int(entry["concept_id"])))]


def _finalize_thumbnail_outputs(video_dir: Path, prompt_entries: list[dict[str, Any]], generation_log: dict[str, Any]) -> dict[str, Any]:
    failed_ids: list[int] = []
    for entry in _overlay_pending_entries(video_dir, prompt_entries, set()):
        concept_id = int(entry["concept_id"])
        try:
            output_path = render_thumbnail_overlay(video_dir, entry)
            if not _image_is_readable(output_path):
                raise ThumbnailGenerationError(f"Unreadable thumbnail JPG for concept {concept_id}")
            generation_log[str(concept_id)] = {
                "status": "completed",
                "background_path": str(_background_path(video_dir, concept_id)),
                "thumbnail_path": str(output_path),
                "errors": [],
            }
        except Exception as exc:
            failed_ids.append(concept_id)
            generation_log[str(concept_id)] = {
                "status": "failed",
                "error": str(exc),
                "background_path": str(_background_path(video_dir, concept_id)),
            }
            logger.error("Thumbnail overlay {} failed: {}", concept_id, exc)

    existing_entries = _existing_thumbnail_entries(video_dir, prompt_entries)
    contact_sheet_path = None
    contact_sheet_ok = False
    if existing_entries:
        try:
            contact_sheet = build_contact_sheet(video_dir, existing_entries)
            contact_sheet_ok = _image_is_readable(contact_sheet)
            contact_sheet_path = str(contact_sheet) if contact_sheet_ok else None
        except Exception as exc:
            logger.error("Thumbnail contact sheet failed: {}", exc)
            contact_sheet_ok = False
    diagnostics = {
        "thumbnail_prompt_count": len(prompt_entries),
        "thumbnail_generated_count": len(existing_entries),
        "thumbnail_failed_ids": sorted(set(failed_ids)),
        "validation_passed": not failed_ids and contact_sheet_ok,
        "warnings": [],
        "contact_sheet_path": contact_sheet_path,
        "contact_sheet_ok": contact_sheet_ok,
    }
    _atomic_write_json(_thumbnail_log_path(video_dir), generation_log)
    _atomic_write_json(_thumbnail_diagnostics_path(video_dir), diagnostics)
    return diagnostics


def generate_thumbnail_backgrounds(
    video_id: str,
    *,
    regenerate: list[int] | None = None,
    allow_stale_package: bool = False,
    backend_override=None,
    teardown_override=None,
    manage_backend: bool = True,
    lifecycle=None,
) -> dict[str, Any]:
    video_dir = Path(config.OUTPUT_DIR) / video_id
    load_validated_package(video_dir, allow_stale_package=allow_stale_package)
    prompt_entries = _load_json(_thumbnail_prompts_path(video_dir), [])
    if not isinstance(prompt_entries, list) or not prompt_entries:
        raise ThumbnailGenerationError("thumbnail_prompts.json must contain a non-empty list")
    regenerate_set = {int(value) for value in regenerate or []}
    log_path = _thumbnail_log_path(video_dir)
    generation_log = _load_json(log_path, {})
    pending_entries = _gpu_pending_entries(video_dir, prompt_entries, regenerate_set)
    if not pending_entries:
        _atomic_write_json(log_path, generation_log)
        return {"background_generated_count": 0, "thumbnail_failed_ids": []}
    backend = backend_override
    teardown = teardown_override
    owns_backend = backend is None and manage_backend
    if backend is None:
        backend, teardown = _build_backend()
    failed_ids: list[int] = []
    generated = 0
    try:
        for entry in pending_entries:
            concept_id = int(entry["concept_id"])
            bg_path = _background_path(video_dir, concept_id)
            temp_scene_id = str(9000 + concept_id)
            candidate_dir = Path(config.OUTPUT_DIR) / video_id / "images" / f"scene_{int(temp_scene_id):03d}"
            request = SceneRequest(
                video_id=video_id,
                scene_id=temp_scene_id,
                prompt=entry["image_prompt"],
                clip_prompt=entry.get(
                    "clip_prompt",
                    f"YouTube thumbnail, {entry['type'].replace('_', ' ')}, {entry['thumbnail_text']}",
                ),
                negative_prompt=entry.get("negative_prompt", ""),
                width=config.IMAGE_WIDTH,
                height=config.IMAGE_HEIGHT,
                steps=config.IMAGE_STEPS,
                guidance_scale=config.IMAGE_GUIDANCE_SCALE,
                candidate_seeds=[config.THUMBNAIL_CANDIDATE_SEED],
                output_format=config.IMAGE_OUTPUT_FORMAT,
                quality=config.IMAGE_QUALITY,
                output_mode="base64",
            )
            try:
                if lifecycle is not None:
                    lifecycle.thumbnail_request_count += 1
                result = backend.generate(request)
                if not result.candidates:
                    raise ThumbnailGenerationError(f"No thumbnail candidates returned for concept {concept_id}")
                candidate_path = Path(result.candidates[0].local_path or "")
                if not candidate_path.exists():
                    raise ThumbnailGenerationError(f"Missing generated candidate file for concept {concept_id}")
                _copy_candidate_to_background(candidate_path, bg_path)
                if not _image_is_readable(bg_path):
                    raise ThumbnailGenerationError(f"Unreadable thumbnail background for concept {concept_id}")
                generation_log[str(concept_id)] = {
                    "status": "background_ready",
                    "background_path": str(bg_path),
                    "thumbnail_path": str(_thumbnail_path(video_dir, concept_id)),
                    "errors": result.errors,
                }
                generated += 1
            except Exception as exc:
                failed_ids.append(concept_id)
                generation_log[str(concept_id)] = {
                    "status": "failed",
                    "error": str(exc),
                }
                logger.error("Thumbnail concept {} failed: {}", concept_id, exc)
            finally:
                _cleanup_candidate_dir(candidate_dir)
        _atomic_write_json(log_path, generation_log)
    finally:
        if owns_backend and teardown:
            teardown()
    return {"background_generated_count": generated, "thumbnail_failed_ids": sorted(set(failed_ids))}


def generate_thumbnail_assets(
    video_id: str,
    *,
    regenerate: list[int] | None = None,
    allow_stale_package: bool = False,
    backend_override=None,
    teardown_override=None,
    manage_backend: bool = True,
    lifecycle=None,
) -> dict[str, Any]:
    video_dir = Path(config.OUTPUT_DIR) / video_id
    package = load_validated_package(video_dir, allow_stale_package=allow_stale_package)
    prompts_path = _thumbnail_prompts_path(video_dir)
    if not prompts_path.exists():
        raise FileNotFoundError(f"Missing thumbnail_prompts.json: {prompts_path}")
    prompt_entries = _load_json(prompts_path, [])
    if not isinstance(prompt_entries, list) or not prompt_entries:
        raise ThumbnailGenerationError("thumbnail_prompts.json must contain a non-empty list")
    regenerate_set = {int(value) for value in regenerate or []}
    log_path = _thumbnail_log_path(video_dir)
    generation_log = _load_json(log_path, {})
    if _gpu_pending_entries(video_dir, prompt_entries, regenerate_set):
        result = generate_thumbnail_backgrounds(
            video_id,
            regenerate=regenerate,
            allow_stale_package=allow_stale_package,
            backend_override=backend_override,
            teardown_override=teardown_override,
            manage_backend=manage_backend,
            lifecycle=lifecycle,
        )
        generation_log = _load_json(log_path, {})
        if result["thumbnail_failed_ids"]:
            diagnostics = _finalize_thumbnail_outputs(video_dir, prompt_entries, generation_log)
            diagnostics["thumbnail_failed_ids"] = sorted(set(diagnostics["thumbnail_failed_ids"]) | set(result["thumbnail_failed_ids"]))
            diagnostics["validation_passed"] = False
            _atomic_write_json(_thumbnail_diagnostics_path(video_dir), diagnostics)
            return diagnostics
    diagnostics = _finalize_thumbnail_outputs(video_dir, prompt_entries, generation_log)
    expected_count = _expected_thumbnail_count(package)
    diagnostics["expected_thumbnail_count"] = expected_count
    diagnostics["validation_passed"] = (
        diagnostics["validation_passed"]
        and diagnostics["thumbnail_generated_count"] == expected_count
        and expected_count in {3, 5}
        and diagnostics["contact_sheet_ok"]
    )
    return diagnostics
