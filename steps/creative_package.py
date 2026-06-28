from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from loguru import logger

import config


PACKAGE_VERSION = "creative-package-v1"
ALLOWED_LANGUAGES = {"vi", "en"}
ALLOWED_CONCEPT_COUNTS = {3, 5}
ALLOWED_CONCEPT_TYPES = {"human_closeup", "mystery_reveal", "scale_or_danger"}
DEFAULT_CONCEPT_DISTRIBUTIONS = {
    3: {"human_closeup": 1, "mystery_reveal": 1, "scale_or_danger": 1},
    5: {"human_closeup": 2, "mystery_reveal": 2, "scale_or_danger": 1},
}
TITLE_ID_PATTERN = re.compile(r"^title_[1-3]$")
SUBJECT_SIDE_VALUES = {"left", "right"}
TEXT_SIDE_VALUES = {"left", "right"}


class CreativePackageError(ValueError):
    pass


def _publishing_dir(video_dir: Path) -> Path:
    return video_dir / config.PUBLISHING_DIRNAME


def _validated_package_path(video_dir: Path) -> Path:
    return _publishing_dir(video_dir) / "creative_package.validated.json"


def _diagnostics_path(video_dir: Path) -> Path:
    return _publishing_dir(video_dir) / "creative_package_diagnostics.json"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8").lstrip("\ufeff"))


def compute_script_sha256(script_path: Path) -> str:
    return hashlib.sha256(script_path.read_bytes()).hexdigest()


def _looks_vietnamese(text: str) -> bool:
    vi_chars = set("ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")
    lowered = text.lower()
    hits = sum(1 for ch in lowered if ch in vi_chars)
    letters = sum(1 for ch in lowered if ch.isalpha())
    return hits >= 5 or (letters > 0 and hits / letters >= 0.01)


def _detect_script_language(script_text: str) -> str:
    return "vi" if _looks_vietnamese(script_text) else "en"


def _validate_word_limit(text: str, limit: int, field: str, errors: list[str]) -> None:
    if len(text.split()) > limit:
        errors.append(f"{field} exceeds {limit} words")


def _validate_title_options(title_options: list[dict[str, Any]], errors: list[str]) -> list[str]:
    if len(title_options) != 3:
        errors.append(f"title_options must contain exactly 3 entries, got {len(title_options)}")
    ids: list[str] = []
    seen_text: set[str] = set()
    for idx, item in enumerate(title_options, start=1):
        if not isinstance(item, dict):
            errors.append(f"title_options[{idx}] must be an object")
            continue
        title_id = str(item.get("id", "")).strip()
        text = str(item.get("text", "")).strip()
        angle = str(item.get("angle", "")).strip()
        if not TITLE_ID_PATTERN.fullmatch(title_id):
            errors.append(f"title_options[{idx}].id must match title_1..title_3")
        if title_id in ids:
            errors.append(f"Duplicate title id: {title_id}")
        if not text:
            errors.append(f"title_options[{idx}].text is required")
        elif text.lower() in seen_text:
            errors.append(f"Duplicate title text: {text}")
        if not angle:
            errors.append(f"title_options[{idx}].angle is required")
        ids.append(title_id)
        if text:
            seen_text.add(text.lower())
    return ids


def _validate_chapter_plan(chapter_plan: list[dict[str, Any]], errors: list[str], warnings: list[str]) -> None:
    if not isinstance(chapter_plan, list) or not chapter_plan:
        errors.append("chapter_plan must be a non-empty list")
        return
    for idx, item in enumerate(chapter_plan, start=1):
        if not isinstance(item, dict):
            errors.append(f"chapter_plan[{idx}] must be an object")
            continue
        sentence_index = item.get("sentence_index")
        label = str(item.get("label", "")).strip()
        if not isinstance(sentence_index, int) or sentence_index < 1:
            errors.append(f"chapter_plan[{idx}].sentence_index must be an integer >= 1")
        if not label:
            errors.append(f"chapter_plan[{idx}].label is required")
        elif len(label.split()) > 6:
            warnings.append(f"chapter_plan[{idx}].label is longer than 6 words")


def _validate_thumbnail_concepts(
    concepts: list[dict[str, Any]],
    title_ids: list[str],
    errors: list[str],
    warnings: list[str],
) -> None:
    if len(concepts) not in ALLOWED_CONCEPT_COUNTS:
        errors.append(f"thumbnail_concepts count must be one of {sorted(ALLOWED_CONCEPT_COUNTS)}, got {len(concepts)}")
        return
    expected_distribution = DEFAULT_CONCEPT_DISTRIBUTIONS[len(concepts)]
    counts = Counter()
    seen_ids: set[int] = set()
    for idx, concept in enumerate(concepts, start=1):
        if not isinstance(concept, dict):
            errors.append(f"thumbnail_concepts[{idx}] must be an object")
            continue
        concept_id = concept.get("id")
        concept_type = str(concept.get("type", "")).strip()
        thumbnail_text = str(concept.get("thumbnail_text", "")).strip()
        paired = concept.get("paired_title_ids", [])
        subject_side = str(concept.get("subject_side", "")).strip().lower()
        text_side = str(concept.get("text_side", "")).strip().lower()
        if not isinstance(concept_id, int):
            errors.append(f"thumbnail_concepts[{idx}].id must be an integer")
        elif concept_id in seen_ids:
            errors.append(f"Duplicate thumbnail concept id: {concept_id}")
        else:
            seen_ids.add(concept_id)
        if concept_type not in ALLOWED_CONCEPT_TYPES:
            errors.append(f"thumbnail_concepts[{idx}].type must be one of {sorted(ALLOWED_CONCEPT_TYPES)}")
        else:
            counts[concept_type] += 1
        if not str(concept.get("visual_hook", "")).strip():
            errors.append(f"thumbnail_concepts[{idx}].visual_hook is required")
        if not str(concept.get("emotional_goal", "")).strip():
            errors.append(f"thumbnail_concepts[{idx}].emotional_goal is required")
        if not thumbnail_text:
            errors.append(f"thumbnail_concepts[{idx}].thumbnail_text is required")
        else:
            _validate_word_limit(thumbnail_text, 4, f"thumbnail_concepts[{idx}].thumbnail_text", errors)
        if subject_side not in SUBJECT_SIDE_VALUES:
            errors.append(f"thumbnail_concepts[{idx}].subject_side must be left or right")
        if text_side not in TEXT_SIDE_VALUES:
            errors.append(f"thumbnail_concepts[{idx}].text_side must be left or right")
        if subject_side and text_side and subject_side == text_side:
            errors.append(f"thumbnail_concepts[{idx}] subject_side and text_side must differ")
        if not isinstance(paired, list) or not paired:
            errors.append(f"thumbnail_concepts[{idx}].paired_title_ids must be a non-empty list")
        else:
            for title_id in paired:
                if title_id not in title_ids:
                    errors.append(f"thumbnail_concepts[{idx}] references unknown title id: {title_id}")
        must_show = concept.get("must_show", [])
        must_avoid = concept.get("must_avoid", [])
        if not isinstance(must_show, list):
            errors.append(f"thumbnail_concepts[{idx}].must_show must be a list")
        if not isinstance(must_avoid, list):
            errors.append(f"thumbnail_concepts[{idx}].must_avoid must be a list")
        if len(must_show) > 6:
            warnings.append(f"thumbnail_concepts[{idx}].must_show is unusually long")
        if len(must_avoid) > 6:
            warnings.append(f"thumbnail_concepts[{idx}].must_avoid is unusually long")
    if counts != expected_distribution:
        errors.append(
            f"thumbnail_concepts distribution must be {expected_distribution}, got {dict(counts)}"
        )


def load_validated_package(
    video_dir: Path,
    *,
    allow_stale_package: bool = False,
    write_validated_copy: bool = True,
) -> dict[str, Any]:
    script_path = video_dir / "script.txt"
    package_path = video_dir / "creative_package.json"
    if not script_path.exists():
        raise CreativePackageError(f"Missing script.txt: {script_path}")
    if not package_path.exists():
        raise FileNotFoundError(f"Missing creative_package.json: {package_path}")

    script_text = script_path.read_text(encoding="utf-8").strip()
    package = _load_json(package_path)
    publishing_dir = _publishing_dir(video_dir)
    diagnostics: dict[str, Any] = {
        "package_version": package.get("package_version"),
        "script_sha256": compute_script_sha256(script_path),
        "language": package.get("language"),
        "title_count": len(package.get("title_options", []) or []),
        "concept_count": len(package.get("thumbnail_concepts", []) or []),
        "validation_passed": False,
        "warnings": [],
    }
    errors: list[str] = []
    warnings: list[str] = []

    if package.get("package_version") != PACKAGE_VERSION:
        errors.append(f"package_version must be {PACKAGE_VERSION}")

    language = str(package.get("language", "")).strip().lower()
    if language not in ALLOWED_LANGUAGES:
        errors.append(f"language must be one of {sorted(ALLOWED_LANGUAGES)}")
    else:
        detected = _detect_script_language(script_text)
        if detected != language:
            warnings.append(f"language looks like {detected} from script.txt")

    for key in ("core_promise", "target_viewer", "primary_hook", "description_draft"):
        if not str(package.get(key, "")).strip():
            errors.append(f"{key} is required")

    if not isinstance(package.get("search_keywords", []), list):
        errors.append("search_keywords must be a list")

    title_ids = _validate_title_options(package.get("title_options", []) or [], errors)
    _validate_chapter_plan(package.get("chapter_plan", []) or [], errors, warnings)
    _validate_thumbnail_concepts(package.get("thumbnail_concepts", []) or [], title_ids, errors, warnings)

    validated_path = _validated_package_path(video_dir)
    existing_hash = None
    if validated_path.exists():
        try:
            existing_hash = _load_json(validated_path).get("script_sha256")
        except Exception:
            warnings.append("Existing validated package could not be parsed; regenerating validated copy")
    if existing_hash and existing_hash != diagnostics["script_sha256"] and not allow_stale_package:
        errors.append("creative_package.json is stale for the current script.txt")

    diagnostics["warnings"] = warnings
    diagnostics["errors"] = errors
    _atomic_write_json(_diagnostics_path(video_dir), diagnostics)
    if errors:
        raise CreativePackageError("; ".join(errors))

    validated = dict(package)
    validated["script_sha256"] = diagnostics["script_sha256"]
    validated["validated_package_version"] = PACKAGE_VERSION
    validated["detected_language"] = _detect_script_language(script_text)
    validated["validation_warnings"] = warnings
    if write_validated_copy:
        _atomic_write_json(validated_path, validated)
    diagnostics["validation_passed"] = True
    diagnostics["warnings"] = warnings
    diagnostics["script_sha256"] = validated["script_sha256"]
    _atomic_write_json(_diagnostics_path(video_dir), diagnostics)
    logger.info("Creative package validated -> {}", validated_path)
    return validated
