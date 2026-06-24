"""
Vision QA module — runs Gemini multimodal checks on generated images.

Each candidate is evaluated for anatomy correctness, prompt adherence,
composition, and style consistency. Results are cached by sha256 + qa_prompt_version.

Public API
----------
    from image_generation.vision_qa import VisionQA, QAResult
    qa = VisionQA()
    result = qa.evaluate(image_path, prompt, scene_meta)
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Bump this version when the QA prompt changes — invalidates old cache entries.
QA_PROMPT_VERSION = "anatomy-qa-v1"

# Scoring weights (must sum to 100)
_WEIGHT_ANATOMY = 40
_WEIGHT_PROMPT = 25
_WEIGHT_COMPOSITION = 20
_WEIGHT_STYLE = 15

# Resize long edge to this before sending to Vision API (saves cost, no quality loss for QA)
_MAX_QA_EDGE = 1024
_QA_JPEG_QUALITY = 82


# ---------------------------------------------------------------------------
# Hard-fail conditions: any True → candidate is automatically rejected
# ---------------------------------------------------------------------------

_HARD_FAIL_ANATOMY_FIELDS = (
    "extra_limbs",
    "fused_bodies",
    "duplicate_faces",
    "malformed_hands",
)

_HARD_FAIL_COMPOSITION_FIELDS = (
    "important_subject_cropped",
)

_HARD_FAIL_STYLE_FIELDS = (
    "contains_text_or_watermark",
    "looks_childish",
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AnatomyResult:
    extra_limbs: bool = False
    missing_limbs: bool = False
    fused_bodies: bool = False
    malformed_hands: bool = False
    duplicate_faces: bool = False

    def score(self) -> int:
        """Returns anatomy sub-score 0-40."""
        deductions = 0
        if self.extra_limbs:
            deductions += 20
        if self.fused_bodies:
            deductions += 15
        if self.duplicate_faces:
            deductions += 10
        if self.malformed_hands:
            deductions += 5
        if self.missing_limbs:
            deductions += 5
        return max(0, _WEIGHT_ANATOMY - deductions)

    def hard_fail(self) -> bool:
        return any(getattr(self, f) for f in _HARD_FAIL_ANATOMY_FIELDS)


@dataclass
class CompositionResult:
    matches_prompt: bool = True
    main_subject_clear: bool = True
    bodies_separated: bool = True
    important_subject_cropped: bool = False

    def score(self) -> int:
        deductions = 0
        if not self.matches_prompt:
            deductions += 10
        if not self.main_subject_clear:
            deductions += 5
        if not self.bodies_separated:
            deductions += 5
        if self.important_subject_cropped:
            deductions += 15
        return max(0, _WEIGHT_COMPOSITION - deductions)

    def hard_fail(self) -> bool:
        return self.important_subject_cropped


@dataclass
class StyleResult:
    matches_2d_painted_documentary: bool = True
    looks_childish: bool = False
    contains_text_or_watermark: bool = False

    def score(self) -> int:
        deductions = 0
        if not self.matches_2d_painted_documentary:
            deductions += 10
        if self.looks_childish:
            deductions += 10
        if self.contains_text_or_watermark:
            deductions += 15
        return max(0, _WEIGHT_STYLE - deductions)

    def hard_fail(self) -> bool:
        return any(getattr(self, f) for f in _HARD_FAIL_STYLE_FIELDS)


@dataclass
class QAResult:
    scene_id: str
    candidate_index: int
    seed: int
    sha256: str
    image_path: str

    passed: bool = False
    score: int = 0
    people_detected: int = 0

    anatomy: AnatomyResult = field(default_factory=AnatomyResult)
    composition: CompositionResult = field(default_factory=CompositionResult)
    style: StyleResult = field(default_factory=StyleResult)

    issues: list[str] = field(default_factory=list)
    regeneration_instruction: str = ""

    qa_error: Optional[str] = None
    cached: bool = False
    qa_prompt_version: str = QA_PROMPT_VERSION

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "candidate_index": self.candidate_index,
            "seed": self.seed,
            "sha256": self.sha256,
            "image_path": self.image_path,
            "passed": self.passed,
            "score": self.score,
            "people_detected": self.people_detected,
            "anatomy": {
                "extra_limbs": self.anatomy.extra_limbs,
                "missing_limbs": self.anatomy.missing_limbs,
                "fused_bodies": self.anatomy.fused_bodies,
                "malformed_hands": self.anatomy.malformed_hands,
                "duplicate_faces": self.anatomy.duplicate_faces,
            },
            "composition": {
                "matches_prompt": self.composition.matches_prompt,
                "main_subject_clear": self.composition.main_subject_clear,
                "bodies_separated": self.composition.bodies_separated,
                "important_subject_cropped": self.composition.important_subject_cropped,
            },
            "style": {
                "matches_2d_painted_documentary": self.style.matches_2d_painted_documentary,
                "looks_childish": self.style.looks_childish,
                "contains_text_or_watermark": self.style.contains_text_or_watermark,
            },
            "issues": self.issues,
            "regeneration_instruction": self.regeneration_instruction,
            "qa_error": self.qa_error,
            "cached": self.cached,
            "qa_prompt_version": self.qa_prompt_version,
        }


@dataclass
class SceneMeta:
    """Optional metadata to help QA evaluate a scene more accurately."""
    expected_people: int = 1
    shot_type: str = "medium"        # full-body, waist-up, chest-up, wide, extreme-close-up
    important_objects: list[str] = field(default_factory=list)
    track: str = "vi"                # "vi" or "en"


# ---------------------------------------------------------------------------
# QA cache
# ---------------------------------------------------------------------------

class _QACache:
    """Disk cache keyed by sha256 + qa_prompt_version."""

    def __init__(self, cache_dir: Path):
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _key(self, sha256: str, version: str) -> Path:
        return self._dir / f"{sha256[:16]}_{version}.json"

    def get(self, sha256: str, version: str) -> Optional[dict]:
        path = self._key(sha256, version)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def put(self, sha256: str, version: str, data: dict) -> None:
        path = self._key(sha256, version)
        try:
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError as e:
            logger.warning("QA cache write failed: %s", e)


# ---------------------------------------------------------------------------
# Image preparation
# ---------------------------------------------------------------------------

def _prepare_image_bytes(image_path: Path) -> bytes:
    """Resize to ≤1024px long edge, convert to JPEG for API call."""
    from PIL import Image as PilImage
    with PilImage.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        scale = min(1.0, _MAX_QA_EDGE / max(w, h))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), PilImage.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=_QA_JPEG_QUALITY)
        return buf.getvalue()


# ---------------------------------------------------------------------------
# QA prompt builder
# ---------------------------------------------------------------------------

_QA_SYSTEM_PROMPT = """You are a strict image quality evaluator for a history documentary video.
You must return ONLY valid JSON — no markdown, no code fences, no explanation text.
Evaluate the image against the prompt and criteria provided."""

_QA_USER_TEMPLATE = """Evaluate this image for use in a documentary video about prehistoric humans.

Original prompt: {prompt}

Expected number of clearly visible foreground people: {expected_people}
Shot type: {shot_type}
Important objects: {important_objects}
Visual style expected: {style_expectation}

Score the image on these dimensions:
- Anatomy (40 pts): correct number of arms, legs, fingers, no extra or fused limbs
- Prompt adherence (25 pts): does the image match the intended scene?
- Composition (20 pts): clear subjects, bodies not overlapping, nothing important cropped
- Style consistency (15 pts): matches the expected visual style, no text/watermarks

Return ONLY this JSON structure (no markdown, no backticks):
{{
  "pass": true,
  "score": 90,
  "people_detected": 1,
  "anatomy": {{
    "extra_limbs": false,
    "missing_limbs": false,
    "fused_bodies": false,
    "malformed_hands": false,
    "duplicate_faces": false
  }},
  "composition": {{
    "matches_prompt": true,
    "main_subject_clear": true,
    "bodies_separated": true,
    "important_subject_cropped": false
  }},
  "style": {{
    "matches_2d_painted_documentary": true,
    "looks_childish": false,
    "contains_text_or_watermark": false
  }},
  "issues": [],
  "regeneration_instruction": ""
}}"""


def _style_expectation(track: str) -> str:
    if track == "vi":
        return (
            "cinematic 2D painted documentary illustration, semi-realistic prehistoric humans, "
            "hand-painted texture, warm golden-amber lighting, serious educational tone, "
            "NOT photorealistic, NOT anime, NOT cartoon, NOT Pixar"
        )
    return (
        "ink sketch illustration on aged parchment paper, hand-drawn, sepia tones, "
        "educational diagram style, no color fill, NOT photorealistic"
    )


# ---------------------------------------------------------------------------
# JSON parser
# ---------------------------------------------------------------------------

def _parse_qa_json(raw: str) -> dict:
    """Strip markdown fences if present, parse JSON."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first and last fence lines
        inner = []
        in_block = False
        for line in lines:
            if line.startswith("```") and not in_block:
                in_block = True
                continue
            if line.startswith("```") and in_block:
                break
            if in_block:
                inner.append(line)
        text = "\n".join(inner).strip()
    return json.loads(text)


def _validate_qa_dict(d: dict) -> dict:
    """Ensure required keys exist with correct types; fill defaults if missing."""
    defaults = {
        "pass": False,
        "score": 0,
        "people_detected": 0,
        "anatomy": {},
        "composition": {},
        "style": {},
        "issues": [],
        "regeneration_instruction": "",
    }
    for k, v in defaults.items():
        if k not in d:
            d[k] = v

    anat_defaults = {
        "extra_limbs": False, "missing_limbs": False,
        "fused_bodies": False, "malformed_hands": False, "duplicate_faces": False,
    }
    comp_defaults = {
        "matches_prompt": True, "main_subject_clear": True,
        "bodies_separated": True, "important_subject_cropped": False,
    }
    style_defaults = {
        "matches_2d_painted_documentary": True,
        "looks_childish": False, "contains_text_or_watermark": False,
    }
    for k, v in anat_defaults.items():
        d["anatomy"].setdefault(k, v)
    for k, v in comp_defaults.items():
        d["composition"].setdefault(k, v)
    for k, v in style_defaults.items():
        d["style"].setdefault(k, v)

    # Ensure score is int (handle "90.5" float strings from model)
    try:
        d["score"] = int(float(d.get("score", 0)))
    except (TypeError, ValueError):
        d["score"] = 0
    d["pass"] = bool(d.get("pass", False))
    return d


# ---------------------------------------------------------------------------
# Scoring and pass/fail logic
# ---------------------------------------------------------------------------

def _compute_score(anat: AnatomyResult, comp: CompositionResult, sty: StyleResult) -> int:
    return anat.score() + comp.score() + sty.score() + _WEIGHT_PROMPT


def _build_result_from_dict(
    d: dict,
    scene_id: str,
    candidate_index: int,
    seed: int,
    sha256: str,
    image_path: str,
    min_score: int,
    anatomy_min: int = 34,
) -> QAResult:
    anat = AnatomyResult(**{k: bool(d["anatomy"][k]) for k in AnatomyResult.__dataclass_fields__})
    comp = CompositionResult(**{k: bool(d["composition"][k]) for k in CompositionResult.__dataclass_fields__})
    sty = StyleResult(**{k: bool(d["style"][k]) for k in StyleResult.__dataclass_fields__})

    score = d["score"]  # use model's own score as base, but override if hard-fail
    hard_fail = anat.hard_fail() or comp.hard_fail() or sty.hard_fail()

    if hard_fail:
        score = min(score, min_score - 1)

    anat_score = anat.score()
    passed = (
        not hard_fail
        and score >= min_score
        and anat_score >= anatomy_min
    )

    return QAResult(
        scene_id=scene_id,
        candidate_index=candidate_index,
        seed=seed,
        sha256=sha256,
        image_path=image_path,
        passed=passed,
        score=score,
        people_detected=d.get("people_detected", 0),
        anatomy=anat,
        composition=comp,
        style=sty,
        issues=list(d.get("issues", [])),
        regeneration_instruction=d.get("regeneration_instruction", ""),
    )


# ---------------------------------------------------------------------------
# Gemini client wrapper
# ---------------------------------------------------------------------------

def _call_gemini(image_bytes: bytes, prompt_text: str, api_key: str, model: str) -> str:
    """Call Gemini multimodal API and return raw text response."""
    import httpx

    b64_image = base64.b64encode(image_bytes).decode()
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        f"?key={api_key}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": _QA_SYSTEM_PROMPT}]},
        "contents": [{
            "parts": [
                {"inline_data": {"mime_type": "image/jpeg", "data": b64_image}},
                {"text": prompt_text},
            ]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
        },
    }
    resp = httpx.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


# ---------------------------------------------------------------------------
# Main VisionQA class
# ---------------------------------------------------------------------------

class VisionQA:
    """
    Evaluate a generated image for anatomy, composition, and style.

    Parameters
    ----------
    api_key : str, optional
        Gemini API key. Defaults to GEMINI_API_KEY env var.
    model : str, optional
        Gemini model to use.
    min_score : int
        Minimum score for pass (default 80).
    anatomy_min : int
        Minimum anatomy sub-score for pass (default 34 / 40).
    cache_dir : Path, optional
        Directory for QA result cache.
    max_retries : int
        Number of API retries on transient errors.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash",
        min_score: int = 80,
        anatomy_min: int = 34,
        cache_dir: Optional[Path] = None,
        max_retries: int = 3,
    ):
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._model = model
        self._min_score = min_score
        self._anatomy_min = anatomy_min
        self._cache = _QACache(cache_dir or Path("output") / ".qa_cache")
        self._max_retries = max_retries

    def evaluate(
        self,
        image_path: Path,
        prompt: str,
        scene_id: str,
        candidate_index: int,
        seed: int,
        sha256: str,
        scene_meta: Optional[SceneMeta] = None,
    ) -> QAResult:
        """
        Evaluate one candidate image. Returns QAResult.
        Never raises — on API error sets qa_error and marks as not passed.
        """
        meta = scene_meta or SceneMeta()

        # Check cache first
        cached = self._cache.get(sha256, QA_PROMPT_VERSION)
        if cached:
            logger.debug("QA cache hit for scene %s candidate %d", scene_id, candidate_index)
            result = _build_result_from_dict(
                cached, scene_id, candidate_index, seed, sha256, str(image_path),
                self._min_score, self._anatomy_min,
            )
            result.cached = True
            return result

        if not self._api_key:
            logger.warning("GEMINI_API_KEY not set — skipping QA for scene %s", scene_id)
            return QAResult(
                scene_id=scene_id, candidate_index=candidate_index, seed=seed,
                sha256=sha256, image_path=str(image_path),
                qa_error="GEMINI_API_KEY not configured",
            )

        # Prepare image
        try:
            img_bytes = _prepare_image_bytes(image_path)
        except Exception as e:
            return QAResult(
                scene_id=scene_id, candidate_index=candidate_index, seed=seed,
                sha256=sha256, image_path=str(image_path),
                qa_error=f"Image preparation failed: {e}",
            )

        prompt_text = _QA_USER_TEMPLATE.format(
            prompt=prompt,
            expected_people=meta.expected_people,
            shot_type=meta.shot_type,
            important_objects=", ".join(meta.important_objects) or "none specified",
            style_expectation=_style_expectation(meta.track),
        )

        # Call API with retry
        raw = None
        for attempt in range(1, self._max_retries + 1):
            try:
                raw = _call_gemini(img_bytes, prompt_text, self._api_key, self._model)
                break
            except Exception as e:
                logger.warning(
                    "QA API error scene %s attempt %d/%d: %s",
                    scene_id, attempt, self._max_retries, e,
                )
                if attempt < self._max_retries:
                    time.sleep(2 ** attempt)

        if raw is None:
            return QAResult(
                scene_id=scene_id, candidate_index=candidate_index, seed=seed,
                sha256=sha256, image_path=str(image_path),
                qa_error="Gemini API failed after all retries",
            )

        # Parse and validate
        try:
            d = _parse_qa_json(raw)
            d = _validate_qa_dict(d)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            return QAResult(
                scene_id=scene_id, candidate_index=candidate_index, seed=seed,
                sha256=sha256, image_path=str(image_path),
                qa_error=f"JSON parse failed: {e}. Raw: {raw[:200]}",
            )

        # Cache the raw parsed dict
        self._cache.put(sha256, QA_PROMPT_VERSION, d)

        return _build_result_from_dict(
            d, scene_id, candidate_index, seed, sha256, str(image_path),
            self._min_score, self._anatomy_min,
        )

    def select_best(self, results: list[QAResult]) -> Optional[QAResult]:
        """
        Pick the best candidate from a list of QAResults.

        Priority:
        1. Must pass (no hard-fail, score >= min, anatomy >= min)
        2. Highest score
        3. Fewest clearly visible people (simpler composition)
        4. Highest prompt adherence (score as proxy)
        5. Smallest seed (deterministic tiebreak)
        """
        passing = [r for r in results if r.passed]
        if not passing:
            return None
        return min(
            passing,
            key=lambda r: (-r.score, r.people_detected, r.seed),
        )

    def build_corrective_prompt(self, original_prompt: str, failed_results: list[QAResult]) -> str:
        """Synthesize issues from failed candidates into a corrective instruction."""
        all_issues: list[str] = []
        for r in failed_results:
            all_issues.extend(r.issues)
            if r.anatomy.extra_limbs:
                all_issues.append("extra limbs detected")
            if r.anatomy.fused_bodies:
                all_issues.append("bodies fused together")
            if r.anatomy.duplicate_faces:
                all_issues.append("duplicate faces")
            if r.composition.important_subject_cropped:
                all_issues.append("subject cropped")
            if r.style.looks_childish:
                all_issues.append("style too childish")

        # Deduplicate
        seen = set()
        unique = []
        for issue in all_issues:
            if issue.lower() not in seen:
                seen.add(issue.lower())
                unique.append(issue)

        correction = (
            "Simplify the composition. Show only one clearly visible foreground character "
            "in a stable natural pose. Keep both arms and both legs fully coherent and "
            "separated from the torso. Move all additional people into the distant background "
            "as small silhouettes. Do not overlap bodies or limbs."
        )
        if unique:
            correction += f" Specifically fix: {'; '.join(unique[:5])}."

        return f"{original_prompt}. COMPOSITION FIX: {correction}"
