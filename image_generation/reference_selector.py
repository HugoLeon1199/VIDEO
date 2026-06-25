"""Select one reference image per scene based on classification."""
from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from image_generation.scene_classifier import SceneClassification

logger = logging.getLogger(__name__)

MAX_CHAIN_DEPTH = 4

STRENGTH_MAP = {
    "pose":                         0.55,
    "angle":                        0.72,
    "master_character_male":        0.65,
    "master_character_female":      0.65,
    "master_character_full_body":   0.65,
    "master_night_fire":            0.68,
    "master_day_wilderness":        0.68,
    "master_cosmic_sky":            0.65,
    "master_scientific_diagram":    0.65,
    "master_timeline_cycle":        0.65,
    "master_object_macro":          0.62,
}


@dataclass
class ReferenceDecision:
    mode: str                       # "text_to_image" | "img2img"
    reference_path: Optional[str]   # local file path or None
    reference_source: str           # "previous_scene" | "master_seed" | "none"
    reference_key: Optional[str]    # key used for lookup
    strength: float                 # denoising strength (0.0 for t2i)
    chain_depth: int
    reset_reason: Optional[str]     # why chain was reset, if any


def select_reference(
    classification: SceneClassification,
    previous_scene_image: Optional[str],
    master_seed_dir: Optional[str],
    master_seed_manifest: Optional[dict],
    chain_depth: int,
    previous_qa_passed: bool = True,
) -> ReferenceDecision:
    """Select exactly one reference image (or none) for this scene.

    Priority rules:
    1. Same-scene pose/angle change AND previous image valid AND chain_depth < MAX -> use previous
    2. Named character -> use matching character master seed
    3. Known environment -> use matching environment master seed
    4. Low confidence -> text-to-image
    5. No valid reference -> text-to-image
    """
    reset_reason = None

    # Force reset conditions
    if chain_depth >= MAX_CHAIN_DEPTH:
        reset_reason = f"chain_depth={chain_depth} >= MAX_CHAIN_DEPTH={MAX_CHAIN_DEPTH}"
        logger.info("Reference reset: %s", reset_reason)

    if not previous_qa_passed and classification.use_previous_image:
        reset_reason = "previous_qa_failed — never reuse bad image"
        logger.warning("Reference reset: %s", reset_reason)

    if classification.confidence < 0.55:
        reset_reason = f"low_confidence={classification.confidence:.2f}"
        logger.info("Reference reset: %s", reset_reason)

    # Rule 1: same-scene continuity (pose or angle)
    if (
        classification.use_previous_image
        and previous_scene_image
        and Path(previous_scene_image).exists()
        and reset_reason is None
    ):
        strength = STRENGTH_MAP.get(classification.change_type, 0.65)
        logger.info(
            "Reference: previous_scene=%s chain_depth=%d strength=%.2f",
            previous_scene_image, chain_depth, strength,
        )
        return ReferenceDecision(
            mode="img2img",
            reference_path=previous_scene_image,
            reference_source="previous_scene",
            reference_key=f"scene_{chain_depth}",
            strength=strength,
            chain_depth=chain_depth,
            reset_reason=reset_reason,
        )

    # Rules 2-5: master seed lookup
    ref_key = classification.reference_key

    if ref_key and master_seed_manifest and master_seed_dir:
        seed_file = master_seed_manifest.get("seeds", {}).get(ref_key)
        if seed_file:
            seed_path = Path(master_seed_dir) / seed_file
            if seed_path.exists():
                strength_key = f"master_{ref_key}"
                strength = STRENGTH_MAP.get(strength_key, 0.65)
                # Clamp to safe range
                strength = max(0.50, min(0.75, strength))
                logger.info(
                    "Reference: master_seed=%s key=%s strength=%.2f",
                    seed_path, ref_key, strength,
                )
                return ReferenceDecision(
                    mode="img2img",
                    reference_path=str(seed_path),
                    reference_source="master_seed",
                    reference_key=ref_key,
                    strength=strength,
                    chain_depth=0,  # reset chain depth on master seed
                    reset_reason=reset_reason,
                )
            else:
                logger.warning("Master seed not found: %s", seed_path)

    # Text-to-image fallback
    logger.info(
        "Reference: none (text_to_image) reason=%s",
        reset_reason or "no_valid_reference",
    )
    return ReferenceDecision(
        mode="text_to_image",
        reference_path=None,
        reference_source="none",
        reference_key=None,
        strength=0.0,
        chain_depth=0,
        reset_reason=reset_reason or "no_valid_reference",
    )


def load_master_manifest(master_seed_dir: str) -> dict:
    """Load and validate master_style_seeds/manifest.json."""
    import json
    manifest_path = Path(master_seed_dir) / "manifest.json"
    if not manifest_path.exists():
        logger.warning("No master seed manifest found at %s", manifest_path)
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Validate all referenced files exist
    seeds = manifest.get("seeds", {})
    for key, fname in seeds.items():
        p = Path(master_seed_dir) / fname
        if not p.exists():
            logger.warning("Master seed missing: %s -> %s", key, p)
    return manifest
