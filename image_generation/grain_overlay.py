"""Apply film grain overlay to generated images in post-processing."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def apply_grain_overlay(
    image_path: str,
    overlay_path: str,
    opacity: float = 0.10,
    output_path: Optional[str] = None,
) -> str:
    """Composite grain overlay onto image using Pillow.

    opacity: 0.08 to 0.12 recommended. Clamped to 0.01-0.30.
    output_path: if None, overwrites image_path.
    Returns path to output file.
    """
    from PIL import Image

    opacity = max(0.01, min(0.30, opacity))
    out = output_path or image_path

    base = Image.open(image_path).convert("RGBA")
    grain = Image.open(overlay_path).convert("RGBA")

    # Resize grain to match base if needed
    if grain.size != base.size:
        grain = grain.resize(base.size, Image.LANCZOS)

    # Blend: grain at `opacity`, base at 1-opacity
    blended = Image.blend(base, grain, alpha=opacity)
    blended.convert("RGB").save(out)

    logger.info("Grain overlay applied: opacity=%.2f -> %s", opacity, out)
    return out
