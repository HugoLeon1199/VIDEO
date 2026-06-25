"""
Patch image_prompts.json: inject Style B (Flat 2D animated) prefix into all prompts.
Style chốt cho VI track — dùng cho mọi video VI.

Usage: edit VIDEO_ID bên dưới rồi chạy:
    python scripts/patch_vi_prompts_style.py
"""
import json
import re
from pathlib import Path

VIDEO_ID = "to-tien-ban-lam-gi-ca-ngay-vi"

# ── Style B: Flat 2D Animated (CHỐT 2026-06-25) ──────────────────────────────
PREFIX_NEW = (
    "2D animated illustration, flat design, "
    "every character including children wearing full hide robes and draped cloaks covering body, "
    "bold clean outlines, warm color palette, simple shapes, "
    "Disney nature documentary style, earthy browns and oranges, 16:9, "
)

ANATOMY_SUFFIX = ", correct anatomy, clean poses"

NEGATIVE_NEW = (
    "nudity, bare chest, bare breasts, bare back, topless, shirtless, "
    "exposed torso, exposed upper body, visible chest skin, cleavage, "
    "loincloth, grass skirt, minimal clothing, half-naked, "
    "extra limbs, fused limbs, deformed anatomy, malformed body, "
    "text, watermark, logo, photorealistic, 3D render, blurry, overexposed"
)
# ─────────────────────────────────────────────────────────────────────────────

# Match any existing prefix (anything before the first scene-specific content)
PREFIX_OLD_PATTERN = re.compile(
    r"^(?:cinematic documentary photograph|2D animated illustration|watercolor illustration)"
    r".*?(?:no text, no watermark, |16:9, )",
    re.DOTALL,
)

# Match anatomy suffix at end
ANATOMY_OLD_PATTERN = re.compile(
    r",?\s*correct(?:ly)? (?:human )?anatomy.*$",
    re.DOTALL,
)

path = Path(f"output/{VIDEO_ID}/image_prompts.json")
scenes = json.loads(path.read_text(encoding="utf-8"))

changed = 0
for s in scenes:
    p = s["prompt"]

    # Strip old prefix
    p_new = PREFIX_OLD_PATTERN.sub(PREFIX_NEW, p)

    # Strip old anatomy suffix then add new one
    p_new = ANATOMY_OLD_PATTERN.sub("", p_new).rstrip(", ")
    p_new = p_new + ANATOMY_SUFFIX

    if p_new != p:
        changed += 1
    s["prompt"] = p_new
    s["negative_prompt"] = NEGATIVE_NEW

path.write_text(json.dumps(scenes, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Done: {len(scenes)} scenes — {changed} prompts updated")

print("\n--- Scene 1 prompt ---")
print(scenes[0]["prompt"])
print("\n--- Scene 1 negative ---")
print(scenes[0]["negative_prompt"])
