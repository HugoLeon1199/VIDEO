"""Rewrite image_prompts.json for VI track — replace photorealistic suffix with
cinematic 2D painted documentary style matching image_prompt_vi.txt and master seeds.

Usage:
    python scripts/rewrite_vi_prompts.py --video-id to-tien-ban-lam-gi-ca-ngay-vi
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PAINTED_PREFIX = (
    "cinematic 2D painted documentary illustration, semi-realistic prehistoric humans, "
    "painted in the style of a serious historical documentary, "
    "characters wearing thick full-coverage animal fur and hide clothing, "
    "fur pelts draped over both shoulders covering entire chest and torso, "
    "no exposed chest no exposed breasts no bare torso on any character, "
    "warm natural earthy lighting, detailed rocky prehistoric landscape, "
    "serious mature tone, cinematic 16:9 composition, no text, no watermark, "
)

ANATOMY_CLAUSE = (
    ", anatomically coherent human figures, exactly two arms and two legs per visible person, "
    "natural human proportions, clean separated silhouettes, clearly readable limbs, natural hands"
)

NEGATIVE = (
    "nudity, bare chest, bare breasts, topless, shirtless, exposed torso, exposed skin, "
    "cleavage, bikini top, revealing clothes, loincloth only, skimpy outfit, "
    "large breasts, oversized chest, sexualized body, sexual content, suggestive pose, "
    "Pixar, Disney, 3D cartoon, anime, chibi, cute style, "
    "extra limbs, extra arms, extra legs, fused limbs, malformed hands, deformed anatomy, "
    "duplicate people, merged faces, text, watermark, logo, "
    "photorealistic, photograph, hyperrealistic, 3D render, CGI, "
    "blurry, foggy, haze, overexposed, washed out"
)

# Patterns that identify the old photorealistic suffix to strip
_PHOTO_SUFFIX_PATTERN = re.compile(
    r",?\s*cinematic documentary photograph.*$",
    re.IGNORECASE | re.DOTALL,
)

# Female keywords — triggers clothing guard injection
_FEMALE_RE = re.compile(
    r"\b(woman|women|female|mother|girl|grandmother|elder woman|huntress|gatherer woman)\b",
    re.IGNORECASE,
)
FEMALE_CLOTHING = (
    "wearing a sewn animal-hide top fully covering her chest and shoulders, "
    "long wraparound hide skirt reaching her knees, fully clothed, "
)


def _extract_core(prompt: str) -> str:
    """Strip old photorealistic suffix and return the scene description core."""
    stripped = _PHOTO_SUFFIX_PATTERN.sub("", prompt).strip().rstrip(",").strip()
    return stripped


def _rewrite_prompt(core: str) -> str:
    """Build new painted-style prompt from scene core description."""
    has_female = bool(_FEMALE_RE.search(core))
    if has_female:
        prefix = PAINTED_PREFIX + FEMALE_CLOTHING
    else:
        prefix = PAINTED_PREFIX
    return prefix + core + ANATOMY_CLAUSE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    prompts_path = Path(args.output_root) / args.video_id / "image_prompts.json"
    if not prompts_path.exists():
        print(f"ERROR: {prompts_path} not found")
        sys.exit(1)

    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(prompts)} prompts from {prompts_path}")

    changed = 0
    for p in prompts:
        old_prompt = p["prompt"]
        core = _extract_core(old_prompt)
        new_prompt = _rewrite_prompt(core)
        p["prompt"] = new_prompt
        p["negative_prompt"] = NEGATIVE

        if args.dry_run and changed < 3:
            print(f"\n--- Scene {p['index']} ---")
            print(f"OLD: {old_prompt[:120]}...")
            print(f"NEW: {new_prompt[:120]}...")

        changed += 1

    print(f"\nRewritten {changed} prompts")

    if not args.dry_run:
        backup = prompts_path.with_suffix(".json.bak")
        backup.write_text(prompts_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Backup saved to {backup}")
        prompts_path.write_text(json.dumps(prompts, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved to {prompts_path}")
    else:
        print("Dry run — no files written")


if __name__ == "__main__":
    main()
