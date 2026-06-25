"""
Full pipeline test: style_dna + scene_classifier + reference_selector + generation.

This script demonstrates the complete system end-to-end:
  1. Each scene text is sanitized (remove realism terms)
  2. Prompt is built with fixed style DNA prefix/suffix
  3. Scene is classified (night_fire, day_wilderness, same_scene_pose, etc.)
  4. One reference is selected (previous image OR master seed OR none)
  5. Image is generated (t2i or img2img)
  6. Output folder is opened for review

Usage:
    $python scripts/run_scene_pipeline.py --video-id "test-series-01"
    $python scripts/run_scene_pipeline.py --video-id "test-series-01" --master-seed-dir master_style_seeds
    $python scripts/run_scene_pipeline.py --video-id "test-series-01" --dry-run
"""
import argparse
import base64
import logging
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from image_generation.style_dna import build_scene_prompt, NEGATIVE_PROMPT
from image_generation.character_bible import detect_characters_in_text, get_character_block
from image_generation.scene_classifier import classify_scene
from image_generation.reference_selector import select_reference, load_master_manifest
from image_generation.runpod_serverless_backend import RunPodServerlessBackend
from image_generation.schemas import SceneRequest

# ── Test scenes — 6 scenes across 2 continuity groups ──────────────────────
TEST_SCENES = [
    {
        "scene_id": "001",
        "text": "A wide establishing shot of a prehistoric tribe gathered around a large campfire at night, African savanna, starry sky",
        "characters": [],
        "continuity_group": "camp_night_a",
    },
    {
        "scene_id": "002",
        "text": "Close-up portrait of Karo sitting at the campfire, warm orange firelight on his face, thoughtful expression",
        "characters": ["karo"],
        "continuity_group": "camp_night_a",
    },
    {
        "scene_id": "003",
        "text": "Karo raises one arm and points toward distant smoke on the horizon",
        "characters": ["karo"],
        "continuity_group": "camp_night_a",
    },
    {
        "scene_id": "004",
        "text": "A completely new scene: wide daytime shot of an expansive prehistoric valley, mountains in the distance, morning light",
        "characters": [],
        "continuity_group": "valley_day_b",
    },
    {
        "scene_id": "005",
        "text": "Scientific diagram showing a human brain cross-section with labeled regions, dark background, educational infographic",
        "characters": [],
        "continuity_group": "diagram_c",
    },
    {
        "scene_id": "006",
        "text": "Extreme close-up of a primitive stone axe lying on rock ground, carved flint edge, leather binding",
        "characters": [],
        "continuity_group": "object_d",
    },
]


def _load_as_b64(path: str) -> str:
    raw = Path(path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(raw).decode()


def run_pipeline(video_id: str, master_seed_dir: str, dry_run: bool = False):
    manifest = load_master_manifest(master_seed_dir)
    if not manifest:
        logger.warning("No master seed manifest — will use text-to-image for all scenes")

    backend = RunPodServerlessBackend() if not dry_run else None
    out_dir = Path("output") / video_id / "images_pipeline_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    prev_image_path = None
    prev_scene_text = None
    prev_group = None
    prev_qa_passed = True
    chain_depth = 0
    results = []

    print("\n" + "=" * 65)
    print(f"  Pipeline test: {video_id}")
    print(f"  Master seeds: {master_seed_dir}")
    print(f"  Dry run: {dry_run}")
    print("=" * 65 + "\n")

    for scene in TEST_SCENES:
        sid = scene["scene_id"]
        text = scene["text"]
        chars = scene["characters"]
        group = scene["continuity_group"]

        print(f"-- Scene {sid} {'-'*45}")
        print(f"   Text: {text[:70]}...")

        # 1. Detect + inject characters
        detected = detect_characters_in_text(text)
        all_chars = list(set(chars + detected))
        char_blocks = [get_character_block(c) for c in all_chars if c in ["karo", "luma"]]

        # 2. Classify
        clf = classify_scene(
            scene_text=text,
            previous_scene_text=prev_scene_text,
            characters=all_chars,
            continuity_group=group,
            previous_continuity_group=prev_group,
        )
        print(f"   Classify: {clf.scene_type} | shot={clf.shot_type} | confidence={clf.confidence:.2f}")

        # 3. Select reference
        ref = select_reference(
            classification=clf,
            previous_scene_image=prev_image_path,
            master_seed_dir=master_seed_dir,
            master_seed_manifest=manifest,
            chain_depth=chain_depth,
            previous_qa_passed=prev_qa_passed,
        )
        print(f"   Reference: mode={ref.mode} source={ref.reference_source} "
              f"key={ref.reference_key} strength={ref.strength:.2f} chain={ref.chain_depth}")
        if ref.reset_reason:
            print(f"   Reset: {ref.reset_reason}")

        # 4. Build prompt
        prompt = build_scene_prompt(
            scene_text=text,
            character_blocks=char_blocks,
            shot_block=f"{clf.shot_type} shot",
        )
        print(f"   Prompt ({len(prompt)} chars): {prompt[:100]}...")

        if dry_run:
            print(f"   [DRY RUN — skipping generation]\n")
            results.append({"scene_id": sid, "mode": ref.mode, "status": "dry_run"})
            prev_scene_text = text
            prev_group = group
            chain_depth = ref.chain_depth + (1 if ref.reference_source == "previous_scene" else 0)
            continue

        # 5. Load reference image as base64 if img2img
        ref_b64 = None
        if ref.mode == "img2img" and ref.reference_path:
            try:
                ref_b64 = _load_as_b64(ref.reference_path)
                print(f"   Ref loaded: {Path(ref.reference_path).name} ({len(ref_b64)//1024}KB b64)")
            except Exception as e:
                logger.warning("Could not load reference %s: %s — falling back to t2i", ref.reference_path, e)
                ref_b64 = None

        # 6. Generate
        req = SceneRequest(
            video_id=video_id,
            scene_id=sid,
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            width=1280,
            height=720,
            steps=22,
            guidance_scale=3.5,
            candidate_seeds=[11001],
            output_format="WEBP",
            quality=92,
            output_mode="base64",
            img2img_base64=ref_b64,
            strength=ref.strength if ref_b64 else 0.0,
        )

        t0 = time.time()
        result = backend.generate(req)
        elapsed = time.time() - t0

        if result.errors:
            print(f"   ERROR: {result.errors}\n")
            results.append({"scene_id": sid, "mode": ref.mode, "status": "failed", "errors": result.errors})
            prev_image_path = None
            prev_qa_passed = False
            chain_depth = 0
        elif result.candidates and result.candidates[0].local_path:
            # Copy to output dir as PNG
            src = Path(result.candidates[0].local_path)
            dst = out_dir / f"img_{sid}.png"
            from PIL import Image
            Image.open(src).convert("RGB").save(dst, format="PNG")
            print(f"   OK — {ref.mode} | {elapsed:.1f}s -> {dst.name}\n")
            results.append({"scene_id": sid, "mode": ref.mode, "status": "ok", "path": str(dst)})
            prev_image_path = str(dst)
            prev_qa_passed = True
            chain_depth = ref.chain_depth + 1
        else:
            print(f"   No candidate returned\n")
            results.append({"scene_id": sid, "mode": ref.mode, "status": "no_output"})
            prev_image_path = None
            chain_depth = 0

        prev_scene_text = text
        prev_group = group

    # Summary
    print("\n" + "=" * 65)
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"  Done: {ok}/{len(results)} scenes OK")
    for r in results:
        icon = "OK" if r["status"] == "ok" else ("DRY" if r["status"] == "dry_run" else "FAIL")
        print(f"    [{icon}] Scene {r['scene_id']} — {r['mode']}")
    print(f"\n  Output: {out_dir}")
    print("=" * 65 + "\n")

    return ok == len(results) or dry_run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full pipeline test with style DNA + master seeds")
    parser.add_argument("--video-id", default="test-series-01")
    parser.add_argument("--master-seed-dir", default="master_style_seeds")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show classification + reference decisions without generating images")
    args = parser.parse_args()

    success = run_pipeline(args.video_id, args.master_seed_dir, args.dry_run)
    sys.exit(0 if success else 1)
