"""
Run full flat-2D pipeline for an existing video.

Reads image_prompts.json, builds style-DNA prompt for each scene (pure t2i,
no img2img chaining), generates via RunPod, saves to images_flat2d/.

Resume-safe: skips scenes already in generation_log_flat2d.json with status=ok.

Usage:
    $python scripts/run_full_video_pipeline.py --video-id <id>
    $python scripts/run_full_video_pipeline.py --video-id <id> --dry-run
    $python scripts/run_full_video_pipeline.py --video-id <id> --from-scene 44
    $python scripts/run_full_video_pipeline.py --video-id <id> --force
"""
import argparse
import logging
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, ".")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

import re

from image_generation.style_dna import build_scene_prompt, NEGATIVE_PROMPT
from image_generation.character_bible import detect_characters_in_text, get_character_block
from image_generation.runpod_serverless_backend import RunPodServerlessBackend
from image_generation.schemas import SceneRequest

# Strip old-style realism suffixes baked into photorealistic backup prompts
_STRIP_SUFFIXES = re.compile(
    r",?\s*(photorealistic|photoreal|cinematic photorealistic|cinematic documentary style"
    r"|earthy cinematic wide shot|photorealistic wide shot|photorealistic\.?)\s*$",
    re.IGNORECASE,
)


def run_full_pipeline(
    video_id: str,
    dry_run: bool = False,
    from_scene: int = 1,
    to_scene: int = 9999,
    force: bool = False,
):
    # Prefer photorealistic backup — it has proper English scene descriptions
    prompts_path = Path("output") / video_id / "image_prompts_photorealistic_backup.json"
    if not prompts_path.exists():
        prompts_path = Path("output") / video_id / "image_prompts.json"
    if not prompts_path.exists():
        logger.error("Not found: %s", prompts_path)
        return False

    logger.info("Reading prompts from: %s", prompts_path.name)
    prompts_data = json.loads(prompts_path.read_text(encoding="utf-8"))

    # Filter to requested range
    scenes = [e for e in prompts_data if from_scene <= e["index"] <= to_scene]
    total = len(scenes)

    out_dir = Path("output") / video_id / "images_flat2d"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path("output") / video_id / "generation_log_flat2d.json"
    gen_log = {}
    if log_path.exists() and not force:
        try:
            gen_log = json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            gen_log = {}

    backend = None if dry_run else RunPodServerlessBackend()

    ok = skipped = failed = 0
    logger.info("Video: %s | %d scenes | dry_run=%s", video_id, total, dry_run)
    logger.info("Output: %s", out_dir)

    for i, entry in enumerate(scenes):
        idx = entry["index"]
        sid = f"{idx:03d}"
        # Use English prompt field (scene description) — strip old realism suffix
        raw = entry.get("prompt", "") or entry.get("text", "")
        text = _STRIP_SUFFIXES.sub("", raw).strip().rstrip(",")
        n = i + 1

        dst = out_dir / f"img_{sid}.png"
        if dst.exists() and not force and gen_log.get(sid, {}).get("status") == "ok":
            logger.info("[%d/%d] SKIP %s", n, total, sid)
            skipped += 1
            continue

        chars = detect_characters_in_text(text)
        char_blocks = [get_character_block(c) for c in chars if c in ("karo", "luma")]

        prompt = build_scene_prompt(
            scene_text=text,
            character_blocks=char_blocks,
            shot_block="wide shot",
            include_anatomy=True,
            include_group_rule=False,
        )

        logger.info("[%d/%d] Scene %s | %d chars", n, total, sid, len(prompt))

        if dry_run:
            continue

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
        )

        t0 = time.time()
        try:
            result = backend.generate(req)
            elapsed = time.time() - t0

            if result.errors and not result.candidates:
                raise RuntimeError(str(result.errors))

            if not result.candidates or not result.candidates[0].local_path:
                raise RuntimeError("No candidate returned")

            from PIL import Image
            Image.open(result.candidates[0].local_path).convert("RGB").save(dst, format="PNG")

            gen_log[sid] = {
                "status": "ok",
                "duration_seconds": round(elapsed, 1),
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            tmp = log_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(gen_log, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(log_path)

            ok += 1
            logger.info("  -> OK  %s  %.1fs", dst.name, elapsed)

        except Exception as e:
            elapsed = time.time() - t0
            logger.error("  -> FAIL %s: %s (%.1fs)", sid, e, elapsed)
            gen_log[sid] = {"status": "failed", "error": str(e)}
            failed += 1

    print("\n" + "=" * 60)
    print(f"  Video : {video_id}")
    print(f"  Total : {total} | OK: {ok} | Skip: {skipped} | Fail: {failed}")
    print(f"  Output: {out_dir}")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--from-scene", type=int, default=1)
    parser.add_argument("--to-scene", type=int, default=9999)
    parser.add_argument("--force", action="store_true", help="Regenerate even if done")
    args = parser.parse_args()

    success = run_full_pipeline(
        video_id=args.video_id,
        dry_run=args.dry_run,
        from_scene=args.from_scene,
        to_scene=args.to_scene,
        force=args.force,
    )
    sys.exit(0 if success else 1)
