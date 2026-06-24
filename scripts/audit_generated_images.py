"""
Final audit of all generated images for a video track.

Runs Vision QA independently on every selected image,
compares to initial QA results, flags uncertain cases.

Usage
-----
# Audit all VI images
python scripts/audit_generated_images.py --video-id my-vi-video --track vi

# Audit and regenerate failed/uncertain scenes
python scripts/audit_generated_images.py --video-id my-vi-video --track vi --regenerate-failed
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("audit_generated_images")


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit generated images with Vision QA")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--track", choices=["vi", "en"], required=True)
    parser.add_argument("--output-root", default=os.environ.get("IMAGE_OUTPUT_ROOT", "output"))
    parser.add_argument("--regenerate-failed", action="store_true",
                        help="Regenerate scenes that fail or are uncertain in audit")
    parser.add_argument("--audit-min-score", type=int, default=None,
                        help="Minimum score for audit pass (default: IMAGE_QA_AUDIT_MIN_SCORE from config)")
    parser.add_argument("--max-regen-rounds", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Load .env
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    import config as _config
    from image_generation.vision_qa import VisionQA, SceneMeta

    audit_min = args.audit_min_score or _config.IMAGE_QA_AUDIT_MIN_SCORE
    max_regen = args.max_regen_rounds or _config.IMAGE_QA_MAX_REGENERATIONS

    # Load generation log
    log_suffix = f"_{args.track}"
    log_path = Path(args.output_root) / args.video_id / f"generation_log{log_suffix}.json"
    gen_log = _load_json(log_path)

    if not gen_log:
        logger.error("No generation log found at %s", log_path)
        sys.exit(1)

    logger.info("Loaded %d scene entries from %s", len(gen_log), log_path)

    # Load prompts for corrective regen
    prompts_path = Path(args.output_root) / args.video_id / "image_prompts.json"
    prompts_by_id: dict[str, dict] = {}
    if prompts_path.exists():
        try:
            prompts_list = json.loads(prompts_path.read_text(encoding="utf-8"))
            prompts_by_id = {f"{p['index']:03d}": p for p in prompts_list}
        except Exception as e:
            logger.warning("Could not load prompts: %s", e)

    qa_engine = VisionQA(min_score=audit_min, anatomy_min=35)
    scene_meta = SceneMeta(track=args.track)

    audit_results: dict[str, dict] = {}
    n_passed = 0
    n_failed = 0
    n_uncertain = 0
    n_missing = 0

    for scene_id, entry in sorted(gen_log.items()):
        selected = entry.get("selected_image", "")
        prev_passed = entry.get("qa_passed")
        prev_score = entry.get("selected_score", 0) or 0

        # Check file exists
        if not selected or not Path(selected).exists():
            logger.warning("Scene %s: image missing — %s", scene_id, selected)
            audit_results[scene_id] = {
                "audit_status": "failed",
                "reason": "image file missing",
                "prev_qa_passed": prev_passed,
            }
            n_missing += 1
            n_failed += 1
            continue

        # Integrity check
        try:
            from PIL import Image as PilImage
            with PilImage.open(selected) as img:
                w, h = img.size
                if w < 64 or h < 64:
                    raise ValueError(f"Suspicious size {w}x{h}")
        except Exception as e:
            logger.error("Scene %s: image corrupt — %s", scene_id, e)
            audit_results[scene_id] = {
                "audit_status": "failed",
                "reason": f"image corrupt: {e}",
                "prev_qa_passed": prev_passed,
            }
            n_failed += 1
            continue

        # Read sha256 from sidecar if available
        sha = entry.get("sha256", "")
        sidecar = Path(selected).with_suffix(".json")
        if not sha and sidecar.exists():
            try:
                sha = json.loads(sidecar.read_text()).get("sha256", "")
            except Exception:
                pass
        seed = entry.get("selected_seed", 11001)

        # Run fresh QA
        prompt = prompts_by_id.get(scene_id, {}).get("prompt", "")
        qa_res = qa_engine.evaluate(
            image_path=Path(selected),
            prompt=prompt,
            scene_id=scene_id,
            candidate_index=0,
            seed=seed,
            sha256=sha,
            scene_meta=scene_meta,
        )

        # Compare with initial QA
        audit_status = "passed" if qa_res.passed else "failed"

        # Mark uncertain when results are inconsistent
        if prev_passed is not None and bool(prev_passed) != qa_res.passed:
            audit_status = "uncertain"
            n_uncertain += 1
            logger.warning(
                "Scene %s UNCERTAIN: initial_pass=%s audit_pass=%s score=%d->%d",
                scene_id, prev_passed, qa_res.passed, prev_score, qa_res.score,
            )
        elif qa_res.passed:
            n_passed += 1
            logger.info("Scene %s audit PASSED score=%d", scene_id, qa_res.score)
        else:
            n_failed += 1
            logger.warning(
                "Scene %s audit FAILED score=%d issues=%s",
                scene_id, qa_res.score, qa_res.issues,
            )

        audit_results[scene_id] = {
            "audit_status": audit_status,
            "audit_score": qa_res.score,
            "audit_passed": qa_res.passed,
            "audit_issues": qa_res.issues,
            "qa_error": qa_res.qa_error,
            "prev_qa_passed": prev_passed,
            "prev_score": prev_score,
        }

    # Save audit report
    audit_report_path = Path(args.output_root) / args.video_id / f"audit_report_{args.track}.json"
    report = {
        "video_id": args.video_id,
        "track": args.track,
        "audit_min_score": audit_min,
        "total": len(gen_log),
        "passed": n_passed,
        "failed": n_failed,
        "uncertain": n_uncertain,
        "missing": n_missing,
        "scenes": audit_results,
        "audited_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _save_json(audit_report_path, report)
    logger.info("Audit report saved: %s", audit_report_path)
    logger.info(
        "Audit summary: %d passed, %d failed, %d uncertain, %d missing",
        n_passed, n_failed, n_uncertain, n_missing,
    )

    # Regenerate failed/uncertain scenes
    if args.regenerate_failed and (n_failed > 0 or n_uncertain > 0):
        logger.info("Regenerating %d failed/uncertain scenes...", n_failed + n_uncertain)
        _regen_scenes(audit_results, gen_log, log_path, prompts_by_id, args, qa_engine, scene_meta, max_regen, audit_min, audit_report_path)

    # Exit 1 if any failed or uncertain
    if n_failed > 0 or n_uncertain > 0:
        sys.exit(1)


def _regen_scenes(
    audit_results: dict,
    gen_log: dict,
    log_path: Path,
    prompts_by_id: dict,
    args,
    qa_engine,
    scene_meta,
    max_regen: int,
    audit_min: int,
    audit_report_path: Path,
) -> None:
    """Regenerate failed/uncertain scenes, up to max_regen rounds each."""
    import config as _config
    from image_generation.runpod_serverless_backend import (
        RunPodServerlessBackend,
        promote_candidate_to_render_image,
    )
    from image_generation.schemas import SceneRequest

    tc = _config.TRACK_CONFIG[args.track]
    backend = RunPodServerlessBackend()

    regen_scenes = [
        sid for sid, r in audit_results.items()
        if r["audit_status"] in ("failed", "uncertain")
    ]

    for scene_id in regen_scenes:
        p = prompts_by_id.get(scene_id)
        if not p:
            logger.warning("Scene %s: no prompt found, skipping regen", scene_id)
            continue

        logger.info("Regenerating scene %s...", scene_id)
        prompt = p["prompt"]
        succeeded = False

        for regen_round in range(1, max_regen + 2):
            seeds = [11001 + 100000 * regen_round + i for i in range(_config.IMAGE_QA_REGEN_CANDIDATES)]

            req = SceneRequest(
                video_id=args.video_id,
                scene_id=scene_id,
                prompt=prompt,
                global_style=p.get("global_style", ""),
                negative_prompt=p.get("negative_prompt", ""),
                width=p.get("width", 1024),
                height=p.get("height", 576),
                steps=tc["steps"],
                guidance_scale=tc["guidance_scale"],
                candidate_seeds=seeds,
                output_mode="base64",
            )

            try:
                result = backend.generate(req)
            except Exception as e:
                logger.error("Scene %s regen round %d failed: %s", scene_id, regen_round, e)
                continue

            if not result.candidates:
                logger.warning("Scene %s regen round %d: no candidates", scene_id, regen_round)
                continue

            # QA each candidate
            qa_results = []
            for c in result.candidates:
                if not c.local_path or not Path(c.local_path).exists():
                    continue
                r = qa_engine.evaluate(
                    image_path=Path(c.local_path),
                    prompt=prompt,
                    scene_id=scene_id,
                    candidate_index=c.candidate_index,
                    seed=c.seed,
                    sha256=c.sha256,
                    scene_meta=scene_meta,
                )
                qa_results.append((c, r))

            best = max(qa_results, key=lambda x: x[1].score, default=None)
            if best:
                best_c, best_r = best
                if best_r.passed:
                    selected = promote_candidate_to_render_image(
                        best_c,
                        video_id=args.video_id,
                        scene_id=scene_id,
                        output_root=args.output_root,
                        images_subdir=tc["output_subdir"],
                    )
                    gen_log[scene_id]["selected_image"] = selected
                    gen_log[scene_id]["selected_seed"] = best_c.seed
                    gen_log[scene_id]["selected_score"] = best_r.score
                    gen_log[scene_id]["qa_passed"] = True
                    gen_log[scene_id]["status"] = "completed"
                    gen_log[scene_id]["audit_regen_round"] = regen_round

                    tmp = log_path.with_suffix(".tmp")
                    tmp.write_text(json.dumps(gen_log, indent=2), encoding="utf-8")
                    tmp.replace(log_path)

                    audit_results[scene_id]["audit_status"] = "passed"
                    audit_results[scene_id]["regen_round"] = regen_round

                    logger.info("Scene %s REGEN SUCCESS round=%d score=%d",
                                scene_id, regen_round, best_r.score)
                    succeeded = True
                    break
                else:
                    logger.warning(
                        "Scene %s regen round %d: best score=%d still failing. Issues: %s",
                        scene_id, regen_round, best_r.score, best_r.issues,
                    )
                    prompt = qa_engine.build_corrective_prompt(p["prompt"], [best_r])

        if not succeeded:
            logger.error("Scene %s: still failing after %d regen rounds", scene_id, max_regen)
            audit_results[scene_id]["audit_status"] = "failed"

    # Re-save audit report
    report_data = json.loads(audit_report_path.read_text()) if audit_report_path.exists() else {}
    report_data["scenes"] = audit_results
    report_data["regen_completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = audit_report_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    tmp.replace(audit_report_path)
    logger.info("Audit report updated after regen.")


if __name__ == "__main__":
    main()
