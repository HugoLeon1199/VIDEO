"""
CLI to generate images for all scenes using the RunPod Serverless backend.

Usage examples
--------------
# Generate with Vision QA enabled (VI track, 3 candidates)
python scripts/generate_images.py --video-id my-vi-video --track vi --qa --candidates 3 --workers 10

# Generate without QA (fast, EN track)
python scripts/generate_images.py --video-id my-en-video --track en --no-qa --candidates 1 --workers 10

# QA-only: evaluate existing images without regenerating
python scripts/generate_images.py --video-id my-vi-video --track vi --qa-only

# Resume (default): skips scenes already QA-passed
python scripts/generate_images.py --video-id my-vi-video --track vi --qa
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("generate_images")


# ---------------------------------------------------------------------------
# Generation log helpers
# ---------------------------------------------------------------------------

def _log_path(video_id: str, output_root: str, track: Optional[str]) -> Path:
    suffix = f"_{track}" if track else ""
    return Path(output_root) / video_id / f"generation_log{suffix}.json"


def _summary_path(video_id: str, output_root: str, track: Optional[str]) -> Path:
    suffix = f"_{track}" if track else ""
    return Path(output_root) / video_id / f"generation_summary{suffix}.json"


def _load_log(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def _save_log(path: Path, log: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(log, indent=2), encoding="utf-8")
    tmp.replace(path)


def _scene_done(log: dict, scene_id: str, n_candidates: int, require_qa: bool = False) -> bool:
    """
    Returns True if scene can be skipped.
    If require_qa=True, also checks qa_passed=True.
    """
    entry = log.get(scene_id, {})
    selected_image = entry.get("selected_image")
    basic = (
        entry.get("status") == "completed"
        and entry.get("candidates_saved", 0) >= n_candidates
        and bool(selected_image)
        and Path(selected_image).exists()
    )
    if not basic:
        return False
    if require_qa:
        return bool(entry.get("qa_passed", False))
    return True


# ---------------------------------------------------------------------------
# QA helpers
# ---------------------------------------------------------------------------

def _run_qa_on_candidates(
    candidates,
    prompt: str,
    scene_id: str,
    scene_meta,
    qa_engine,
) -> list:
    """Evaluate all candidates and return list of QAResult."""
    from image_generation.vision_qa import SceneMeta
    results = []
    for c in candidates:
        if not c.local_path or not Path(c.local_path).exists():
            logger.warning("Scene %s candidate %d has no local file — skipping QA", scene_id, c.candidate_index)
            continue
        result = qa_engine.evaluate(
            image_path=Path(c.local_path),
            prompt=prompt,
            scene_id=scene_id,
            candidate_index=c.candidate_index,
            seed=c.seed,
            sha256=c.sha256,
            scene_meta=scene_meta,
        )
        results.append(result)
    return results


def _make_log_entry(
    status: str,
    candidates,
    selected_image: str,
    errors: list,
    job_id: Optional[str],
    duration: float,
    qa_results: Optional[list] = None,
    selected_seed: Optional[int] = None,
    selected_score: Optional[int] = None,
    qa_passed: Optional[bool] = None,
    qa_round: int = 0,
    style_version: str = "",
    qa_prompt_version: str = "",
) -> dict:
    entry: dict = {
        "status": status,
        "candidates_saved": len(candidates) if candidates else 0,
        "selected_image": selected_image,
        "errors": errors,
        "job_id": job_id,
        "duration_seconds": duration,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if selected_seed is not None:
        entry["selected_seed"] = selected_seed
    if selected_score is not None:
        entry["selected_score"] = selected_score
    if qa_passed is not None:
        entry["qa_passed"] = qa_passed
    if qa_round:
        entry["qa_round"] = qa_round
    if style_version:
        entry["style_version"] = style_version
    if qa_prompt_version:
        entry["qa_prompt_version"] = qa_prompt_version
    if qa_results:
        entry["candidate_reviews"] = [
            {
                "seed": r.seed,
                "score": r.score,
                "pass": r.passed,
                "issues": r.issues,
                "qa_error": r.qa_error,
                "cached": r.cached,
            }
            for r in qa_results
        ]
    return entry


# ---------------------------------------------------------------------------
# Backend factories
# ---------------------------------------------------------------------------

def _build_vast_backend():
    """Rent a Vast.ai instance, wait until ready, return (backend, teardown_fn).

    If VAST_INSTANCE_HOST + VAST_INSTANCE_PORT are set, skip rent and connect
    directly (useful for manual testing or resume after crash).
    teardown_fn destroys the instance; None if rent was skipped.
    """
    import config as _cfg
    from image_generation.vast_manager import VastManager
    from image_generation.vast_backend import VastInstanceBackend

    if not _cfg.VAST_API_KEY:
        raise RuntimeError("VAST_API_KEY not set — add it to .env")

    manager = VastManager(api_key=_cfg.VAST_API_KEY, worker_port=_cfg.VAST_WORKER_PORT)

    # Manual / resume mode: instance already running
    if _cfg.VAST_INSTANCE_HOST and _cfg.VAST_INSTANCE_PORT:
        logger.info("Vast: connecting to existing instance %s:%d",
                    _cfg.VAST_INSTANCE_HOST, _cfg.VAST_INSTANCE_PORT)
        backend = VastInstanceBackend(
            host=_cfg.VAST_INSTANCE_HOST,
            port=_cfg.VAST_INSTANCE_PORT,
            timeout=_cfg.VAST_REQUEST_TIMEOUT,
        )
        manager.wait_worker_ready(_cfg.VAST_INSTANCE_HOST, _cfg.VAST_INSTANCE_PORT, timeout=120)
        return backend, None  # caller manages lifetime

    # Auto-rent flow
    logger.info("Vast: searching for GPU (vram>=%dGB, max $%.2f/hr, inet>=%dMbps)...",
                _cfg.VAST_MIN_VRAM_GB, _cfg.VAST_MAX_PRICE_PER_HOUR, _cfg.VAST_MIN_INET_DOWN_MBPS)
    offer = manager.find_offer(
        min_vram_gb=_cfg.VAST_MIN_VRAM_GB,
        gpu_name=_cfg.VAST_GPU_NAME,
        max_price_per_hour=_cfg.VAST_MAX_PRICE_PER_HOUR,
        min_inet_down_mbps=_cfg.VAST_MIN_INET_DOWN_MBPS,
        min_reliability=_cfg.VAST_MIN_RELIABILITY,
    )
    env_vars = {}
    if _cfg.VAST_HF_TOKEN:
        env_vars["HF_TOKEN"] = _cfg.VAST_HF_TOKEN
    env_vars["USE_FP8"] = _cfg.VAST_USE_FP8
    instance = manager.rent(
        offer_id=offer["id"],
        image=_cfg.VAST_WORKER_IMAGE,
        env_vars=env_vars,
        disk_gb=_cfg.VAST_DISK_GB,
    )

    # From this point on, the instance is billed. Destroy on any failure.
    try:
        instance = manager.wait_until_running(instance.instance_id, timeout=1200)
        # Vast often needs a little longer to publish the host port mapping than
        # it does to reach 'created'. Wait for it instead of failing immediately.
        if not instance.direct_port:
            instance = manager.wait_for_port(instance.instance_id, timeout=180)
        # 1200s: the worker downloads the ~24GB FLUX model from HF at startup
        # (model is not baked into the image) and then loads it into VRAM.
        manager.wait_worker_ready(instance.public_ipaddr, instance.direct_port, timeout=1200)
    except Exception:
        logger.error("Vast setup failed — destroying instance %d", instance.instance_id)
        manager.destroy(instance.instance_id)
        raise

    backend = VastInstanceBackend(
        host=instance.public_ipaddr,
        port=instance.direct_port,
        timeout=_cfg.VAST_REQUEST_TIMEOUT,
    )

    def teardown():
        logger.info("Vast: destroying instance %d...", instance.instance_id)
        manager.destroy(instance.instance_id)

    return backend, teardown


def _build_vast_backends_parallel(n: int) -> tuple[list, callable]:
    """Rent N Vast instances in parallel, return (backends_list, teardown_all_fn).

    All instances are rented simultaneously then waited on in parallel — boot
    time stays the same as renting 1 instance. On any failure, all already-rented
    instances are destroyed before re-raising.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed
    from image_generation.vast_manager import VastManager

    if n == 1:
        backend, teardown = _build_vast_backend()
        return [backend], teardown

    logger.info("Vast: renting %d instances in parallel...", n)
    results = [None] * n
    errors = []

    def _rent_one(idx):
        return idx, _build_vast_backend()

    teardowns = []
    with ThreadPoolExecutor(max_workers=n) as ex:
        futures = {ex.submit(_rent_one, i): i for i in range(n)}
        for fut in _as_completed(futures):
            idx = futures[fut]
            try:
                _, (backend, td) = fut.result()
                results[idx] = backend
                if td:
                    teardowns.append(td)
                logger.info("Vast: instance %d/%d ready", idx + 1, n)
            except Exception as e:
                errors.append(e)
                logger.error("Vast: instance %d/%d failed: %s", idx + 1, n, e)

    backends = [b for b in results if b is not None]

    # Fault-tolerant: a failed instance no longer kills the whole batch. As long
    # as at least one machine came up, we run on those; the failed ones were
    # already destroyed inside _build_vast_backend. Only abort if ALL failed.
    if not backends:
        # Nothing usable — make absolutely sure nothing is left billing.
        try:
            import config as _cfg2
            VastManager(api_key=_cfg2.VAST_API_KEY,
                        worker_port=_cfg2.VAST_WORKER_PORT).destroy_all()
        except Exception:
            pass
        raise RuntimeError(f"Vast: all {n} instances failed to start: {errors[0]}")

    if errors:
        logger.warning(
            "Vast: %d/%d instances failed but %d are ready — continuing on those",
            len(errors), n, len(backends),
        )

    def teardown_all():
        for td in teardowns:
            try:
                td()
            except Exception as e:
                logger.warning("Vast teardown error: %s", e)
        # Belt-and-suspenders: force a verified destroy-all so no machine is left.
        try:
            import config as _cfg3
            VastManager(api_key=_cfg3.VAST_API_KEY,
                        worker_port=_cfg3.VAST_WORKER_PORT).destroy_all()
        except Exception as e:
            logger.error("Vast destroy_all error: %s — CHECK DASHBOARD", e)

    logger.info("Vast: %d/%d instances ready", len(backends), n)
    return backends, teardown_all


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate images via RunPod Serverless")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--prompts", help="Path to image_prompts.json")
    parser.add_argument("--backend", default="runpod_serverless",
                        choices=["runpod_serverless", "vast_instance"])
    parser.add_argument("--track", choices=["vi", "en"], default=None,
                        help="Image track: 'vi'=2D documentary, 'en'=ink sketch")
    parser.add_argument("--scene-id", help="Process only this scene_id")
    parser.add_argument("--from-scene", type=int)
    parser.add_argument("--to-scene", type=int)
    parser.add_argument("--candidates", type=int, default=None,
                        help="Candidates per scene (default: 3 with QA, 1 without)")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--force", action="store_true", help="Regenerate even if already done")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--vast-instances", type=int, default=1,
                        help="Number of Vast.ai instances to rent in parallel (vast_instance backend only)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-root", default=os.environ.get("IMAGE_OUTPUT_ROOT", "output"))

    # QA flags
    qa_group = parser.add_mutually_exclusive_group()
    qa_group.add_argument("--qa", action="store_true", default=None,
                          help="Enable Vision QA after generation (default for --track vi)")
    qa_group.add_argument("--no-qa", action="store_true",
                          help="Disable Vision QA")
    qa_group.add_argument("--qa-only", action="store_true",
                          help="Run QA on existing images only, do not call RunPod")
    parser.add_argument("--qa-min-score", type=int, default=None)
    parser.add_argument("--max-regenerations", type=int, default=None)
    parser.add_argument("--allow-qa-fallback", action="store_true",
                        help="Promote best failing image when all retries exhausted")

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

    # Determine QA mode
    if args.qa_only:
        qa_enabled = True
        generation_enabled = False
    elif args.no_qa:
        qa_enabled = False
        generation_enabled = True
    elif args.qa:
        qa_enabled = True
        generation_enabled = True
    else:
        # Default: QA on for VI track, off for EN
        qa_enabled = (args.track == "vi") and _config.IMAGE_QA_ENABLED
        generation_enabled = True

    qa_min_score = args.qa_min_score or _config.IMAGE_QA_MIN_SCORE
    max_regenerations = args.max_regenerations or _config.IMAGE_QA_MAX_REGENERATIONS
    allow_fallback = args.allow_qa_fallback or _config.IMAGE_QA_ALLOW_FALLBACK

    # Apply track-specific config
    _track_steps = None
    _track_guidance = None
    _track_output_subdir = "images"
    if args.track:
        tc = _config.TRACK_CONFIG[args.track]
        _track_steps = tc["steps"]
        _track_guidance = tc["guidance_scale"]
        _track_output_subdir = tc["output_subdir"]
        logger.info(
            "Track '%s': %d steps, guidance %.1f -> %s | QA: %s",
            args.track, _track_steps, _track_guidance, _track_output_subdir,
            "enabled" if qa_enabled else "disabled",
        )

    # Candidate count: default 3 when QA enabled, 1 otherwise
    if args.seeds:
        seeds = args.seeds
    else:
        n = args.candidates
        if n is None:
            n = 3 if qa_enabled else 1
        base = 11000
        seeds = [base + i + 1 for i in range(n)]

    # Load prompts
    prompts_path = Path(args.prompts) if args.prompts else (
        Path(args.output_root) / args.video_id / "image_prompts.json"
    )
    if not prompts_path.exists():
        logger.error("Prompts file not found: %s", prompts_path)
        sys.exit(1)

    prompts: list[dict] = json.loads(prompts_path.read_text(encoding="utf-8"))
    logger.info("Loaded %d prompts from %s", len(prompts), prompts_path)

    if args.validate_only:
        logger.info("Validation only — %d prompts look good.", len(prompts))
        return

    # Filter scenes
    if args.scene_id:
        prompts = [p for p in prompts if str(p["index"]).zfill(3) == args.scene_id.zfill(3)]
    else:
        if args.from_scene:
            prompts = prompts[args.from_scene - 1:]
        if args.to_scene:
            prompts = prompts[:args.to_scene]

    if not prompts:
        logger.warning("No prompts matched filter.")
        return

    logger.info("Processing %d scenes, %d candidates each", len(prompts), len(seeds))

    if args.dry_run:
        for p in prompts:
            print(f"  [dry-run] scene {p['index']:03d}: {p['prompt'][:80]}...")
        return

    # Setup backend (only needed when generation_enabled)
    backend = None          # single backend (runpod or vast n=1)
    _vast_backends = []     # list of backends when vast n>1
    _vast_teardown = None
    if generation_enabled:
        if args.backend == "vast_instance":
            n_inst = args.vast_instances
            if n_inst > 1:
                _vast_backends, _vast_teardown = _build_vast_backends_parallel(n_inst)
            else:
                backend, _vast_teardown = _build_vast_backend()
        else:
            from image_generation.runpod_serverless_backend import RunPodServerlessBackend
            backend = RunPodServerlessBackend()

    from image_generation.runpod_serverless_backend import promote_candidate_to_render_image
    from image_generation.schemas import SceneRequest

    # Setup QA engine
    qa_engine = None
    if qa_enabled:
        from image_generation.vision_qa import VisionQA, SceneMeta
        qa_engine = VisionQA(min_score=qa_min_score)

    # Generation log (per-track)
    log_path = _log_path(args.video_id, args.output_root, args.track)
    gen_log = _load_log(log_path)
    log_lock = threading.Lock()

    t_start = time.time()
    total_ok = 0
    total_fail = 0
    total_needs_review = 0

    def _generate_candidates(p: dict, attempt_seeds: list[int], _backend=None) -> tuple:
        """Submit one generation job and return (result, error_str)."""
        scene_id = f"{p['index']:03d}"
        active_backend = _backend or backend

        req = SceneRequest(
            video_id=args.video_id,
            scene_id=scene_id,
            prompt=p["prompt"],
            global_style=p.get("global_style", ""),
            negative_prompt=p.get("negative_prompt", ""),
            width=p.get("width", 1024),
            height=p.get("height", 576),
            steps=_track_steps if _track_steps is not None else p.get("steps", 4),
            guidance_scale=_track_guidance if _track_guidance is not None else p.get("guidance_scale", 1.0),
            candidate_seeds=attempt_seeds,
            output_mode="base64",
        )
        result = active_backend.generate(req)
        return result, None

    def process_scene(p: dict, _backend=None) -> tuple[str, bool]:
        scene_id = f"{p['index']:03d}"
        nonlocal total_needs_review

        # --- QA-only mode: evaluate existing image ---
        if not generation_enabled:
            entry = gen_log.get(scene_id, {})
            selected = entry.get("selected_image", "")
            if not selected or not Path(selected).exists():
                logger.warning("Scene %s: no existing image for QA-only", scene_id)
                return scene_id, False

            if not qa_engine:
                logger.warning("Scene %s: QA-only but no QA engine", scene_id)
                return scene_id, False

            from image_generation.vision_qa import SceneMeta
            from image_generation.schemas import CandidateResult
            # Build a minimal candidate to evaluate
            seed = entry.get("selected_seed", 11001)
            sha = ""
            # Try to read sha from sidecar
            img_path = Path(selected)
            sidecar = img_path.with_suffix(".json")
            if sidecar.exists():
                try:
                    meta = json.loads(sidecar.read_text())
                    sha = meta.get("sha256", "")
                except Exception:
                    pass

            qa_res = qa_engine.evaluate(
                image_path=img_path,
                prompt=p["prompt"],
                scene_id=scene_id,
                candidate_index=0,
                seed=seed,
                sha256=sha,
                scene_meta=SceneMeta(track=args.track or "vi"),
            )

            with log_lock:
                gen_log[scene_id]["qa_passed"] = qa_res.passed
                gen_log[scene_id]["qa_score"] = qa_res.score
                gen_log[scene_id]["qa_issues"] = qa_res.issues
                if qa_res.qa_error:
                    gen_log[scene_id]["qa_error"] = qa_res.qa_error
                _save_log(log_path, gen_log)

            status = "pass" if qa_res.passed else "FAIL"
            logger.info(
                "Scene %s QA-only: %s score=%d issues=%s",
                scene_id, status, qa_res.score, qa_res.issues,
            )
            return scene_id, qa_res.passed

        # --- Normal generation mode ---
        # Skip if already done (with QA check when QA enabled)
        if args.resume and not args.force and _scene_done(gen_log, scene_id, len(seeds), require_qa=qa_enabled):
            logger.info("Scene %s already done — skipping", scene_id)
            return scene_id, True

        t_scene = time.time()
        all_qa_results = []
        best_candidate = None
        best_qa_result = None
        qa_round = 0

        current_seeds = list(seeds)
        current_prompt = p["prompt"]

        for regen_round in range(max_regenerations + 1):
            qa_round = regen_round

            # Generate candidates
            try:
                result, err = _generate_candidates(p | {"prompt": current_prompt}, current_seeds, _backend)
            except Exception as e:
                logger.error("Scene %s FAILED (round %d): %s", scene_id, regen_round, e)
                with log_lock:
                    gen_log[scene_id] = {
                        "status": "failed",
                        "error": str(e),
                        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                    _save_log(log_path, gen_log)
                return scene_id, False

            if not result.candidates:
                logger.warning("Scene %s round %d: no candidates returned. Errors: %s",
                               scene_id, regen_round, result.errors)
                if regen_round < max_regenerations:
                    current_seeds = [s + 100000 * (regen_round + 1) for s in seeds]
                    continue
                break

            if not qa_enabled:
                # No QA — just take first candidate
                best_candidate = result.candidates[0]
                break

            # Run QA on all candidates
            from image_generation.vision_qa import SceneMeta
            scene_meta = SceneMeta(track=args.track or "vi")
            qa_results = _run_qa_on_candidates(
                result.candidates, current_prompt, scene_id, scene_meta, qa_engine
            )
            all_qa_results.extend(qa_results)

            # Select best passing candidate
            best_qa_result = qa_engine.select_best(qa_results)

            if best_qa_result:
                # Find matching candidate object
                for c in result.candidates:
                    if c.seed == best_qa_result.seed:
                        best_candidate = c
                        break
                logger.info(
                    "Scene %s round %d: best candidate seed=%d score=%d PASS",
                    scene_id, regen_round, best_qa_result.seed, best_qa_result.score,
                )
                break
            else:
                # All failed this round
                failed_scores = [(r.seed, r.score, r.issues[:2]) for r in qa_results]
                logger.warning(
                    "Scene %s round %d: all %d candidates failed QA. %s",
                    scene_id, regen_round, len(qa_results), failed_scores,
                )
                if regen_round < max_regenerations:
                    # Build corrective prompt and retry with new seeds
                    current_prompt = qa_engine.build_corrective_prompt(p["prompt"], qa_results)
                    current_seeds = [s + 100000 * (regen_round + 1) for s in seeds]
                    logger.info("Scene %s: corrective prompt applied, retrying with seeds %s",
                                scene_id, current_seeds[:3])

        # --- Promote best candidate ---
        selected_image = ""
        if best_candidate:
            try:
                selected_image = promote_candidate_to_render_image(
                    best_candidate,
                    video_id=args.video_id,
                    scene_id=scene_id,
                    output_root=args.output_root,
                    images_subdir=_track_output_subdir,
                )
            except Exception as e:
                logger.error("Scene %s promote failed: %s", scene_id, e)

        elif not qa_enabled:
            # No QA: should not reach here
            pass
        else:
            # All retries exhausted, no passing candidate
            if allow_fallback and all_qa_results:
                # Find highest scoring (even if failing)
                best_failing = max(all_qa_results, key=lambda r: r.score)
                for c in result.candidates if result.candidates else []:
                    if c.seed == best_failing.seed:
                        review_path = (
                            Path(args.output_root) / args.video_id /
                            "needs_review" / f"scene_{scene_id}"
                        )
                        review_path.mkdir(parents=True, exist_ok=True)
                        # Copy to review dir but do NOT promote to render dir
                        import shutil
                        if c.local_path and Path(c.local_path).exists():
                            shutil.copy2(c.local_path, review_path / f"best_fallback_seed{c.seed}.webp")
                        logger.warning(
                            "Scene %s: FALLBACK enabled — best failing score=%d. Saved to needs_review/.",
                            scene_id, best_failing.score,
                        )
                        break
            else:
                logger.error(
                    "Scene %s: needs_review — no candidate passed QA after %d rounds",
                    scene_id, max_regenerations + 1,
                )
                with log_lock:
                    gen_log[scene_id] = _make_log_entry(
                        status="needs_review",
                        candidates=[],
                        selected_image="",
                        errors=["All QA rounds failed"],
                        job_id=None,
                        duration=time.time() - t_scene,
                        qa_results=all_qa_results,
                        qa_round=qa_round,
                        style_version=_config.IMAGE_STYLE_VERSION,
                        qa_prompt_version=_config.IMAGE_QA_PROMPT_VERSION,
                    )
                    _save_log(log_path, gen_log)
                    total_needs_review += 1
                return scene_id, False

        # Determine final QA status
        qa_passed = None
        selected_seed = best_candidate.seed if best_candidate else None
        selected_score = best_qa_result.score if best_qa_result else None

        if qa_enabled:
            qa_passed = bool(selected_image and best_qa_result and best_qa_result.passed)

        # Build log entry
        status = "completed" if selected_image else "partial"
        import config as _config2
        entry = _make_log_entry(
            status=status,
            candidates=result.candidates if result else [],
            selected_image=selected_image,
            errors=result.errors if result else [],
            job_id=result.job_id if result else None,
            duration=time.time() - t_scene,
            qa_results=all_qa_results if qa_enabled else None,
            selected_seed=selected_seed,
            selected_score=selected_score,
            qa_passed=qa_passed,
            qa_round=qa_round,
            style_version=_config2.IMAGE_STYLE_VERSION,
            qa_prompt_version=_config2.IMAGE_QA_PROMPT_VERSION if qa_enabled else "",
        )

        with log_lock:
            gen_log[scene_id] = entry
            _save_log(log_path, gen_log)

        if selected_image:
            logger.info(
                "Scene %s done — seed=%s score=%s qa=%s in %.1fs",
                scene_id,
                selected_seed,
                selected_score if selected_score is not None else "N/A",
                qa_passed if qa_passed is not None else "N/A",
                time.time() - t_scene,
            )
        else:
            logger.warning("Scene %s: no image selected", scene_id)

        return scene_id, bool(selected_image)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            if _vast_backends:
                # Multi-instance: round-robin assign scenes to backends
                # Each backend gets its own subset; scenes for the same backend
                # run sequentially on that GPU (GPU is the bottleneck anyway).
                n_b = len(_vast_backends)
                logger.info("Vast: distributing %d scenes across %d instances", len(prompts), n_b)
                futures = {
                    executor.submit(process_scene, p, _vast_backends[i % n_b]): p
                    for i, p in enumerate(prompts)
                }
            else:
                futures = {executor.submit(process_scene, p): p for p in prompts}

            for future in as_completed(futures):
                scene_id, ok = future.result()
                if ok:
                    total_ok += 1
                else:
                    total_fail += 1
                    if args.fail_fast:
                        logger.error("--fail-fast: cancelling remaining jobs.")
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
    finally:
        if _vast_teardown:
            _vast_teardown()
        # Safety net: the Vast API has been seen reporting 0 instances while
        # machines were still alive and billing (one V100 was stranded ~16h).
        # After the per-instance teardown, force a destroy-all + verify so no
        # rented machine is ever left running. Only for the vast backend.
        if args.backend == "vast_instance":
            try:
                from image_generation.vast_manager import VastManager
                _mgr = VastManager(api_key=_config.VAST_API_KEY,
                                   worker_port=_config.VAST_WORKER_PORT)
                _mgr.destroy_all()
            except Exception as _e:  # noqa: BLE001
                logger.error(
                    "Vast safety-net cleanup error: %s — CHECK DASHBOARD "
                    "https://cloud.vast.ai/instances/", _e,
                )

    # Write summary
    import config as _config3
    summary = {
        "video_id": args.video_id,
        "track": args.track,
        "total_scenes": len(prompts),
        "completed": total_ok,
        "failed": total_fail,
        "needs_review": total_needs_review,
        "qa_enabled": qa_enabled,
        "qa_min_score": qa_min_score if qa_enabled else None,
        "style_version": _config3.IMAGE_STYLE_VERSION,
        "total_seconds": round(time.time() - t_start, 1),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _save_log(_summary_path(args.video_id, args.output_root, args.track), summary)

    # Generate QA report if QA was enabled
    if qa_enabled and args.track:
        _write_qa_report(gen_log, args.video_id, args.output_root, args.track)

    logger.info(
        "Done: %d ok, %d failed, %d needs_review in %.0fs",
        total_ok, total_fail, total_needs_review, time.time() - t_start,
    )
    if total_fail or total_needs_review:
        sys.exit(1)


def _write_qa_report(gen_log: dict, video_id: str, output_root: str, track: str) -> None:
    """Write QA report JSON and Markdown."""
    out_dir = Path(output_root) / video_id
    report_json_path = out_dir / f"image_qa_report_{track}.json"
    report_md_path = out_dir / f"image_qa_report_{track}.md"

    scenes = []
    n_pass_first = 0
    n_regen = 0
    n_needs_review = 0
    issue_counts: dict[str, int] = {}

    for scene_id, entry in sorted(gen_log.items()):
        status = entry.get("status", "")
        qa_passed = entry.get("qa_passed", None)
        qa_round = entry.get("qa_round", 0)
        score = entry.get("selected_score", 0) or 0
        seed = entry.get("selected_seed")
        img = entry.get("selected_image", "")

        for review in entry.get("candidate_reviews", []):
            for issue in review.get("issues", []):
                issue_counts[issue] = issue_counts.get(issue, 0) + 1

        if status == "completed" and qa_passed:
            if qa_round == 0:
                n_pass_first += 1
            else:
                n_regen += 1
        elif status == "needs_review":
            n_needs_review += 1

        scenes.append({
            "scene_id": scene_id,
            "status": status,
            "qa_passed": qa_passed,
            "qa_round": qa_round,
            "selected_seed": seed,
            "selected_score": score,
            "image": img,
        })

    top_issues = sorted(issue_counts.items(), key=lambda x: -x[1])[:10]

    report = {
        "video_id": video_id,
        "track": track,
        "total_scenes": len(scenes),
        "pass_first_round": n_pass_first,
        "pass_after_regen": n_regen,
        "needs_review": n_needs_review,
        "top_issues": top_issues,
        "scenes": scenes,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _save_log(report_json_path, report)

    # Markdown report
    lines = [
        f"# Image QA Report — {video_id} ({track})",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Summary",
        f"- Total scenes: {len(scenes)}",
        f"- Pass first round: {n_pass_first}",
        f"- Pass after regen: {n_regen}",
        f"- Needs review: {n_needs_review}",
        "",
        "## Top Issues",
    ]
    for issue, count in top_issues:
        lines.append(f"- `{issue}`: {count}x")

    lines += [
        "",
        "## Scene Detail",
        "",
        "| Scene | Status | QA Pass | Round | Seed | Score | Image |",
        "|-------|--------|---------|-------|------|-------|-------|",
    ]
    for s in scenes:
        lines.append(
            f"| {s['scene_id']} | {s['status']} | {s['qa_passed']} | "
            f"{s['qa_round']} | {s['selected_seed']} | {s['selected_score']} | "
            f"`{Path(s['image']).name if s['image'] else '-'}` |"
        )

    report_md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("QA report written: %s", report_md_path)


if __name__ == "__main__":
    main()
