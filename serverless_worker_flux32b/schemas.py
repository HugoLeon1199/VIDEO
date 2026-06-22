"""Request/response schemas and validation for the RunPod serverless worker."""

from __future__ import annotations

SUPPORTED_MODES = {"text_to_image"}
SUPPORTED_FORMATS = {"WEBP", "PNG"}
MAX_CANDIDATES = 8
MAX_WIDTH = 2048
MAX_HEIGHT = 2048
MIN_DIM = 64
MAX_STEPS = 50
MAX_SEEDS = 8


def validate_input(job_input: dict) -> tuple[dict, list[str]]:
    """
    Validate and normalise job input.
    Returns (cleaned_input, errors). If errors is non-empty, reject the job.
    """
    errors: list[str] = []

    # --- mode ---
    mode = job_input.get("mode", "text_to_image")
    if mode not in SUPPORTED_MODES:
        errors.append(f"Unsupported mode '{mode}'. Supported: {sorted(SUPPORTED_MODES)}")

    # --- prompt ---
    prompt = job_input.get("prompt", "").strip()
    if not prompt:
        errors.append("'prompt' must be a non-empty string")

    # --- dimensions ---
    width = job_input.get("width", 1024)
    height = job_input.get("height", 576)
    if not isinstance(width, int) or not (MIN_DIM <= width <= MAX_WIDTH):
        errors.append(f"'width' must be int in [{MIN_DIM}, {MAX_WIDTH}], got {width!r}")
    if not isinstance(height, int) or not (MIN_DIM <= height <= MAX_HEIGHT):
        errors.append(f"'height' must be int in [{MIN_DIM}, {MAX_HEIGHT}], got {height!r}")

    # --- steps ---
    steps = job_input.get("steps", 4)
    if not isinstance(steps, int) or not (1 <= steps <= MAX_STEPS):
        errors.append(f"'steps' must be int in [1, {MAX_STEPS}], got {steps!r}")

    # --- guidance_scale ---
    guidance_scale = job_input.get("guidance_scale", 1.0)
    if not isinstance(guidance_scale, (int, float)) or not (0.0 <= float(guidance_scale) <= 20.0):
        errors.append(f"'guidance_scale' must be float in [0, 20], got {guidance_scale!r}")

    # --- candidate_seeds ---
    seeds = job_input.get("candidate_seeds", [11001])
    if not isinstance(seeds, list) or len(seeds) == 0:
        errors.append("'candidate_seeds' must be a non-empty list")
    elif len(seeds) > MAX_SEEDS:
        errors.append(f"'candidate_seeds' exceeds max {MAX_SEEDS}, got {len(seeds)}")
    else:
        for s in seeds:
            if not isinstance(s, int) or s < 0:
                errors.append(f"Each seed must be a non-negative int, got {s!r}")
                break

    # --- output_format ---
    output_format = job_input.get("output_format", "WEBP").upper()
    if output_format not in SUPPORTED_FORMATS:
        errors.append(f"'output_format' must be one of {sorted(SUPPORTED_FORMATS)}, got {output_format!r}")

    # --- quality ---
    quality = job_input.get("quality", 92)
    if not isinstance(quality, int) or not (1 <= quality <= 100):
        errors.append(f"'quality' must be int in [1, 100], got {quality!r}")

    # --- output_mode ---
    output_mode = job_input.get("output_mode", "base64")
    if output_mode not in ("base64", "volume"):
        errors.append(f"'output_mode' must be 'base64' or 'volume', got {output_mode!r}")

    if errors:
        return {}, errors

    return {
        "video_id": str(job_input.get("video_id", "unknown")),
        "scene_id": str(job_input.get("scene_id", "000")),
        "mode": mode,
        "prompt": prompt,
        "global_style": str(job_input.get("global_style", "")),
        "negative_prompt": str(job_input.get("negative_prompt", "")),
        "width": int(width),
        "height": int(height),
        "steps": int(steps),
        "guidance_scale": float(guidance_scale),
        "candidate_seeds": [int(s) for s in seeds],
        "output_format": output_format,
        "quality": int(quality),
        "output_mode": output_mode,
    }, []
