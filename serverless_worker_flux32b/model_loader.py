"""Load FLUX.2 Dev 32B once at worker startup and keep it in memory."""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger(__name__)

MODEL_ID = os.environ.get("MODEL_ID", "black-forest-labs/FLUX.2-dev")

_pipeline = None


def load_model():
    """
    Load FluxPipeline (32B) in bfloat16 full precision.
    Requires 80GB GPU (~64GB VRAM peak).
    Called once at cold-start. Subsequent calls return cached pipeline.
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    logger.info("Loading model %s (token=%s) ...", MODEL_ID, "set" if hf_token else "MISSING")
    t0 = time.time()

    try:
        import torch
        from diffusers import FluxPipeline
    except ImportError as e:
        logger.error("Missing dependency: %s", e)
        raise RuntimeError(f"Missing dependency: {e}") from e

    if not torch.cuda.is_available():
        logger.error("No CUDA device found")
        raise RuntimeError("No CUDA device available")

    try:
        pipe = FluxPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            token=hf_token or None,
        )
        pipe = pipe.to("cuda")
    except Exception as e:
        logger.error("Failed to load model '%s': %s", MODEL_ID, e)
        raise

    elapsed = time.time() - t0
    logger.info("Model loaded in %.1fs (bfloat16, device=cuda)", elapsed)

    _pipeline = pipe
    return _pipeline


def get_pipeline():
    if _pipeline is None:
        raise RuntimeError("Model not loaded — call load_model() first")
    return _pipeline
