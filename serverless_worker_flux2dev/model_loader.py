"""Load FLUX.2 Dev 32B once at worker startup and keep it in memory."""

from __future__ import annotations

import logging
import os
import sys
import time

logger = logging.getLogger(__name__)

MODEL_ID = os.environ.get("MODEL_ID", "black-forest-labs/FLUX.2-klein-9B")
HF_TOKEN = os.environ.get("HF_TOKEN")

_pipeline = None


def load_model():
    """
    Load FluxPipeline (FLUX.2 Dev 32B) onto GPU in bfloat16.
    Called once at cold-start. Subsequent calls return the cached pipeline.
    Requires H100/A100 80GB — model weights ~60GB in bfloat16.
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    logger.info("Loading model %s ...", MODEL_ID)
    t0 = time.time()

    try:
        import torch
        from diffusers import Flux2KleinPipeline as FluxPipeline
    except ImportError as e:
        logger.error("Missing dependency: %s", e)
        sys.exit(1)

    if not torch.cuda.is_available():
        logger.error("No CUDA device found — cannot run FLUX.2 Dev")
        sys.exit(1)

    try:
        pipe = FluxPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            token=HF_TOKEN or None,
        )
        pipe = pipe.to("cuda")
    except Exception as e:
        logger.error("Failed to load model '%s': %s", MODEL_ID, e)
        sys.exit(1)

    elapsed = time.time() - t0
    logger.info("Model loaded in %.1fs (dtype=bfloat16, device=cuda)", elapsed)

    _pipeline = pipe
    return _pipeline


def get_pipeline():
    """Return the already-loaded pipeline. Raises if load_model() was not called."""
    if _pipeline is None:
        raise RuntimeError("Model not loaded — call load_model() first")
    return _pipeline
