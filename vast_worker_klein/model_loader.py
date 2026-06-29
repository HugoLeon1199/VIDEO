"""Model download and validation for the Klein 9B worker.

Called synchronously at worker startup before the model-load thread spawns.
Adapted from vast_worker/model_loader.py — no FLUX.1-dev-specific ignore_patterns.
"""

from __future__ import annotations

import time
from pathlib import Path

_REQUIRED_SUBDIRS = [
    "transformer",
    "text_encoder",
    "text_encoder_2",
    "tokenizer",
    "tokenizer_2",
    "vae",
    "scheduler",
]


def validate_local(model_path: str) -> bool:
    """Return True if the model directory looks like a complete FLUX layout."""
    root = Path(model_path)
    if not root.is_dir():
        return False
    for subdir in _REQUIRED_SUBDIRS:
        if not (root / subdir).is_dir():
            return False
    transformer_dir = root / "transformer"
    weight_files = (
        list(transformer_dir.glob("*.safetensors"))
        + list(transformer_dir.glob("*.bin"))
    )
    return bool(weight_files)


def download_and_validate(
    model_id: str,
    model_path: str,
    revision: str,
    hf_token: str | None = None,
) -> str:
    """Download Klein model to model_path if not already present, return local path."""
    from huggingface_hub import snapshot_download

    if validate_local(model_path):
        print(f"[klein_loader] Model already present at {model_path!r}, skipping download.", flush=True)
        return model_path

    t0 = time.time()
    print(f"[klein_loader] Downloading {model_id} rev={revision or 'main'} to {model_path!r} ...", flush=True)
    downloaded = snapshot_download(
        repo_id=model_id,
        revision=revision or None,
        token=hf_token or None,
        local_dir=model_path,
    )
    elapsed = time.time() - t0
    print(f"[klein_loader] Download complete in {elapsed:.0f}s: {downloaded}", flush=True)
    return downloaded
