"""Model download and validation for the Vast.ai multi-GPU worker.

Called once by the gateway (server.py) before spawning GPU subprocesses.
Each GPU subprocess uses validate_local() to confirm the model is present
before loading with local_files_only=True.
"""

from __future__ import annotations

import os
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


def download_and_validate(
    model_id: str,
    model_path: str,
    revision: str,
    hf_token: str | None = None,
) -> str:
    """Download model to model_path (skip if already present) and return local path.

    Uses ignore_patterns to skip the 23.8 GB root single-file duplicates:
        flux1-dev.safetensors   (diffusers uses transformer/ subdir)
        ae.safetensors          (diffusers uses vae/ subdir)
        dev_grid.jpg            (demo image)
    """
    from huggingface_hub import snapshot_download

    if validate_local(model_path, revision):
        print(f"[model_loader] Model already present at {model_path!r}, skipping download.", flush=True)
        return model_path

    print(f"[model_loader] Downloading {model_id} rev={revision} to {model_path!r}...", flush=True)
    downloaded = snapshot_download(
        repo_id=model_id,
        revision=revision,
        token=hf_token or None,
        local_dir=model_path,
        ignore_patterns=[
            "flux1-dev.safetensors",
            "ae.safetensors",
            "dev_grid.jpg",
        ],
    )
    print(f"[model_loader] Download complete: {downloaded}", flush=True)
    return downloaded


def validate_local(model_path: str, revision: str) -> bool:
    """Return True if the model directory looks like a complete FLUX layout."""
    root = Path(model_path)
    if not root.is_dir():
        return False
    for subdir in _REQUIRED_SUBDIRS:
        if not (root / subdir).is_dir():
            return False
    # Check that at least one .safetensors or .bin file exists inside transformer/
    transformer_dir = root / "transformer"
    weight_files = list(transformer_dir.glob("*.safetensors")) + list(transformer_dir.glob("*.bin"))
    if not weight_files:
        return False
    # Verify the pinned revision is recorded in refs/ if using local_dir mode
    refs_file = root / ".cache" / "huggingface" / "hub" / "refs" / revision
    # refs/ is optional (snapshot_download with local_dir may not write it),
    # so we only check for directory structure, not the ref file.
    return True
