"""Image serialisation, hashing, and volume-save utilities."""

from __future__ import annotations

import base64
import hashlib
import io
import os
import tempfile
from pathlib import Path

from PIL import Image

# ~9 MB base64 payload safety limit (RunPod max response is ~10 MB)
_BASE64_SIZE_LIMIT = 9 * 1024 * 1024


def pil_to_bytes(image: Image.Image, fmt: str = "WEBP", quality: int = 92) -> bytes:
    buf = io.BytesIO()
    if fmt == "WEBP":
        image.save(buf, format="WEBP", quality=quality, method=4)
    else:
        image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def check_base64_size(data: bytes) -> None:
    """Raise ValueError if the payload would exceed the safe base64 limit."""
    encoded_size = (len(data) * 4 + 2) // 3
    if encoded_size > _BASE64_SIZE_LIMIT:
        raise ValueError(
            f"Image payload ({encoded_size // 1024} KB base64) exceeds safe limit "
            f"({_BASE64_SIZE_LIMIT // 1024} KB). Use output_mode='volume' instead."
        )


def save_to_volume(
    data: bytes,
    video_id: str,
    scene_id: str,
    candidate_index: int,
    seed: int,
    fmt: str = "WEBP",
) -> str:
    """
    Atomically write image bytes to the RunPod network volume.
    Returns the relative path (relative to RUNPOD_VOLUME_PATH).
    """
    volume_root = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume")
    ext = fmt.lower()
    rel_dir = f"emberlore/{video_id}/images/scene_{scene_id}"
    filename = f"candidate_{candidate_index:02d}_seed_{seed}.{ext}"

    abs_dir = Path(volume_root) / rel_dir
    abs_dir.mkdir(parents=True, exist_ok=True)

    dest = abs_dir / filename
    # Atomic write via temp file + rename
    fd, tmp_path = tempfile.mkstemp(dir=abs_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp_path, dest)
    except Exception:
        os.unlink(tmp_path)
        raise

    return str(Path(rel_dir) / filename)
