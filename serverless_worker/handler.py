"""
DIAGNOSTIC handler — temporary, to isolate why the serverless loop wasn't
picking up jobs (worker "ready" but job stuck IN_QUEUE).

Only imports `runpod` + stdlib at module level (nothing heavy), then starts the
serverless loop immediately. Heavy imports (torch/diffusers/local modules) are
tested INSIDE the handler and reported back in the job output — so if the job
completes we learn (a) the loop works and (b) exactly which import is broken.

If even this minimal handler leaves jobs stuck IN_QUEUE, the problem is the
base image / ENTRYPOINT / build, not our Python code.

Restore the real handler from git (commit bbf8f81) once diagnosis is done.
"""

import os
import sys
import traceback

import runpod


def _try_import(modname):
    try:
        __import__(modname)
        return "ok"
    except Exception as e:  # noqa: BLE001
        return f"FAIL: {type(e).__name__}: {e}"


def handler(job):
    inp = job.get("input", {})
    diag = {
        "echo": inp,
        "python": sys.version,
        "cwd": os.getcwd(),
        "app_listing": sorted(os.listdir("/app")) if os.path.isdir("/app") else "no /app",
        "env_model_id": os.environ.get("MODEL_ID"),
        "env_hf_token_set": bool(os.environ.get("HF_TOKEN")),
        "volume_exists": os.path.isdir("/runpod-volume"),
    }

    imports = {}
    for m in ("torch", "diffusers", "transformers", "PIL",
              "image_utils", "model_loader", "schemas"):
        imports[m] = _try_import(m)
    diag["imports"] = imports

    # CUDA check
    try:
        import torch
        diag["cuda_available"] = torch.cuda.is_available()
        diag["cuda_device_count"] = torch.cuda.device_count()
    except Exception as e:  # noqa: BLE001
        diag["cuda_error"] = f"{type(e).__name__}: {e}"

    # Can we reference the Flux pipeline class?
    try:
        from diffusers import Flux2KleinPipeline  # noqa: F401
        diag["flux2klein_class"] = "importable"
    except Exception as e:  # noqa: BLE001
        diag["flux2klein_class"] = f"FAIL: {type(e).__name__}: {e}"

    return diag


print("DIAGNOSTIC handler.py loaded — starting serverless loop", flush=True)
runpod.serverless.start({"handler": handler})
