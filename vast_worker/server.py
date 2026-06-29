"""Gateway server for multi-GPU Vast.ai workers.

Downloads the FLUX model once, then spawns one gpu_worker subprocess per GPU.
Routes /generate to the least-busy worker. /health reports all workers.

Start (injected by deploy_worker via env):
    NUM_GPUS=2 python vast_worker/server.py --port 8080 --preload
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
NUM_GPUS = int(os.getenv("NUM_GPUS", "1"))
GATEWAY_PORT = int(os.getenv("VAST_WORKER_PORT", "8080"))
# localhost ports allocated per GPU worker: 8090, 8091, 8092, ...
_GPU_BASE_PORT = 8090
WORKER_TIMEOUT = int(os.getenv("WORKER_TIMEOUT", "600"))

app = FastAPI(title="Vast FLUX Gateway")

# ---------------------------------------------------------------------------
# Worker registry
# ---------------------------------------------------------------------------

class _WorkerEntry:
    def __init__(self, gpu_index: int, port: int, proc: subprocess.Popen):
        self.gpu_index = gpu_index
        self.port = port
        self.proc = proc
        self.pending = 0          # in-flight request count
        self.ready = False        # /health model_loaded confirmed
        self.dead = False

_workers: list[_WorkerEntry] = []
_workers_lock = threading.Lock()
_load_error: Optional[str] = None
_model_download_done = threading.Event()

# ---------------------------------------------------------------------------
# Model download (once, gateway only)
# ---------------------------------------------------------------------------

def _download_model() -> str:
    from vast_worker.model_loader import download_and_validate
    model_id = os.getenv("MODEL_ID", "black-forest-labs/FLUX.1-dev")
    revision = _validate_model_revision()
    hf_token = os.getenv("HF_TOKEN", "")
    model_path = download_and_validate(
        model_id=model_id,
        model_path="/workspace/model",
        revision=revision,
        hf_token=hf_token or None,
    )
    return model_path


def _validate_model_revision() -> str:
    revision = (os.getenv("HF_MODEL_REVISION", "") or "").strip()
    if not revision or revision.lower() == "main":
        raise RuntimeError("HF_MODEL_REVISION must be pinned to a commit SHA")
    return revision


# ---------------------------------------------------------------------------
# GPU worker subprocess management
# ---------------------------------------------------------------------------

def _worker_port(gpu_index: int) -> int:
    return _GPU_BASE_PORT + gpu_index


def _spawn_worker(gpu_index: int, model_path: str) -> subprocess.Popen:
    port = _worker_port(gpu_index)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    env["MODEL_PATH"] = model_path
    env["WORKER_PORT"] = str(port)
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "gpu_worker.py"),
        "--model-path", model_path,
        "--port", str(port),
    ]
    proc = subprocess.Popen(cmd, env=env)
    print(f"[gateway] Spawned GPU worker {gpu_index} on localhost:{port} (pid={proc.pid})", flush=True)
    return proc


def _poll_worker_health(entry: _WorkerEntry, timeout: int = 300) -> bool:
    url = f"http://127.0.0.1:{entry.port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if entry.proc.poll() is not None:
            print(f"[gateway] GPU worker {entry.gpu_index} exited early (rc={entry.proc.returncode})", flush=True)
            return False
        try:
            resp = httpx.get(url, timeout=5)
            data = resp.json()
            if data.get("model_loaded"):
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def _start_all_workers(model_path: str) -> None:
    global _load_error
    procs = []
    for gpu_index in range(NUM_GPUS):
        proc = _spawn_worker(gpu_index, model_path)
        entry = _WorkerEntry(gpu_index=gpu_index, port=_worker_port(gpu_index), proc=proc)
        with _workers_lock:
            _workers.append(entry)
        procs.append(entry)

    for entry in procs:
        ready = _poll_worker_health(entry, timeout=300)
        with _workers_lock:
            entry.ready = ready
            if not ready:
                entry.dead = True
                print(f"[gateway] GPU worker {entry.gpu_index} failed to become ready", flush=True)

    alive = sum(1 for e in _workers if not e.dead)
    if alive == 0:
        _load_error = "All GPU workers failed to start"
        print(f"[gateway] FATAL: {_load_error}", flush=True)
    else:
        print(f"[gateway] {alive}/{NUM_GPUS} GPU workers ready", flush=True)


def _background_startup() -> None:
    global _load_error
    try:
        model_path = _download_model()
        _model_download_done.set()
        _start_all_workers(model_path)
    except Exception as exc:
        _load_error = str(exc)
        _model_download_done.set()
        print(f"[gateway] Startup failed: {exc}", flush=True)


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def _least_busy_worker() -> Optional[_WorkerEntry]:
    with _workers_lock:
        alive = [e for e in _workers if not e.dead and e.ready]
        if not alive:
            return None
        return min(alive, key=lambda e: e.pending)


def _mark_dead_if_exited() -> None:
    with _workers_lock:
        for entry in _workers:
            if not entry.dead and entry.proc.poll() is not None:
                entry.dead = True
                entry.ready = False
                print(f"[gateway] GPU worker {entry.gpu_index} process died (rc={entry.proc.returncode})", flush=True)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _require_worker_token(request: Request) -> None:
    expected = (os.getenv("WORKER_API_TOKEN", "local-worker-token") or "").strip()
    if not expected:
        raise HTTPException(status_code=401, detail="Missing worker token configuration")
    header_token = request.headers.get("x-worker-token", "").strip()
    auth_header = request.headers.get("authorization", "").strip()
    if not header_token and auth_header.lower().startswith("bearer "):
        header_token = auth_header.split(" ", 1)[1].strip()
    if header_token != expected:
        raise HTTPException(status_code=401, detail="Invalid worker token")


# ---------------------------------------------------------------------------
# Request model (mirrors gpu_worker)
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    video_id: str
    scene_id: str
    prompt: str
    clip_prompt: str = ""
    negative_prompt: str = ""
    width: int = 1024
    height: int = 576
    steps: int = 20
    guidance_scale: float = 3.5
    candidate_seeds: list[int] = [11001]
    output_format: str = "WEBP"
    quality: int = 92
    img2img_base64: Optional[str] = None
    strength: float = 0.75


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    _mark_dead_if_exited()
    with _workers_lock:
        alive = sum(1 for e in _workers if not e.dead)
        ready = sum(1 for e in _workers if e.ready and not e.dead)
    model_downloaded = _model_download_done.is_set() and _load_error is None
    return {
        "status": "ok",
        "model_loaded": model_downloaded and ready >= 1,
        "num_gpus": NUM_GPUS,
        "workers_alive": alive,
        "workers_ready": ready,
        "load_error": _load_error,
    }


@app.post("/generate")
def generate(req: GenerateRequest, request: Request) -> JSONResponse:
    _require_worker_token(request)
    _mark_dead_if_exited()
    worker = _least_busy_worker()
    if worker is None:
        raise HTTPException(status_code=503, detail="No healthy GPU workers available")

    url = f"http://127.0.0.1:{worker.port}/generate"
    worker_token = (os.getenv("WORKER_API_TOKEN", "local-worker-token") or "").strip()
    with _workers_lock:
        worker.pending += 1
    try:
        resp = httpx.post(
            url,
            json=req.model_dump(),
            headers={"x-worker-token": worker_token},
            timeout=WORKER_TIMEOUT,
        )
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception as exc:
        with _workers_lock:
            worker.dead = True
            worker.ready = False
        raise HTTPException(status_code=503, detail=f"GPU worker {worker.gpu_index} failed: {exc}") from exc
    finally:
        with _workers_lock:
            worker.pending = max(0, worker.pending - 1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

signal.signal(signal.SIGTERM, lambda sig, frame: sys.exit(0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=GATEWAY_PORT)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--preload", action="store_true", help="Download model and start GPU workers at startup")
    args = parser.parse_args()

    _validate_model_revision()

    if args.preload:
        threading.Thread(target=_background_startup, daemon=True).start()

    uvicorn.run(app, host=args.host, port=args.port)
