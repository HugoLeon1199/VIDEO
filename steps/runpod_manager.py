"""RunPod pod lifecycle management — create, wait for ready, terminate."""

import time

import httpx
from loguru import logger

import config

_BASE = "https://rest.runpod.io/v1"


def _headers() -> dict:
    return {"Authorization": f"Bearer {config.RUNPOD_API_KEY}", "Content-Type": "application/json"}


def create_pod() -> dict:
    """Spin up an on-demand pod with the ComfyUI template. Returns pod dict with 'id'."""
    payload = {
        "name": "comfyui-youtube-pipeline",
        "imageName": "runpod/stable-diffusion:web-ui-10.2.0",
        "gpuTypeId": config.RUNPOD_GPU_TYPE,
        "cloudType": "SECURE",
        "templateId": config.RUNPOD_TEMPLATE_ID,
        "containerDiskInGb": config.RUNPOD_DISK_SIZE,
        "volumeInGb": 0,
        "ports": "8188/http",
        "env": [],
    }
    resp = httpx.post(f"{_BASE}/pods", json=payload, headers=_headers(), timeout=30)
    resp.raise_for_status()
    pod = resp.json()
    logger.info("Pod created: id={} gpu={}", pod["id"], pod.get("gpuTypeId", "?"))
    return pod


def wait_for_ready(pod_id: str, timeout: int = None) -> str:
    """Poll until pod status=RUNNING and ComfyUI port is open. Returns ComfyUI base URL."""
    if timeout is None:
        timeout = config.RUNPOD_POD_READY_TIMEOUT

    comfyui_url = f"https://{pod_id}-8188.proxy.runpod.net"
    deadline = time.monotonic() + timeout

    logger.info("Waiting for pod {} to be ready (timeout={}s)...", pod_id, timeout)

    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{_BASE}/pods/{pod_id}", headers=_headers(), timeout=10)
            resp.raise_for_status()
            pod = resp.json()
            status = pod.get("desiredStatus", "")
            runtime = pod.get("runtime")

            if status == "RUNNING" and runtime:
                # Verify ComfyUI HTTP is actually up
                try:
                    ping = httpx.get(f"{comfyui_url}/system_stats", timeout=10)
                    if ping.status_code == 200:
                        logger.info("Pod {} ready — ComfyUI at {}", pod_id, comfyui_url)
                        return comfyui_url
                except httpx.RequestError:
                    pass  # not up yet

            logger.debug("Pod {} status={} — waiting...", pod_id, status)
        except httpx.HTTPError as e:
            logger.debug("Poll error: {}", e)

        time.sleep(10)

    raise TimeoutError(f"Pod {pod_id} did not become ready within {timeout}s")


def terminate_pod(pod_id: str) -> None:
    """Terminate pod immediately to stop billing."""
    try:
        resp = httpx.delete(f"{_BASE}/pods/{pod_id}/terminate", headers=_headers(), timeout=15)
        resp.raise_for_status()
        logger.info("Pod {} terminated.", pod_id)
    except httpx.HTTPError as e:
        logger.warning("Failed to terminate pod {}: {} — terminate it manually in RunPod console!", pod_id, e)
