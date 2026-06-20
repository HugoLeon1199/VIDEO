"""ComfyUI API client — queue prompt, wait for result, download image bytes."""

import copy
import json
import time
import uuid

import httpx
from loguru import logger

import config

# ComfyUI workflow for FLUX.2 Klein 4B Distilled FP8
# Node IDs are stable strings that ComfyUI uses to wire inputs/outputs.
_WORKFLOW_TEMPLATE: dict = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": config.COMFYUI_MODEL},
    },
    "2": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "__PROMPT__",   # replaced at runtime
            "clip": ["1", 1],
        },
    },
    "3": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "",             # negative prompt (empty for FLUX)
            "clip": ["1", 1],
        },
    },
    "4": {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "width": config.COMFYUI_WIDTH,
            "height": config.COMFYUI_HEIGHT,
            "batch_size": 1,
        },
    },
    "5": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 0,              # replaced at runtime with idx for reproducibility
            "steps": config.COMFYUI_STEPS,
            "cfg": config.COMFYUI_CFG,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
            "model": ["1", 0],
            "positive": ["2", 0],
            "negative": ["3", 0],
            "latent_image": ["4", 0],
        },
    },
    "6": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["5", 0],
            "vae": ["1", 2],
        },
    },
    "7": {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": "__FILENAME__",  # replaced at runtime
            "images": ["6", 0],
        },
    },
}


def _build_workflow(prompt_text: str, idx: int) -> dict:
    wf = copy.deepcopy(_WORKFLOW_TEMPLATE)
    wf["2"]["inputs"]["text"] = prompt_text
    wf["5"]["inputs"]["seed"] = idx * 42  # deterministic seed per image
    wf["7"]["inputs"]["filename_prefix"] = f"img_{idx:03d}"
    return wf


def queue_prompt(comfyui_url: str, prompt_text: str, idx: int) -> str:
    """Queue a generation job. Returns prompt_id string."""
    workflow = _build_workflow(prompt_text, idx)
    client_id = str(uuid.uuid4())
    payload = {"prompt": workflow, "client_id": client_id}

    resp = httpx.post(
        f"{comfyui_url}/prompt",
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    prompt_id = resp.json()["prompt_id"]
    return prompt_id


def wait_for_image(comfyui_url: str, prompt_id: str, timeout: int = 180) -> bytes:
    """
    Poll /history/{prompt_id} until generation is done.
    Download and return image bytes (PNG).
    """
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{comfyui_url}/history/{prompt_id}", timeout=15)
            resp.raise_for_status()
            history = resp.json()

            if prompt_id not in history:
                time.sleep(2)
                continue

            job = history[prompt_id]
            outputs = job.get("outputs", {})

            # Find the SaveImage node output
            for node_id, node_output in outputs.items():
                images = node_output.get("images", [])
                if images:
                    img_info = images[0]
                    filename = img_info["filename"]
                    subfolder = img_info.get("subfolder", "")
                    img_type = img_info.get("type", "output")

                    params = {"filename": filename, "type": img_type}
                    if subfolder:
                        params["subfolder"] = subfolder

                    img_resp = httpx.get(
                        f"{comfyui_url}/view",
                        params=params,
                        timeout=30,
                    )
                    img_resp.raise_for_status()
                    return img_resp.content

            # Job exists in history but no images yet — still processing
            time.sleep(2)

        except httpx.RequestError as e:
            logger.debug("ComfyUI poll error for {}: {}", prompt_id, e)
            time.sleep(3)

    raise TimeoutError(f"Image generation timed out after {timeout}s for prompt_id={prompt_id}")
