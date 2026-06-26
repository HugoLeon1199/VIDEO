"""Vast.ai instance lifecycle manager.

Handles rent → wait_ready → destroy for on-demand GPU instances.
The instance runs a FastAPI worker (vast_worker/) that accepts /generate requests.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import requests
from loguru import logger

VAST_API_BASE = "https://console.vast.ai/api/v0"


@dataclass
class VastInstance:
    instance_id: int
    ssh_host: str
    ssh_port: int
    direct_port: Optional[int] = None   # mapped external port for FastAPI
    public_ipaddr: str = ""             # public IP for HTTP connections (≠ ssh_host)


class VastManager:
    def __init__(self, api_key: str, worker_port: int = 8888):
        self.api_key = api_key
        self.worker_port = worker_port
        self._headers = {"Authorization": f"Bearer {api_key}"}

    # ── Search & Rent ─────────────────────────────────────────────────────────

    def find_offer(
        self,
        min_vram_gb: int = 24,
        gpu_name: str = "",
        max_price_per_hour: float = 1.0,
        min_inet_down_mbps: int = 500,
    ) -> dict:
        """Find the cheapest available offer matching requirements.

        min_inet_down_mbps filters out slow-internet machines; at 500 Mbps a
        24 GB model download takes ~6 min instead of hours.
        """
        params = {
            "q": {
                "gpu_ram": {"gte": min_vram_gb * 1024},
                "rentable": {"eq": True},
                "num_gpus": {"eq": 1},
                "inet_down": {"gte": min_inet_down_mbps},
            }
        }
        if gpu_name:
            params["q"]["gpu_name"] = {"eq": gpu_name}

        resp = requests.get(
            f"{VAST_API_BASE}/bundles",
            headers=self._headers,
            params={"q": json.dumps(params["q"])},
            timeout=30,
        )
        resp.raise_for_status()
        offers = resp.json().get("offers", [])

        # V100 only supports CUDA ≤11.x; our image requires 11.8+ and the
        # container silently fails to start, so exclude it entirely.
        _GPU_BLACKLIST = {"Tesla V100"}

        eligible = [
            o for o in offers
            if o.get("dph_total", 999) <= max_price_per_hour
            and o.get("rentable", False)
            and (o.get("inet_down") or 0) >= min_inet_down_mbps
            and o.get("gpu_name", "") not in _GPU_BLACKLIST
        ]
        if not eligible:
            raise RuntimeError(
                f"No Vast.ai offers found: vram>={min_vram_gb}GB, "
                f"price<=${max_price_per_hour}/hr, inet_down>={min_inet_down_mbps}Mbps"
            )

        best = min(eligible, key=lambda o: o["dph_total"])
        logger.info(
            "Vast offer selected: id={} gpu={} vram={}GB ${:.3f}/hr inet_down={:.0f}Mbps",
            best["id"], best.get("gpu_name"), best.get("gpu_ram", 0) // 1024,
            best["dph_total"], best.get("inet_down") or 0,
        )
        return best

    def rent(
        self,
        offer_id: int,
        image: str,
        env_vars: dict[str, str] | None = None,
        extra_ports: list[int] | None = None,
        disk_gb: float = 40.0,
    ) -> VastInstance:
        """Rent an instance using a pre-built Docker image.

        We use runtype="args": it preserves the image's own ENTRYPOINT/CMD
        (our server.py runs exactly as built) and still provisions the
        "-p PORT:PORT" port mapping, without appending a "/ssh" or "/jupyter"
        suffix to the image name. Earlier runtypes broke here: "jupyter_direct"
        launched Vast's Jupyter wrapper instead of our CMD (status_msg
        ".../jupyter"), and "ssh_direct" tried to pull a nonexistent
        "<image>/ssh" derived image (pull access denied). With "args" the
        image's CMD already starts the worker, so no onstart is needed.
        """
        # Vast API env field is a JSON dict; port mappings go inside it as "-p HOST:CONTAINER" keys
        env_dict = dict(env_vars or {})
        env_dict[f"-p {self.worker_port}:{self.worker_port}"] = "1"
        for p in (extra_ports or []):
            env_dict[f"-p {p}:{p}"] = "1"

        payload = {
            "client_id": "me",
            "image": image,
            "disk": disk_gb,
            "env": env_dict,
            "runtype": "args",  # keep image CMD/ENTRYPOINT; map ports; no wrapper
        }

        resp = requests.put(
            f"{VAST_API_BASE}/asks/{offer_id}/",
            headers=self._headers,
            json=payload,
            timeout=30,
        )
        if not resp.ok:
            raise RuntimeError(f"Vast rent failed {resp.status_code}: {resp.text}")
        data = resp.json()
        instance_id = data.get("new_contract")
        if not instance_id:
            raise RuntimeError(f"Vast rent failed: {data}")

        logger.info("Vast instance rented: id={}", instance_id)
        return self._get_instance_info(instance_id)

    # ── Status & Wait ─────────────────────────────────────────────────────────

    def _get_instance_info(self, instance_id: int) -> VastInstance:
        resp = requests.get(
            f"{VAST_API_BASE}/instances/{instance_id}/",
            headers=self._headers,
            timeout=15,
        )
        resp.raise_for_status()
        inst = resp.json().get("instances", {})
        if isinstance(inst, list):
            inst = inst[0] if inst else {}

        ssh_host = inst.get("ssh_host", "")
        ssh_port = inst.get("ssh_port", 22)
        public_ipaddr = inst.get("public_ipaddr", "") or ssh_host

        # Find external port mapped to our worker port
        port_map = inst.get("ports", {}) or {}
        direct_port = None
        key = f"{self.worker_port}/tcp"
        if key in port_map and port_map[key]:
            direct_port = int(port_map[key][0]["HostPort"])

        return VastInstance(
            instance_id=instance_id,
            ssh_host=ssh_host,
            ssh_port=ssh_port,
            direct_port=direct_port,
            public_ipaddr=public_ipaddr,
        )

    def wait_until_running(self, instance_id: int, timeout: int = 300) -> VastInstance:
        """Poll until instance status == 'running' or 'created'.

        With runtype='jupyter_direct', Vast only marks an instance 'running' when
        its internal jupyter health check passes (port 8888). Our FastAPI worker
        uses port 8080 so the Vast health check never passes — the instance stays
        at 'created' forever even though the container is up. We accept 'created'
        as sufficient here; wait_worker_ready() then polls our /health endpoint.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            resp = requests.get(
                f"{VAST_API_BASE}/instances/{instance_id}/",
                headers=self._headers,
                timeout=15,
            )
            resp.raise_for_status()
            inst = resp.json().get("instances", {})
            if isinstance(inst, list):
                inst = inst[0] if inst else {}
            status = inst.get("actual_status") or ""
            logger.debug("Vast instance {} status: {}", instance_id, status)
            if status in ("running", "created"):
                port_map = inst.get("ports", {}) or {}
                key = f"{self.worker_port}/tcp"
                direct_port = None
                if key in port_map and port_map[key]:
                    direct_port = int(port_map[key][0]["HostPort"])
                public_ipaddr = inst.get("public_ipaddr", "") or inst.get("ssh_host", "")
                info = VastInstance(
                    instance_id=instance_id,
                    ssh_host=inst.get("ssh_host", ""),
                    ssh_port=inst.get("ssh_port", 22),
                    direct_port=direct_port,
                    public_ipaddr=public_ipaddr,
                )
                logger.info(
                    "Vast instance {}: ip={} ssh={} worker_port={}",
                    status, info.public_ipaddr, info.ssh_host, info.direct_port,
                )
                return info
            if status in ("exited", "dead", "error"):
                raise RuntimeError(f"Vast instance {instance_id} failed with status={status!r}")
            time.sleep(10)
        raise TimeoutError(f"Vast instance {instance_id} not running after {timeout}s")

    def wait_worker_ready(self, host: str, port: int, timeout: int = 600) -> None:
        """Poll /health until the FLUX model is loaded and ready to generate.

        The worker preloads the model in a background thread, so /health responds
        200 immediately (HTTP up) but reports model_loaded=False until the ~13GB
        model finishes streaming in. We wait for model_loaded=True; if the worker
        reports a load_error we fail fast instead of burning the full timeout.
        """
        url = f"http://{host}:{port}/health"
        deadline = time.time() + timeout
        http_up = False
        while time.time() < deadline:
            try:
                r = requests.get(url, timeout=5)
                if r.status_code == 200:
                    if not http_up:
                        http_up = True
                        logger.info("Vast worker HTTP up at {}:{} — waiting for model load", host, port)
                    body = r.json()
                    if body.get("load_error"):
                        raise RuntimeError(f"Vast worker model load failed: {body['load_error']}")
                    if body.get("model_loaded"):
                        logger.info("Vast worker ready (model loaded) at {}:{}", host, port)
                        return
            except (requests.RequestException, ValueError):
                pass
            time.sleep(5)
        raise TimeoutError(f"Vast worker at {host}:{port} not model-ready after {timeout}s")

    # ── Deploy worker via SSH ─────────────────────────────────────────────────

    def deploy_worker(
        self,
        instance: VastInstance,
        worker_dir: str = "vast_worker",
        hf_token: str = "",
    ) -> None:
        """SCP worker files to instance and start the FastAPI server."""
        dest = f"root@{instance.ssh_host}"
        ssh_opts = ["-o", "StrictHostKeyChecking=no", "-p", str(instance.ssh_port)]

        # Upload worker directory
        logger.info("Uploading worker files to Vast instance...")
        subprocess.run(
            ["scp", *ssh_opts, "-r", worker_dir, f"{dest}:/workspace/vast_worker"],
            check=True,
        )

        # Install deps + start server in background
        hf_export = f"export HF_TOKEN={hf_token} && " if hf_token else ""
        cmd = (
            f"cd /workspace && "
            f"pip install -q fastapi uvicorn diffusers transformers accelerate "
            f"safetensors pillow torch && "
            f"{hf_export}"
            f"nohup python vast_worker/server.py --port {self.worker_port} "
            f"> /workspace/worker.log 2>&1 &"
        )
        subprocess.run(
            ["ssh", *ssh_opts, dest, cmd],
            check=True,
        )
        logger.info("Worker started on Vast instance")

    # ── Destroy ───────────────────────────────────────────────────────────────

    def destroy(self, instance_id: int) -> None:
        """Terminate and delete the instance."""
        resp = requests.delete(
            f"{VAST_API_BASE}/instances/{instance_id}/",
            headers=self._headers,
            timeout=15,
        )
        if resp.status_code in (200, 204):
            logger.info("Vast instance {} destroyed", instance_id)
        else:
            logger.warning("Vast destroy returned {}: {}", resp.status_code, resp.text)
