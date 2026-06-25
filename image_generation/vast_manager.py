"""Vast.ai instance lifecycle manager.

Handles rent → wait_ready → destroy for on-demand GPU instances.
The instance runs a FastAPI worker (vast_worker/) that accepts /generate requests.
"""

from __future__ import annotations

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
    direct_port: Optional[int] = None  # mapped port for FastAPI (default 8080)


class VastManager:
    def __init__(self, api_key: str, worker_port: int = 8080):
        self.api_key = api_key
        self.worker_port = worker_port
        self._headers = {"Authorization": f"Bearer {api_key}"}

    # ── Search & Rent ─────────────────────────────────────────────────────────

    def find_offer(
        self,
        min_vram_gb: int = 24,
        gpu_name: str = "",
        max_price_per_hour: float = 1.0,
    ) -> dict:
        """Find the cheapest available offer matching requirements."""
        params = {
            "q": {
                "gpu_ram": {"gte": min_vram_gb * 1024},
                "rentable": {"eq": True},
                "num_gpus": {"eq": 1},
            }
        }
        if gpu_name:
            params["q"]["gpu_name"] = {"eq": gpu_name}

        resp = requests.get(
            f"{VAST_API_BASE}/bundles",
            headers=self._headers,
            params={"q": str(params["q"])},
            timeout=30,
        )
        resp.raise_for_status()
        offers = resp.json().get("offers", [])

        eligible = [
            o for o in offers
            if o.get("dph_total", 999) <= max_price_per_hour
            and o.get("rentable", False)
        ]
        if not eligible:
            raise RuntimeError(
                f"No Vast.ai offers found: vram>={min_vram_gb}GB, price<=${max_price_per_hour}/hr"
            )

        best = min(eligible, key=lambda o: o["dph_total"])
        logger.info(
            "Vast offer selected: id={} gpu={} vram={}GB ${:.3f}/hr",
            best["id"], best.get("gpu_name"), best.get("gpu_ram", 0) // 1024, best["dph_total"],
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
        """Rent an instance and return its connection info."""
        onstart = (
            f"cd /workspace && pip install -q fastapi uvicorn diffusers transformers "
            f"accelerate safetensors pillow torch && "
            f"python vast_worker/server.py --port {self.worker_port} &"
        )

        # Build port mapping: {internal: null} asks Vast to auto-assign external port
        ports: dict[str, None] = {f"{self.worker_port}/tcp": None}
        if extra_ports:
            for p in extra_ports:
                ports[f"{p}/tcp"] = None

        env_str = " ".join(f"-e {k}={v}" for k, v in (env_vars or {}).items())

        payload = {
            "client_id": "me",
            "image": image,
            "disk": disk_gb,
            "onstart": onstart,
            "env": env_str,
            "ports": " ".join(f"{p}" for p in ports),
            "runtype": "ssh",
        }

        resp = requests.put(
            f"{VAST_API_BASE}/asks/{offer_id}/",
            headers=self._headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
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

        ssh_host = inst.get("ssh_host", "")
        ssh_port = inst.get("ssh_port", 22)

        # Find external port mapped to our worker port
        port_map = inst.get("ports", {})
        direct_port = None
        key = f"{self.worker_port}/tcp"
        if key in port_map and port_map[key]:
            direct_port = int(port_map[key][0]["HostPort"])

        return VastInstance(
            instance_id=instance_id,
            ssh_host=ssh_host,
            ssh_port=ssh_port,
            direct_port=direct_port,
        )

    def wait_until_running(self, instance_id: int, timeout: int = 300) -> VastInstance:
        """Poll until instance status == 'running'."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            resp = requests.get(
                f"{VAST_API_BASE}/instances/{instance_id}/",
                headers=self._headers,
                timeout=15,
            )
            resp.raise_for_status()
            inst = resp.json().get("instances", {})
            status = inst.get("actual_status", "")
            logger.debug("Vast instance {} status: {}", instance_id, status)
            if status == "running":
                info = self._get_instance_info(instance_id)
                logger.info(
                    "Vast instance running: {}:{} worker_port={}",
                    info.ssh_host, info.ssh_port, info.direct_port,
                )
                return info
            time.sleep(10)
        raise TimeoutError(f"Vast instance {instance_id} not running after {timeout}s")

    def wait_worker_ready(self, host: str, port: int, timeout: int = 300) -> None:
        """Poll /health until FastAPI worker responds."""
        url = f"http://{host}:{port}/health"
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = requests.get(url, timeout=5)
                if r.status_code == 200:
                    logger.info("Vast worker ready at {}:{}", host, port)
                    return
            except requests.RequestException:
                pass
            time.sleep(5)
        raise TimeoutError(f"Vast worker at {host}:{port} not ready after {timeout}s")

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
