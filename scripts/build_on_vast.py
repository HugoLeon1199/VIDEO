"""Build the vast-flux image (with embedded FLUX model) ON a rented Vast machine.

Why: building locally re-downloads the 13GB model from HuggingFace at the local
machine's slow speed (~12 MB/s → ~30 min) and then pushes 23GB up at the local
upload speed. A Vast machine has fast datacenter internet, so HF download and the
Docker Hub push are both far quicker. We rent a cheap CPU box, build there, push
to Docker Hub, then destroy it.

Requires in .env:
  VAST_API_KEY        – to rent/destroy
  VAST_HF_TOKEN       – HuggingFace token (FLUX.1-dev is gated)
  DOCKERHUB_USER      – e.g. leon1199
  DOCKERHUB_TOKEN     – Docker Hub Personal Access Token (Read & Write)

Usage:
  python scripts/build_on_vast.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
SSH_KEY = os.path.expanduser("~/.ssh/vast_build")
IMAGE = "leon1199/vast-flux:latest"
VAST_API = "https://console.vast.ai/api/v0"


def _load_env() -> dict:
    env = {}
    for line in (ROOT / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip()
    return env


def _ssh(host: str, port: int, cmd: str, check: bool = True) -> int:
    full = [
        "ssh", "-i", SSH_KEY,
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-p", str(port), f"root@{host}", cmd,
    ]
    return subprocess.run(full, check=check).returncode


def main() -> None:
    env = _load_env()
    api_key = env.get("VAST_API_KEY", "")
    hf_token = env.get("VAST_HF_TOKEN") or env.get("HF_TOKEN", "")
    dh_user = env.get("DOCKERHUB_USER", "leon1199")
    dh_token = env.get("DOCKERHUB_TOKEN", "")
    if not (api_key and hf_token and dh_token):
        sys.exit("Missing VAST_API_KEY / HF token / DOCKERHUB_TOKEN in .env")

    h = {"Authorization": f"Bearer {api_key}"}

    # 1) Find a cheap box with fast internet and enough disk for a 23GB image.
    q = {
        "rentable": {"eq": True},
        "num_gpus": {"eq": 1},
        "inet_down": {"gte": 1000},
        "disk_space": {"gte": 60},
    }
    import json as _json
    offers = requests.get(
        f"{VAST_API}/bundles/", headers=h,
        params={"q": _json.dumps(q)}, timeout=30,
    ).json().get("offers", [])
    offers = [o for o in offers if o.get("dph_total", 9) <= 0.4]
    if not offers:
        sys.exit("No suitable Vast offer found")
    best = min(offers, key=lambda o: o["dph_total"])
    print(f"Offer {best['id']}: {best.get('gpu_name')} ${best['dph_total']:.3f}/hr "
          f"inet {best.get('inet_down'):.0f}Mbps")

    # 2) Rent it (ssh_direct so we can SSH in and run docker build).
    rent = requests.put(
        f"{VAST_API}/asks/{best['id']}/", headers=h,
        json={"client_id": "me", "image": "docker:dind", "disk": 70,
              "runtype": "ssh_direct"},
        timeout=30,
    ).json()
    iid = rent.get("new_contract")
    if not iid:
        sys.exit(f"Rent failed: {rent}")
    print(f"Rented instance {iid}")

    try:
        # 3) Wait for SSH to come up.
        host = port = None
        deadline = time.time() + 600
        while time.time() < deadline:
            inst = requests.get(f"{VAST_API}/instances/{iid}/", headers=h, timeout=20).json()
            inst = inst.get("instances", {})
            if isinstance(inst, list):
                inst = inst[0] if inst else {}
            status = inst.get("actual_status", "")
            if status == "running" and inst.get("ssh_host"):
                host, port = inst["ssh_host"], inst["ssh_port"]
                # SSH may still need a few seconds
                time.sleep(15)
                break
            if status in ("exited", "error"):
                sys.exit(f"Instance failed: {inst.get('status_msg')}")
            print(f"  status={status}...")
            time.sleep(10)
        if not host:
            sys.exit("Instance never became reachable")
        print(f"SSH ready at {host}:{port}")

        # 4) Upload build context.
        scp = [
            "scp", "-i", SSH_KEY,
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-P", str(port),
            str(ROOT / "vast_worker" / "Dockerfile"),
            str(ROOT / "vast_worker" / "requirements.txt"),
            str(ROOT / "vast_worker" / "server.py"),
            f"root@{host}:/root/",
        ]
        subprocess.run(scp, check=True)
        print("Uploaded build context")

        # 5) Build + push on the remote box.
        remote = (
            "set -e; cd /root; "
            f"echo '{hf_token}' > hf_token.txt; "
            f"echo '{dh_token}' | docker login -u {dh_user} --password-stdin; "
            f"DOCKER_BUILDKIT=1 docker build --secret id=hf_token,src=hf_token.txt "
            f"-t {IMAGE} .; "
            f"docker push {IMAGE}; "
            "rm -f hf_token.txt; echo BUILD_PUSH_DONE"
        )
        rc = _ssh(host, port, remote, check=False)
        if rc != 0:
            sys.exit(f"Remote build/push failed (rc={rc})")
        print("✅ Image built and pushed from Vast")

    finally:
        # 6) Always destroy to stop billing.
        d = requests.delete(f"{VAST_API}/instances/{iid}/", headers=h, timeout=20)
        print(f"Destroyed instance {iid}: {d.status_code}")


if __name__ == "__main__":
    main()
