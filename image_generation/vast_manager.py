"""Vast.ai instance lifecycle manager.

Handles rent → wait_ready → destroy for on-demand GPU instances.
The instance runs a FastAPI worker (vast_worker/) that accepts /generate requests.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import requests
from loguru import logger

VAST_API_BASE = "https://console.vast.ai/api/v0"
# v1 is required for listing instances — v0 /instances/ now returns 410 Gone,
# which used to parse as "0 instances" and stranded a billing machine for ~16h.
VAST_API_V1 = "https://console.vast.ai/api/v1"


@dataclass
class VastInstance:
    instance_id: int
    ssh_host: str
    ssh_port: int
    direct_port: Optional[int] = None   # mapped external port for FastAPI
    public_ipaddr: str = ""             # public IP for HTTP connections (≠ ssh_host)


class VastManager:
    # Class-level registry of every instance id this process has rented, so cleanup
    # can destroy them BY ID even when the list/destroy_all API returns 410/403 and
    # falsely reports "0 instances" (that false-clean left 3 boxes billing 14-17h =
    # $4.79). A persistent on-disk log (rented_instances.log) survives a crash too.
    _RENTED_LOG = os.path.join(os.path.dirname(__file__), "rented_instances.log")
    rented_ids: set[int] = set()

    # Real measured warm seconds-per-image (20 steps, 1024×576, FLUX 8-bit). Add a
    # GPU here once you've clocked it — used by find_offer to estimate job time
    # accurately (dlperf/TFLOPS only used as a fallback for unknown cards).
    MEASURED_SPEED: dict[str, float] = {
        "RTX 3090": 19.0,   # measured: 50-image batch, ~18.5-19s/img
    }
    _BASELINE_TFLOPS = 35.5   # RTX 3090 FP16 TFLOPS — the card MEASURED_SPEED is keyed to

    def __init__(self, api_key: str, worker_port: int = 8888):
        self.api_key = api_key
        self.worker_port = worker_port
        self._headers = {"Authorization": f"Bearer {api_key}"}

    def _record_rented(self, instance_id: int) -> None:
        """Remember a rented id (in-memory + on-disk) for guaranteed cleanup."""
        VastManager.rented_ids.add(int(instance_id))
        try:
            with open(self._RENTED_LOG, "a", encoding="utf-8") as fh:
                fh.write(f"{int(instance_id)}\n")
        except Exception:  # noqa: BLE001 — best-effort persistence
            pass

    # ── Search & Rent ─────────────────────────────────────────────────────────

    def find_offer(
        self,
        min_vram_gb: int = 24,
        max_vram_gb: int = 200,  # no real VRAM cap — any card ≥24GB is fine. The arch
                                 # gate (max_compute_cap) guarantees it runs FLUX, and
                                 # _true_cost (GPU + download + upload + storage) picks
                                 # the cheapest TOTAL, so a big-VRAM card only wins if
                                 # it's genuinely cheaper overall (e.g. free download).
        gpu_name: str = "",
        max_price_per_hour: float = 1.0,
        min_inet_down_mbps: int = 500,
        min_reliability: float = 0.98,
        require_verified: bool = True,
        min_direct_ports: int = 1,
        min_disk_gb: float = 60.0,
        min_cuda: float = 12.2,
        min_tflops: float = 25.0,    # reject cards too slow for FLUX (FP16 TFLOPS).
                                     # RTX 3090=35, 4090=81; below ~25 the per-image
                                     # time balloons and a "cheap $/hr" card ends up
                                     # costing MORE in total GPU-hours.
        max_compute_cap: int = 900,  # torch 2.4.1 supports sm_50..sm_90 → compute_cap
                                     # ≤900. Reject sm_100 (B200=1000) & sm_120 (RTX
                                     # 50xx / RTX PRO 4000+) — they boot but torch
                                     # refuses them, so the model downloads (costs $)
                                     # then NEVER generates. This is the robust gate;
                                     # the name blacklist below is just belt-and-braces.
        on_demand_only: bool = True,
        prefer_datacenter: bool = True,
        max_inet_cost_per_gb: float = 0.005,   # HARD CAP on download $/GB
        preferred_inet_cost_per_gb: float = 0.003,  # tie-breaker only, NOT a hard filter
        expected_download_gb: float = 37.0,    # model(34) + docker/setup(3) pulled per rental
        expected_upload_gb: float = 2.0,       # output images sent back
        n_images: int = 100,                   # batch size — drives the GPU-hours estimate
        sec_per_image: float = 19.0,           # measured warm gen time (fallback)
        boot_seconds: float = 120.0,           # fixed boot/init before model download
        allowed_countries: set | None = None,
        exclude_machine_ids: set | None = None,
    ) -> dict:
        """Find the cheapest *good* offer matching requirements.

        "Good, not just cheap": we filter out machines that won't actually run
        BEFORE renting, so we never pay to boot + download the model on a box that
        then fails. The two decisive checks (per Vast docs):
          - require_verified: only "verified" hosts — Vast has already tested them
            end to end (datacenter GPUs, stable, many open ports). Unverified hosts
            "may offer bad connections and may be unavailable once rebooted".
          - min_direct_ports: hosts with too few open ports can't publish the
            worker port mapping — that was the exact "running but no port mapping"
            failure that killed earlier batches. Vast requires >=3, recommends 100.
        We then also drop low-reliability / slow-internet machines and pick the
        cheapest of what survives.

        - min_reliability: drop hosts below this uptime score (0.95 = 95%).
        - min_inet_down_mbps: drop slow machines; faster = model downloads in
          minutes, not hours, and far fewer load timeouts.
        - exclude_machine_ids: machine_ids already rented in this batch, so the
          N parallel instances land on N distinct physical hosts.
        """
        # gpu_ram cap: lte uses a small margin (24GB cards report ~24576MB, but some
        # read slightly under, so allow up to max_vram_gb+1 GB to not exclude a real
        # 24GB box while still rejecting 32/40/48GB cards).
        params = {
            "q": {
                "gpu_ram": {"gte": min_vram_gb * 1024, "lte": (max_vram_gb + 1) * 1024},
                "rentable": {"eq": True},
                "num_gpus": {"eq": 1},
                "inet_down": {"gte": min_inet_down_mbps},
                "disk_space": {"gte": min_disk_gb},
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

        # Exclude GPUs that look fine by cuda_max_good but don't actually run our
        # FLUX/bf16 image well (old pre-Ampere architectures: no proper bf16, the
        # container fails or is painfully slow). cuda_max_good lies for these —
        # e.g. V100 advertises CUDA 13 but the worker silently fails to start.
        # This is the "don't pick a lemon GPU and then cry" guard.
        # Two reasons a GPU is blacklisted:
        #  - pre-Ampere (no proper bf16, slow/failing): old Teslas, Quadros, TITANs.
        #  - RTX 50xx (Blackwell, sm_120): NOT supported by our torch 2.4.1 wheel
        #    (it only covers sm_50..sm_90). The card boots but torch refuses it:
        #    "CUDA capability sm_120 is not compatible". Proven by an RTX 5070 Ti
        #    that loaded the worker but could never generate. Use Ada (RTX 40xx /
        #    RTX 6000 Ada) or Ampere (RTX 30xx) instead — those generate fine.
        #    (To use 50xx later, rebuild the worker on torch >= 2.7 / cu124+.)
        _GPU_BLACKLIST = {
            "Tesla V100", "Tesla P100", "Tesla P40", "Tesla K80",
            "Tesla T4", "Quadro RTX 6000", "Quadro RTX 8000", "TITAN V",
            "TITAN Xp", "GTX 1080 Ti",
            "RTX 5090", "RTX 5080", "RTX 5070 Ti", "RTX 5070", "RTX 5060 Ti", "RTX 5060",
        }
        exclude_machine_ids = exclude_machine_ids or set()

        # Geo gate: HuggingFace + Docker Hub origin storage lives in US/EU AWS/GCP
        # datacenters. A machine in China/Asia pulls the 24GB FLUX model across the
        # Pacific — the advertised inet_down (e.g. 800Mbps) is the LAN link, but the
        # real HF egress over throttled subsea cables drops to a few MB/s and the
        # download hangs (exactly what stranded the China box 120.238.149.205 here).
        # Restricting to US/EU keeps the machine on the same network backbone as the
        # model origin, so a 24GB pull finishes in seconds-to-minutes, not never.
        if allowed_countries is None:
            allowed_countries = {
                "US", "CA",                                      # North America
                "GB", "DE", "NL", "FR", "FI", "SE", "NO", "IE",  # West/North EU
                "CZ", "PL", "BG", "HU", "RO", "AT", "CH", "BE", "ES", "IT", "PT",
            }

        def _country(o: dict) -> str:
            geo = o.get("geolocation") or ""
            m = geo.rsplit(",", 1)
            return m[-1].strip().upper() if m else ""

        def _reliability(o: dict) -> float:
            return o.get("reliability2", o.get("reliability", 0)) or 0

        # Every check here answers "will this machine actually do the work?" —
        # we apply them BEFORE renting, so we never pay to boot a box that fails.
        eligible = [
            o for o in offers
            if o.get("dph_total", 999) <= max_price_per_hour
            and o.get("rentable", False)
            # VRAM band: ≥min (fits 8-bit) AND ≤max (don't overpay for 48GB cards).
            and (o.get("gpu_ram") or 0) >= min_vram_gb * 1024
            and (o.get("gpu_ram") or 0) <= (max_vram_gb + 1) * 1024
            and (o.get("inet_down") or 0) >= min_inet_down_mbps
            and (o.get("disk_space") or 0) >= min_disk_gb
            and _reliability(o) >= min_reliability
            and (o.get("cuda_max_good") or 0) >= min_cuda
            # Arch gate: torch 2.4.1 only supports up to sm_90. compute_cap is sm×10,
            # so ≤900. This rejects Blackwell (sm_120) & B200 (sm_100) reliably —
            # the real "card must run our FLUX" check, independent of the name list.
            and 800 <= (o.get("compute_cap") or 0) <= max_compute_cap
            # Speed gate: reject cards too slow for FLUX (FP16 TFLOPS). A slow card
            # is a false bargain — cheap $/hr but many more GPU-hours = higher total.
            and (o.get("total_flops") or 0) >= min_tflops
            and (not require_verified or o.get("verification") == "verified")
            and (o.get("direct_port_count") or 0) >= min_direct_ports
            # On-demand only: bid/interruptible machines can be yanked mid-batch.
            and (not on_demand_only or not o.get("is_bid", False))
            # Bandwidth fee: SOME hosts charge $/GB for download. The worker pulls
            # the ~30GB FLUX model every rental, so a host at $0.0156/GB silently
            # adds ~$0.47 PER RUN — that's why a "$0.25/hr" box actually cost ~$1.86/hr
            # for a 28-min job. Reject hosts with a high inet_down_cost; the good
            # datacenter boxes charge $0 or a tiny amount.
            and (o.get("inet_down_cost") or 0) <= max_inet_cost_per_gb
            and o.get("gpu_name", "") not in _GPU_BLACKLIST
            and _country(o) in allowed_countries
            and o.get("machine_id") not in exclude_machine_ids
        ]
        if not eligible:
            raise RuntimeError(
                f"No good Vast.ai offers: vram>={min_vram_gb}GB disk>={min_disk_gb}GB, "
                f"price<=${max_price_per_hour}/hr, inet_down>={min_inet_down_mbps}Mbps, "
                f"reliability>={min_reliability}, verified={require_verified}, "
                f"ports>={min_direct_ports}, cuda>={min_cuda}, on_demand={on_demand_only}, "
                f"countries={sorted(allowed_countries)}"
            )

        # PREFER real datacenters (hosting_type==1, the 🏢 icon) over home hosts
        # (hosting_type==0, the 🏠 icon). A home host can DESTROY your instance the
        # moment they decide the price is too low for them ("bẻ kèo" — they manually
        # kill it); datacenters run on Vast's automation and never hand-check a single
        # box. So: if any datacenter box passes the gates, choose the cheapest of
        # THOSE; only fall back to home hosts when no datacenter is available in range.
        # (All survivors already passed reliability/inet/on-demand/disk/arch gates.)
        pool = eligible
        if prefer_datacenter:
            datacenters = [o for o in eligible if o.get("hosting_type") == 1]
            if datacenters:
                pool = datacenters
                logger.info("Vast: {} datacenter (🏢) offers in range — preferring those over home hosts",
                            len(datacenters))
            else:
                logger.warning("Vast: no datacenter offers in range — falling back to home hosts "
                               "(higher risk of host-initiated destroy)")

        # Pick by TRUE total cost of THIS batch, not $/hr alone. The invoice proved
        # download dominates (a $0.25/hr box at $0.012/GB cost $0.86 for 28 min,
        # 85% download). But cost depends on the batch: 3 images = download-bound,
        # 400 = GPU-bound; a slow card with cheap bandwidth can lose. So estimate
        # the actual wall-time per offer:
        #   job_hours = boot + (download_gb / this box's link) + n_images × sec/image
        #   cost = gpu×job_hours + download_gb×$/GB + upload_gb×$/GB + storage
        # SPEED: dlperf is a bad proxy for FLUX inference (it weights training/mixed
        # workloads — an A100 shows high dlperf but low FP16 TFLOPS and draws slowly;
        # an A40 looked fine on dlperf but rendered ~3-4× slower than a 4090 in
        # practice). So we estimate sec/image from:
        #   1) MEASURED_SPEED — real per-GPU times we've actually clocked (most exact),
        #   2) else scale the baseline by FP16 TFLOPS (total_flops) — far better than
        #      dlperf for diffusion, since denoising is compute-bound.
        def _job_hours(o: dict) -> float:
            name = o.get("gpu_name", "")
            spi = self.MEASURED_SPEED.get(name)
            if spi is None:
                tflops = o.get("total_flops") or self._BASELINE_TFLOPS
                spi = sec_per_image * (self._BASELINE_TFLOPS / max(tflops, 1.0))
            link_mbps = max(o.get("inet_down") or 1.0, 1.0)
            dl_sec = expected_download_gb * 8 * 1024 / link_mbps          # GB→Mb / Mbps = s
            return (boot_seconds + dl_sec + n_images * spi) / 3600.0
        def _true_cost(o: dict) -> float:
            jh = _job_hours(o)
            gpu = (o.get("dph_total") or 0) * jh
            dl = expected_download_gb * (o.get("inet_down_cost") or 0)
            ul = expected_upload_gb * (o.get("inet_up_cost") or 0)
            # storage is billed per hour the box exists; default search assumes 5GB,
            # but we rent ~60GB → add it explicitly. storage_cost is $/GB/month.
            store = (o.get("storage_cost") or 0) * min_disk_gb / 720.0 * jh
            return gpu + dl + ul + store
        # Rank by true cost. Tie-break (costs within ~$0.02): prefer a host at or
        # below preferred_inet_cost_per_gb — so a near-tie goes to the cheaper-
        # bandwidth box, WITHOUT hard-filtering the pool to that threshold.
        def _sort_key(o: dict):
            tc = _true_cost(o)
            bucket = round(tc / 0.02)  # group near-equal costs
            below_pref = 0 if (o.get("inet_down_cost") or 0) <= preferred_inet_cost_per_gb else 1
            return (bucket, below_pref, o.get("inet_down_cost") or 0, tc)
        best = min(pool, key=_sort_key)
        host_kind = "datacenter🏢" if best.get("hosting_type") == 1 else "home🏠"
        logger.info(
            "Vast offer selected: id={} machine={} gpu={} vram={}GB ${:.3f}/hr "
            "inet_down={:.0f}Mbps reliability={:.3f} {} verified={} ports={} geo={}",
            best["id"], best.get("machine_id"), best.get("gpu_name"),
            best.get("gpu_ram", 0) // 1024, best["dph_total"],
            best.get("inet_down") or 0, _reliability(best), host_kind,
            best.get("verification"), best.get("direct_port_count"),
            best.get("geolocation"),
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

        self._record_rented(instance_id)  # track for guaranteed cleanup
        logger.info("Vast instance rented: id={}", instance_id)
        return self._get_instance_info(instance_id)

    # ── Status & Wait ─────────────────────────────────────────────────────────

    def _get_instance_info(self, instance_id: int) -> VastInstance:
        # Retry through transient Vast rate-limits (429) instead of crashing.
        for attempt in range(6):
            resp = requests.get(
                f"{VAST_API_BASE}/instances/{instance_id}/",
                headers=self._headers,
                timeout=15,
            )
            if resp.status_code == 429:
                logger.warning("Vast 429 on instance info — backing off 20s")
                time.sleep(20)
                continue
            break
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
            # 429 = Vast rate-limit (many calls across a long session). Don't crash
            # the whole run — back off and retry; the instance is still booting.
            if resp.status_code == 429:
                logger.warning("Vast 429 rate-limit on status poll — backing off 20s")
                time.sleep(20)
                continue
            resp.raise_for_status()
            inst = resp.json().get("instances", {})
            if isinstance(inst, list):
                inst = inst[0] if inst else {}
            # Vast can return "instances": null right after creation — treat as
            # "not ready yet" and keep polling instead of crashing with NoneType.
            if not inst:
                logger.debug("Vast instance {} not in API yet — waiting", instance_id)
                time.sleep(15)
                continue
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
            time.sleep(15)  # gentle poll to avoid Vast API rate-limit over a session
        raise TimeoutError(f"Vast instance {instance_id} not running after {timeout}s")

    def wait_for_port(self, instance_id: int, timeout: int = 180) -> VastInstance:
        """Poll until Vast has mapped the worker port (direct_port is set).

        After an instance reaches 'created'/'running', Vast often needs another
        10-60s to publish the host port mapping. The old code failed immediately
        when direct_port was still None, killing perfectly good instances. This
        waits for the mapping instead.
        """
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            info = self._get_instance_info(instance_id)
            last = info
            if info.direct_port:
                logger.info(
                    "Vast instance {} port mapped: {}:{}",
                    instance_id, info.public_ipaddr, info.direct_port,
                )
                return info
            logger.debug("Vast instance {} waiting for port mapping...", instance_id)
            time.sleep(5)
        raise TimeoutError(
            f"Vast instance {instance_id} got no port mapping after {timeout}s "
            f"(ip={last.public_ipaddr if last else '?'})"
        )

    def list_instances(self) -> list[dict]:
        """Return ALL instances for this account (raw dicts), robust to API shape.

        IMPORTANT: uses the v1 endpoint. The old v0 `/instances/` is DEPRECATED
        and now returns HTTP 410 — which silently parsed as "0 instances" and let
        a machine stay alive billing for ~16h. v1 reports the real list.
        """
        resp = requests.get(
            f"{VAST_API_V1}/instances/",
            headers=self._headers,
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json().get("instances", [])
        if isinstance(data, dict):
            data = [data]
        return data or []

    def destroy_all(self, known_ids: list[int] | None = None, verify_rounds: int = 3) -> None:
        """Safety net: destroy every instance on the account, then verify.

        The Vast API sometimes reports zero instances while machines are still
        alive and billing (this stranded a V100 for ~16h). So we (a) always try
        the explicitly known ids from this run, (b) also list+destroy anything
        the API does report, and (c) re-query several times to catch stragglers
        the API was briefly hiding.
        """
        # Seed from EVERY source we trust more than the (often-broken) list API:
        # explicit known_ids, this process's in-memory registry, and the on-disk log
        # of everything ever rented (survives a crash). These get destroyed by id
        # regardless of what list_instances returns.
        ids: set[int] = set(known_ids or [])
        ids |= set(VastManager.rented_ids)
        try:
            if os.path.exists(self._RENTED_LOG):
                with open(self._RENTED_LOG, encoding="utf-8") as fh:
                    ids |= {int(x) for x in fh.read().split() if x.strip().isdigit()}
        except Exception:  # noqa: BLE001
            pass
        for _ in range(verify_rounds):
            try:
                for inst in self.list_instances():
                    if inst.get("id"):
                        ids.add(int(inst["id"]))
            except Exception as e:  # noqa: BLE001
                logger.warning("Vast list_instances failed: {}", e)
            for iid in list(ids):
                try:
                    self.destroy(iid)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Vast destroy {} failed: {}", iid, e)
            time.sleep(4)
            # Verify each id is really gone by GETting it (404 = destroyed). Don't
            # rely on list_instances alone — it's the thing that lied before.
            still_alive = []
            for iid in list(ids):
                try:
                    resp = requests.get(f"{VAST_API_BASE}/instances/{iid}/",
                                        headers=self._headers, timeout=15)
                    # ONLY 404 = confirmed destroyed. Per Vast destroy/show-instance
                    # docs: 401/403 = auth error (NOT clean — could still be billing),
                    # 410/429/5xx = transient/ambiguous (NOT clean). Treat anything
                    # that isn't a clear 404 as still-alive and keep retrying — the
                    # old "410 == clean" assumption is exactly what stranded $4.79.
                    if resp.status_code == 404:
                        ids.discard(iid)  # confirmed gone
                        continue
                    inst = resp.json().get("instances") if resp.ok else None
                    if isinstance(inst, list):
                        inst = inst[0] if inst else None
                    if resp.ok and not inst:
                        ids.discard(iid)  # 200 but empty body = gone
                    else:
                        still_alive.append(iid)  # 200-alive / 401 / 403 / 410 / 5xx
                except Exception:  # noqa: BLE001 — network blip: assume still alive, retry
                    still_alive.append(iid)
            if not still_alive:
                # clear the on-disk log so old ids don't linger
                try:
                    open(self._RENTED_LOG, "w").close()
                except Exception:  # noqa: BLE001
                    pass
                logger.info("Vast: all instances destroyed and verified clean (by id)")
                return
            logger.warning("Vast: still alive {} — retrying destroy", still_alive)
        logger.error(
            "Vast: instances may still be alive after cleanup — CHECK DASHBOARD: "
            "https://cloud.vast.ai/instances/"
        )

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
