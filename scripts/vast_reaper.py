"""Independent Vast.ai orphan reaper — destroys instances that outlive their lease.

WHY this exists: the in-process cleanup (destroy_all in generate_images.py) only
runs while the pipeline runs. If the PC loses power, the network drops, or the
process is killed and you don't re-run for a day, a rented box keeps billing. The
invoice on 2026-06-26 showed exactly this: 3 boxes ran 14-17h ($4.79) after a prior
session never cleaned them up.

This script is meant to run INDEPENDENTLY of the pipeline, on a schedule:
  - Windows Task Scheduler every 5 min:
      python d:\\CODE\\VIDEO\\YOUTUBE\\scripts\\vast_reaper.py
  - (Better, survives PC-off) a cloud cron / GitHub Actions calling the Vast API.

Logic:
  1. Collect candidate instance ids from rented_instances.log + the live API list.
  2. For each, read its age (start_date) and current status.
  3. Destroy any instance older than MAX_LEASE_MINUTES (still running/billing).
  4. Verify by GET: only 404 means gone (401/403/410 are NOT "clean").

Safe to run anytime: it never touches instances younger than the lease, so a healthy
batch in progress is left alone (set MAX_LEASE_MINUTES > your longest batch).
"""

from __future__ import annotations

import os
import sys
import time

import requests

# Make `config` and the vast manager importable when run as a standalone script.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import config as cfg  # noqa: E402
from image_generation.vast_manager import VastManager, VAST_API_BASE  # noqa: E402

_HEADERS = {"Authorization": f"Bearer {cfg.VAST_API_KEY}"}
_RENTED_LOG = VastManager._RENTED_LOG


def _candidate_ids() -> set[int]:
    """All ids worth checking: the on-disk rental log + whatever the API lists."""
    ids: set[int] = set()
    # on-disk log (survives a crash / PC-off)
    try:
        if os.path.exists(_RENTED_LOG):
            with open(_RENTED_LOG, encoding="utf-8") as fh:
                ids |= {int(x) for x in fh.read().split() if x.strip().isdigit()}
    except Exception as e:  # noqa: BLE001
        print(f"[reaper] read log error: {e}")
    # live API list (v1)
    try:
        mgr = VastManager(api_key=cfg.VAST_API_KEY, worker_port=cfg.VAST_WORKER_PORT)
        for inst in mgr.list_instances():
            if inst.get("id"):
                ids.add(int(inst["id"]))
    except Exception as e:  # noqa: BLE001
        print(f"[reaper] list_instances error: {e}")
    return ids


def _instance(iid: int) -> dict | None:
    """GET one instance; return its dict, or None if 404 (gone)."""
    try:
        resp = requests.get(f"{VAST_API_BASE}/instances/{iid}/", headers=_HEADERS, timeout=15)
        if resp.status_code == 404:
            return None
        if not resp.ok:
            # 401/403/410/5xx — ambiguous, treat as "still there, recheck next run"
            print(f"[reaper] GET {iid} -> HTTP {resp.status_code} (ambiguous, leaving)")
            return {"_ambiguous": True}
        inst = resp.json().get("instances")
        if isinstance(inst, list):
            inst = inst[0] if inst else None
        return inst
    except Exception as e:  # noqa: BLE001
        print(f"[reaper] GET {iid} error: {e}")
        return {"_ambiguous": True}


def main() -> int:
    if not cfg.VAST_API_KEY:
        print("[reaper] VAST_API_KEY not set — nothing to do")
        return 0
    lease_sec = cfg.MAX_LEASE_MINUTES * 60
    now = time.time()
    mgr = VastManager(api_key=cfg.VAST_API_KEY, worker_port=cfg.VAST_WORKER_PORT)

    ids = _candidate_ids()
    if not ids:
        print("[reaper] no candidate instances — clean")
        return 0

    reaped, kept, gone = [], [], []
    for iid in sorted(ids):
        inst = _instance(iid)
        if inst is None:
            gone.append(iid)
            continue
        if inst.get("_ambiguous"):
            kept.append((iid, "ambiguous"))
            continue
        start = inst.get("start_date") or inst.get("created_at") or 0
        age_min = (now - start) / 60 if start else None
        status = inst.get("actual_status") or "?"
        if age_min is None:
            # can't determine age → leave it, recheck next run (don't kill blindly)
            kept.append((iid, f"{status}, age=unknown"))
            continue
        if age_min * 60 >= lease_sec:
            print(f"[reaper] DESTROY {iid} (age {age_min:.0f}min ≥ lease {cfg.MAX_LEASE_MINUTES}min, status={status})")
            try:
                mgr.destroy(iid)
                # verify
                time.sleep(3)
                if _instance(iid) is None:
                    reaped.append(iid)
                else:
                    kept.append((iid, "destroy-unconfirmed"))
            except Exception as e:  # noqa: BLE001
                kept.append((iid, f"destroy-error:{e}"))
        else:
            kept.append((iid, f"{status}, age={age_min:.0f}min < lease"))

    print(f"[reaper] reaped={reaped} gone={gone} kept={kept}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
