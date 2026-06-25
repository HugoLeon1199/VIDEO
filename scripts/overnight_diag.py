"""
Overnight orchestrator:
  1. Wait until the endpoint version increases (new build deployed).
  2. Force-recycle workers so the new build runs.
  3. Submit one test job and watch.
  4. Print the FULL job output (the diagnostic handler returns import/CUDA
     probe info) so we learn exactly what's wrong without UI logs.

Self-contained; uses public REST + queue API only.
"""
import base64
import json
import os
import pathlib
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
for line in (ROOT / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

API = os.environ["RUNPOD_API_KEY"]
EP = os.environ["RUNPOD_ENDPOINT_ID"]
QBASE = f"https://api.runpod.ai/v2/{EP}"
REST = f"https://rest.runpod.io/v1/endpoints/{EP}"
H = {"Authorization": f"Bearer {API}", "Content-Type": "application/json"}


def req(method, url, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, headers=H, method=method)
    try:
        with urllib.request.urlopen(r, timeout=40) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, e.read().decode()[:300]
    except Exception as e:
        return -1, str(e)


def log(m):
    print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


def get_version():
    st, d = req("GET", REST)
    return d.get("version") if st == 200 and isinstance(d, dict) else None


def health():
    st, d = req("GET", f"{QBASE}/health")
    return (d.get("jobs", {}), d.get("workers", {})) if st == 200 else ({}, {})


def wait_version_above(base_ver, max_wait=900):
    log(f"waiting for build deploy (version > {base_ver}) ...")
    t0 = time.time()
    while time.time() - t0 < max_wait:
        v = get_version()
        if v is not None and v > base_ver:
            log(f"version changed {base_ver} -> {v} (build deployed)")
            return v
        time.sleep(20)
    log(f"version still {get_version()} after {max_wait}s (build slow/failed)")
    return get_version()


def force_fresh():
    log("=== force-recycle workers ===")
    req("POST", f"{QBASE}/purge-queue", {})
    req("PATCH", REST, {"workersMax": 0})
    for i in range(40):
        time.sleep(15)
        _, w = health()
        total = sum(w.get(k, 0) for k in ("idle", "running", "initializing", "unhealthy", "ready"))
        if total == 0:
            log(f"  drained after {i*15}s")
            break
    req("PATCH", REST, {"workersMax": 3})


def submit():
    body = {"input": {
        "video_id": "test", "scene_id": "001", "mode": "text_to_image",
        "prompt": "ancient human sitting by fire, cave painting style, ochre on stone",
        "global_style": "prehistoric cave painting, ochre and charcoal, no text",
        "width": 512, "height": 288, "steps": 4, "guidance_scale": 1.0,
        "candidate_seeds": [42], "output_format": "WEBP", "quality": 80,
        "output_mode": "base64",
    }}
    st, d = req("POST", f"{QBASE}/run", body)
    if st == 200:
        log(f"submitted {d['id']}")
        return d["id"]
    log(f"submit FAILED {st}: {d}")
    return None


def watch(job_id, deadline=420):
    last = None
    t0 = time.time()
    i = 0
    while time.time() - t0 < deadline:
        time.sleep(5)
        i += 1
        st, res = req("GET", f"{QBASE}/status/{job_id}")
        j, w = health()
        cur = res.get("status") if st == 200 else f"HTTP{st}"
        if cur != last or i % 6 == 0:
            log(f"[{int(time.time()-t0)}s] job={cur} q={j.get('inQueue')} prog={j.get('inProgress')} "
                f"run={w.get('running')} idle={w.get('idle')} init={w.get('initializing')} unhlth={w.get('unhealthy')}")
            last = cur
        if st == 200 and res.get("status") in ("COMPLETED", "FAILED", "TIMED_OUT", "CANCELLED"):
            return res["status"], res
    return "DEADLINE", None


def main():
    log("########## overnight diag start ##########")
    base = get_version() or 0
    log(f"current version {base}")
    wait_version_above(base, max_wait=2400)  # torch build is large (~15-20 min)
    force_fresh()
    jid = submit()
    if not jid:
        log("RESULT: SUBMIT_FAILED")
        return
    status, res = watch(jid, deadline=900)  # first run downloads 13GB model
    log(f"=== watch ended: {status} ===")
    if status == "COMPLETED":
        out = res.get("output", {})
        imgs = out.get("images", [])
        errs = out.get("errors", [])
        log(f"RESULT: JOB_COMPLETED — images={len(imgs)} duration={out.get('duration_seconds')}s errors={errs}")
        if imgs and imgs[0].get("base64"):
            data = base64.b64decode(imgs[0]["base64"])
            p = ROOT / "test_scene001.webp"
            p.write_bytes(data)
            log(f">>> SAVED REAL IMAGE {p} ({len(data)} bytes) sha={imgs[0].get('sha256','')[:16]} <<<")
            log("RESULT: SUCCESS — FLUX image generated!")
        else:
            log("FULL OUTPUT:")
            log(json.dumps(out, indent=2)[:1500])
    elif status in ("FAILED", "TIMED_OUT", "CANCELLED"):
        log(f"RESULT: JOB_{status}")
        log(json.dumps(res, indent=2)[:1500])
    else:
        _, w = health()
        log(f"RESULT: STUCK_IN_QUEUE workers={w}")
        log("=> minimal handler also ignored job: base image / ENTRYPOINT / build issue, not Python imports")
    log("########## overnight diag end ##########")


if __name__ == "__main__":
    main()
