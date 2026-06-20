"""
Overnight autopilot test for the RunPod Serverless FLUX worker.

One full cycle:
  1. Force-recycle workers (workersMax 0 -> wait drain -> workersMax 3) so the
     latest build (version N) is what actually runs. This avoids stale workers
     from a rolled-back/old build sticking around as "idle" and never picking
     up jobs.
  2. Submit one low-res test job.
  3. Poll job + worker health until COMPLETED / FAILED / TIMED_OUT, or a hard
     deadline. First run downloads the 13GB model to the network volume, so
     allow a long deadline.
  4. On success, save the image and print SUCCESS so the parent can detect it.

Everything goes through the public REST/queue API; no UI needed.
Prints verbose, timestamped state so the result file is self-explanatory.
"""
import base64
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENV = ROOT / ".env"
for line in ENV.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

API = os.environ["RUNPOD_API_KEY"]
EP = os.environ["RUNPOD_ENDPOINT_ID"]
QBASE = f"https://api.runpod.ai/v2/{EP}"
REST = f"https://rest.runpod.io/v1/endpoints/{EP}"
QH = {"Authorization": f"Bearer {API}", "Content-Type": "application/json"}


def _req(method, url, body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, headers=headers or QH, method=method)
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


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def health():
    st, d = _req("GET", f"{QBASE}/health")
    if st == 200:
        return d["jobs"], d["workers"]
    return {}, {}


def set_max(n):
    st, d = _req("PATCH", REST, {"workersMax": n})
    ver = d.get("version") if isinstance(d, dict) else "?"
    log(f"PATCH workersMax={n} -> HTTP {st} (version={ver})")
    return st == 200


def force_fresh_workers():
    """workersMax 0 -> wait until no workers -> workersMax 3."""
    log("=== Force-recycling workers so the latest build runs ===")
    _req("POST", f"{QBASE}/purge-queue", {})
    set_max(0)
    for i in range(40):  # up to 10 min to drain
        time.sleep(15)
        _, w = health()
        total = sum(w.get(k, 0) for k in ("idle", "running", "initializing", "unhealthy", "ready"))
        log(f"  draining [{i*15}s] total_workers={total} {w}")
        if total == 0:
            log("  all workers drained")
            break
    set_max(3)


def submit():
    body = {"input": {
        "video_id": "test", "scene_id": "001", "mode": "text_to_image",
        "prompt": "ancient human sitting by fire, cave painting style, ochre on stone",
        "global_style": "prehistoric cave painting, ochre and charcoal, no text",
        "width": 512, "height": 288, "steps": 4, "guidance_scale": 1.0,
        "candidate_seeds": [42], "output_format": "WEBP", "quality": 80,
        "output_mode": "base64",
    }}
    st, d = _req("POST", f"{QBASE}/run", body)
    if st == 200:
        log(f"submitted job {d['id']}")
        return d["id"]
    log(f"submit FAILED HTTP {st}: {d}")
    return None


def watch(job_id, deadline_s=600):
    last = None
    run_streak = 0
    t0 = time.time()
    i = 0
    while time.time() - t0 < deadline_s:
        time.sleep(5)
        i += 1
        st, res = _req("GET", f"{QBASE}/status/{job_id}")
        j, w = health()
        cur = res.get("status") if st == 200 else f"HTTP{st}"
        run_streak = run_streak + 5 if w.get("running", 0) > 0 else 0
        if cur != last or i % 6 == 0:
            log(f"[{int(time.time()-t0)}s] job={cur} | q={j.get('inQueue')} prog={j.get('inProgress')} "
                f"done={j.get('completed')} fail={j.get('failed')} retry={j.get('retried')} | "
                f"run={w.get('running')} idle={w.get('idle')} init={w.get('initializing')} "
                f"unhlth={w.get('unhealthy')} | run_streak={run_streak}s")
            last = cur
        if st == 200 and res.get("status") == "COMPLETED":
            return "COMPLETED", res
        if st == 200 and res.get("status") in ("FAILED", "TIMED_OUT", "CANCELLED"):
            return res["status"], res
    return "DEADLINE", None


def main():
    log("########## RunPod autofix cycle start ##########")
    st, d = _req("GET", REST)
    if st == 200:
        log(f"endpoint version={d.get('version')} workersMax={d.get('workersMax')} "
            f"volume={d.get('networkVolumeId')} idleTimeout={d.get('idleTimeout')}")

    force_fresh_workers()

    job_id = submit()
    if not job_id:
        log("RESULT: SUBMIT_FAILED")
        return

    status, res = watch(job_id, deadline_s=600)
    log(f"=== watch ended: {status} ===")

    if status == "COMPLETED":
        out = res.get("output", {})
        imgs = out.get("images", [])
        errs = out.get("errors", [])
        log(f"images={len(imgs)} duration={out.get('duration_seconds')}s errors={errs}")
        if imgs:
            data = base64.b64decode(imgs[0]["base64"])
            p = ROOT / "test_scene001.webp"
            p.write_bytes(data)
            log(f"SAVED {p} ({len(data)} bytes) sha={imgs[0].get('sha256','')[:16]}")
            log("RESULT: SUCCESS")
        else:
            log(f"RESULT: COMPLETED_BUT_NO_IMAGE errors={errs}")
    elif status in ("FAILED", "TIMED_OUT", "CANCELLED"):
        log(f"RESULT: JOB_{status} detail={json.dumps(res)[:800]}")
    else:
        # DEADLINE — job never ran. Capture why.
        _, w = health()
        log(f"RESULT: STUCK_IN_QUEUE final_workers={w}")
        log("Interpretation: workers never processed the job within deadline. "
            "Either build still exits 127 (worker dies before handler) or the "
            "serverless loop is not attaching to the queue.")
    log("########## cycle end ##########")


if __name__ == "__main__":
    main()
