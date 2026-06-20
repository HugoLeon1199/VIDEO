# RunPod Serverless Deployment Guide

FLUX.2 Klein 4B image generation backend for the YouTube autopilot pipeline.

---

## Prerequisites

- RunPod account with credit (runpod.io)
- RunPod Network Volume (for model cache between cold starts)
- Hugging Face account with access to `black-forest-labs/FLUX.2-klein-4B`
- This repository pushed to GitHub

---

## One-time setup steps

### 1. Connect RunPod to GitHub

RunPod Console → **Settings** → **Integrations** → Connect GitHub → Authorize.

### 2. Create a Network Volume (model cache)

Serverless → **Storage** → **+ Network Volume**
- Name: `emberlore-model-cache`
- Size: 50 GB
- Region: same as your endpoint

### 3. Create the Serverless Endpoint

Serverless → **New Endpoint** → **Custom deployment** (Import Git Repository)

| Setting | Value |
|---------|-------|
| Repository | `your-github-user/your-repo` |
| Branch | `main` |
| Dockerfile path | `serverless_worker/Dockerfile` |
| Endpoint type | Queue |
| Active workers | 0 |
| Maximum workers | 1 |
| GPUs per worker | 1 |
| GPU priority | L4 → A5000 → RTX 3090 24GB → RTX 4090 24GB |
| Idle timeout | 5 seconds |
| Execution timeout | 1800 seconds |
| FlashBoot | Enabled |
| Network Volume | Attach `emberlore-model-cache` → mount at `/runpod-volume` |

### 4. Add HF_TOKEN via RunPod Secrets

Endpoint → **Environment Variables** → **+ Secret**
- Key: `HF_TOKEN`
- Value: your Hugging Face token (read access to gated model)

### 5. Deploy

Click **Deploy**. Wait for the endpoint to show **Ready**.

### 6. Copy the Endpoint ID

From the endpoint detail page, copy the ID (e.g. `abc123def456`).

### 7. Create a scoped RunPod API key

Settings → **API Keys** → **+ API Key** → copy the `rpa_...` key.

### 8. Store credentials locally

```bash
cp .env.example .env
# Edit .env — fill in RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID
```

**Never commit `.env` to git.**

---

## Validation and testing

### A. Validate environment (no GPU call)

```powershell
$python = "C:\Users\LEON_RM\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $python scripts/validate_runpod_serverless.py --video-id ancient-humans-without-medicine
```

### B. Submit one real test image

```powershell
& $python scripts/validate_runpod_serverless.py --video-id ancient-humans-without-medicine --generate-test
```

### C. Run one scene

```powershell
& $python scripts/generate_images.py `
    --video-id ancient-humans-without-medicine `
    --scene-id 001 `
    --candidates 3
```

### D. Run 20-scene benchmark

```powershell
& $python scripts/generate_images.py `
    --video-id ancient-humans-without-medicine `
    --from-scene 1 --to-scene 20 `
    --candidates 3 --resume
```

### E. Run full video

```powershell
& $python scripts/generate_images.py `
    --video-id ancient-humans-without-medicine `
    --candidates 3 --resume
```

---

## Docker build (local test)

```bash
docker build -f serverless_worker/Dockerfile -t emberlore-worker:latest .
docker run --rm emberlore-worker:latest python -c "import handler; print('imports OK')"
```

---

## Unit tests

```powershell
& $python -m pytest tests/ -v
```

---

## Acceptance checklist

- [ ] A. Docker image builds without error
- [ ] B. Unit tests pass (`pytest tests/ -v`)
- [ ] C. Dry-run passes (`--dry-run` flag)
- [ ] D. Endpoint deploys and shows Ready in RunPod console
- [ ] E. One real text-to-image request succeeds (`--generate-test`)
- [ ] F. Three deterministic candidates succeed (same seeds → same images)
- [ ] G. Full `image_prompts.json` resume flow succeeds
- [ ] H. (Phase 4) Multi-reference generation succeeds

---

## Cost estimates

| Scenario | GPU | Time | Cost |
|----------|-----|------|------|
| 1 image (4 steps) | RTX A5000 | ~3s | ~$0.001 |
| 3 candidates | RTX A5000 | ~9s | ~$0.003 |
| 144 scenes × 3 candidates | RTX A5000 | ~22 min | ~$0.44 |

Idle workers cost $0 (Min Workers = 0, Idle Timeout = 5s).
