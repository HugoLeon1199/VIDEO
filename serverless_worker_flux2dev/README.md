# serverless_worker

RunPod Serverless worker for FLUX.2 Klein 4B image generation.

## Files

| File | Purpose |
|------|---------|
| `handler.py` | RunPod entry point — validates input, runs inference, returns result |
| `model_loader.py` | Loads `Flux2KleinPipeline` once at cold-start |
| `schemas.py` | Input validation |
| `image_utils.py` | PIL → bytes, SHA-256, base64, volume save |
| `requirements.txt` | Pinned Python deps |
| `Dockerfile` | Production image for RunPod |
| `test_input.json` | Sample job payload for local testing |

## Local test (no GPU)

```bash
pip install runpod pillow
python handler.py --rp_serve_api   # starts local test server on :8000
curl -X POST http://localhost:8000/runsync \
  -H "Content-Type: application/json" \
  -d @test_input.json
```

## Docker build

```bash
docker build -f serverless_worker/Dockerfile -t emberlore-worker:latest .
```

## Environment variables (set via RunPod Secrets / env)

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Yes | Hugging Face token for gated model download |
| `MODEL_ID` | No | Defaults to `black-forest-labs/FLUX.2-klein-4B` |
| `HF_HOME` | No | Defaults to `/runpod-volume/hf-cache` |
| `RUNPOD_VOLUME_PATH` | No | Defaults to `/runpod-volume` |
