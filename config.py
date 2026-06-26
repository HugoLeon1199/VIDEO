import os

# API Keys (set as environment variables before running)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Vision QA config — used by generate_images.py --qa
IMAGE_QA_ENABLED = os.getenv("IMAGE_QA_ENABLED", "false").lower() == "true"
IMAGE_QA_MIN_SCORE = int(os.getenv("IMAGE_QA_MIN_SCORE", "80"))
IMAGE_QA_MAX_REGENERATIONS = int(os.getenv("IMAGE_QA_MAX_REGENERATIONS", "2"))
IMAGE_QA_REGEN_CANDIDATES = int(os.getenv("IMAGE_QA_REGEN_CANDIDATES", "2"))
IMAGE_QA_WORKERS = int(os.getenv("IMAGE_QA_WORKERS", "4"))
IMAGE_QA_ALLOW_FALLBACK = os.getenv("IMAGE_QA_ALLOW_FALLBACK", "false").lower() == "true"
IMAGE_QA_AUDIT_MIN_SCORE = int(os.getenv("IMAGE_QA_AUDIT_MIN_SCORE", "85"))

# Versioning — bump when style/QA prompt changes to invalidate old cache
IMAGE_STYLE_VERSION = "vi-2d-documentary-v1"
IMAGE_QA_PROMPT_VERSION = "anatomy-qa-v1"

# Master seed library
MASTER_SEED_DIR = os.getenv("MASTER_SEED_DIR", "master_style_seeds")
MASTER_SEED_VERSION = "master-seeds-v1"

# Anti-drift
MAX_CHAIN_DEPTH = int(os.getenv("MAX_CHAIN_DEPTH", "4"))

# Grain overlay
GRAIN_ENABLED = os.getenv("GRAIN_ENABLED", "true").lower() == "true"
GRAIN_OPACITY = float(os.getenv("GRAIN_OPACITY", "0.10"))

# Models
CLAUDE_MODEL = "claude-sonnet-4-6"
GEMINI_TEXT_MODEL = "gemini-2.5-flash"  # text only (step 4 image prompts)

# TTS Settings
TTS_VOICE = "am_fenrir"  # Kokoro voice ID (dramatic male, good for history storytelling)
TTS_SPEED = 0.95         # Slightly slower for clarity

# RunPod Serverless (step 5 image generation)
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "")          # unified endpoint — FLUX.1-dev 12B
RUNPOD_REQUEST_TIMEOUT = int(os.getenv("RUNPOD_REQUEST_TIMEOUT", "1800"))
RUNPOD_POLL_INTERVAL = float(os.getenv("RUNPOD_POLL_INTERVAL", "3"))
RUNPOD_MAX_RETRIES = int(os.getenv("RUNPOD_MAX_RETRIES", "3"))
IMAGE_BACKEND = os.getenv("IMAGE_BACKEND", "runpod_serverless")  # runpod_serverless | vast_instance

# Vast.ai settings (only used when IMAGE_BACKEND=vast_instance)
VAST_API_KEY = os.getenv("VAST_API_KEY", "")
VAST_WORKER_PORT = int(os.getenv("VAST_WORKER_PORT", "8080"))
VAST_MIN_VRAM_GB = int(os.getenv("VAST_MIN_VRAM_GB", "24"))
VAST_MAX_PRICE_PER_HOUR = float(os.getenv("VAST_MAX_PRICE_PER_HOUR", "1.0"))
VAST_GPU_NAME = os.getenv("VAST_GPU_NAME", "")             # e.g. "RTX 4090", "" = any
VAST_WORKER_IMAGE = os.getenv("VAST_WORKER_IMAGE", "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime")
VAST_INSTANCE_ID = os.getenv("VAST_INSTANCE_ID", "")       # set this to skip rent step
VAST_INSTANCE_HOST = os.getenv("VAST_INSTANCE_HOST", "")   # set this + VAST_INSTANCE_ID to skip rent
VAST_INSTANCE_PORT = int(os.getenv("VAST_INSTANCE_PORT", "0"))  # external port mapped to VAST_WORKER_PORT
VAST_HF_TOKEN = os.getenv("VAST_HF_TOKEN", os.getenv("HF_TOKEN", ""))
VAST_DISK_GB = float(os.getenv("VAST_DISK_GB", "60.0"))
VAST_USE_FP8 = os.getenv("VAST_USE_FP8", "1")  # "1" = FP8 (~12GB VRAM), "0" = bfloat16 (~24GB VRAM)
VAST_MIN_INET_DOWN_MBPS = int(os.getenv("VAST_MIN_INET_DOWN_MBPS", "500"))  # min download speed Mbps
VAST_REQUEST_TIMEOUT = int(os.getenv("VAST_REQUEST_TIMEOUT", "600"))

# Per-track config — used by scripts/generate_images.py --track vi|en
# Both tracks share the same unified endpoint (FLUX.1-dev 12B, 24GB GPU)
TRACK_CONFIG = {
    "vi": {
        "endpoint_id_env": "RUNPOD_ENDPOINT_ID",
        "model": "black-forest-labs/FLUX.1-dev",
        "steps": 22,
        "guidance_scale": 3.5,
        "system_prompt_file": "prompts/image_prompt_vi.txt",
        "output_subdir": "images_vi",
        "style_version": "prehistoric-flat-vector-v1",
    },
    "en": {
        "endpoint_id_env": "RUNPOD_ENDPOINT_ID",
        "model": "black-forest-labs/FLUX.1-dev",
        "steps": 22,
        "guidance_scale": 3.5,
        "system_prompt_file": "prompts/image_prompt_en.txt",
        "output_subdir": "images_en",
        "style_version": "prehistoric-flat-vector-v1",
    },
}

# Image generation defaults (sent to serverless worker)
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 576
IMAGE_STEPS = 20
IMAGE_GUIDANCE_SCALE = 3.5
IMAGE_CANDIDATES = 3
IMAGE_CANDIDATE_SEEDS = [11001, 11002, 11003]
IMAGE_OUTPUT_FORMAT = "WEBP"
IMAGE_QUALITY = 92

# Video Settings
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
VIDEO_FPS = 30
VIDEO_BITRATE = "8M"
KEN_BURNS_ZOOM = 0.05    # 5% zoom for Ken Burns effect
FADE_DURATION = 0.3      # seconds for fade transition

# Subtitle settings (burn-in)
SUBTITLE_FONT_SIZE = 36
SUBTITLE_FONT_COLOR = "white"
SUBTITLE_SHADOW = True

# Paths
OUTPUT_DIR = "output"
LOGS_FILE = "pipeline.log"
PROMPTS_DIR = "prompts"

# Claude API retry
CLAUDE_MAX_RETRIES = 2
CLAUDE_RETRY_SLEEP = 5   # seconds
