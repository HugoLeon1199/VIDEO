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
VAST_MIN_VRAM_GB = int(os.getenv("VAST_MIN_VRAM_GB", "24"))   # FLUX 8-bit needs ~15GB → 24GB (16GB OOMs)
VAST_MAX_VRAM_GB = int(os.getenv("VAST_MAX_VRAM_GB", "200"))   # no real cap; arch gate + _true_cost decide
# GPU hourly price cap only; bandwidth/download fees are gated separately by
# VAST_MAX_INET_DOWN_COST below. Keep the default cheap-first and do not silently
# raise this in production runs without Leon explicitly approving the cost.
VAST_MAX_PRICE_PER_HOUR = float(os.getenv("VAST_MAX_PRICE_PER_HOUR", "0.20"))
VAST_GPU_NAME = os.getenv("VAST_GPU_NAME", "")             # e.g. "RTX 4090", "" = any (blacklist handles 50xx/Tesla)
VAST_MAX_RENT_ATTEMPTS = int(os.getenv("VAST_MAX_RENT_ATTEMPTS", "4"))  # try N cheapest machines before giving up
VAST_WORKER_IMAGE = os.getenv("VAST_WORKER_IMAGE", "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime")
VAST_INSTANCE_ID = os.getenv("VAST_INSTANCE_ID", "")       # set this to skip rent step
VAST_INSTANCE_HOST = os.getenv("VAST_INSTANCE_HOST", "")   # set this + VAST_INSTANCE_ID to skip rent
VAST_INSTANCE_PORT = int(os.getenv("VAST_INSTANCE_PORT", "0"))  # external port mapped to VAST_WORKER_PORT
VAST_HF_TOKEN = os.getenv("VAST_HF_TOKEN", os.getenv("HF_TOKEN", ""))
VAST_DISK_GB = float(os.getenv("VAST_DISK_GB", "60.0"))
VAST_USE_FP8 = os.getenv("VAST_USE_FP8", "1")  # "1" = FP8 (~12GB VRAM), "0" = bfloat16 (~24GB VRAM)
# Bandwidth (download) fee — the #1 hidden cost. Invoice showed a host at $0.012/GB
# charging $0.73 just to pull the model (7× the GPU charge). HARD CAP at 0.005;
# PREFERRED (tie-breaker, NOT a hard filter) 0.003. Don't hard-filter to 0.003 — a
# slightly pricier-bandwidth box that's a cheaper/faster GPU can win on true cost.
VAST_MAX_INET_DOWN_COST = float(os.getenv("VAST_MAX_INET_DOWN_COST", "0.005"))
VAST_PREFERRED_INET_DOWN_COST = float(os.getenv("VAST_PREFERRED_INET_DOWN_COST", "0.003"))
VAST_MAX_ESTIMATED_TOTAL_COST = float(os.getenv("VAST_MAX_ESTIMATED_TOTAL_COST", "0.20"))
VAST_ESTIMATED_TOTAL_COST_FALLBACKS = tuple(
    float(value.strip())
    for value in os.getenv("VAST_ESTIMATED_TOTAL_COST_FALLBACKS", "0.20,0.30,0.40").split(",")
    if value.strip()
)
# Expected download per rental, used by find_offer true-cost ranking. After adding
# ignore_patterns (skip 23.8GB single-file dup), the HF model is ~34GB; Docker image
# pull + pip/apt setup adds ~3GB → ~37GB total. Tune IMAGE_AND_SETUP after invoices.
VAST_EXPECTED_MODEL_GB = float(os.getenv("VAST_EXPECTED_MODEL_GB", "34.0"))
VAST_EXPECTED_IMAGE_AND_SETUP_GB = float(os.getenv("VAST_EXPECTED_IMAGE_AND_SETUP_GB", "3.0"))
VAST_EXPECTED_DOWNLOAD_GB = float(os.getenv(
    "VAST_EXPECTED_DOWNLOAD_GB",
    str(VAST_EXPECTED_MODEL_GB + VAST_EXPECTED_IMAGE_AND_SETUP_GB)))
VAST_EXPECTED_UPLOAD_GB = float(os.getenv("VAST_EXPECTED_UPLOAD_GB", "2.0"))
HF_MODEL_REVISION = os.getenv("HF_MODEL_REVISION", "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21")  # pinned Flux rev
WORKER_API_TOKEN = os.getenv("WORKER_API_TOKEN", "local-worker-token")
MAX_LEASE_MINUTES = int(os.getenv("MAX_LEASE_MINUTES", "90"))  # reaper kills boxes older than this
# Keep production cheap-first: `find_offer` already ranks eligible machines by
# estimated true total cost. A 10Gbps floor can force expensive GPUs; 500Mbps keeps
# good low-cost RTX 3090/4090 offers in the pool while reliability/disk/arch gates
# still reject bad machines.
VAST_MIN_INET_DOWN_MBPS = int(os.getenv("VAST_MIN_INET_DOWN_MBPS", "500"))
VAST_MIN_RELIABILITY = float(os.getenv("VAST_MIN_RELIABILITY", "0.98"))     # drop low-uptime hosts (avoid lemons)
VAST_REQUEST_TIMEOUT = int(os.getenv("VAST_REQUEST_TIMEOUT", "600"))

# Per-track config — used by scripts/generate_images.py --track vi|en
# Both tracks share the same unified endpoint (FLUX.1-dev 12B, 24GB GPU)
TRACK_CONFIG = {
    "vi": {
        "endpoint_id_env": "RUNPOD_ENDPOINT_ID",
        "model": "black-forest-labs/FLUX.1-dev",
        # 20 steps: the flat-vector / 2D illustration style converges well below
        # FLUX-dev's default 28; dropping 22->20 saves ~7% GPU time per image with
        # no visible quality loss for this art style.
        "steps": 20,
        "guidance_scale": 3.5,
        "system_prompt_file": "prompts/image_prompt_vi.txt",
        "output_subdir": "images_vi",
        "style_version": "prehistoric-flat-vector-v1",
    },
    "en": {
        "endpoint_id_env": "RUNPOD_ENDPOINT_ID",
        "model": "black-forest-labs/FLUX.1-dev",
        "steps": 20,
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
FLUX_CLIP_TOKEN_LIMIT = int(os.getenv("FLUX_CLIP_TOKEN_LIMIT", "77"))
FLUX_T5_TOKEN_LIMIT = int(os.getenv("FLUX_T5_TOKEN_LIMIT", "512"))

# Video Settings
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
VIDEO_FPS = 30
VIDEO_BITRATE = "8M"
KEN_BURNS_ZOOM = 0.05    # 5% zoom for Ken Burns effect
FADE_DURATION = 0.3      # seconds for fade transition

# Subtitle settings (burn-in)
SUBTITLE_FONT_FAMILY = os.getenv("SUBTITLE_FONT_FAMILY", "Segoe UI Semibold")
SUBTITLE_FONT_SIZE = int(os.getenv("SUBTITLE_FONT_SIZE", "38"))
SUBTITLE_OUTLINE = float(os.getenv("SUBTITLE_OUTLINE", "3.0"))
SUBTITLE_SHADOW = float(os.getenv("SUBTITLE_SHADOW", "1.2"))
SUBTITLE_MARGIN_V = int(os.getenv("SUBTITLE_MARGIN_V", "120"))
SUBTITLE_MAX_CHARS_PER_LINE = int(os.getenv("SUBTITLE_MAX_CHARS_PER_LINE", "46"))
SUBTITLE_MIN_WORDS = int(os.getenv("SUBTITLE_MIN_WORDS", "3"))
SUBTITLE_MAX_WORDS = int(os.getenv("SUBTITLE_MAX_WORDS", "8"))
SUBTITLE_TARGET_MIN_SECONDS = float(os.getenv("SUBTITLE_TARGET_MIN_SECONDS", "1.2"))
SUBTITLE_TARGET_MAX_SECONDS = float(os.getenv("SUBTITLE_TARGET_MAX_SECONDS", "3.5"))
SUBTITLE_FADE_MS = int(os.getenv("SUBTITLE_FADE_MS", "100"))
SUBTITLE_PREVIEW_SECONDS = int(os.getenv("SUBTITLE_PREVIEW_SECONDS", "45"))
SUBTITLE_DEFAULT_STYLE = os.getenv("SUBTITLE_DEFAULT_STYLE", "cinematic_clean")
SUBTITLE_PRIMARY_COLOR_ASS = os.getenv("SUBTITLE_PRIMARY_COLOR_ASS", "&H00F2F5F8")
SUBTITLE_SECONDARY_COLOR_ASS = os.getenv("SUBTITLE_SECONDARY_COLOR_ASS", "&H0066C7F2")
SUBTITLE_OUTLINE_COLOR_ASS = os.getenv("SUBTITLE_OUTLINE_COLOR_ASS", "&H00301812")
SUBTITLE_SHADOW_COLOR_ASS = os.getenv("SUBTITLE_SHADOW_COLOR_ASS", "&H80000000")

# Paths
OUTPUT_DIR = "output"
LOGS_FILE = "pipeline.log"
PROMPTS_DIR = "prompts"
PUBLISHING_DIRNAME = os.getenv("PUBLISHING_DIRNAME", "publishing")

# Creative package and thumbnail workflow
CREATIVE_PACKAGE_ALLOWED_CONCEPT_COUNTS = (3, 5)
CREATIVE_PACKAGE_DEFAULT_CONCEPT_COUNT = int(os.getenv("CREATIVE_PACKAGE_DEFAULT_CONCEPT_COUNT", "5"))
THUMBNAIL_FONT_FAMILY = os.getenv("THUMBNAIL_FONT_FAMILY", "Segoe UI Bold")
THUMBNAIL_FONT_SIZE = int(os.getenv("THUMBNAIL_FONT_SIZE", "96"))
THUMBNAIL_SAFE_MARGIN = int(os.getenv("THUMBNAIL_SAFE_MARGIN", "72"))
THUMBNAIL_LINE_SPACING = int(os.getenv("THUMBNAIL_LINE_SPACING", "12"))
THUMBNAIL_TEXT_PANEL_RATIO = float(os.getenv("THUMBNAIL_TEXT_PANEL_RATIO", "0.36"))
THUMBNAIL_JPEG_QUALITY = int(os.getenv("THUMBNAIL_JPEG_QUALITY", "92"))
THUMBNAIL_STROKE_WIDTH = int(os.getenv("THUMBNAIL_STROKE_WIDTH", "5"))
THUMBNAIL_CANDIDATE_SEED = int(os.getenv("THUMBNAIL_CANDIDATE_SEED", "21001"))
THUMBNAIL_TEXT_COLOR = (255, 246, 231)
THUMBNAIL_STROKE_COLOR = (24, 16, 12)
THUMBNAIL_SHADOW_COLOR = (0, 0, 0)
THUMBNAIL_SHADOW_OFFSET = (3, 4)
THUMBNAIL_CONTACT_SHEET_BG = (245, 238, 225)
THUMBNAIL_CONTACT_SHEET_LABEL_COLOR = (44, 34, 24)

# Claude API retry
CLAUDE_MAX_RETRIES = 2
CLAUDE_RETRY_SLEEP = 5   # seconds


def require_pinned_hf_model_revision() -> str:
    revision = (HF_MODEL_REVISION or "").strip()
    if not revision or revision.lower() == "main":
        raise RuntimeError(
            "HF_MODEL_REVISION must be pinned to a commit SHA before Vast image generation"
        )
    return revision
