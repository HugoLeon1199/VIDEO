import os

# API Keys (set as environment variables before running)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Models
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
GEMINI_TEXT_MODEL = "gemini-2.5-flash"  # text only (step 4 image prompts)

# TTS Settings
TTS_VOICE = "am_fenrir"  # Kokoro voice ID (dramatic male, good for history storytelling)
TTS_SPEED = 0.95         # Slightly slower for clarity

# RunPod Serverless (step 5 image generation)
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "")
RUNPOD_REQUEST_TIMEOUT = int(os.getenv("RUNPOD_REQUEST_TIMEOUT", "1800"))
RUNPOD_POLL_INTERVAL = float(os.getenv("RUNPOD_POLL_INTERVAL", "3"))
RUNPOD_MAX_RETRIES = int(os.getenv("RUNPOD_MAX_RETRIES", "3"))
IMAGE_BACKEND = os.getenv("IMAGE_BACKEND", "runpod_serverless")

# Image generation defaults (sent to serverless worker)
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 576
IMAGE_STEPS = 4
IMAGE_GUIDANCE_SCALE = 1.0
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
