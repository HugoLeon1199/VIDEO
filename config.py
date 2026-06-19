import os

# API Keys (set as environment variables before running)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Models
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
GEMINI_IMAGE_MODEL = "gemini-2.5-flash-image"

# TTS Settings
TTS_VOICE = "am_fenrir"  # Kokoro voice ID (dramatic male, good for history storytelling)
TTS_SPEED = 0.95         # Slightly slower for clarity

# Image Settings
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024
IMAGES_PER_VIDEO = 200
GEMINI_RATE_LIMIT_SLEEP = 31   # seconds between requests (<2 req/min)
GEMINI_RETRY_SLEEP = 60        # seconds to wait on 429
GEMINI_MAX_RETRIES = 3

# Video Settings
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
VIDEO_FPS = 30
VIDEO_BITRATE = "4M"
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
