"""Central configuration for the novel-to-audiobook pipeline.

All paths are resolved relative to the project root (parent of src/).
Override via environment variables or command-line args.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path so scripts can run from any directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_str_root = str(PROJECT_ROOT)
if _str_root not in sys.path:
    sys.path.insert(0, _str_root)

# ── Data paths ──────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
NOVEL_DIR = DATA_DIR / "novel"
CHAPTER_DIR = DATA_DIR / "chapter"
CHAPTERS_JSON_DIR = DATA_DIR / "chapters_json"
ROLE_BATCHES_DIR = DATA_DIR / "role_batches"
VOICE_PROFILES_PATH = DATA_DIR / "voice_profiles.json"

# ── Output paths ────────────────────────────────────────────
OUTPUT_DIR = PROJECT_ROOT / "output"

# ── External dependencies ───────────────────────────────────
QWEN3_TTS_ROOT = Path(os.environ.get("QWEN3_TTS_ROOT", r"D:\Qwen3-TTS"))

# ── API config ──────────────────────────────────────────────
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.5-122b-a10b")
