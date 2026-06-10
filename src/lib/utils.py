"""Shared utility functions for the novel-to-audiobook pipeline."""

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            backup = path.with_suffix(path.suffix + ".bad")
            backup.write_text(path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        except Exception:
            pass
        return default


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_name(name: str) -> str:
    s = re.sub(r"[\\/:*?\"<>|]+", "_", name.strip())
    s = re.sub(r"\s+", "_", s)
    return s or "unknown"


def role_key_from_name(role_name: str) -> str:
    return re.sub(r"\s+", "_", role_name).strip("_") or "unknown_role"


def chapter_num(filename: str) -> Optional[int]:
    m = re.match(r"^(\d+)_.*\.json$", filename, re.I)
    return int(m.group(1)) if m else None


def dominant(counter: Counter) -> Optional[str]:
    return counter.most_common(1)[0][0] if counter else None


def suggest_voice(gender: Optional[str], feature: Optional[str]) -> tuple:
    f = (feature or "").lower()
    if gender == "女":
        if any(k in f for k in ["柔", "甜", "轻", "少"]):
            return "custom_voice", "Serena"
        return "custom_voice", "Vivian"
    if gender == "男":
        if any(k in f for k in ["老", "低", "沙"]):
            return "custom_voice", "Uncle_Fu"
        if any(k in f for k in ["少", "清", "年"]):
            return "custom_voice", "Dylan"
        return "custom_voice", "Eric"
    return "voice_design", "neutral clear narrator-like voice"


def normalize_gender(value: Optional[str]) -> str:
    x = (value or "").strip().lower()
    if x in {"男", "male", "m", "man", "boy"}:
        return "男"
    if x in {"女", "female", "f", "woman", "girl"}:
        return "女"
    return "中性"


def map_language(lang: Optional[str], default_lang: str) -> str:
    x = (lang or "").strip().lower()
    if not x:
        return default_lang
    if x in {"zh", "zh-cn", "cn", "chinese"}:
        return "Chinese"
    if x in {"en", "en-us", "english"}:
        return "English"
    if x in {"auto"}:
        return "Auto"
    return lang.strip()
