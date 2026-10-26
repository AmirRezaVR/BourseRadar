import json
import sys
from pathlib import Path

FONT_SIZES = {"regular": 10, "large": 13, "extra_large": 16}  # pt
FONT_LABELS = {"regular": "عادی", "large": "بزرگ", "extra_large": "خیلی بزرگ"}
DEFAULT_FONT = "regular"


def _app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


SETTINGS_FILE = _app_dir() / "settings.json"


def load_settings() -> dict:
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_settings(data: dict) -> None:
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError:
        pass
