"""Где лежат файлы.

Из исходников всё рядом с программой: уроки в lessons/, результаты в records.json.
В собранной программе встроенные уроки лежат внутри неё (только чтение), а результаты
и свои мелодии — в папке пользователя: на macOS ~/Library/Application Support/NotoSvet,
на Windows %APPDATA%\\NotoSvet.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

FROZEN = getattr(sys, "frozen", False)
ROOT = Path(__file__).resolve().parent.parent
BUNDLE = Path(getattr(sys, "_MEIPASS", ROOT))


def _user_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "NotoSvet"
    if os.name == "nt":
        return Path(os.environ.get("APPDATA", Path.home())) / "NotoSvet"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "NotoSvet"


DATA_DIR = _user_dir() if FROZEN else ROOT
BUILTIN_LESSONS = BUNDLE / "lessons"
USER_LESSONS = DATA_DIR / "lessons"  # из исходников совпадает со встроенными
RECORDS = DATA_DIR / "records.json"
SETTINGS = DATA_DIR / "settings.json"
ERROR_LOG = DATA_DIR / "error.log"
