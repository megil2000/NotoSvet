"""Настройки между запусками (settings.json рядом с результатами): пока только полный экран."""
from __future__ import annotations

import json

from . import paths


def load() -> dict:
    try:
        data = json.loads(paths.SETTINGS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save(**changes) -> None:
    try:
        paths.SETTINGS.parent.mkdir(parents=True, exist_ok=True)
        paths.SETTINGS.write_text(json.dumps(load() | changes, ensure_ascii=False), encoding="utf-8")
    except OSError:  # нет прав на запись — просто не запомним
        pass
