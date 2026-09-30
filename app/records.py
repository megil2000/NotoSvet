"""Последние 10 результатов по каждой мелодии — в records.json (см. paths.py)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from . import paths

KEEP = 10


def load(path: Path | None = None) -> dict:
    try:
        with open(path or paths.RECORDS, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def add(title: str, result: dict, path: Path | None = None) -> list[dict]:
    path = Path(path or paths.RECORDS)
    data = load(path)
    runs = data.setdefault(title, [])
    runs.append({"date": datetime.now().strftime("%Y-%m-%d %H:%M"), **result})
    del runs[:-KEEP]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    return runs
