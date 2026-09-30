"""Уроки в формате JSON из плана: {"title", "bpm", "notes": [{"t", "d", "n", "h", "s"}]}.

t — начало в долях, d — длительность в долях, n — MIDI-номер 48..72,
h — рука ("R"/"L"), s — номер шага (ноты с одним s — аккорд).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from . import paths
from .config import HIGH, LOW

CHORD = 0.05  # ноты ближе 1/20 доли — один шаг, если s не размечен


@dataclass
class LessonNote:
    t: float
    d: float
    n: int
    h: str
    s: int


@dataclass
class Lesson:
    title: str
    bpm: float
    notes: list[LessonNote]
    path: Path | None = None

    @property
    def hands(self) -> list[str]:
        return sorted({x.h for x in self.notes})

    @property
    def steps(self) -> int:
        return len({x.s for x in self.notes})

    def seconds(self, tempo: float = 1.0) -> float:
        end = max((x.t + x.d for x in self.notes), default=0.0)
        return end * 60.0 / (self.bpm * tempo)


def parse_lesson(data: dict, path: Path | None = None) -> Lesson:
    notes = []
    for x in data["notes"]:
        n = int(x["n"])
        while n < LOW:   # не влезшее в 25 клавиш переносим в ближайшую октаву
            n += 12
        while n > HIGH:
            n -= 12
        hand = str(x.get("h", "R")).upper()[:1]
        s = x.get("s")
        notes.append(LessonNote(float(x["t"]), float(x.get("d", 1.0)), n,
                                hand if hand in ("R", "L") else "R",
                                None if s is None else int(s)))
    if not notes:
        raise ValueError("в уроке нет нот")
    notes.sort(key=lambda x: (x.t, x.n))
    if any(x.s is None for x in notes):
        step, first = -1, None
        for x in notes:
            if first is None or x.t - first > CHORD:
                step, first = step + 1, x.t
            x.s = step
    title = str(data.get("title") or (path.stem if path else "Без названия"))
    return Lesson(title, float(data.get("bpm", 100)), notes, path)


def load_lesson(path: Path) -> Lesson:
    with open(path, encoding="utf-8") as f:
        return parse_lesson(json.load(f), Path(path))


def load_lessons(folders: list[Path] | None = None) -> tuple[list[Lesson], list[str]]:
    """Встроенные уроки, затем свои; битые файлы не роняют программу, а попадают в список ошибок."""
    if folders is None:
        folders = [paths.BUILTIN_LESSONS, paths.USER_LESSONS]
    lessons, errors, seen = [], [], set()
    for folder in folders:
        folder = Path(folder).resolve()
        if folder in seen:
            continue
        seen.add(folder)
        for p in sorted(folder.glob("*.json")):
            try:
                lessons.append(load_lesson(p))
            except (OSError, ValueError, KeyError, TypeError) as e:
                errors.append(f"{p.name}: {e}")
    return lessons, errors


def free_name(folder: Path, stem: str) -> Path:
    """Свободное имя файла урока: «Песня.json», «Песня (2).json»…"""
    stem = re.sub(r"[^\w\- ]+", "_", stem).strip() or "мелодия"
    target, k = folder / f"{stem}.json", 2
    while target.exists():
        target, k = folder / f"{stem} ({k}).json", k + 1
    return target


def save_lesson(data: dict, path: Path) -> None:
    """Записать урок: одна нота — одна строка, как в примерах формата."""
    path.parent.mkdir(parents=True, exist_ok=True)
    notes = data.get("notes", [])
    lines = ["{", f' "title": {json.dumps(data.get("title", ""), ensure_ascii=False)},',
             f' "bpm": {json.dumps(data.get("bpm", 100))},', ' "notes": [']
    lines += [f"  {json.dumps(x, ensure_ascii=False)}{',' if i < len(notes) - 1 else ''}"
              for i, x in enumerate(notes)]
    lines += [" ]", "}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
