"""MIDI → урок НотоСвет и добавление своих мелодий.

convert() — конвертер из плана: собирает ноты всех дорожек, сдвигает мелодию октавами
в 25 клавиш, делит по рукам (ниже «до» первой октавы — левая) и склеивает одновременные
ноты в шаги.
"""
from __future__ import annotations

import json
from pathlib import Path

import mido

from . import paths
from .lesson import free_name, load_lesson, parse_lesson, save_lesson

LOW, HIGH = 48, 72      # клавиатура: C3..C5
CHORD = 0.05            # ноты ближе 1/20 доли — один шаг (аккорд)
MIDI_EXT = {".mid", ".midi"}


def convert(path, split=60):
    mid = mido.MidiFile(path)
    bpm, notes, pressed, tick = 120.0, [], {}, 0
    for msg in mido.merge_tracks(mid.tracks):
        tick += msg.time
        beat = tick / mid.ticks_per_beat
        if msg.type == "set_tempo" and not notes and not pressed:
            bpm = round(mido.tempo2bpm(msg.tempo), 1)
        elif msg.type == "note_on" and msg.velocity > 0:
            pressed[msg.note] = beat
        elif msg.type in ("note_on", "note_off") and msg.note in pressed:
            start = pressed.pop(msg.note)
            notes.append([start, beat - start, msg.note])
    # сдвиг октавами: больше всего нот в диапазоне, при равенстве — меньший сдвиг
    fits = lambda s: sum(LOW <= n + s <= HIGH for _, _, n in notes)
    shift = max((-36, -24, -12, 0, 12, 24, 36), key=lambda s: (fits(s), -abs(s)))
    out = []
    for t, d, n in notes:
        n += shift
        while n < LOW: n += 12          # не влезшее переносим
        while n > HIGH: n -= 12         # в ближайшую октаву
        out.append({"t": round(t, 3), "d": round(d, 3), "n": n,
                    "h": "R" if n >= split else "L"})
    out.sort(key=lambda x: (x["t"], x["n"]))
    step, first = -1, None              # номер шага: аккорд = один шаг
    for x in out:
        if first is None or x["t"] - first > CHORD:
            step, first = step + 1, x["t"]
        x["s"] = step
    return {"title": Path(path).stem, "bpm": bpm, "notes": out}


def import_file(src: Path, folder: Path | None = None) -> Path:
    """Добавить мелодию из .mid/.midi или .json в папку своих уроков. Возвращает путь к уроку."""
    src = Path(src)
    ext = src.suffix.lower()
    if ext in MIDI_EXT:
        try:
            data = convert(src)
        except (OSError, EOFError, ValueError, KeyError, IndexError) as e:
            raise ValueError(f"не удалось прочитать MIDI: {e}") from e
    elif ext == ".json":
        try:
            with open(src, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            raise ValueError(f"не удалось прочитать JSON: {e}") from e
    else:
        raise ValueError("нужен файл .mid, .midi или .json")
    if not isinstance(data, dict) or not data.get("notes"):
        raise ValueError("в файле нет нот")
    try:
        parse_lesson(data)  # проверка формата до сохранения
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"неверный формат урока: {e}") from e
    folder = Path(folder or paths.USER_LESSONS)
    target = free_name(folder, src.stem)
    save_lesson(data, target)
    load_lesson(target)
    return target
