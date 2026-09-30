"""Редактор своей мелодии: ноты на бит-сетке тактов и долей, как в rekordbox.

Три способа ввода:
  • мышью — клик по сетке ставит ноту, её можно тянуть, удлинять, удалять;
  • по шагам — клавиша ставит ноту у курсора, курсор уходит дальше на «Длину»;
  • записью — играете под метроном после такта отсчёта, ноты выравниваются по сетке («Q»).

«Q» — привязка к сетке, как кнопка QUANTIZE в rekordbox: всё, что ставится или двигается,
встаёт на линии сетки. Уже стоящие ноты ставит на сетку команда «Выровнять».

Здесь только логика — без pygame, поэтому её проверяют тесты. Как и сессия урока,
модель копит события для звука: ("on", p), ("off", p), ("click", сильная доля),
а для «Оригинала» — ("start", доля) и ("stop",).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from . import paths
from .config import HIGH, LOW
from .lesson import CHORD, Lesson, free_name, save_lesson

BEATS_PER_BAR = 4
GRIDS = ((1.0, "1/4"), (0.5, "1/8"), (0.25, "1/16"))
LENGTHS = ((0.25, "1/16"), (0.5, "1/8"), (1.0, "1/4"), (2.0, "1/2"), (4.0, "целая"))
HAND_MODES = {"auto": "авто", "R": "правая", "L": "левая"}
FREE_STEP = 1 / 48          # без квантования ноты встают с точностью 1/48 доли
SPLIT = 60                  # «авто»: ниже «до» первой октавы — левая рука, как в конвертере
BPM_MIN, BPM_MAX = 30, 240
UNDO_LIMIT = 100
EPS = 1e-6


@dataclass(eq=False)  # ноты сравниваются как объекты: две одинаковые ноты — разные ноты
class ENote:
    t: float    # начало, доли
    d: float    # длительность, доли
    n: int      # MIDI-номер
    h: str      # рука: "R" / "L"

    @property
    def end(self) -> float:
        return self.t + self.d


def _num(v: float):
    v = round(v, 4)
    return int(v) if v == int(v) else v


def _nearest(t: float, step: float) -> float:
    """Ближайшая линия сетки. Ровно посередине — вперёд: round() в Python округлял бы
    к чётной линии, и ноты 1/16 на сетке 1/8 разъезжались бы то вперёд, то назад."""
    return round(math.floor(t / step + 0.5 + EPS) * step, 4)


class EditorModel:
    def __init__(self, title: str = "Моя мелодия", bpm: float = 90.0,
                 notes: list[ENote] | None = None, path: Path | None = None):
        self.title, self.bpm, self.path = title, bpm, path
        self.notes: list[ENote] = sorted(notes or [], key=lambda x: (x.t, x.n))
        self.selected: set[ENote] = set()
        self.playhead = 0.0
        self.grid_i, self.length_i, self.hand = 1, 2, "auto"
        self.quantize = True
        self.metronome = True
        self.step_input = True
        self.playing = self.recording = False
        self.record_from = 0.0
        self.pending: dict[int, float] = {}   # запись: высота → начало нажатия (доли)
        self.chord: set[int] = set()          # ввод по шагам: зажатые клавиши аккорда
        self.chord_t = 0.0
        self.sounding: set[ENote] = set()
        self.events: list[tuple] = []
        self.dirty = False
        self._last = 0.0
        self._undo: list[tuple] = []
        self._redo: list[tuple] = []
        self._taps: list[float] = []

    @classmethod
    def from_lesson(cls, lesson: Lesson) -> EditorModel:
        return cls(lesson.title, lesson.bpm, [ENote(x.t, x.d, x.n, x.h) for x in lesson.notes], lesson.path)

    # --- сетка ---

    @property
    def grid(self) -> float:
        return GRIDS[self.grid_i][0]

    @property
    def length(self) -> float:
        return LENGTHS[self.length_i][0]

    def snap(self, t: float) -> float:
        return max(0.0, _nearest(t, self.grid if self.quantize else FREE_STEP))

    def snap_down(self, t: float) -> float:
        step = self.grid if self.quantize else FREE_STEP
        return max(0.0, round(math.floor(t / step + EPS) * step, 4))

    def hand_for(self, pitch: int) -> str:
        if self.hand == "auto":
            return "L" if pitch < SPLIT else "R"
        return self.hand

    def length_beats(self) -> float:
        return max((x.end for x in self.notes), default=0.0)

    def set_bpm(self, bpm: float) -> None:
        bpm = max(BPM_MIN, min(BPM_MAX, round(bpm)))
        if bpm != self.bpm:
            self.bpm, self.dirty = bpm, True

    def tap(self, now: float) -> None:
        """Тап-темп: стучите в такт — темп считается по последним ударам."""
        self._taps = [t for t in self._taps if now - t < 2.0][-3:] + [now]
        if len(self._taps) >= 3:
            gaps = [b - a for a, b in zip(self._taps, self._taps[1:])]
            self.set_bpm(60 / (sum(gaps) / len(gaps)))

    # --- отмена ---

    def snapshot(self) -> tuple:
        """Ноты и курсор: отмена шага возвращает и курсор, чтобы не оставалось дыры."""
        return [(x.t, x.d, x.n, x.h) for x in self.notes], self.playhead

    def push_undo(self, snap: tuple) -> None:
        self._undo.append(snap)
        del self._undo[:-UNDO_LIMIT]
        self._redo.clear()
        self.dirty = True

    def checkpoint(self) -> None:
        self.push_undo(self.snapshot())

    def _restore(self, snap: tuple) -> None:
        notes, playhead = snap
        self.notes = [ENote(*x) for x in notes]
        self.selected.clear()
        self.dirty = True
        if not self.playing:
            self.seek(playhead)

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self.snapshot())
        self._restore(self._undo.pop())
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self.snapshot())
        self._restore(self._redo.pop())
        return True

    # --- правка нот ---

    def add(self, t: float, pitch: int, d: float | None = None, hand: str | None = None) -> ENote:
        """Нота в точке t; нота той же высоты с тем же началом заменяется."""
        note = ENote(round(max(0.0, t), 4), round(d or self.length, 4), pitch, hand or self.hand_for(pitch))
        self.notes = [x for x in self.notes if not (x.n == pitch and abs(x.t - note.t) < EPS)]
        self.notes.append(note)
        self.notes.sort(key=lambda x: (x.t, x.n))
        self.dirty = True
        return note

    def note_at(self, t: float, pitch: int) -> ENote | None:
        for x in reversed(self.notes):
            if x.n == pitch and x.t - EPS <= t < x.end:
                return x
        return None

    def delete(self, notes) -> None:
        gone = set(notes)
        self.notes = [x for x in self.notes if x not in gone]
        self.selected -= gone
        self.sounding -= gone
        self.dirty = True

    def normalize(self) -> None:
        """После перетаскивания: по порядку и без дублей одной высоты в одной точке."""
        seen, out = set(), []
        for x in sorted(self.notes, key=lambda x: (x.t, x.n, x not in self.selected)):
            key = (x.n, round(x.t, 4))
            if key not in seen:
                seen.add(key)
                out.append(x)
        self.notes = out
        self.selected &= set(out)

    def transpose(self, dn: int) -> bool:
        sel = self.selected
        if not sel or not all(LOW <= x.n + dn <= HIGH for x in sel):
            return False
        self.checkpoint()
        for x in sel:
            x.n += dn
        self.normalize()
        return True

    def align(self) -> int:
        """«Выровнять»: уже стоящие ноты — на сетку (выбранные, а если ничего не выбрано — все).

        «Q» привязывает только то, что ставится сейчас; эта команда чинит ноты, сыгранные
        без «Q» или пришедшие из MIDI. Начало и конец встают на ближайшие линии текущей
        сетки, нота не короче шага сетки. Возвращает, сколько нот сдвинулось.
        """
        step = self.grid
        moved = []
        for x in (self.selected or self.notes):
            t = max(0.0, _nearest(x.t, step))
            end = max(t + step, _nearest(x.end, step))
            if abs(t - x.t) > EPS or abs(end - x.end) > EPS:
                moved.append((x, t, round(end - t, 4)))
        if moved:
            self.checkpoint()  # всё выравнивание — одна отмена
            for x, t, d in moved:
                x.t, x.d = t, d
            self.normalize()   # две ноты одной высоты сошлись в одну точку — остаётся одна
        return len(moved)

    def set_hand(self, hand: str) -> None:
        if self.selected:
            self.checkpoint()
            for x in self.selected:
                x.h = hand

    # --- ввод с клавиатуры ---

    def key_down(self, pitch: int) -> None:
        if self.recording:
            self.pending[pitch] = self.playhead
            return
        if self.playing or not self.step_input:
            return
        if not self.chord:  # новый шаг — одна точка отмены на весь аккорд
            self.checkpoint()
            self.selected.clear()
            self.chord_t = self.snap(self.playhead)
        self.chord.add(pitch)
        self.selected.add(self.add(self.chord_t, pitch))

    def key_up(self, pitch: int) -> None:
        if pitch in self.pending:
            self._close_pending(pitch)
        elif pitch in self.chord:
            self.chord.discard(pitch)
            if not self.chord:  # аккорд отпущен — курсор на следующий шаг
                self.seek(self.chord_t + self.length)

    def step_back(self) -> None:
        """Backspace при вводе по шагам: курсор на шаг назад, ноты на этом шаге удаляются."""
        t = max(0.0, self.snap(self.playhead) - self.length)
        victims = [x for x in self.notes if t - EPS <= x.t < t + self.length - EPS]
        if victims:
            self.checkpoint()
            self.delete(victims)
        self.seek(t)

    def _close_pending(self, pitch: int) -> None:
        start = self.pending.pop(pitch)
        if start < self.record_from - self.grid / 2:
            return  # нажали во время отсчёта — не нота
        step = self.grid if self.quantize else FREE_STEP
        d = max(step, _nearest(self.playhead - start, step))
        self.selected.add(self.add(self.snap(max(start, self.record_from)), pitch, d))

    # --- воспроизведение и запись ---

    def scale_tempo(self, factor: float) -> bool:
        """«×2» / «÷2»: темп и доли меняются вместе — мелодия звучит так же, меняется только запись."""
        bpm = self.bpm * factor
        if not BPM_MIN <= bpm <= BPM_MAX:
            return False
        self.stop()

        def scaled(snap):
            notes, playhead = snap
            return [(t * factor, d * factor, n, h) for t, d, n, h in notes], playhead * factor

        for x in self.notes:
            x.t, x.d = round(x.t * factor, 4), round(x.d * factor, 4)
        self._undo = [scaled(x) for x in self._undo]
        self._redo = [scaled(x) for x in self._redo]
        self.bpm = round(bpm, 2)
        self.seek(self.playhead * factor)
        self.dirty = True
        return True

    def seek(self, t: float) -> None:
        self.playhead = max(-BEATS_PER_BAR, round(t, 4))
        self._last = self.playhead - EPS
        self._silence()
        if self.playing:
            self.events.append(("start", self.playhead))

    def _silence(self) -> None:
        for x in self.sounding:
            self.events.append(("off", x.n))
        self.sounding.clear()

    def toggle_play(self) -> None:
        if self.playing:
            self.stop()
        else:
            self.playing = True
            self._last = self.playhead - EPS
            self.events.append(("start", self.playhead))

    def toggle_record(self) -> None:
        if self.recording:
            self.stop()
            return
        self.checkpoint()  # вся запись — одна отмена
        self.selected.clear()
        self.record_from = self.snap(self.playhead)
        self.seek(self.record_from - BEATS_PER_BAR)  # такт отсчёта
        self.recording = self.playing = True
        self.events.append(("start", self.playhead))

    def stop(self) -> None:
        for p in list(self.pending):
            self._close_pending(p)
        if self.recording and self.playhead < self.record_from:
            self.playhead = self.record_from
        if self.playing:
            self.events.append(("stop",))
        self.recording = self.playing = False
        self.chord.clear()
        self._silence()

    def update(self, dt: float) -> None:
        if not self.playing:
            return
        prev, now = self._last, self.playhead + dt * self.bpm / 60
        self.playhead = self._last = now
        count_in = self.recording and now < self.record_from
        if self.metronome or count_in:
            beats = range(math.floor(prev) + 1, math.floor(now) + 1)
            for k in beats[-2:]:
                self.events.append(("click", k % BEATS_PER_BAR == 0))
        for x in self.notes:
            if x.t > now:
                break
            if prev < x.t <= now and x not in self.sounding:
                self.sounding.add(x)
                self.events.append(("on", x.n))
        for x in [x for x in self.sounding if now >= x.end]:
            self.sounding.discard(x)
            self.events.append(("off", x.n))
        if not self.recording and now > self.length_beats() + 1:
            self.stop()

    def pop_events(self) -> list[tuple]:
        events, self.events = self.events, []
        return events

    def lit(self) -> set[int]:
        """Что горит на ленте: звучащие ноты, а в паузе — ноты под курсором."""
        if self.playing:
            return {x.n for x in self.sounding}
        return {x.n for x in self.notes if x.t - EPS <= self.playhead < x.end - EPS}

    # --- сохранение ---

    def to_lesson(self) -> dict:
        notes, out, step, first = sorted(self.notes, key=lambda x: (x.t, x.n)), [], -1, None
        for x in notes:
            if first is None or x.t - first > CHORD:
                step, first = step + 1, x.t
            out.append({"t": _num(x.t), "d": _num(x.d), "n": x.n, "h": x.h, "s": step})
        return {"title": self.title.strip() or "Моя мелодия", "bpm": _num(self.bpm), "notes": out}

    def save(self, folder: Path | None = None) -> Path:
        """Свои уроки перезаписываем, встроенные и новые сохраняем новым файлом."""
        if not self.notes:
            raise ValueError("добавьте хотя бы одну ноту")
        folder = Path(folder or paths.USER_LESSONS)
        own = self.path is not None and Path(self.path).resolve().parent == folder.resolve()
        target = Path(self.path) if own else free_name(folder, self.title.strip() or "Моя мелодия")
        save_lesson(self.to_lesson(), target)
        self.path, self.dirty = target, False
        return target
