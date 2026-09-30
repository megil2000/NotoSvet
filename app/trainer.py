"""Режимы обучения: «Ожидание», «Ритм», «Демонстрация».

Здесь только логика — без pygame, звука и отрисовки, поэтому её проверяют тесты.
Сессия копит события (звук ноты, щелчок метронома, ошибка, оценка нажатия),
а программа сама решает, как их озвучить и показать.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .lesson import Lesson

PERFECT = 0.060     # «Отлично» — ±60 мс
GOOD = 0.150        # «Хорошо» — ±150 мс, дальше — «Мимо»
FADE = 0.2          # верно нажатая подсказка гаснет за 0,2 с
NEXT_LEVEL = 20 / 127  # следующая нота горит на ~15 % (velocity 20 в протоколе)
TAIL = 1.0          # пауза после последней ноты перед итогом, с

MODES = {"wait": "Ожидание", "rhythm": "Ритм", "demo": "Демонстрация"}
HANDS = {"both": "обе руки", "R": "правая рука", "L": "левая рука"}
TEMPOS = (0.5, 0.75, 1.0)


@dataclass
class PlayNote:
    start: float                 # начало, с (с учётом темпа)
    dur: float                   # длительность, с
    pitch: int
    hand: str
    step: int
    auto: bool                   # играет программа: демо или «чужая» рука
    result: str | None = None    # ok / perfect / good / miss / auto
    hit_at: float | None = None  # реальное время нажатия — для угасания подсказки
    sounding: bool = False


@dataclass
class Step:
    time: float
    notes: list[PlayNote]
    required: frozenset[int]     # что нажимает ученик
    pressed: set[int] = field(default_factory=set)
    done: bool = False


def grade(accuracy: float) -> str:
    if accuracy >= 0.95:
        return "S"
    if accuracy >= 0.85:
        return "A"
    if accuracy >= 0.70:
        return "B"
    return "C"


class Session:
    def __init__(self, lesson: Lesson, mode: str = "wait", tempo: float = 1.0, hand: str = "both"):
        if hand != "both" and len(lesson.hands) < 2:
            hand = "both"
        self.lesson, self.mode, self.tempo, self.hand = lesson, mode, tempo, hand
        self.spb = 60.0 / (lesson.bpm * tempo)  # секунд в доле
        self.notes = [PlayNote(x.t * self.spb, x.d * self.spb, x.n, x.h, x.s,
                               auto=mode == "demo" or (hand != "both" and x.h != hand))
                      for x in lesson.notes]
        groups: dict[int, list[PlayNote]] = {}
        for n in self.notes:
            groups.setdefault(n.step, []).append(n)
        self.steps = sorted((Step(min(n.start for n in g), g, frozenset(n.pitch for n in g if not n.auto))
                             for g in groups.values()), key=lambda st: st.time)
        self._step_of = {st.notes[0].step: st for st in self.steps}
        self.end_time = max(n.start + n.dur for n in self.notes)
        self.first_required = min((n.start for n in self.notes if not n.auto), default=0.0)

        lead = 4 * self.spb if mode == "rhythm" else 2.0  # в «Ритме» — такт отсчёта под метроном
        self.song_time = -lead - 1e-6
        self.metronome = mode == "rhythm"
        self.catch_to = -math.inf
        self.cur = self._next_required(0)
        self.events: list[tuple] = []
        self.finished = False
        self.elapsed = 0.0
        self.started_at: float | None = None
        self.done_at: float | None = None
        self.presses = self.correct = self.errors = 0
        self.perfect = self.good = self.miss = 0
        self.score = self.combo = self.max_combo = 0

    # --- время ---

    def update(self, dt: float) -> None:
        if self.finished:
            return
        self.elapsed += dt
        prev = self.song_time
        t = prev + dt
        if self.mode == "wait":
            if t < self.catch_to:  # ученик сыграл раньше — ноты плавно догоняют
                t = max(t, prev + (self.catch_to - prev) * min(1.0, dt * 12))
            if self.cur < len(self.steps):
                t = min(t, self.steps[self.cur].time)  # ноты замирают у линии
        self.song_time = t
        if self.started_at is None and t >= self.first_required:
            self.started_at = self.elapsed
        self._metronome(prev, t)
        self._auto_notes()
        if self.mode == "rhythm":
            self._misses()
        if t >= self.end_time + TAIL and (self.mode != "wait" or self.cur >= len(self.steps)):
            self._finish()

    def _metronome(self, a: float, b: float) -> None:
        if not self.metronome or b <= a:
            return
        beats = range(math.floor(a / self.spb) + 1, math.floor(b / self.spb) + 1)
        for k in beats[-2:]:  # после резкого скачка времени — не пулемётом
            self.events.append(("click", k % 4 == 0))

    def _step_open(self, n: PlayNote) -> bool:
        st = self._step_of[n.step]
        return self.mode != "wait" or st.done or not st.required

    def _auto_notes(self) -> None:
        t = self.song_time
        for n in self.notes:
            if n.start > t:
                break  # ноты отсортированы по началу
            if not n.auto:
                continue
            if n.result is None and self._step_open(n):
                n.result, n.sounding = "auto", True
                self.events.append(("on", n.pitch))
            elif n.sounding and t >= n.start + n.dur:
                n.sounding = False
                self.events.append(("off", n.pitch))

    def _misses(self) -> None:
        t = self.song_time
        for n in self.notes:
            if n.start > t:
                break
            if not n.auto and n.result is None and t > n.start + GOOD:
                n.result = "miss"
                self.miss += 1
                self.combo = 0
                self.events.append(("judge", "miss", n.pitch))

    def _finish(self) -> None:
        self.finished = True
        if self.done_at is None:
            self.done_at = self.elapsed
        for n in self.notes:
            if n.sounding:
                n.sounding = False
                self.events.append(("off", n.pitch))

    # --- нажатия ---

    def press(self, pitch: int) -> bool | None:
        """Ученик нажал клавишу. True — верно, False — ошибка, None — не оценивается."""
        if self.finished or self.mode == "demo":
            return None
        if self.mode == "wait":
            return self._press_wait(pitch)
        return self._press_rhythm(pitch)

    def _next_required(self, i: int) -> int:
        while i < len(self.steps) and (self.steps[i].done or not self.steps[i].required):
            i += 1
        return i

    def _press_wait(self, pitch: int) -> bool | None:
        if self.cur >= len(self.steps):
            return None  # мелодия пройдена — дальше играем свободно
        if self.started_at is None:
            self.started_at = self.elapsed
        step = self.steps[self.cur]
        if pitch in step.pressed:
            return None  # повторное нажатие уже засчитанной ноты — не ошибка
        self.presses += 1
        if pitch not in step.required:
            self.errors += 1
            self.events.append(("error", pitch))
            return False
        self.correct += 1
        step.pressed.add(pitch)
        for n in step.notes:
            if n.pitch == pitch and not n.auto:
                n.result, n.hit_at = "ok", self.elapsed
        if step.required <= step.pressed:  # аккорд засчитан, когда нажаты все ноты
            step.done = True
            self.catch_to = max(self.catch_to, step.time)
            self.cur = self._next_required(self.cur + 1)
            if self.cur >= len(self.steps):
                self.done_at = self.elapsed
        return True

    def _press_rhythm(self, pitch: int) -> bool | None:
        t = self.song_time
        best, best_dt = None, 0.0
        for n in self.notes:
            if n.start > t + GOOD:
                break
            if n.auto or n.result is not None or n.pitch != pitch:
                continue
            dt = t - n.start
            if abs(dt) <= GOOD and (best is None or abs(dt) < abs(best_dt)):
                best, best_dt = n, dt
        if best is None:
            if t < self.first_required - GOOD:
                return None  # ещё идёт отсчёт — разминка не считается
            self.presses += 1
            self.errors += 1
            self.combo = 0
            self.events.append(("error", pitch))
            return False
        self.presses += 1
        self.correct += 1
        kind = "perfect" if abs(best_dt) <= PERFECT else "good"
        best.result, best.hit_at = kind, self.elapsed
        if kind == "perfect":
            self.perfect += 1
        else:
            self.good += 1
        self.combo += 1
        self.max_combo = max(self.max_combo, self.combo)
        bonus = min(1 + (self.combo // 10) * 0.5, 3.0)  # каждые 10 нот серии — +50 % очков
        self.score += round((300 if kind == "perfect" else 100) * bonus)
        self.events.append(("judge", kind, pitch))
        return True

    # --- что показать ---

    def pop_events(self) -> list[tuple]:
        events, self.events = self.events, []
        return events

    def current_notes(self) -> list[int]:
        """Что нажать сейчас (режим «Ожидание»)."""
        if self.mode != "wait" or self.cur >= len(self.steps):
            return []
        step = self.steps[self.cur]
        return sorted(step.required - step.pressed)

    def sounding(self) -> set[int]:
        return {n.pitch for n in self.notes if n.sounding}

    def hints(self) -> dict[int, float]:
        """Подсказки на ленте: высота → яркость 0..1."""
        out: dict[int, float] = {}
        if self.mode == "demo":
            return {p: 1.0 for p in self.sounding()}
        if self.mode == "wait":
            for p in self.current_notes():
                out[p] = 1.0
            nxt = self._next_required(self.cur + 1)
            if self.cur < len(self.steps) and nxt < len(self.steps):
                for p in self.steps[nxt].required:
                    out.setdefault(p, NEXT_LEVEL)
        else:
            t = self.song_time
            for n in self.notes:
                if n.start - self.spb > t:
                    break
                if n.auto or n.result is not None:
                    continue
                lvl = (t - (n.start - self.spb)) / self.spb  # разгорается за долю до нажатия
                if lvl > 0:
                    out[n.pitch] = max(out.get(n.pitch, 0.0), min(1.0, lvl))
        for n in self.notes:
            if n.hit_at is not None and self.elapsed - n.hit_at < FADE:
                out[n.pitch] = max(out.get(n.pitch, 0.0), 1.0 - (self.elapsed - n.hit_at) / FADE)
        return out

    def progress(self) -> float:
        return max(0.0, min(1.0, self.song_time / self.end_time)) if self.end_time > 0 else 0.0

    def result(self) -> dict:
        r = {"mode": self.mode, "tempo": round(self.tempo * 100), "hand": self.hand,
             "time": round((self.done_at or self.elapsed) - (self.started_at or 0.0), 1),
             "errors": self.errors,
             "accuracy": round(100 * self.correct / self.presses) if self.presses else 0}
        if self.mode == "rhythm":
            total = self.perfect + self.good + self.miss
            acc = (300 * self.perfect + 100 * self.good) / (300 * total) if total else 0.0
            r.update(accuracy=round(acc * 100), errors=self.errors + self.miss, grade=grade(acc),
                     score=self.score, max_combo=self.max_combo,
                     perfect=self.perfect, good=self.good, miss=self.miss)
        return r
