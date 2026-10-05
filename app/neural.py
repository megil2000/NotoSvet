"""Распознавание нейросетями: песня → дорожки инструментов → ноты каждой дорожки, как в Songsterr.

1. Demucs (separate.py) делит фрагмент на шесть дорожек: вокал, гитара, пианино, бас,
   барабаны и «прочее» (синтезаторы, струнные, духовые).
2. Basic Pitch (basic_pitch.py) слушает каждую дорожку отдельно и по кадрам говорит,
   какие ноты звучат и где начинаются.
3. Из этих вероятностей собираются ноты. Вокал и бас — одноголосо: путь Витерби, одна нота
   в момент времени, скачок высоты стоит штраф; повтор одной ноты делится по атаке.
   Гитара, пианино и «прочее» — аккордами, как в самой Basic Pitch: нота начинается
   на пике атаки и тянется, пока сеть её слышит.
4. Темп и доли — по дорожке барабанов (там атаки чище), сетка у всех дорожек общая.
5. Каждая дорожка выравнивается по сетке и сдвигается октавами в 25 клавиш.

Нейросети работают на PyTorch. Если его нет, остаётся старый способ из transcribe.py:
одна дорожка «Мелодия» без разделения.
"""
from __future__ import annotations

import bisect
import functools
import importlib.util
import math
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import transcribe as T
from .config import HIGH, LOW

if TYPE_CHECKING:
    from .audio import Audio

FPS = 22050 / 256           # кадров Basic Pitch в секунду
LOWEST = 21                 # MIDI первой высоты сети (A0)
STEMS = {                   # дорожка → (название, как собирать ноты)
    "vocals": ("Вокал", "melody"),
    "guitar": ("Гитара", "chords"),
    "piano": ("Пианино", "chords"),
    "bass": ("Бас", "bass"),
    "other": ("Прочее", "chords"),
    "drums": ("Барабаны", "drums"),
}
ORDER = ("vocals", "guitar", "piano", "bass", "other", "drums")   # порядок в списке, как в Songsterr
RANGES = {"melody": (40, 88), "bass": (24, 64), "chords": (28, 100)}  # где искать ноты, MIDI
PREFER = {"melody": (60, HIGH), "bass": (LOW, 59), "chords": (LOW, HIGH)}  # куда сдвигать октавами
QUIET = 0.04                # дорожка тише 4 % песни — пустая: туда только просочились соседи
DRUMS_FOR_TEMPO = 0.1       # барабаны громче 10 % песни — темп по ним, иначе по всей песне
MAX_CHORD = 4               # в аккорде не больше 4 нот — чтобы его можно было сыграть
MAX_SECONDS = 6 * 60        # длиннее фрагмент — не хватит памяти на шесть дорожек
# одноголосая партия: штрафы Витерби (в логарифмах вероятности) и порог «нота или пауза»
VOICING, JUMP, STEP, SWITCH = 0.3, 2.5, 0.05, 1.2
SPLIT = 0.7                 # повтор ноты: атака сильнее этого (вибрато даёт ложные атаки до ~0,6)
MIN_NOTE = 0.08             # короче — не нота, а призвук, с
# аккорды: пороги Basic Pitch
ONSET_THRESH, FRAME_THRESH, ENERGY_TOL, MIN_CHORD_NOTE = 0.5, 0.3, 11, 0.12
MIN_AMPLITUDE = 0.35        # слабее — отзвук соседней дорожки
OVERTONE = 1.3              # нота над другой, слабее её в 1,3 раза и больше, — её гармоника


@dataclass
class Track:
    key: str                                    # vocals / guitar / … / melody (старый способ)
    name: str                                   # «Вокал»
    kind: str                                   # melody / chords / bass / drums
    notes: list[tuple[float, float, int, str]]  # (доля, длительность в долях, MIDI, рука): на сетке, в 25 клавишах
    start: float                                # секунда фрагмента, где у этой дорожки доля 0
    shift: int                                  # на сколько полутонов перенесено в 25 клавиш
    raw: list[tuple[float, float, int]]         # (начало, длительность в секундах, MIDI) — как услышано
    level: float = 1.0                          # громкость дорожки относительно песни
    pcm: np.ndarray | None = None               # звук дорожки во фрагменте, int16 как у микшера
    env: np.ndarray | None = None               # громкость по 10 мс — волна в редакторе

    @property
    def playable(self) -> bool:
        return bool(self.notes)


@dataclass
class Recognition:
    bpm: float
    tracks: list[Track]
    neural: bool
    grid: float = 0.5        # шаг сетки, по которой выровнены ноты, доли
    offset: int = 0          # с какого сэмпла песни (в частоте микшера) начинается звук дорожек
    duration: float = 0.0    # длина фрагмента, с

    def best(self) -> int:
        """Какую дорожку предложить первой: голос, если он есть, иначе ту, где больше нот."""
        playable = [i for i, t in enumerate(self.tracks) if t.playable]
        if not playable:
            return 0
        for i in playable:
            if self.tracks[i].kind == "melody":
                return i
        return max(playable, key=lambda i: len(self.tracks[i].notes))


@functools.cache
def available() -> bool:
    """Есть ли PyTorch и Demucs. Сами библиотеки не грузим — это секунды и сотни мегабайт."""
    try:
        return all(importlib.util.find_spec(m) is not None for m in ("torch", "demucs"))
    except (ImportError, ValueError):
        return False


def _check(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise T.Cancelled


# --- ноты из вероятностей сети ---

def viterbi(emis: np.ndarray, trans: np.ndarray, cancel=None) -> np.ndarray:
    """Самый вероятный путь по состояниям: emis — кадры × состояния, trans — из × в (логарифмы)."""
    n, states = emis.shape
    back = np.empty((n, states), np.int16)
    score = emis[0].copy()
    cols = np.arange(states)
    for t in range(1, n):
        if t % 2000 == 0:
            _check(cancel)
        cand = score[:, None] + trans
        b = cand.argmax(axis=0)
        back[t] = b
        score = cand[b, cols] + emis[t]
    path = np.empty(n, np.int16)
    path[-1] = int(score.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def melody_notes(note: np.ndarray, onset: np.ndarray, lo: int, hi: int,
                 cancel=None) -> list[tuple[int, int, int, float]]:
    """Одноголосая партия → ноты (кадр начала, кадр конца, MIDI, сила).

    Состояния — высоты lo..hi и пауза. Остаться на ноте ничего не стоит, сменить её —
    штраф JUMP плюс STEP за каждый полутон, нота↔пауза — SWITCH. Поэтому вибрато
    и случайный кадр чужой высоты не рвут ноту. Одна высота подряд делится на ноты
    там, где сеть слышит новую атаку.
    """
    obs = note[:, lo - LOWEST:hi - LOWEST + 1]
    n, k = obs.shape
    if n == 0:
        return []
    emis = np.empty((n, k + 1), np.float32)
    emis[:, :k] = np.log(np.maximum(obs, 1e-4))
    emis[:, k] = math.log(VOICING)
    idx = np.arange(k)
    trans = np.zeros((k + 1, k + 1), np.float32)
    trans[:k, :k] = -(JUMP + STEP * np.abs(idx[:, None] - idx[None, :]))
    trans[idx, idx] = 0.0
    trans[:k, k] = trans[k, :k] = -SWITCH
    path = viterbi(emis, trans, cancel)

    min_frames = max(2, round(MIN_NOTE * FPS))
    notes, t = [], 0
    while t < n:
        s = int(path[t])
        if s == k:
            t += 1
            continue
        a = t
        while t < n and path[t] == s:
            t += 1
        col = onset[:, s + lo - LOWEST]
        cuts = [a]
        for i in range(a + min_frames, t - min_frames):
            if col[i] >= SPLIT and col[i] >= col[i - 1] and col[i] > col[i + 1] and i - cuts[-1] >= min_frames:
                cuts.append(i)
        cuts.append(t)
        for x, y in zip(cuts, cuts[1:]):
            if y - x >= min_frames:
                notes.append((x, y, lo + s, float(obs[x:y, s].mean())))
    return notes


def _onsets_with_jumps(onset: np.ndarray, note: np.ndarray) -> np.ndarray:
    """Атаки сети плюс резкие подъёмы громкости ноты — как get_infered_onsets в Basic Pitch."""
    diffs = [note - np.vstack([np.zeros((d, note.shape[1]), note.dtype), note[:-d]]) for d in (1, 2)]
    jump = np.maximum(np.minimum(*diffs), 0)
    jump[:2] = 0
    if jump.max() > 0:
        jump *= onset.max() / jump.max()
    return np.maximum(onset, jump)


def chord_notes(note: np.ndarray, onset: np.ndarray, lo: int, hi: int,
                cancel=None) -> list[tuple[int, int, int, float]]:
    """Многоголосая партия → ноты (кадр начала, кадр конца, MIDI, сила). Алгоритм Basic Pitch.

    Сначала ноты от пиков атак (с конца песни к началу): нота тянется, пока сеть её слышит.
    Потом «трюк Melodia»: из оставшегося — самые громкие длинные ноты без явной атаки.
    """
    n = len(note)
    frames, onsets = note.copy(), onset.copy()
    for x in (frames, onsets):  # вне диапазона инструмента не ищем
        x[:, :lo - LOWEST] = 0
        x[:, hi - LOWEST + 1:] = 0
    onsets = _onsets_with_jumps(onsets, frames)
    peaks = np.zeros_like(onsets, dtype=bool)
    peaks[1:-1] = (onsets[1:-1] > onsets[:-2]) & (onsets[1:-1] > onsets[2:]) & (onsets[1:-1] >= ONSET_THRESH)
    remaining = frames.copy()
    min_len = round(MIN_CHORD_NOTE * FPS)
    out = []

    def add(a: int, b: int, f: int) -> None:
        if b - a > min_len:
            out.append((a, b, f + LOWEST, float(frames[a:b, f].mean())))

    times, freqs = np.nonzero(peaks)
    for a, f in sorted(zip(times.tolist(), freqs.tolist()), reverse=True):
        if a >= n - 1:
            continue
        i, quiet = a + 1, 0
        while i < n - 1 and quiet < ENERGY_TOL:
            quiet = quiet + 1 if remaining[i, f] < FRAME_THRESH else 0
            i += 1
        b = i - quiet
        if b - a > min_len:  # нота взята — её энергия и соседние полутона больше не в счёт
            remaining[a:b, max(0, f - 1):f + 2] = 0
            add(a, b, f)
    _check(cancel)

    # клетки только обнуляются, поэтому «взять самую громкую» по очереди = обойти их по убыванию
    cells = np.flatnonzero(remaining > FRAME_THRESH)
    for cell in cells[np.argsort(-remaining.flat[cells], kind="stable")]:
        if remaining.flat[cell] <= FRAME_THRESH:
            continue
        mid, f = divmod(int(cell), remaining.shape[1])
        end, quiet = mid + 1, 0
        while end < n - 1 and quiet < ENERGY_TOL:
            quiet = quiet + 1 if remaining[end, f] < FRAME_THRESH else 0
            end += 1
        start, quiet_back = mid - 1, 0
        while start > 0 and quiet_back < ENERGY_TOL:
            quiet_back = quiet_back + 1 if remaining[start, f] < FRAME_THRESH else 0
            start -= 1
        remaining[start + 1:end, max(0, f - 1):f + 2] = 0  # просмотренное — уже не в счёт
        add(start + 1 + quiet_back, end - quiet, f)
    return drop_overtones(out)


def drop_overtones(notes: list[tuple[int, int, int, float]]) -> list[tuple[int, int, int, float]]:
    """Убрать обертоны: нота на октаву, дуодециму или две октавы выше другой, начатая вместе
    с ней и заметно слабее, — это её гармоника, а не отдельная нота."""
    together = round(0.06 * FPS)
    notes = sorted(notes)
    starts = [x[0] for x in notes]
    out = []
    for a, b, p, s in notes:
        near = notes[bisect.bisect_left(starts, a - together):bisect.bisect_right(starts, a + together)]
        if not any(p - p2 in (12, 19, 24) and s2 >= OVERTONE * s for _, _, p2, s2 in near):
            out.append((a, b, p, s))
    return out


# --- сетка и клавиатура ---

def fit_octaves(pitches: list[int], prefer: tuple[int, int]) -> int:
    """Сдвиг октавами: больше всего нот в 25 клавишах, из них побольше в удобной зоне, сдвиг поменьше."""
    def key(s):
        inside = sum(LOW <= p + s <= HIGH for p in pitches)
        zone = sum(prefer[0] <= p + s <= prefer[1] for p in pitches)
        return inside, zone, -abs(s)
    return max(range(-48, 49, 12), key=key)


def place(notes: list[tuple[float, float, int, float]], kind: str, bpm: float, beat0: float,
          grid: float) -> tuple[list[tuple[float, float, int, str]], float, int]:
    """Ноты в секундах (начало, конец, MIDI, сила) → на сетку, в 25 клавиш, по рукам.

    Доля 0 — там, где начинается первая нота дорожки. Возвращает ноты, секунду доли 0, сдвиг.
    """
    if not notes:
        return [], beat0, 0
    spb = 60.0 / bpm
    beats = [((a - beat0) / spb, (b - beat0) / spb, p, s) for a, b, p, s in notes]
    base = math.floor(min(x[0] for x in beats) + grid / 2)
    cells: dict[tuple, tuple[float, float, int, float]] = {}
    one_voice = kind in ("melody", "bass")
    for sb, eb, p, s in beats:
        t = max(0.0, round((sb - base) / grid) * grid)
        d = max(grid, round((eb - base) / grid) * grid - t)
        key = t if one_voice else (t, p)          # одноголосо: в точке одна нота
        old = cells.get(key)
        if old is None or s * d > old[3] * old[1]:
            cells[key] = (t, d, p, s)
    picked = sorted(cells.values())
    if not one_voice:  # в аккорде — самые сильные ноты
        by_t: dict[float, list] = {}
        for x in picked:
            by_t.setdefault(x[0], []).append(x)
        picked = sorted(x for group in by_t.values() for x in sorted(group, key=lambda x: -x[3])[:MAX_CHORD])

    shift = fit_octaves([p for _, _, p, _ in picked], PREFER[kind])
    placed: dict[tuple[float, int], tuple[float, float, int]] = {}
    for t, d, p, _ in picked:
        p += shift
        while p < LOW:
            p += 12
        while p > HIGH:
            p -= 12
        if (t, p) not in placed or d > placed[t, p][1]:  # после переноса две ноты совпали — оставляем длинную
            placed[t, p] = (t, d, p)
    out = sorted(placed.values())
    starts = sorted({t for t, _, _ in out})
    nxt = {a: b for a, b in zip(starts, starts[1:])}
    same: dict[int, float] = {}
    result = []
    for t, d, p in reversed(out):                  # нота не тянется за начало следующей той же высоты,
        if one_voice and t in nxt:                 # а одноголосая — за начало любой следующей
            d = min(d, nxt[t] - t)
        if p in same:
            d = min(d, same[p] - t)
        same[p] = t
        hand = "R" if kind == "melody" else "L" if kind == "bass" else ("L" if p < 60 else "R")
        result.append((round(t, 4), round(max(d, grid), 4), p, hand))
    return sorted(result), beat0 + base * spb, shift


# --- звук ---

def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0


def _halfband(x: np.ndarray) -> np.ndarray:
    """44 100 → 22 050 Гц: фильтр ниже 10 кГц и каждый второй сэмпл."""
    taps = np.arange(-32, 33)
    h = np.sinc(taps * 0.45) * 0.45 * np.hamming(len(taps))
    return np.convolve(x, (h / h.sum()).astype(np.float32), mode="same")[::2].astype(np.float32)


def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return x
    pos = np.arange(0, len(x), src / dst)
    return np.stack([np.interp(pos, np.arange(len(x)), x[:, c]) for c in range(x.shape[1])], axis=1).astype(np.float32)


def _envelope(mono: np.ndarray, rate: int) -> np.ndarray:
    step = max(1, rate // 100)
    blocks = np.abs(mono[:len(mono) // step * step]).reshape(-1, step).max(axis=1)
    return (blocks / (blocks.max() + 1e-9)).astype(np.float32)


# --- распознавание ---

def recognize(audio: Audio, start: float, end: float, grid: float = 0.5, engine: str = "neural",
              progress=None, cancel: threading.Event | None = None) -> Recognition:
    """Фрагмент песни → дорожки с нотами. engine: neural — нейросети, classic — старый способ."""
    report = progress or (lambda stage, frac: None)
    if engine != "neural":
        r = T.transcribe(audio.mono_segment(start, end), grid, report, cancel)
        track = Track("melody", "Мелодия", "melody", [(t, d, p, "R") for t, d, p in r.notes],
                      r.first_beat, r.shift, r.raw)
        return Recognition(r.bpm, [track], neural=False, grid=grid, duration=end - start)
    if end - start > MAX_SECONDS:
        raise ValueError(f"нейросеть берёт не больше {MAX_SECONDS // 60} минут — выделите кусок покороче")

    from . import basic_pitch as BP
    from . import separate as S

    if S.weights_path() is None:
        S.download(lambda f: report(f"Скачиваю нейросеть Demucs ({S.SIZE_MB} МБ)", f), cancel)
    report("Загружаю нейросети", 0.02)
    S.load(cancel=cancel)
    BP.model()

    rate, channels = audio.rate, audio.pcm.shape[1]
    first_sample = int(start * rate)
    pcm = audio.pcm[first_sample:int(end * rate)].astype(np.float32) / 32768.0
    stereo = _resample(pcm if channels == 2 else np.repeat(pcm[:, :1], 2, axis=1), rate, S.RATE)
    stems = S.separate(stereo, lambda f: report("Разделяю песню на дорожки", 0.03 + 0.77 * f), cancel)

    mix_rms = _rms(stereo) + 1e-9
    levels = {key: _rms(stem) / mix_rms for key, stem in stems.items()}
    heard: dict[str, list[tuple[float, float, int, float]]] = {}
    pitched = [k for k in ORDER if STEMS[k][1] != "drums" and levels[k] >= QUIET]
    for i, key in enumerate(pitched):
        name, kind = STEMS[key]
        lo, hi = 0.8 + 0.15 * i / len(pitched), 0.8 + 0.15 * (i + 1) / len(pitched)
        note, onset, _ = BP.predict(_halfband(stems[key].mean(axis=1)),
                                    lambda f: report(f"Ноты: {name.lower()}", lo + (hi - lo) * f), cancel)
        decode = chord_notes if kind == "chords" else melody_notes
        heard[key] = [(x / FPS, y / FPS, p, s) for x, y, p, s in decode(note, onset, *RANGES[kind], cancel=cancel)
                      if kind != "chords" or s >= MIN_AMPLITUDE]

    report("Темп и сетка", 0.96)
    beat_src = (_halfband(stems["drums"].mean(axis=1)) if levels["drums"] >= DRUMS_FOR_TEMPO
                else audio.mono_segment(start, end))
    _, flux = T.spectrogram(beat_src, lambda stage, frac: None, cancel)
    bpm, phase = T.tempo(flux)
    beat0 = T.frame_time(phase)

    tracks = []
    for key in ORDER:
        name, kind = STEMS[key]
        notes = heard.get(key, [])
        placed, first, shift = place(notes, kind, bpm, beat0, grid)
        sound = _resample(stems.pop(key), S.RATE, rate)  # pop — шесть дорожек float32 в памяти не держим
        sound = sound if channels == 2 else sound.mean(axis=1, keepdims=True)
        tracks.append(Track(key, name, kind, placed, first, shift,
                            [(round(x, 3), round(y - x, 3), p) for x, y, p, _ in notes], round(levels[key], 3),
                            np.clip(sound * 32768.0, -32768, 32767).astype(np.int16),
                            _envelope(sound.mean(axis=1), rate)))
    report("Готово", 1.0)
    return Recognition(bpm, tracks, neural=True, grid=grid, offset=first_sample, duration=end - start)
