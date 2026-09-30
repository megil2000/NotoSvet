"""Распознавание мелодии из песни: звук → ноты на бит-сетке.

Только numpy, без нейросетей:
1. Спектр кадрами по 93 мс с шагом 11,6 мс.
2. Для каждой высоты от G2 до E6 — «заметность»: сумма амплитуд её гармоник с убывающими
   весами. Голос или соло-инструмент обычно дают самую заметную высоту в кадре.
3. Путь мелодии через кадры ищется алгоритмом Витерби: скачок высоты и пауза стоят штраф,
   поэтому мелодия не дрожит между соседними нотами (вибрато, шум).
4. Кадры одной высоты склеиваются в ноты; повтор одной ноты отделяется по провалу громкости.
5. Темп и доли — по автокорреляции огибающей атак, как анализ BPM в диджейских программах.
6. Ноты выравниваются по сетке и сдвигаются октавами в 25 клавиш.

Работает на чёткой мелодии; в плотном миксе возможны ошибки — их правят в редакторе.
"""
from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .config import HIGH, LOW

SR = 22050
WIN, HOP, NFFT = 2048, 256, 8192
FPS = SR / HOP                       # ≈ 86 кадров в секунду
PITCH_LO, PITCH_HI = 43, 88          # G2..E6 — где обычно звучит мелодия
K = PITCH_HI - PITCH_LO + 1
SUB = 3                              # сетка высот — треть полутона
HARMONICS = 10
H_DECAY = 0.84
F_MAX = 5000.0
VOICING = 0.2                        # ниже этой заметности кадр — пауза
JUMP, STEP, SWITCH = 2.0, 0.08, 1.5  # штрафы Витерби: смена ноты, за полутон, нота↔пауза
MIN_NOTE = 0.07                      # короче — не нота, а призвук, с
BPM_LO, BPM_HI = 60.0, 180.0

Progress = Callable[[str, float], None]


class Cancelled(Exception):
    """Распознавание отменили."""


@dataclass
class Transcription:
    bpm: float
    notes: list[tuple[float, float, int]]   # (доля, длительность в долях, MIDI): на сетке и в 25 клавишах
    first_beat: float                       # секунда фрагмента, где стоит доля 0
    shift: int                              # на сколько полутонов перенесено в 25 клавиш
    raw: list[tuple[float, float, int]]     # (начало, длительность в секундах, MIDI) — как услышано


def frame_time(frame: float) -> float:
    return (frame * HOP + WIN / 2) / SR


def _check(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled


# --- 1. спектр ---

def spectrogram(x: np.ndarray, progress: Progress, cancel=None) -> tuple[np.ndarray, np.ndarray]:
    """Амплитудный спектр до 5 кГц (кадры × частоты) и огибающая атак (spectral flux)."""
    x = np.asarray(x, np.float32)
    if len(x) < WIN:
        x = np.pad(x, (0, WIN - len(x)))
    n = 1 + (len(x) - WIN) // HOP
    nb = int(F_MAX / (SR / NFFT)) + 2
    window = np.hanning(WIN).astype(np.float32)
    mag = np.empty((n, nb), np.float32)
    offsets = np.arange(WIN)
    for a in range(0, n, 256):
        _check(cancel)
        b = min(n, a + 256)
        frames = x[np.arange(a, b)[:, None] * HOP + offsets] * window
        mag[a:b] = np.abs(np.fft.rfft(frames, n=NFFT, axis=1)[:, :nb])
        progress("Спектр", 0.5 * b / n)
    log = np.log1p(mag * (1000.0 / (mag.max() + 1e-9)))
    flux = np.maximum(0.0, np.diff(log, axis=0, prepend=log[:1])).sum(axis=1)
    return mag, flux


# --- 2. заметность высот ---

def steady_floor(mag: np.ndarray, seconds: float = 1.5) -> np.ndarray:
    """Медленный «фон» спектра: то, что звучит почти всё время (аккорды-подложка, гул)."""
    n, block = len(mag), max(8, int(seconds * FPS))
    if n < 2 * block:
        return np.zeros_like(mag)
    starts = np.arange(0, n - block // 2, block // 2)
    floors = np.stack([np.percentile(mag[a:a + block], 30, axis=0) for a in starts])
    centers = starts + block / 2
    t = np.arange(n)
    idx = np.clip(np.searchsorted(centers, t) - 1, 0, len(centers) - 2)
    w = np.clip((t - centers[idx]) / (centers[idx + 1] - centers[idx]), 0, 1)[:, None]
    return (floors[idx] * (1 - w) + floors[idx + 1] * w).astype(np.float32)


def salience(mag: np.ndarray) -> np.ndarray:
    """Заметность каждой высоты (кадры × K полутонов) с поправкой на строй записи."""
    mag = np.maximum(mag - 0.9 * steady_floor(mag), 0)  # мелодия меняется, подложка — нет
    nb = mag.shape[1]
    freqs_bins = np.arange(nb) * (SR / NFFT)
    tilt = freqs_bins ** 2 / (freqs_bins ** 2 + 150.0 ** 2)  # приглушаем бас и бочку
    m = np.sqrt(mag) * tilt.astype(np.float32)

    cands = PITCH_LO + (np.arange(K * SUB) - SUB // 2) / SUB
    f0 = 440.0 * 2 ** ((cands - 69) / 12)
    sal = np.zeros((mag.shape[0], len(cands)), np.float32)

    def at(freq: np.ndarray) -> np.ndarray:
        pos = np.clip(freq / (SR / NFFT), 0, nb - 2)
        i0 = pos.astype(int)
        frac = (pos - i0).astype(np.float32)
        return m[:, i0] * (1 - frac) + m[:, i0 + 1] * frac

    for h in range(1, HARMONICS + 1):
        f = f0 * h
        w = (H_DECAY ** (h - 1)) * (f < F_MAX)
        sal += at(f) * w.astype(np.float32)
    for h in range(1, 4):  # энергия «между гармониками» — признак, что нота на октаву ниже
        sal -= 0.5 * at(f0 * (h - 0.5)) * np.float32(H_DECAY ** (h - 1))
    np.maximum(sal, 0, out=sal)

    # строй записи бывает не A=440: выбираем сдвиг сетки, на котором больше всего энергии
    phase = int(np.argmax([sal[:, j::SUB].sum() for j in range(SUB)]))
    sal = np.roll(sal, SUB // 2 - phase, axis=1)
    return sal.reshape(len(sal), K, SUB).max(axis=2)


# --- 3. путь мелодии ---

def track(obs: np.ndarray, loud: np.ndarray, cancel=None) -> np.ndarray:
    """Витерби по состояниям «полутон 0..K-1» и «пауза» (K). obs — заметность, 1 ≈ громкая нота."""
    n = len(obs)
    states = K + 1
    emis = np.empty((n, states), np.float32)
    emis[:, :K] = np.log(obs + 1e-4)
    emis[:, K] = math.log(VOICING)
    emis[~loud, :K] = -30.0
    idx = np.arange(K)
    trans = np.zeros((states, states), np.float32)
    trans[:K, :K] = -(JUMP + STEP * np.abs(idx[:, None] - idx[None, :]))
    trans[idx, idx] = 0.0
    trans[:K, K] = trans[K, :K] = -SWITCH
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


def segment(path: np.ndarray, obs: np.ndarray) -> list[tuple[int, int, int]]:
    """Путь → ноты (кадр начала, кадр конца, MIDI). Повтор одной ноты делим по провалу заметности."""
    notes, n, t = [], len(path), 0
    min_frames = max(2, round(MIN_NOTE * FPS))
    while t < n:
        k = int(path[t])
        if k == K:
            t += 1
            continue
        a = t
        while t < n and path[t] == k:
            t += 1
        s = np.convolve(obs[a:t, k], np.ones(3) / 3, mode="same")
        cuts = [a]
        for i in range(5, len(s) - 5):
            left, right = s[max(0, i - 25):i].max(), s[i + 1:i + 26].max()
            if s[i] <= s[i - 1] and s[i] <= s[i + 1] and s[i] < 0.5 * min(left, right) and a + i - cuts[-1] >= 5:
                cuts.append(a + i)
        cuts.append(t)
        for x, y in zip(cuts, cuts[1:]):
            if y - x >= min_frames:
                notes.append((x, y, PITCH_LO + k))
    return notes


# --- 5. темп и доли ---

def tempo(flux: np.ndarray) -> tuple[float, float]:
    """Темп (уд/мин) и кадр первой доли по огибающей атак."""
    n = len(flux)
    if n < FPS * 3 or not np.any(flux > 0):
        return 100.0, 0.0
    w = int(FPS)
    env = np.maximum(flux - np.convolve(flux, np.ones(w) / w, mode="same"), 0.0).astype(np.float64)
    spec = np.fft.rfft(env - env.mean(), n=2 * n)
    ac = np.fft.irfft(np.abs(spec) ** 2)[:n]
    ac /= ac[0] + 1e-12
    grid = np.arange(n)

    def acv(lag):
        return np.interp(lag, grid, ac, right=0.0)

    bpms = np.arange(BPM_LO, BPM_HI + 0.01, 0.25)
    lags = 60 * FPS / bpms
    score = sum(acv(k * lags) / k for k in range(1, 5))
    # темп «вдвое» неоднозначен даже для людей; при сомнении берём быстрый: так ритм не теряется,
    # а медленный легко получить в редакторе кнопкой «÷2»
    prior = np.exp(-0.5 * (np.log2(bpms / 130.0) / 0.6) ** 2)
    bpm = float(bpms[np.argmax(np.maximum(score, 0) * prior)])
    fine = np.arange(bpm - 1, bpm + 1.001, 0.02)  # уточняем по длинной гребёнке
    bpm = float(fine[np.argmax(sum(acv(k * 60 * FPS / fine) for k in range(1, 17)))])
    bpm = float(round(bpm)) if abs(bpm - round(bpm)) < 0.25 else round(bpm, 1)

    period = 60 * FPS / bpm
    beats = np.arange(0, n / period + 1) * period
    phases = np.arange(0, period, 0.5)
    phase = float(phases[np.argmax([np.interp(p + beats, grid, env, right=0.0).sum() for p in phases])])
    return bpm, phase


# --- 6. сетка и клавиатура ---

def fit_keyboard(pitches: list[int]) -> int:
    """Сдвиг октавами: больше всего нот в C3–C5, из них побольше в правой руке (от C4), сдвиг поменьше."""
    def key(s):
        inside = sum(LOW <= p + s <= HIGH for p in pitches)
        right = sum(60 <= p + s <= HIGH for p in pitches)
        return inside, right, -abs(s)
    return max(range(-36, 37, 12), key=key)


def to_grid(notes: list[tuple[int, int, int]], obs: np.ndarray, bpm: float, phase: float,
            grid: float) -> tuple[list[tuple[float, float, int]], float, int]:
    period = 60 * FPS / bpm
    beats = [((a - phase) / period, (b - phase) / period, p, float(obs[a:b, p - PITCH_LO].mean()))
             for a, b, p in notes]
    # сильную долю такта по звуку не угадать — мелодия просто начинается с доли 0
    base = math.floor(min(x[0] for x in beats) + grid / 2) if beats else 0
    by_start: dict[float, tuple[float, float, int, float]] = {}
    for sb, eb, p, strength in beats:
        t = max(0.0, round((sb - base) / grid) * grid)
        e = round((eb - base) / grid) * grid
        d = max(grid, e - t)
        old = by_start.get(t)
        if old is None or strength * d > old[3] * old[1]:  # мелодия одноголосая: в точке — одна нота
            by_start[t] = (t, d, p, strength)
    starts = sorted(by_start)
    out = []
    for i, t in enumerate(starts):
        _, d, p, _ = by_start[t]
        if i + 1 < len(starts):
            d = min(d, starts[i + 1] - t)
        out.append((round(t, 4), round(d, 4), p))
    shift = fit_keyboard([p for *_, p in out])
    fitted = []
    for t, d, p in out:
        p += shift
        while p < LOW:
            p += 12
        while p > HIGH:
            p -= 12
        fitted.append((t, d, p))
    first_beat = frame_time(phase + base * period)
    return fitted, first_beat, shift


def transcribe(samples: np.ndarray, grid: float = 0.5, progress: Progress | None = None,
               cancel: threading.Event | None = None) -> Transcription:
    """samples — моно, SR = 22050 Гц. grid — шаг сетки в долях (0,5 = восьмые)."""
    report = progress or (lambda stage, frac: None)
    mag, flux = spectrogram(samples, report, cancel)
    _check(cancel)
    report("Высоты", 0.55)
    sal = salience(mag)
    peak = sal.max(axis=1)
    obs = sal / (np.percentile(peak, 95) + 1e-9)
    energy = mag.sum(axis=1)
    loud = energy > 0.03 * (np.percentile(energy, 95) + 1e-9)
    _check(cancel)
    report("Мелодия", 0.7)
    path = track(obs, loud, cancel)
    notes = segment(path, obs)
    report("Темп и сетка", 0.9)
    bpm, phase = tempo(flux)
    fitted, first_beat, shift = to_grid(notes, obs, bpm, phase, grid)
    raw = [(round(frame_time(a), 3), round((b - a) / FPS, 3), p) for a, b, p in notes]
    report("Готово", 1.0)
    return Transcription(bpm, fitted, first_beat, shift, raw)
