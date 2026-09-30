"""Синтетические «песни» для проверки распознавания: мелодия голосом с вибрато,
аккорды, бас и барабаны — известны точные ноты и темп."""
import numpy as np

SR = 22050

ODE = [(b, d, n) for b, d, n in [
    (0, 1, 64), (1, 1, 64), (2, 1, 65), (3, 1, 67), (4, 1, 67), (5, 1, 65), (6, 1, 64), (7, 1, 62),
    (8, 1, 60), (9, 1, 60), (10, 1, 62), (11, 1, 64), (12, 1.5, 64), (13.5, 0.5, 62), (14, 2, 62),
    (16, 1, 64), (17, 1, 64), (18, 1, 65), (19, 1, 67), (20, 1, 67), (21, 1, 65), (22, 1, 64), (23, 1, 62),
    (24, 1, 60), (25, 1, 60), (26, 1, 62), (27, 1, 64), (28, 1.5, 62), (29.5, 0.5, 60), (30, 2, 60)]]


def hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def voice(freq: float, dur: float, vibrato: bool = True) -> np.ndarray:
    t = np.arange(int(dur * SR)) / SR
    depth = 0.017 * np.clip((t - 0.15) / 0.2, 0, 1) if vibrato else 0.0  # ±30 центов
    phase = 2 * np.pi * np.cumsum(freq * (1 + depth * np.sin(2 * np.pi * 5.5 * t))) / SR
    wave = sum(np.sin(h * phase) / h for h in range(1, 13) if freq * h < SR / 2)
    env = np.clip(np.minimum(t / 0.03, (dur - t) / 0.06), 0, 1)
    return wave * env


def song(melody, bpm: float, lead: float = 1.3, band: bool = True, transpose: int = 0, seed: int = 1,
         chords: float = 0.08, bass: float = 0.25, noise: float = 0.0, vibrato: bool = True, octave: int = 0) -> np.ndarray:
    spb = 60.0 / bpm
    beats = max(b + d for b, d, _ in melody)
    out = np.zeros(int((lead + beats * spb + 1.0) * SR))
    rng = np.random.default_rng(seed)

    def add(sig, at):
        i = int(at * SR)
        out[i:i + len(sig)] += sig[:max(0, len(out) - i)]

    for b, d, m in melody:
        add(0.5 * voice(hz(m + transpose), d * spb * 0.92, vibrato), lead + b * spb)
    if band:
        roots = [r + 12 * octave for r in (48, 55, 48, 55, 48, 55, 48, 55)]
        for bar in range(int(np.ceil(beats / 4))):
            root = roots[bar % len(roots)] + transpose
            t = np.arange(int(4 * spb * SR)) / SR
            env = np.clip(np.minimum(t / 0.05, (4 * spb - t) / 0.1), 0, 1)
            chord = sum(np.sin(2 * np.pi * hz(root + i) * t) + 0.3 * np.sin(4 * np.pi * hz(root + i) * t)
                        for i in (0, 4, 7))
            add(chords * chord * env + bass * np.sin(2 * np.pi * hz(root - 12) * t) * env, lead + bar * 4 * spb)
        for k in range(int(beats * 2)):
            at = lead + k * spb / 2
            if k % 4 == 0:  # бочка на 1 и 3
                t = np.arange(int(0.15 * SR)) / SR
                add(0.6 * np.sin(2 * np.pi * (60 + 60 * np.exp(-t * 30)) * t) * np.exp(-t * 20), at)
            if k % 4 == 2:  # малый на 2 и 4
                add(0.25 * rng.standard_normal(int(0.1 * SR)) * np.exp(-np.arange(int(0.1 * SR)) / SR * 30), at)
            hat = np.diff(rng.standard_normal(int(0.03 * SR) + 1))
            add(0.05 * hat * np.exp(-np.arange(len(hat)) / SR * 150), at)
    out += noise * rng.standard_normal(len(out))
    return (0.9 * out / np.abs(out).max()).astype(np.float32)


def score(result_notes, truth, grid: float = 0.5, shift: int = 0, ratio: float = 1.0):
    """Доля нот мелодии, найденных с верной высотой и началом (± шаг сетки), и доля лишних.
    ratio — истинный темп / найденный: темп «вдвое быстрее» не ошибка, ноты просто длиннее в долях."""
    found = 0
    if not result_notes:
        return 0.0, 0.0
    result_notes = [(t * ratio, d * ratio, p) for t, d, p in result_notes]
    grid *= max(1.0, ratio)
    t0 = result_notes[0][0] - truth[0][0]
    for b, d, n in truth:
        if any(abs((t - t0) - b) <= grid + 1e-6 and p == n + shift for t, _, p in result_notes):
            found += 1
    extra = sum(1 for t, _, p in result_notes
                if not any(abs((t - t0) - b) <= grid + 1e-6 and p == n + shift for b, _, n in truth))
    return found / len(truth), extra / len(result_notes)
