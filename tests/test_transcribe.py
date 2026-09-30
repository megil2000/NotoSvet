import threading

import numpy as np
import pytest

from app import transcribe as T
from app.config import HIGH, LOW
from songs import ODE, SR, score, song


def run(bpm, grid=0.5, **kw):
    r = T.transcribe(song(ODE, bpm, **kw), grid=grid)
    found, extra = score(r.notes, ODE, grid, shift=kw.get("transpose", 0) + r.shift, ratio=bpm / r.bpm)
    return r, found, extra


@pytest.mark.parametrize("bpm, kw, min_found", [
    (100, dict(band=False), 1.0),                    # голос без группы
    (100, dict(), 1.0),                              # голос, аккорды, бас, барабаны
    (97.3, dict(), 1.0),                             # дробный темп
    (110, dict(chords=0.45, octave=1), 0.9),         # громкие аккорды в регистре мелодии
    (100, dict(noise=0.15, vibrato=False), 1.0),     # шум, ровный голос
    (120, dict(transpose=12), 1.0),                  # высокий голос — переносится в 25 клавиш
])
def test_melody_is_recognized(bpm, kw, min_found):
    r, found, extra = run(bpm, **kw)
    assert found >= min_found and extra <= 0.1
    assert all(LOW <= p <= HIGH for _, _, p in r.notes)


def test_tempo_found_exactly_or_doubled():
    for bpm in (66, 84, 128, 160):
        r, found, _ = run(bpm)
        assert r.bpm == pytest.approx(bpm, rel=0.01) or r.bpm == pytest.approx(2 * bpm, rel=0.01)
        assert found == 1.0


def test_melody_starts_on_beat_zero_and_grid_is_respected():
    r, _, _ = run(100, lead=3.7, grid=0.25)
    assert r.notes[0][0] == 0
    assert all(abs(t / 0.25 - round(t / 0.25)) < 1e-6 and d >= 0.25 for t, d, _ in r.notes)
    assert r.first_beat == pytest.approx(3.7, abs=0.08)  # доля 0 — там, где началась мелодия


def test_silence_gives_no_notes():
    r = T.transcribe(np.zeros(SR * 5, np.float32))
    assert r.notes == []


def test_cancel_stops_work():
    stop = threading.Event()
    stop.set()
    with pytest.raises(T.Cancelled):
        T.transcribe(song(ODE, 100), cancel=stop)
