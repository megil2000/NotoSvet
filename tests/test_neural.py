import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import pygame  # noqa: E402

from app import neural as N  # noqa: E402
from app.audio import Audio  # noqa: E402
from app.config import HIGH, LOW  # noqa: E402
from songs import ODE, SR, song  # noqa: E402

needs_torch = pytest.mark.skipif(not N.available(), reason="нужны PyTorch и Demucs: requirements-ai.txt")


def posteriorgram(notes, frames, noise=0.02, seed=0):
    """Выход сети «по заказу»: notes — (кадр начала, кадр конца, MIDI, атака)."""
    rng = np.random.default_rng(seed)
    note = (noise * rng.random((frames, 88))).astype(np.float32)
    onset = (noise * rng.random((frames, 88))).astype(np.float32)
    for a, b, p, attack in notes:
        note[a:b, p - N.LOWEST] = 0.85
        onset[a, p - N.LOWEST] = attack
    return note, onset


def frames(sec):
    return round(sec * N.FPS)


# --- ноты из вероятностей ---

def test_melody_one_voice_with_repeats_and_false_attacks():
    plan = [(frames(0.5), frames(1.0), 64, 0.9), (frames(1.0), frames(1.5), 64, 0.9),  # повтор — новая атака
            (frames(1.5), frames(2.5), 67, 0.9), (frames(3.0), frames(3.5), 60, 0.9)]
    note, onset = posteriorgram(plan, frames(4))
    onset[frames(2.0), 67 - N.LOWEST] = 0.55          # вибрато: слабая «атака» посреди ноты — не разрез
    note[frames(1.8):frames(1.85), 79 - N.LOWEST] = 0.9  # кадр чужой высоты — не нота
    got = N.melody_notes(note, onset, 40, 88)
    assert [p for _, _, p, _ in got] == [64, 64, 67, 60]
    assert [round(a / N.FPS, 1) for a, _, _, _ in got] == [0.5, 1.0, 1.5, 3.0]


def test_chords_found_and_overtones_dropped():
    plan = [(frames(0.5), frames(1.5), p, 0.9) for p in (48, 52, 55)] + [(frames(2.0), frames(3.0), 57, 0.9)]
    note, onset = posteriorgram(plan, frames(4))
    a, b = frames(0.5), frames(1.5)
    note[a:b, 60 - N.LOWEST] = 0.45   # октава над до — слабая: обертон
    onset[a, 60 - N.LOWEST] = 0.6
    got = sorted((round(x / N.FPS, 1), p) for x, _, p, _ in N.chord_notes(note, onset, 28, 100))
    assert got == [(0.5, 48), (0.5, 52), (0.5, 55), (2.0, 57)]


def test_silence_gives_no_notes():
    note, onset = posteriorgram([], frames(3))
    assert N.melody_notes(note, onset, 40, 88) == [] and N.chord_notes(note, onset, 28, 100) == []


# --- сетка и клавиатура ---

def test_place_melody_on_grid_from_beat_zero():
    spb = 0.5  # 120 уд/мин, доля 0 — на 1,0 с
    notes = [(1.0 + b * spb + 0.03, 1.0 + (b + d) * spb, n + 12, 0.8) for b, d, n in ODE[:8]]
    placed, start, shift = N.place(notes, "melody", 120, 1.0, 0.5)
    assert start == pytest.approx(1.0) and placed[0][0] == 0
    assert [(t, p) for t, _, p, _ in placed] == [(b, n + 12 + shift) for b, _, n in ODE[:8]]
    assert all(LOW <= p <= HIGH and h == "R" for *_, p, h in placed)
    assert all(a[0] + a[1] <= b[0] for a, b in zip(placed, placed[1:]))  # одноголосо: ноты не наезжают


def test_place_bass_goes_to_left_hand_and_chords_are_capped():
    bass = [(k * 0.5, k * 0.5 + 0.4, p, 0.9) for k, p in enumerate((28, 33, 35, 40))]
    placed, _, shift = N.place(bass, "bass", 120, 0.0, 0.5)
    assert shift == 24 and [p for *_, p, _ in placed] == [52, 57, 59, 64]
    assert all(h == "L" for *_, h in placed)

    chord = [(0.0, 1.0, p, s) for p, s in ((48, 0.9), (52, 0.8), (55, 0.7), (60, 0.6), (64, 0.5), (67, 0.4))]
    placed, _, _ = N.place(chord, "chords", 120, 0.0, 0.5)
    assert sorted(p for *_, p, _ in placed) == [48, 52, 55, 60]   # четыре самые сильные
    assert {p: h for *_, p, h in placed} == {48: "L", 52: "L", 55: "L", 60: "R"}


def test_classic_engine_is_one_melody_track():
    x = song(ODE, 100)
    pcm = np.repeat((x * 32767).astype(np.int16)[:, None], 2, axis=1)
    audio = Audio("Ода", None, pcm, SR, x, np.ones(len(x) // (SR // 100), np.float32))
    rec = N.recognize(audio, 0, len(x) / SR, engine="classic")
    assert not rec.neural and [t.name for t in rec.tracks] == ["Мелодия"] and rec.bpm == 100
    assert len(rec.tracks[0].notes) == len(ODE) and rec.best() == 0


# --- «минус» и дорожки в редакторе ---

def fake_recognition(rate=100):
    """Песня из трёх «инструментов» по 4 с, фрагмент — с 1-й секунды."""
    n = 6 * rate
    parts = {k: np.full((n, 2), v, np.int16) for k, v in (("vocals", 100), ("bass", 20), ("drums", 3))}
    song_pcm = np.zeros((n + 2 * rate, 2), np.int16)
    song_pcm[rate:rate + n] = sum(p.astype(np.int16) for p in parts.values())
    audio = Audio("Песня", None, song_pcm, rate, np.zeros(1, np.float32), np.ones(80, np.float32))
    tracks = [N.Track("vocals", "Вокал", "melody", [(0.0, 1.0, 64, "R"), (1.0, 1.0, 67, "R")], 0.5, 0, [(0.5, 0.5, 64)],
                      0.6, parts["vocals"], np.ones(600, np.float32)),
              N.Track("bass", "Бас", "bass", [(0.0, 2.0, 50, "L")], 1.0, 24, [(1.0, 1.0, 26)], 0.3, parts["bass"]),
              N.Track("drums", "Барабаны", "drums", [], 0.0, 0, [], 0.2, parts["drums"])]
    return audio, N.Recognition(120, tracks, neural=True, offset=rate, duration=6.0)


def test_reference_plays_song_minus_track_or_track_alone():
    from app.editor_view import Reference
    audio, rec = fake_recognition()
    ref = Reference(audio, 1.5, rec.tracks[0], rec.offset)
    song_part = ref.sound("song", 0.0, 8.0)
    assert (song_part[100:700] == 123).all() and not song_part[:100].any()
    assert (ref.sound("solo", 0.0, 8.0)[100:700] == 100).all()
    minus = ref.sound("minus", 0.5, 2.0)  # кусок до фрагмента и внутри: до — песня как есть (тишина)
    assert not minus[:50].any() and (minus[50:] == 23).all()


def test_app_tracks_screen_and_editor_tabs(tmp_path, monkeypatch):
    from app import draw as D
    from app import paths
    from app.game import App
    monkeypatch.setattr(paths, "RECORDS", tmp_path / "records.json")
    monkeypatch.setattr(paths, "USER_LESSONS", tmp_path / "lessons")
    pygame.init()
    app = App(D.Canvas(pygame.Surface((D.W, D.H))))
    audio, rec = fake_recognition()
    app.open_tracks(rec, audio, 1.0)
    assert app.state == "tracks" and app.tracks.sel == 0  # первым предложен голос
    app.draw()
    app.tracks.play(2)                                    # у барабанов нот нет — остаёмся
    assert app.state == "tracks" and app.toast[1]
    app.tracks.play(1)
    ed = app.editor
    assert app.state == "editor" and [p.track.name for p in ed.parts] == ["Вокал", "Бас"]
    assert ed.m.title == "Песня — Бас" and ed.sound == "minus" and ed.ref.start == pytest.approx(2.0)
    app.draw()
    ed.select_part(0)
    assert ed.m.title == "Песня — Вокал" and ed.ref.track.key == "vocals" and ed.ref.start == pytest.approx(1.5)
    ed.save()
    assert not ed.m.dirty and ed.dirty                    # бас ещё не сохранён — выход с предупреждением
    assert not ed.can_leave() and ed.can_leave()


# --- сами нейросети ---

@needs_torch
def test_basic_pitch_hears_the_melody():
    from app import basic_pitch as BP
    for transpose in (0, 12):
        note, onset, contour = BP.predict(song(ODE, 100, band=False, transpose=transpose))
        assert note.shape[1] == 88 and contour.shape[1] == 264 and len(note) == len(onset)
        got = N.melody_notes(note, onset, 40, 88)
        truth = [(1.3 + b * 0.6, n + transpose) for b, _, n in ODE]
        found = sum(any(abs(a / N.FPS - t) < 0.08 and p == n for a, _, p, _ in got) for t, n in truth)
        assert found == len(ODE) and len(got) == len(ODE)


@needs_torch
def test_basic_pitch_frames_are_on_time():
    from app import basic_pitch as BP
    t = np.arange(int(0.5 * SR)) / SR
    x = np.zeros(SR * 6, np.float32)
    for at in (1.0, 4.4):  # вторая нота — в другом окне сети
        x[int(at * SR):int(at * SR) + len(t)] = 0.5 * np.sin(2 * np.pi * 440 * t)
    note, onset, _ = BP.predict(x)
    peaks = [k for k in range(1, len(onset) - 1) if onset[k, 69 - N.LOWEST] > 0.5
             and onset[k, 69 - N.LOWEST] >= onset[k - 1, 69 - N.LOWEST] and onset[k, 69 - N.LOWEST] > onset[k + 1, 69 - N.LOWEST]]
    assert [round(BP.frame_time(k), 1) for k in peaks] == [1.0, 4.4]


@needs_torch
def test_cancel_stops_basic_pitch():
    import threading

    from app import basic_pitch as BP
    from app.transcribe import Cancelled
    stop = threading.Event()
    stop.set()
    with pytest.raises(Cancelled):
        BP.predict(song(ODE, 100), cancel=stop)


needs_demucs = pytest.mark.skipif(
    not N.available() or __import__("app.separate", fromlist=["x"]).weights_path() is None,
    reason="нужны PyTorch, Demucs и скачанные веса Demucs")


@needs_demucs
def test_cancel_stops_separation():
    import threading

    from app import separate as S
    from app.transcribe import Cancelled
    stop, steps = threading.Event(), []

    def progress(frac):
        steps.append(frac)
        stop.set()  # отменили после первого куска

    stereo = np.repeat(song(ODE, 100)[:, None], 2, axis=1).repeat(2, axis=0)
    with pytest.raises(Cancelled):
        S.separate(stereo, progress, stop)
    assert len(steps) == 1


@needs_demucs
def test_full_song_is_split_into_tracks():
    x = song(ODE, 100)
    pcm = np.repeat((x * 32767).astype(np.int16)[:, None], 2, axis=1).repeat(2, axis=0)  # 44 100 Гц
    audio = Audio("Ода", None, pcm, 44100, x, np.ones(len(x) // (SR // 100), np.float32))
    rec = N.recognize(audio, 0.0, len(x) / SR, 0.5)
    assert rec.neural and [t.key for t in rec.tracks] == list(N.ORDER) and rec.bpm in (100, 200)
    assert all(t.pcm is not None and len(t.pcm) == len(pcm) for t in rec.tracks)
    best = max(rec.tracks, key=lambda t: sum(any(p == n for *_, p in t.raw) for n in {n for *_, n in ODE}))
    heard = {p for *_, p in best.raw}
    assert {n for *_, n in ODE} <= heard  # все высоты мелодии нашлись в одной дорожке
