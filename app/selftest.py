"""Самопроверка без окна: что всё нужное попало в сборку и работает.

Запуск: python main.py --selftest или NotoSvet --selftest. Результаты пишет во временную
папку, чтобы не портить историю попыток.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def selftest() -> int:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    import mido
    import pygame

    from . import draw as D
    from . import paths
    from .converter import import_file
    from .display import Display
    from .game import App

    tmp = Path(tempfile.mkdtemp(prefix="notosvet-selftest-"))
    paths.RECORDS = tmp / "records.json"
    paths.SETTINGS = tmp / "settings.json"
    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.init()
    display = Display("НотоСвет — самопроверка", pygame.Surface((32, 32)))
    app = App(display.canvas, display)
    checks = [("встроенные уроки", len(app.lessons) >= 5 and not app.lesson_errors, f"{len(app.lessons)} шт."),
              ("звук", app.synth.ok, str(pygame.mixer.get_init()))]

    try:
        ports = mido.get_input_names()  # грузит бэкенд rtmidi
        checks.append(("MIDI", True, f"портов: {len(ports)}"))
    except Exception as e:  # noqa: BLE001 — здесь важна любая причина
        checks.append(("MIDI", False, repr(e)))

    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    for n in (60, 64, 67, 72):
        track += [mido.Message("note_on", note=n, velocity=80), mido.Message("note_off", note=n, time=480)]
    mid.save(tmp / "проба.mid")
    try:
        lesson_path = import_file(tmp / "проба.mid", tmp / "lessons")
        checks.append(("импорт MIDI", lesson_path.exists(), lesson_path.name))
    except ValueError as e:
        checks.append(("импорт MIDI", False, str(e)))

    paths.USER_LESSONS = tmp / "my_lessons"  # редактор сохраняет во временную папку
    app.new_melody()
    for chord in ((60,), (64, 67), (72,)):  # ввод по шагам, второй шаг — аккорд
        for p in chord:
            app.note_down(p)
        for p in chord:
            app.note_up(p)
    app.step(1 / 30)
    app.editor.save()
    saved = [x for x in app.lessons if x.path and x.path.parent == paths.USER_LESSONS.resolve()]
    ok = len(saved) == 1 and saved[0].steps == 3 and not app.editor.m.dirty
    checks.append(("редактор: ввод по шагам и сохранение", ok, saved[0].path.name if saved else "нет файла"))
    app.editor.exit()
    app.opt["lesson"] = 0

    # мелодия из песни: WAV → звук → распознавание (гамма до мажор дважды, 120 уд/мин)
    import wave

    import numpy as np

    from .audio import load
    from .transcribe import SR, transcribe
    scale = [60, 62, 64, 65, 67, 69, 71, 72] * 2
    t = np.arange(int(0.45 * SR)) / SR
    parts = [np.zeros(SR // 2)]
    for m in scale:
        tone = sum(np.sin(2 * np.pi * 440 * 2 ** ((m - 69) / 12) * h * t) / h for h in range(1, 6))
        parts += [tone * np.minimum(1, t / 0.02), np.zeros(int(0.05 * SR))]
    x = np.concatenate(parts)
    with wave.open(str(tmp / "гамма.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((0.3 * x / np.abs(x).max() * 32767).astype(np.int16).tobytes())
    try:
        r = transcribe(load(tmp / "гамма.wav").mono)
        pitches = [p for _, _, p in r.notes]
        checks.append(("мелодия из песни", pitches == scale and r.bpm == 120, f"{len(pitches)} нот, {r.bpm:g} уд/мин"))
    except Exception as e:  # noqa: BLE001 — в отчёт любую причину
        checks.append(("мелодия из песни", False, repr(e)))
    checks += neural_checks(np.float32(0.3) * x.astype(np.float32) / np.abs(x).max(), scale)

    app.start_session()  # первый урок в режиме «Ожидание», играем без ошибок
    for _ in range(60 * 120):
        if app.state != "play":
            break
        for p in app.session.current_notes():
            app.note_down(p)
            app.note_up(p)
        app.step(1 / 30)
    ok = app.state == "results" and app.result["accuracy"] == 100
    checks.append(("урок пройден", ok, f"точность {app.result['accuracy'] if app.result else '—'} %"))
    app.draw()
    display.present()
    checks.append(("кадр нарисован и выведен", app.screen.get_at((D.W // 2, D.H - 20))[:3] != D.BG, ""))
    app.screen = D.Canvas(pygame.Surface((2 * D.W, 2 * D.H)), 2.0)  # как на Retina
    app.draw()
    white = app.screen.surface.get_at((app.screen.px(D.W // 2), app.screen.px(D.H - 20)))[:3] == D.KEY_WHITE
    checks.append(("кадр для Retina", white, "2560×1440"))

    for name, passed, info in checks:
        print(f"{'OK  ' if passed else 'FAIL'} {name} {info}")
    pygame.quit()
    return 0 if all(passed for _, passed, _ in checks) else 1


def neural_checks(x, scale: list[int]) -> list[tuple[str, bool, str]]:
    """Нейросети, если они есть: Basic Pitch слышит гамму, Demucs делит звук на дорожки.
    Без PyTorch это не ошибка — песни распознаются старым способом."""
    import numpy as np

    from . import neural
    if not neural.available():
        return [("нейросети", True, "не установлены — «Мелодия из песни» работает старым способом")]
    out = []
    try:
        import torch

        from . import basic_pitch as BP
        note, onset, _ = BP.predict(x)
        pitches = [p for _, _, p, _ in neural.melody_notes(note, onset, 40, 88)]
        out.append(("нейросеть Basic Pitch", pitches == scale, f"{len(pitches)} нот, PyTorch {torch.__version__}"))
    except Exception as e:  # noqa: BLE001 — в отчёт любую причину
        out.append(("нейросеть Basic Pitch", False, repr(e)))
    try:
        from . import separate as S
        if S.weights_path() is None:
            out.append(("нейросеть Demucs", True, "веса скачаются при первом распознавании"))
        else:
            stereo = np.repeat(x[:2 * 22050, None], 2, axis=1).repeat(2, axis=0)  # 2 с, 44 100 Гц
            stems = S.separate(stereo)
            ok = sorted(stems) == sorted(S.SOURCES) and all(v.shape == stereo.shape for v in stems.values())
            out.append(("нейросеть Demucs", ok, f"{len(stems)} дорожек"))
    except Exception as e:  # noqa: BLE001
        out.append(("нейросеть Demucs", False, repr(e)))
    return out
