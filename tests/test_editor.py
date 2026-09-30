import pytest

from app.editor import EditorModel, ENote
from app.lesson import load_lesson, parse_lesson
from app.trainer import Session


def run(m, beats, fps=100):
    """Прокрутить воспроизведение на beats долей; вернуть события."""
    events = []
    for _ in range(round(beats * 60 / m.bpm * fps)):
        m.update(1 / fps)
        events += m.pop_events()
    return events


def test_snapping_follows_grid_and_quantize():
    m = EditorModel()
    m.grid_i = 1  # 1/8 = пол-доли
    assert m.snap(1.26) == 1.5 and m.snap_down(1.26) == 1.0 and m.snap(-3) == 0.0
    m.quantize = False
    assert m.snap(1.26) == pytest.approx(1.25, abs=1 / 48)


def test_align_puts_existing_notes_on_the_grid():
    # сыграно вживую без «Q»: чуть раньше, чуть позже, 1/16 на сетке 1/8, дубль одной высоты
    m = EditorModel(notes=[ENote(0.02, 0.46, 60, "R"), ENote(0.97, 0.9, 62, "R"), ENote(1.25, 0.25, 64, "R"),
                           ENote(1.76, 0.1, 65, "R"), ENote(3.02, 1, 67, "R"), ENote(2.98, 1, 67, "R")])
    m.grid_i = 1  # 1/8
    assert m.align() == 6
    assert [(x.t, x.d, x.n) for x in m.notes] == [(0, 0.5, 60), (1.0, 1.0, 62), (1.5, 0.5, 64),
                                                  (2.0, 0.5, 65), (3.0, 1.0, 67)]
    assert m.align() == 0                      # уже на сетке — ничего не меняется
    m.undo()                                   # всё выравнивание отменяется разом
    assert [x.t for x in m.notes] == [0.02, 0.97, 1.25, 1.76, 2.98, 3.02]


def test_align_touches_only_selected_notes():
    m = EditorModel(notes=[ENote(0.1, 1, 60, "R"), ENote(1.1, 1, 62, "R")])
    m.grid_i = 0  # 1/4
    m.selected = {m.notes[1]}
    assert m.align() == 1
    assert [x.t for x in m.notes] == [0.1, 1.0]


def test_step_input_places_notes_and_chords_one_step_each():
    m = EditorModel()
    m.length_i = 2  # 1/4 = доля
    m.key_down(60); m.key_up(60)
    m.key_down(64); m.key_down(67); m.key_up(64); m.key_up(67)  # аккорд
    m.key_down(72); m.key_up(72)
    assert [(x.t, x.n) for x in m.notes] == [(0, 60), (1, 64), (1, 67), (2, 72)]
    assert m.playhead == 3
    assert [x.h for x in m.notes] == ["R"] * 4
    m.undo()                                   # отмена снимает последний шаг и курсор
    assert len(m.notes) == 3 and m.playhead == 2
    m.step_back()                              # Backspace: курсор назад, аккорд удалён
    assert m.playhead == 1 and [x.n for x in m.notes] == [60]


def test_auto_hand_and_forced_hand():
    m = EditorModel()
    assert m.add(0, 55).h == "L" and m.add(0, 60).h == "R"
    m.hand = "L"
    assert m.add(1, 70).h == "L"


def test_same_pitch_same_start_is_replaced():
    m = EditorModel()
    m.add(0, 60, 1)
    m.add(0, 60, 2)
    assert len(m.notes) == 1 and m.notes[0].d == 2


def test_recording_quantizes_after_count_in():
    m = EditorModel(bpm=120)
    m.grid_i = 1  # 1/8
    m.toggle_record()
    assert m.playhead == -4                    # такт отсчёта
    clicks = [e for e in run(m, 4.1) if e[0] == "click"]
    assert clicks[0] == ("click", True) and len(clicks) >= 4
    m.key_down(62)                             # чуть позже доли 0
    run(m, 0.9)
    m.key_up(62)
    m.key_down(64)                             # держим до остановки
    run(m, 0.3)
    m.stop()
    assert [(x.t, x.n) for x in m.notes] == [(0, 62), (1.0, 64)]
    assert m.notes[0].d == 1.0 and not m.recording
    m.undo()
    assert m.notes == []                       # вся запись отменяется разом


def test_playback_plays_notes_in_time_and_stops_at_end():
    m = EditorModel(bpm=60, notes=[ENote(0, 1, 60, "R"), ENote(1, 1, 64, "R")])
    m.metronome = False
    m.toggle_play()
    events = run(m, 3.5)
    assert [e for e in events if e[0] == "on"] == [("on", 60), ("on", 64)]
    assert ("off", 60) in events and not m.playing


def test_transpose_keeps_notes_on_keyboard():
    m = EditorModel(notes=[ENote(0, 1, 71, "R")])
    m.selected = set(m.notes)
    assert m.transpose(1) and m.notes[0].n == 72
    assert not m.transpose(1)                  # выше C5 нельзя


def test_tap_tempo():
    m = EditorModel()
    for i in range(4):
        m.tap(i * 0.5)
    assert m.bpm == 120


def test_save_roundtrip_is_a_playable_lesson(tmp_path):
    m = EditorModel("Моя песня", 100)
    for p in (60, 64, 67):
        m.key_down(p)
    for p in (60, 64, 67):
        m.key_up(p)
    m.key_down(55); m.key_up(55)
    path = m.save(tmp_path)
    assert path.name == "Моя песня.json" and not m.dirty
    lesson = load_lesson(path)
    assert (lesson.title, lesson.bpm, lesson.steps, lesson.hands) == ("Моя песня", 100, 2, ["L", "R"])
    assert m.save(tmp_path) == path            # свой урок перезаписывается
    s = Session(lesson, "wait")
    assert s.current_notes() == [60, 64, 67]


def test_editing_builtin_lesson_saves_a_copy(tmp_path):
    builtin = tmp_path / "builtin" / "Гамма.json"
    lesson = parse_lesson({"title": "Гамма", "bpm": 80, "notes": [{"t": 0, "d": 1, "n": 60}]}, builtin)
    m = EditorModel.from_lesson(lesson)
    path = m.save(tmp_path / "user")
    assert path.parent == tmp_path / "user" and not builtin.exists()


def test_empty_melody_is_not_saved(tmp_path):
    with pytest.raises(ValueError):
        EditorModel().save(tmp_path)


def test_tempo_x2_and_half_keep_the_sound():
    m = EditorModel(bpm=160, notes=[ENote(0, 1, 60, "R"), ENote(2, 2, 64, "R")])
    m.add(4, 67, 1)
    m.checkpoint()
    assert m.scale_tempo(0.5)
    assert m.bpm == 80 and [(x.t, x.d) for x in m.notes] == [(0, 0.5), (1, 1), (2, 0.5)]
    m.undo()                                   # отмена после «÷2» — в новых долях
    assert [x.t for x in m.notes] == [0, 1, 2]
    assert not EditorModel(bpm=200).scale_tempo(2)


def test_original_follows_play_seek_and_stop():
    m = EditorModel(notes=[ENote(0, 1, 60, "R")])
    m.toggle_play()
    m.seek(2)
    m.stop()
    kinds = [e for e in m.pop_events() if e[0] in ("start", "stop")]
    assert kinds == [("start", 0.0), ("start", 2), ("stop",)]
