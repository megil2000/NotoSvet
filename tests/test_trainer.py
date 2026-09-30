import pytest

from app.config import HIGH, LOW
from app.lesson import load_lessons, parse_lesson
from app.trainer import Session


def lesson(notes, bpm=60):
    return parse_lesson({"title": "тест", "bpm": bpm, "notes": notes})


def advance_to(s, t, step=0.01):
    while s.song_time < t - 1e-9 and not s.finished:
        s.update(min(step, t - s.song_time))


def events(s, kind):
    return [e for e in s.pop_events() if e[0] == kind]


def test_beats_become_seconds_with_tempo():
    s = Session(lesson([{"t": 1, "d": 2, "n": 60}], bpm=60), "rhythm", tempo=0.5)
    assert s.spb == 2.0
    assert (s.notes[0].start, s.notes[0].dur) == (2.0, 4.0)


def test_out_of_range_notes_move_by_octave_and_steps_are_assigned():
    les = lesson([{"t": 0, "d": 1, "n": 36}, {"t": 0.01, "d": 1, "n": 84}, {"t": 1, "d": 1, "n": 60}])
    assert [n.n for n in les.notes] == [48, 72, 60]
    assert [n.s for n in les.notes] == [0, 0, 1]


def test_wait_mode_freezes_until_correct_key():
    s = Session(lesson([{"t": 0, "d": 1, "n": 60}, {"t": 1, "d": 1, "n": 62}]), "wait")
    s.update(10.0)
    assert s.song_time == 0.0                       # ноты замерли у линии
    assert s.hints() == {60: 1.0, 62: pytest.approx(20 / 127)}
    assert s.press(61) is False                     # не та клавиша — ошибка
    assert events(s, "error") == [("error", 61)]
    assert s.press(60) is True
    assert s.current_notes() == [62]
    s.update(10.0)
    assert s.song_time == 1.0
    assert (s.presses, s.correct, s.errors) == (2, 1, 1)


def test_wait_mode_chord_needs_every_note():
    s = Session(lesson([{"t": 0, "d": 1, "n": n} for n in (60, 64, 67)] + [{"t": 1, "d": 1, "n": 60}]), "wait")
    assert s.press(60) and s.press(64)
    assert s.current_notes() == [67]
    assert s.press(64) is None                      # повтор засчитанной ноты — не ошибка
    assert s.press(67)
    assert s.current_notes() == [60]


def test_wait_mode_repeated_note_needs_new_press():
    s = Session(lesson([{"t": 0, "d": 1, "n": 64}, {"t": 1, "d": 1, "n": 64}]), "wait")
    assert s.press(64)
    assert s.current_notes() == [64]
    assert s.press(64)
    assert s.current_notes() == []
    advance_to(s, s.end_time + 2)
    assert s.finished
    r = s.result()
    assert (r["errors"], r["accuracy"]) == (0, 100)


def test_early_press_catches_notes_up():
    s = Session(lesson([{"t": 0, "d": 1, "n": 60}, {"t": 1, "d": 1, "n": 62}, {"t": 2, "d": 1, "n": 64}]), "wait")
    s.update(0.5)                                   # идёт вступление: song_time = -1.5
    s.press(60)
    s.press(62)
    for _ in range(30):
        s.update(1 / 60)
    assert 1.0 <= s.song_time < 1.2                 # за полсекунды догнали сыгранную ноту
    s.update(2.0)
    assert s.song_time == 2.0                       # и ждут третью


def test_rhythm_judgement_windows():
    s = Session(lesson([{"t": 0, "d": 1, "n": 60}, {"t": 1, "d": 1, "n": 62}, {"t": 2, "d": 1, "n": 64}]), "rhythm")
    assert s.song_time < -3.9                       # такт отсчёта
    advance_to(s, 0.03)
    assert s.press(60) is True
    advance_to(s, 1.1)
    assert s.press(62) is True
    advance_to(s, 2.2)                              # третью пропустили
    judged = [e[1] for e in events(s, "judge")]
    assert judged == ["perfect", "good", "miss"]
    assert s.press(67) is False                     # лишнее нажатие
    advance_to(s, s.end_time + 2)
    r = s.result()
    assert r["grade"] == "C" and r["accuracy"] == round(100 * 400 / 900)
    assert (r["perfect"], r["good"], r["miss"], r["errors"]) == (1, 1, 1, 2)


def test_rhythm_count_in_clicks_and_ignores_warmup_presses():
    s = Session(lesson([{"t": 0, "d": 1, "n": 60}]), "rhythm")
    advance_to(s, -0.5)
    clicks = events(s, "click")
    assert clicks[0] == ("click", True) and len(clicks) == 4
    assert s.press(65) is None                     # разминка во время отсчёта не считается


def test_other_hand_plays_itself_after_step_is_done():
    notes = [{"t": 0, "d": 1, "n": 64, "h": "R"}, {"t": 0, "d": 1, "n": 48, "h": "L"}]
    s = Session(lesson(notes), "wait", hand="R")
    s.update(5.0)
    assert events(s, "on") == []                    # левая ждёт, пока ученик сыграет правую
    s.press(64)
    s.update(0.01)
    assert events(s, "on") == [("on", 48)]


def test_one_hand_lesson_ignores_hand_choice():
    s = Session(lesson([{"t": 0, "d": 1, "n": 60, "h": "L"}]), "wait", hand="R")
    assert s.hand == "both" and s.current_notes() == [60]


def test_demo_plays_everything():
    s = Session(lesson([{"t": 0, "d": 1, "n": 60}, {"t": 1, "d": 1, "n": 62}]), "demo")
    advance_to(s, 1.5)
    assert [e[1] for e in events(s, "on")] == [60, 62]
    assert s.hints() == {62: 1.0}
    assert s.press(70) is None


def test_bundled_lessons_load_and_fit_keyboard():
    lessons, errors = load_lessons()
    assert errors == [] and len(lessons) >= 5
    for les in lessons:
        assert all(LOW <= n.n <= HIGH for n in les.notes), les.title
        s = Session(les, "wait")
        while s.cur < len(s.steps):                 # проходим урок без ошибок
            for p in s.current_notes():
                assert s.press(p) is True
        advance_to(s, s.end_time + 2, step=0.05)
        assert s.finished and s.result()["accuracy"] == 100
