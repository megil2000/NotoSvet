import json

import mido
import pytest

from app.converter import convert, import_file
from app.lesson import load_lesson


def make_midi(path, notes, tempo_bpm=90):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm)))
    for n in notes:
        track += [mido.Message("note_on", note=n, velocity=80), mido.Message("note_off", note=n, time=480)]
    # аккорд за пределами 25 клавиш
    track += [mido.Message("note_on", note=n, velocity=80) for n in (84, 88, 91)]
    track.append(mido.Message("note_off", note=84, time=960))
    track += [mido.Message("note_off", note=n) for n in (88, 91)]
    mid.save(path)


def test_converter_matches_plan(tmp_path):
    make_midi(tmp_path / "gamma.mid", [60, 62, 64, 65, 67, 69, 71, 72])
    lesson = convert(tmp_path / "gamma.mid")
    assert (lesson["title"], lesson["bpm"]) == ("gamma", 90.0)
    assert len(lesson["notes"]) == 11 and lesson["notes"][-1]["s"] + 1 == 9  # как в плане


def test_import_midi_and_json_without_overwriting(tmp_path):
    make_midi(tmp_path / "Моя песня.mid", [60, 64, 67])
    first = import_file(tmp_path / "Моя песня.mid", tmp_path / "lessons")
    second = import_file(tmp_path / "Моя песня.mid", tmp_path / "lessons")
    assert first.name == "Моя песня.json" and second.name == "Моя песня (2).json"
    assert load_lesson(first).title == "Моя песня"

    src = tmp_path / "урок.json"
    src.write_text(json.dumps({"title": "Урок", "bpm": 80, "notes": [{"t": 0, "d": 1, "n": 60}]}), encoding="utf-8")
    assert load_lesson(import_file(src, tmp_path / "lessons")).steps == 1


@pytest.mark.parametrize("name, content", [
    ("пусто.json", '{"title": "x", "notes": []}'),
    ("битый.json", "{не json"),
    ("битый.mid", "not a midi file"),
    ("картинка.png", ""),
])
def test_bad_files_are_rejected(tmp_path, name, content):
    (tmp_path / name).write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        import_file(tmp_path / name, tmp_path / "lessons")
    assert not (tmp_path / "lessons").exists() or not any((tmp_path / "lessons").iterdir())
