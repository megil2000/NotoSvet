"""MIDI -> урок НотоСвет (JSON). Запуск: python tools/converter.py song.mid lessons/song.json

То же самое умеет программа: перетащите .mid в окно.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.converter import convert  # noqa: E402

if __name__ == "__main__":
    lesson = convert(sys.argv[1])
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        json.dump(lesson, f, ensure_ascii=False, indent=1)
    print(f"{lesson['title']}: {len(lesson['notes'])} нот, {lesson['notes'][-1]['s'] + 1} шагов")
