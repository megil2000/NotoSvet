import os
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import pygame  # noqa: E402

from app import audio, library  # noqa: E402
from app import transcribe as T  # noqa: E402
from songs import ODE, SR, song  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def mixer():
    pygame.mixer.init(44100, -16, 2, 512)
    yield
    pygame.mixer.quit()


def write_wav(path: Path, x: np.ndarray, rate: int = SR) -> Path:
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return path


def test_wav_loads_and_melody_survives_the_trip(tmp_path):
    a = audio.load(write_wav(tmp_path / "01 - Ода к радости.wav", song(ODE, 100)))
    assert a.title == "Ода к радости" and a.rate == 44100
    assert a.duration == pytest.approx(len(song(ODE, 100)) / SR, abs=0.05)
    r = T.transcribe(a.mono, 0.5)
    assert r.bpm == 100 and len(r.notes) == len(ODE)
    assert len(a.peaks(300)) == 300 and a.peaks(300).max() == pytest.approx(1.0, abs=0.01)


def test_segment_pads_silence_before_start(tmp_path):
    a = audio.load(write_wav(tmp_path / "x.wav", song(ODE, 100)))
    part = a.segment(-0.5, 1.0)
    assert len(part) == int(1.5 * 44100) and not part[:1000].any()


@pytest.mark.skipif(sys.platform != "darwin" or not shutil.which("afconvert"), reason="нужен afconvert (macOS)")
def test_m4a_and_aiff_open_via_macos(tmp_path):
    wav = write_wav(tmp_path / "x.wav", song(ODE[:8], 100))
    m4a = tmp_path / "Песня.m4a"
    subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", str(wav), str(m4a)], check=True)
    assert audio.load(m4a).duration > 4
    glass = Path("/System/Library/Sounds/Glass.aiff")  # системный AIFF
    if glass.exists():
        assert audio.load(glass).duration > 1


def test_bad_files_explain_themselves(tmp_path):
    (tmp_path / "notes.txt").write_text("привет")
    (tmp_path / "broken.mp3").write_bytes(b"\x00" * 5000)
    write_wav(tmp_path / "short.wav", np.zeros(SR // 2, np.float32))  # короче секунды — не песня
    for name in ("notes.txt", "broken.mp3", "missing.wav", "short.wav"):
        with pytest.raises(audio.AudioError):
            audio.load(tmp_path / name)


def test_library_scan_and_search(tmp_path):
    album = tmp_path / "Music" / "Кино" / "Группа крови"
    album.mkdir(parents=True)
    (album / "01 Группа крови.mp3").write_bytes(b"")
    (album / "02 Закрой за мной дверь.m4a").write_bytes(b"")
    (tmp_path / "Music" / ".hidden").mkdir()
    (tmp_path / "Music" / ".hidden" / "x.mp3").write_bytes(b"")
    (tmp_path / "Music" / "readme.txt").write_bytes(b"")
    tracks = library.scan([tmp_path / "Music"])
    assert [t.name for t in tracks] == ["Группа крови", "Закрой за мной дверь"]
    assert tracks[0].where == "Кино · Группа крови" and tracks[1].kind == "M4A"
    assert [t.name for t in library.search(tracks, "кино дверь")] == ["Закрой за мной дверь"]
