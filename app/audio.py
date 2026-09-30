"""Открыть песню: MP3, WAV, FLAC, OGG, AIFF — напрямую; M4A/AAC (Apple Music, iTunes) — через
встроенный в macOS afconvert или через ffmpeg, если он установлен.

Нужен запущенный pygame.mixer: он и раскодирует файл, и потом сыграет «Оригинал».
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygame

from .transcribe import SR

AUDIO_EXT = {".mp3", ".wav", ".flac", ".ogg", ".oga", ".opus", ".aif", ".aiff", ".aifc",
             ".m4a", ".aac", ".mp4", ".caf", ".alac", ".wma"}
DIRECT_EXT = {".mp3", ".wav", ".flac", ".ogg", ".oga", ".opus", ".aif", ".aiff", ".aifc"}
MAX_MB = 200
MAX_MINUTES = 15


class AudioError(Exception):
    """Файл не открылся — текст ошибки показываем как есть."""


@dataclass
class Audio:
    title: str
    path: Path
    pcm: np.ndarray    # int16 (сэмплы × каналы) в формате микшера — чтобы играть
    rate: int
    mono: np.ndarray   # float32, SR = 22050 Гц — чтобы распознавать
    env: np.ndarray    # громкость по 10 мс — для волны

    @property
    def duration(self) -> float:
        return len(self.pcm) / self.rate

    def peaks(self, columns: int, start: float = 0.0, end: float | None = None) -> np.ndarray:
        """Громкость для рисования волны: по столбику на пиксель."""
        a = int(start * 100)
        b = int((self.duration if end is None else end) * 100)
        part = self.env[a:max(a + 1, b)]
        edges = np.linspace(0, len(part), columns + 1).astype(int)
        return np.array([part[x:max(x + 1, y)].max() if x < len(part) else 0.0
                         for x, y in zip(edges, edges[1:])], np.float32)

    def segment(self, start: float, end: float | None = None) -> np.ndarray:
        """Кусок для прослушивания; отрицательное начало — тишина перед звуком."""
        a = int(start * self.rate)
        b = len(self.pcm) if end is None else int(end * self.rate)
        part = self.pcm[max(0, a):max(0, b)]
        if a < 0:
            part = np.concatenate([np.zeros((-a,) + self.pcm.shape[1:], np.int16), part])
        return np.ascontiguousarray(part)

    def mono_segment(self, start: float, end: float) -> np.ndarray:
        return self.mono[int(start * SR):int(end * SR)]


def clean_title(path: Path) -> str:
    """«03 - Song Name.mp3» → «Song Name»."""
    stem = path.stem
    parts = stem.replace("_", " ").split(" ", 1)
    if len(parts) == 2 and parts[0].replace("-", "").replace(".", "").isdigit():
        stem = parts[1].lstrip(" -.")
    return stem.strip() or path.stem


def _to_wav(src: Path, folder: Path) -> Path:
    out = folder / "song.wav"
    if sys.platform == "darwin" and shutil.which("afconvert"):
        cmd = ["afconvert", "-f", "WAVE", "-d", "LEI16", str(src), str(out)]
    elif shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-ac", "2", str(out)]
    else:
        raise AudioError("этот формат открывается только через ffmpeg — установите его "
                         "или сохраните песню в MP3/WAV")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0 or not out.exists():
        if src.suffix.lower() == ".m4p":
            raise AudioError("песня защищена Apple Music и не открывается")
        raise AudioError("не удалось перекодировать файл (защищённые песни из Apple Music не открываются)")
    return out


def _decode(path: Path) -> pygame.mixer.Sound:
    try:
        return pygame.mixer.Sound(str(path))
    except pygame.error as e:
        raise AudioError(f"не удалось прочитать звук: {e}") from e


def load(path: Path) -> Audio:
    path = Path(path)
    if pygame.mixer.get_init() is None:
        raise AudioError("нет звуковой карты")
    if not path.is_file():
        raise AudioError("файл не найден")
    if path.stat().st_size > MAX_MB * 1024 * 1024:
        raise AudioError(f"файл больше {MAX_MB} МБ")
    ext = path.suffix.lower()
    if ext not in AUDIO_EXT and ext != ".m4p":
        raise AudioError("это не песня: нужен MP3, WAV, FLAC, OGG, AIFF или M4A")
    with tempfile.TemporaryDirectory(prefix="notosvet-") as tmp:
        if ext in DIRECT_EXT:
            try:
                sound = _decode(path)
            except AudioError:
                try:  # редкий кодек — пробуем через конвертер
                    sound = _decode(_to_wav(path, Path(tmp)))
                except AudioError:
                    raise AudioError("файл не читается — возможно, он повреждён или это не песня") from None
        else:
            sound = _decode(_to_wav(path, Path(tmp)))
        pcm = pygame.sndarray.array(sound)
    rate, _, channels = pygame.mixer.get_init()
    if pcm.ndim == 1:
        pcm = pcm[:, None]
    if len(pcm) < rate:
        raise AudioError("песня короче секунды")
    if len(pcm) > MAX_MINUTES * 60 * rate:
        raise AudioError(f"песня длиннее {MAX_MINUTES} минут")
    mono = pcm.mean(axis=1, dtype=np.float32) / 32768.0
    if rate % SR == 0:  # 44100 → 22050: среднее соседних сэмплов и прореживание
        k = rate // SR
        mono = mono[:len(mono) // k * k].reshape(-1, k).mean(axis=1)
    else:
        mono = np.interp(np.arange(0, len(mono), rate / SR), np.arange(len(mono)), mono).astype(np.float32)
    step = SR // 100
    blocks = np.abs(mono[:len(mono) // step * step]).reshape(-1, step)
    env = blocks.max(axis=1)
    env = env / (env.max() + 1e-9)
    return Audio(clean_title(path), path, pcm.astype(np.int16), rate, mono.astype(np.float32), env.astype(np.float32))
