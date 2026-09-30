"""Своя музыкальная библиотека: поиск песен в «Музыке» и «Загрузках» и системный выбор файла."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from .audio import AUDIO_EXT, clean_title

LIMIT = 5000
MAX_DEPTH = 8
SKIP_DIRS = {"Library", "node_modules", "Applications", "System", "$RECYCLE.BIN", "Caches"}


@dataclass
class Track:
    path: Path
    name: str
    where: str   # «Исполнитель · Альбом» или папка

    @property
    def kind(self) -> str:
        return self.path.suffix.lstrip(".").upper()


def music_dirs() -> list[Path]:
    home = Path.home()
    return [d for d in (home / "Music", home / "Downloads") if d.is_dir()]


def _where(path: Path, root: Path) -> str:
    parts = path.relative_to(root).parts[:-1]
    # медиатека Apple Music / iTunes: …/Music/Исполнитель/Альбом/трек
    inner = [p for p in parts if p not in ("Music", "Media", "Media.localized", "iTunes", "iTunes Media")]
    if len(inner) >= 2:
        return f"{inner[-2]} · {inner[-1]}"
    return inner[-1] if inner else root.name


def scan(dirs: list[Path] | None = None, cancel: threading.Event | None = None) -> list[Track]:
    tracks: list[Track] = []
    for root in dirs if dirs is not None else music_dirs():
        base_depth = len(root.parts)
        for folder, subdirs, files in os.walk(root, onerror=lambda e: None):
            if cancel is not None and cancel.is_set():
                return tracks
            here = Path(folder)
            subdirs[:] = [d for d in subdirs if not d.startswith(".") and d not in SKIP_DIRS
                          and not d.endswith((".app", ".photoslibrary", ".band", ".logicx"))
                          and len(here.parts) - base_depth < MAX_DEPTH]
            for f in files:
                p = here / f
                if not f.startswith(".") and p.suffix.lower() in AUDIO_EXT:
                    tracks.append(Track(p, clean_title(p), _where(p, root)))
                    if len(tracks) >= LIMIT:
                        return sorted(tracks, key=lambda t: t.name.lower())
    return sorted(tracks, key=lambda t: t.name.lower())


def search(tracks: list[Track], query: str) -> list[Track]:
    words = query.lower().split()
    return [t for t in tracks if all(w in f"{t.name} {t.where} {t.path.name}".lower() for w in words)]


def pick_file() -> str | None:
    """Системное окно выбора файла. Блокирует — вызывать из отдельного потока."""
    try:
        if sys.platform == "darwin":
            script = ('try\nPOSIX path of (choose file with prompt "Выберите песню" of type {"public.audio"})\n'
                      'end try')
            out = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=600)
        elif os.name == "nt":
            ps = ("[Console]::OutputEncoding=[Text.Encoding]::UTF8;Add-Type -AssemblyName System.Windows.Forms;"
                  "$d=New-Object System.Windows.Forms.OpenFileDialog;$d.Title='Выберите песню';"
                  "$d.Filter='Песни|*.mp3;*.wav;*.flac;*.ogg;*.m4a;*.aac;*.aif;*.aiff;*.wma|Все файлы|*.*';"
                  "if($d.ShowDialog() -eq 'OK'){$d.FileName}")
            out = subprocess.run(["powershell", "-NoProfile", "-STA", "-Command", ps],
                                 capture_output=True, text=True, encoding="utf-8", timeout=600)
        else:
            out = subprocess.run(["zenity", "--file-selection", "--title=Выберите песню"],
                                 capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError):
        return None
    path = out.stdout.strip()
    return path or None
