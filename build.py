"""Сборка готовой программы: python build.py [--no-ai]

  macOS   → dist/NotoSvet.app и dist/NotoSvet-macOS.zip (чтобы переслать)
  Windows → dist/NotoSvet.exe — один файл; с нейросетями — папка dist/NotoSvet и NotoSvet-Windows.zip

Если в окружении сборки есть PyTorch и Demucs (requirements-ai.txt), нейросети встраиваются
в программу вместе с весами Demucs из .model-cache — программа станет ~600 МБ, зато
разделяет песни на дорожки без интернета. --no-ai — маленькая сборка со старым распознаванием.

Собирать нужно на той системе, для которой программа: PyInstaller не собирает
Windows-версию на Mac и наоборот. Нужен PyInstaller: pip install pyinstaller
"""
from __future__ import annotations

import importlib.util
import io
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
DIST = ROOT / "dist"
NAME = "NotoSvet"
MAC = sys.platform == "darwin"
WINDOWS = os.name == "nt"
DEMUCS_WEIGHTS = ROOT / ".model-cache" / "checkpoints" / "5c90dfd2-34c22ccb.th"
AI_MODULES = ("torch", "torchaudio", "demucs", "dora", "openunmix", "julius", "einops", "omegaconf",
              "lameenc", "huggingface_hub", "safetensors", "sphn")


def icon_png(size: int) -> bytes:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    sys.path.insert(0, str(ROOT))
    from app.icon import make_icon

    buf = io.BytesIO()
    pygame.image.save(make_icon(size), buf, "icon.png")
    return buf.getvalue()


def make_icns(path: Path) -> None:
    iconset = BUILD / f"{NAME}.iconset"
    iconset.mkdir(parents=True, exist_ok=True)
    for size in (16, 32, 128, 256, 512):
        (iconset / f"icon_{size}x{size}.png").write_bytes(icon_png(size))
        (iconset / f"icon_{size}x{size}@2x.png").write_bytes(icon_png(size * 2))
    subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(path)], check=True)


def make_ico(path: Path) -> None:
    """ICO с PNG внутри — так понимает Windows Vista и новее."""
    sizes = (16, 24, 32, 48, 64, 128, 256)
    images = [icon_png(s) for s in sizes]
    offset = 6 + 16 * len(sizes)
    entries = b""
    for size, data in zip(sizes, images):
        entries += struct.pack("<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32, len(data), offset)
        offset += len(data)
    path.write_bytes(struct.pack("<HHH", 0, 1, len(sizes)) + entries + b"".join(images))


def main() -> int:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("Нужен PyInstaller: pip install pyinstaller")
        return 1
    shutil.rmtree(BUILD, ignore_errors=True)
    shutil.rmtree(DIST, ignore_errors=True)
    BUILD.mkdir()

    icon = BUILD / (f"{NAME}.icns" if MAC else f"{NAME}.ico")
    make_icns(icon) if MAC else make_ico(icon)

    ai = "--no-ai" not in sys.argv and all(importlib.util.find_spec(m) for m in ("torch", "demucs"))
    args = [sys.executable, "-m", "PyInstaller", str(ROOT / "main.py"),
            "--name", NAME, "--windowed", "--noconfirm", "--log-level", "WARN",
            "--distpath", str(DIST), "--workpath", str(BUILD / "work"), "--specpath", str(BUILD),
            "--icon", str(icon),
            "--add-data", f"{ROOT / 'lessons'}{os.pathsep}lessons",
            "--add-data", f"{ROOT / 'app' / 'models'}{os.pathsep}app/models",   # веса Basic Pitch
            "--hidden-import", "mido.backends.rtmidi",  # mido грузит бэкенд по имени — сам PyInstaller его не видит
            "--exclude-module", "tkinter"]
    if ai:
        args += ["--collect-submodules", "demucs"]  # класс модели берётся из файла весов по имени
        if DEMUCS_WEIGHTS.exists():
            args += ["--add-data", f"{DEMUCS_WEIGHTS}{os.pathsep}models"]
        print("Нейросети встраиваются" + ("" if DEMUCS_WEIGHTS.exists() else
                                          " (веса Demucs скачаются при первом распознавании)"))
    else:
        for module in AI_MODULES:
            args += ["--exclude-module", module]
        print("Сборка без нейросетей: «Мелодия из песни» — старым способом")
    one_file = not MAC and not ai  # с PyTorch один файл распаковывал бы ~1 ГБ при каждом запуске
    if MAC:
        args += ["--osx-bundle-identifier", "ru.notosvet.app"]
    elif one_file:
        args += ["--onefile"]
    subprocess.run(args, check=True, cwd=ROOT)

    if MAC:
        app = DIST / f"{NAME}.app"
        exe = app / "Contents" / "MacOS" / NAME
    elif one_file:
        app = exe = DIST / (f"{NAME}.exe" if WINDOWS else NAME)
    else:
        app = DIST / NAME
        exe = app / (f"{NAME}.exe" if WINDOWS else NAME)

    print("\nСамопроверка собранной программы:")
    check = subprocess.run([str(exe), "--selftest"])
    if check.returncode != 0:
        print("Самопроверка не прошла — программу раздавать нельзя.")
        return 1

    if MAC:  # ditto сохраняет подпись и ссылки внутри .app
        archive = DIST / f"{NAME}-macOS.zip"
        subprocess.run(["ditto", "-c", "-k", "--keepParent", str(app), str(archive)], check=True)
        print(f"\nГотово: {app.relative_to(ROOT)} и {archive.relative_to(ROOT)}")
    elif not one_file:  # папку пересылают архивом
        archive = Path(shutil.make_archive(str(DIST / f"{NAME}-{'Windows' if WINDOWS else 'Linux'}"), "zip",
                                           DIST, NAME))
        print(f"\nГотово: {app.relative_to(ROOT)} и {archive.relative_to(ROOT)}")
    else:
        print(f"\nГотово: {app.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
