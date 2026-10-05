"""Разделение песни на дорожки инструментов нейросетью Demucs (Meta, модель htdemucs_6s, PyTorch).

Шесть дорожек: барабаны, бас, прочее, вокал, гитара, пианино. Модель — гибрид свёрток
по звуку и по спектру с трансформером между ними; песню она слушает кусками по 7,8 с внахлёст.
Веса (53 МБ, лицензия MIT) скачиваются один раз в папку моделей и проверяются по SHA-256.
"""
from __future__ import annotations

import hashlib
import threading
import urllib.request
from pathlib import Path

import numpy as np

from . import paths
from .transcribe import Cancelled

FILE = "5c90dfd2-34c22ccb.th"
URL = "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/" + FILE
SHA256 = "34c22ccb381c6f9fdbf324f04e1e2fe21aaaf293f5ded163a162697ff9a02ddd"
SIZE_MB = 53
RATE = 44100
SOURCES = ("drums", "bass", "other", "vocals", "guitar", "piano")
OVERLAP = 0.25

_model = None
_lock = threading.Lock()


def _check(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled


def weights_path() -> Path | None:
    """Где лежат веса: встроенные в программу или скачанные раньше."""
    for folder in (paths.BUILTIN_MODELS, paths.MODELS):
        if (folder / FILE).is_file():
            return folder / FILE
    return None


def download(progress=None, cancel: threading.Event | None = None) -> Path:
    """Скачать веса в paths.MODELS. Недокачанный или испорченный файл не остаётся."""
    target = paths.MODELS / FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(".part")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(URL, timeout=30) as r, open(part, "wb") as f:
            total = int(r.headers.get("Content-Length") or SIZE_MB * 1024 * 1024)
            done = 0
            while chunk := r.read(1 << 16):
                _check(cancel)
                f.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(min(1.0, done / total))
        if digest.hexdigest() != SHA256:
            raise OSError("файл модели скачался с ошибкой — попробуйте ещё раз")
        part.replace(target)
    finally:
        part.unlink(missing_ok=True)
    return target


def load(progress=None, cancel: threading.Event | None = None):
    """Модель в памяти; при первом вызове — скачать (если нужно) и прочитать веса."""
    global _model
    with _lock:
        if _model is None:
            path = weights_path() or download(progress, cancel)
            import torch
            from demucs.states import load_model

            # в файле не только веса, но и класс модели — поэтому weights_only=False;
            # файл проверен по SHA-256 при скачивании
            package = torch.load(path, map_location="cpu", weights_only=False)
            _model = load_model(package).eval()
        return _model


class _Steps:
    """«Пул задач» для demucs.apply_model: куски считаются по одному, между ними —
    прогресс и проверка отмены."""

    def __init__(self, total: int, progress, cancel):
        self.total, self.done, self.progress, self.cancel = max(1, total), 0, progress, cancel

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        """demucs 4.1 зовёт это при ошибке или отмене; ждать нечего — куски считаются по одному."""

    def submit(self, func, *args, **kwargs):
        steps = self

        class Job:
            def result(self):
                _check(steps.cancel)
                out = func(*args, **kwargs)
                steps.done += 1
                if steps.progress is not None:
                    steps.progress(min(1.0, steps.done / steps.total))
                return out
        return Job()


def separate(stereo: np.ndarray, progress=None, cancel: threading.Event | None = None) -> dict[str, np.ndarray]:
    """Стерео 44 100 Гц (сэмплы × 2, float32) → дорожки того же размера по именам SOURCES."""
    import torch
    from demucs.apply import apply_model

    model = load(cancel=cancel)
    x = torch.from_numpy(np.ascontiguousarray(np.asarray(stereo, np.float32).T))
    ref = x.mean(dim=0)
    mean, std = float(ref.mean()), float(ref.std())
    if std < 1e-5:  # тишина: делить нечего
        return {name: np.zeros_like(stereo, dtype=np.float32) for name in model.sources}
    x = (x - mean) / std  # громкость как при обучении модели — так же делает demucs.separate
    segment = int(model.samplerate * float(model.segment))
    stride = int((1 - OVERLAP) * segment)
    pool = _Steps(len(range(0, x.shape[-1], stride)), progress, cancel)
    with torch.inference_mode():
        out = apply_model(model, x[None], shifts=0, split=True, overlap=OVERLAP, pool=pool)[0]
    out = out * std + mean
    return {name: out[i].T.contiguous().numpy() for i, name in enumerate(model.sources)}
