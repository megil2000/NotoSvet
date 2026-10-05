"""Нейросеть Basic Pitch на PyTorch: звук → какие ноты звучат в каждом кадре.

Basic Pitch (Spotify, ICASSP 2022) не привязана к инструменту: слышит и голос, и аккорды
пианино или гитары. Сеть маленькая — 17 тысяч весов, — поэтому быстро работает на процессоре.
Здесь её архитектура переписана на PyTorch один в один, а веса перенесены из оригинальной
модели (nmp.onnx из пакета basic-pitch 0.4.0, лицензия Apache 2.0) скриптом
tools/convert_basic_pitch.py. Выход совпадает с оригиналом с точностью ~1e-4.

Как устроена:
1. CQT — спектр с шагом в треть полутона, 9 октав от A0; считается свёртками
   (верхняя октава, потом сигнал вдвое прореживается для каждой следующей).
2. Логарифм громкости, нормированный в 0..1 на каждом окне.
3. «Стопка гармоник»: к каждой высоте приставляются её 0,5-я, 2-я…7-я гармоники.
4. Свёртки дают три карты (кадры × высоты): contour — высота по трети полутона,
   note — нота звучит, onset — нота начинается.

Вход — моно 22 050 Гц окнами по 2 с; кадр — 256 сэмплов (≈ 86 кадров в секунду), кадр g
стоит на сэмпле g·256.
"""
from __future__ import annotations

import math
import threading

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import paths
from .transcribe import Cancelled

SR = 22050
HOP = 256
WINDOW = 2 * SR - HOP          # 43 844 сэмпла — окно, на котором училась сеть
FRAMES = 172                   # кадров в окне
EDGE = 15                      # у краёв окна сети не хватает контекста — эти кадры отбрасываем
STEP = FRAMES - 2 * EDGE       # новых кадров с каждого окна
BATCH = 4                      # окон за раз: свёртка 3×39 прожорлива, на 16+ окнах начинается своп
LOWEST = 21                    # MIDI первой из 88 высот (A0)
HARMONICS = (0.5, 1, 2, 3, 4, 5, 6, 7)
SHIFTS = [round(36 * math.log2(h)) for h in HARMONICS]  # сдвиг гармоники в бинах CQT
CONVS = ("onset1", "contour1", "contour2", "note1", "note2", "onset2")


class BasicPitch(nn.Module):
    def __init__(self, weights: dict[str, np.ndarray]):
        super().__init__()
        w = {k: torch.from_numpy(np.asarray(v, np.float32)) for k, v in weights.items()}
        self.register_buffer("kernels", torch.cat([w["cqt_a"], w["cqt_b"]])[:, None, :])  # 36 пар «cos, sin»
        self.register_buffer("lowpass", w["lowpass"][None, None, :])
        self.register_buffer("sqrt_len", w["sqrt_len"][:, None])
        self.bn_scale, self.bn_shift = (float(x) for x in w["bn"])
        self.onset1 = nn.Conv2d(8, 32, 5, stride=(1, 3), padding=(2, 1))
        self.contour1 = nn.Conv2d(8, 8, (3, 39), padding=(1, 19))
        self.contour2 = nn.Conv2d(8, 1, 5, padding=2)
        self.note1 = nn.Conv2d(1, 32, 7, stride=(1, 3), padding=(3, 2))
        self.note2 = nn.Conv2d(32, 1, (7, 3), padding=(3, 1))
        self.onset2 = nn.Conv2d(33, 1, 3, padding=1)
        for name in CONVS:
            conv = getattr(self, name)
            conv.weight.data, conv.bias.data = w[f"{name}.weight"], w[f"{name}.bias"]

    def cqt(self, x: torch.Tensor) -> torch.Tensor:
        """(окна, сэмплы) → амплитуды (окна, кадры, 309 бинов по трети полутона от A0)."""
        x = x[:, None, :]
        hop, octaves = HOP, []
        for k in range(9):
            if k:  # следующая октава вниз: фильтр и прореживание вдвое
                x = F.conv1d(F.pad(x, (127, 127)), self.lowpass, stride=2)
                hop //= 2
            c = F.conv1d(F.pad(x, (128, 128), mode="reflect"), self.kernels, stride=hop)
            octaves.insert(0, torch.sqrt(c[:, :36] ** 2 + c[:, 36:] ** 2))
        return (torch.cat(octaves, dim=1)[:, -309:] * self.sqrt_len).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        db = 10 * torch.log10(self.cqt(x) ** 2 + 1e-10)
        db = db - db.amin(dim=(1, 2), keepdim=True)
        top = db.amax(dim=(1, 2), keepdim=True)
        db = torch.where(top > 0, db / top.clamp_min(1e-12), torch.zeros_like(db))
        x = db * self.bn_scale + self.bn_shift
        stack = []
        for s in SHIFTS:
            h = F.pad(x[:, :, s:], (0, s)) if s > 0 else F.pad(x[:, :, :s], (-s, 0)) if s < 0 else x
            stack.append(h[:, :, :264])
        h = torch.stack(stack, dim=1)                                   # (окна, 8 гармоник, кадры, 264)
        onset_features = F.relu(self.onset1(h))
        contour = torch.sigmoid(self.contour2(F.relu(self.contour1(h))))
        note = torch.sigmoid(self.note2(F.relu(self.note1(contour))))
        onset = torch.sigmoid(self.onset2(torch.cat([note, onset_features], dim=1)))
        return note[:, 0], onset[:, 0], contour[:, 0]


_model: BasicPitch | None = None
_lock = threading.Lock()


def model() -> BasicPitch:
    global _model
    with _lock:
        if _model is None:
            with np.load(paths.NET_WEIGHTS / "basic_pitch.npz", allow_pickle=False) as w:
                _model = BasicPitch(dict(w)).eval()
        return _model


def predict(x: np.ndarray, progress=None, cancel: threading.Event | None = None
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Моно 22 050 Гц → (note, onset, contour): кадры × 88, кадры × 88, кадры × 264.

    Окна идут внахлёст: от каждого берутся только средние STEP кадров, края отбрасываются,
    поэтому склейка без швов и кадр g стоит ровно на сэмпле g·256.
    """
    x = np.asarray(x, np.float32)
    n = len(x) // HOP + 1
    count = -(-n // STEP)
    padded = np.zeros(EDGE * HOP + count * STEP * HOP + WINDOW, np.float32)
    padded[EDGE * HOP:EDGE * HOP + len(x)] = x
    net = model()
    outs: list[list[np.ndarray]] = [[], [], []]
    with torch.inference_mode():
        for a in range(0, count, BATCH):
            if cancel is not None and cancel.is_set():
                raise Cancelled
            windows = np.stack([padded[w * STEP * HOP:w * STEP * HOP + WINDOW] for w in range(a, min(count, a + BATCH))])
            for out, y in zip(outs, net(torch.from_numpy(windows))):
                out.append(y[:, EDGE:EDGE + STEP].reshape(-1, y.shape[-1]).numpy())
            if progress is not None:
                progress(min(1.0, (a + BATCH) / count))
    note, onset, contour = (np.concatenate(o)[:n] for o in outs)
    return note, onset, contour


def frame_time(frame: float) -> float:
    return frame * HOP / SR
