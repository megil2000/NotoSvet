"""Звук по рецепту прошивки: 6 гармоник, атака 3 мс, экспоненциальное затухание, отпускание 150 мс."""
from __future__ import annotations

import numpy as np
import pygame

from .config import HIGH, LOW

HARMONICS = (1.0, 0.5, 0.28, 0.16, 0.09, 0.05)
RELEASE_MS = 150


class Synth:
    def __init__(self) -> None:
        self.playing: dict[int, pygame.mixer.Channel] = {}
        self.down: set[int] = set()  # звучащие по нажатию (клавишей или программой)
        self.pedal = False
        self.volume = 0.8
        init = pygame.mixer.get_init()
        self.ok = init is not None
        if not self.ok:  # нет звуковой карты — работаем молча
            return
        self.rate, _, self.channels = init
        pygame.mixer.set_num_channels(48)
        self.sounds = {p: self._sound(self._piano(p)) for p in range(LOW, HIGH + 1)}
        self.click_accent = self._sound(self._blip(1800, 0.45))
        self.click_plain = self._sound(self._blip(1200, 0.3))
        self.beep = self._sound(self._error_tone())

    # --- синтез ---

    def _time(self, seconds: float) -> np.ndarray:
        return np.arange(int(self.rate * seconds)) / self.rate

    def _piano(self, pitch: int) -> np.ndarray:
        f = 440.0 * 2 ** ((pitch - 69) / 12)
        tau = 1.4 * 2 ** (-(pitch - LOW) / 20)  # низкие звучат дольше: C3 ≈ 1,4 с, C5 ≈ 0,6 с
        t = self._time(min(4.0, 4 * tau))
        wave = np.zeros_like(t)
        for k, a in enumerate(HARMONICS, start=1):
            if f * k >= self.rate / 2:
                break
            # верхние гармоники гаснут быстрее — звук похож на электропианино
            wave += a * np.sin(2 * np.pi * f * k * t) * np.exp(-t * (1 + 0.5 * (k - 1)) / tau)
        wave *= np.minimum(1.0, t / 0.003)  # атака 3 мс
        wave[-256:] *= np.linspace(1.0, 0.0, 256)
        return 0.28 * wave / np.max(np.abs(wave))

    def _blip(self, freq: float, vol: float) -> np.ndarray:
        t = self._time(0.03)
        return vol * np.sin(2 * np.pi * freq * t) * np.exp(-t / 0.005)  # щелчок метронома ~5 мс

    def _error_tone(self) -> np.ndarray:
        t = self._time(0.08)  # сигнал ошибки: 80 мс, низкий тон
        wave = sum(np.sin(2 * np.pi * 110 * k * t) / k for k in (1, 3, 5))
        env = np.minimum(1.0, t / 0.002) * np.minimum(1.0, (t[-1] - t) / 0.02)
        return 0.3 * wave * env

    def _sound(self, wave: np.ndarray) -> pygame.mixer.Sound:
        pcm = (np.clip(wave, -1.0, 1.0) * 32767).astype(np.int16)
        if self.channels > 1:
            pcm = np.repeat(pcm[:, None], self.channels, axis=1)
        return pygame.sndarray.make_sound(np.ascontiguousarray(pcm))

    # --- игра ---

    def note_on(self, pitch: int, velocity: float = 1.0) -> None:
        if not self.ok or pitch not in self.sounds:
            return
        old = self.playing.get(pitch)
        if old is not None and old.get_sound() is self.sounds[pitch]:
            old.fadeout(40)  # повторный удар по той же клавише
        ch = pygame.mixer.find_channel(True)
        ch.set_volume(self.volume * velocity)
        ch.play(self.sounds[pitch])
        self.playing[pitch] = ch
        self.down.add(pitch)

    def note_off(self, pitch: int) -> None:
        self.down.discard(pitch)
        if not self.pedal:
            self._release(pitch)

    def _release(self, pitch: int) -> None:
        ch = self.playing.pop(pitch, None)
        if ch is not None and ch.get_sound() is self.sounds.get(pitch):
            ch.fadeout(RELEASE_MS)

    def set_pedal(self, on: bool) -> None:
        """Педаль (сустейн): пока нажата, отпущенные ноты звучат до затухания."""
        self.pedal = on
        if not on:
            for p in list(self.playing):
                if p not in self.down:
                    self._release(p)

    def click(self, accent: bool) -> None:
        if self.ok:
            ch = pygame.mixer.find_channel(True)
            ch.set_volume(self.volume)
            ch.play(self.click_accent if accent else self.click_plain)

    def error(self) -> None:
        if self.ok:
            ch = pygame.mixer.find_channel(True)
            ch.set_volume(self.volume)
            ch.play(self.beep)

    def all_off(self) -> None:
        if self.ok:
            pygame.mixer.fadeout(100)
        self.playing.clear()
        self.down.clear()
