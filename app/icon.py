"""Иконка НотоСвета: три ноты падают на ленту над клавишами. Рисуется кодом — и для окна, и для сборки."""
from __future__ import annotations

import pygame

from .config import note_color

_BASE = 1024


def _draw(s: int) -> pygame.Surface:
    surf = pygame.Surface((s, s), pygame.SRCALPHA)
    u = s / 100  # сетка 100×100

    def rect(x, y, w, h):
        return pygame.Rect(round(x * u), round(y * u), round(w * u), round(h * u))

    pygame.draw.rect(surf, (20, 22, 28), rect(9, 9, 82, 82), border_radius=round(18 * u))
    keys_y, keys_h = 66, 18
    for i in range(7):  # белые клавиши
        pygame.draw.rect(surf, (236, 237, 239), rect(17 + i * 9.43, keys_y, 8.6, keys_h),
                         border_bottom_left_radius=round(1.6 * u), border_bottom_right_radius=round(1.6 * u))
    for i in (0, 1, 3, 4, 5):  # чёрные
        pygame.draw.rect(surf, (24, 26, 30), rect(22.8 + i * 9.43, keys_y, 5.4, 10.5),
                         border_bottom_left_radius=round(1 * u), border_bottom_right_radius=round(1 * u))
    pygame.draw.rect(surf, (36, 40, 47), rect(15, 59, 70, 6), border_radius=round(2 * u))  # профиль ленты
    for pitch, x, top in ((60, 19.5, 26), (64, 38.4, 40), (67, 52.5, 18)):
        c = note_color(pitch)
        pygame.draw.rect(surf, c, rect(x, top, 7, 59 - top - 1.5), border_radius=round(2 * u))
        for dx in (0.8, 4.2):  # горящие светодиоды под нотой
            pygame.draw.rect(surf, tuple(min(255, v + 90) for v in c), rect(x + dx, 61, 2, 2),
                             border_radius=round(0.6 * u))
    return surf


def make_icon(size: int) -> pygame.Surface:
    """Рисуем крупно и уменьшаем — так мелкие размеры сглажены."""
    return pygame.transform.smoothscale(_draw(_BASE), (size, size))
