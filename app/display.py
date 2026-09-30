"""Окно программы: чёткая картинка на Retina/HiDPI и полноэкранный режим.

Раньше кадр рисовался в 1280×720 и растягивался на экран — на Retina были видны пиксели.
Теперь холст (draw.Canvas) имеет столько пикселей, сколько их у окна на самом деле:
2560×1440 на Retina, в полном экране — весь экран с сохранением пропорций 16:9.
Экраны по-прежнему рисуют в логических координатах 1280×720, а кадр выводится 1:1.
"""
from __future__ import annotations

import math

import pygame
from pygame._sdl2 import video

from . import draw as D

MOUSE_EVENTS = (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP)


class Display:
    def __init__(self, title: str, icon: pygame.Surface, fullscreen: bool = False):
        self.window = pygame.Window(title, size=(D.W, D.H), allow_high_dpi=True, resizable=True)
        self.window.minimum_size = (D.W // 2, D.H // 2)
        self.window.set_icon(icon)
        try:
            self.renderer = video.Renderer(self.window, vsync=True)
        except pygame.error:  # без вертикальной синхронизации — лишь бы окно открылось
            self.renderer = video.Renderer(self.window)
        self.renderer.draw_color = D.BG  # поля вокруг кадра — цвета фона
        self.canvas: D.Canvas | None = None
        self.texture: video.Texture | None = None
        self.fullscreen = False
        self._window_size = None
        self.layout()
        if fullscreen:
            self.set_fullscreen(True)

    def layout(self) -> bool:
        """Подогнать холст под окно. Вызывается каждый кадр; True — холст новый."""
        size = self.window.size
        if size != self._window_size:  # полный экран включают и зелёной кнопкой macOS
            self._window_size = size
            self.fullscreen = size in [tuple(s) for s in pygame.display.get_desktop_sizes()]
        self.renderer.set_viewport(None)
        out = self.renderer.get_viewport()          # окно в настоящих пикселях
        self.ratio = out.w / max(1, size[0])        # пикселей в точке: 2 на Retina
        scale = min(out.w / D.W, out.h / D.H)
        frame = (max(1, round(D.W * scale)), max(1, round(D.H * scale)))
        self.dest = pygame.Rect(((out.w - frame[0]) // 2, (out.h - frame[1]) // 2), frame)
        if self.canvas is not None and self.canvas.surface.get_size() == frame:
            return False
        self.canvas = D.Canvas(pygame.Surface(frame), scale)
        self.texture = video.Texture(self.renderer, frame, streaming=True)
        return True

    def present(self) -> None:
        self.texture.update(self.canvas.surface)
        self.renderer.clear()
        self.texture.draw(dstrect=self.dest)
        self.renderer.present()

    def set_fullscreen(self, on: bool) -> None:
        if on:
            self.window.set_fullscreen(desktop=True)  # во весь экран без смены разрешения
        else:
            self.window.set_windowed()
        self.fullscreen = on
        self._window_size = self.window.size

    # --- мышь: точки окна ↔ логические координаты 1280×720 ---

    def to_logical(self, pos) -> tuple[int, int]:
        s = self.canvas.scale
        return (math.floor((pos[0] * self.ratio - self.dest.x) / s),
                math.floor((pos[1] * self.ratio - self.dest.y) / s))

    def to_window(self, r) -> pygame.Rect:
        box = self.canvas.rect(r).move(self.dest.topleft)
        k = self.ratio
        return pygame.Rect(round(box.x / k), round(box.y / k), round(box.w / k), round(box.h / k))

    def mouse_pos(self) -> tuple[int, int]:
        return self.to_logical(pygame.mouse.get_pos())

    def map_event(self, e: pygame.event.Event) -> pygame.event.Event:
        if e.type not in MOUSE_EVENTS:
            return e
        d = dict(e.dict)
        d["pos"] = self.to_logical(e.pos)
        if "rel" in d:
            k = self.ratio / self.canvas.scale
            d["rel"] = (round(e.rel[0] * k), round(e.rel[1] * k))
        return pygame.event.Event(e.type, d)
