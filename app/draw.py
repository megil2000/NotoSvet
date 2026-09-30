"""Отрисовка: клавиши, лента подсветки, падающие ноты.

Геометрия повторяет пианино из плана: спереди белые клавиши широкие, а в задней зоне
все 12 клавиш октавы получают равные слоты. Над каждым слотом — 2 светодиода ленты,
и нота падает ровно в свой слот.

Все экраны рисуют в логических координатах 1280×720, а холст (Canvas) переводит их
в настоящие пиксели экрана: на Retina — вдвое больше, в полном экране — сколько влезет.
Поэтому текст, скругления и линии чёткие на любом экране, а раскладка одна.
"""
from __future__ import annotations

import math

import numpy as np
import pygame

from .config import HIGH, LOW, NOTE_LABEL, is_black, note_color, note_name

W, H = 1280, 720
TOP = 64                         # верхняя панель
KEYS_H = 168                     # видимая длина белой клавиши
KEYS_Y = H - KEYS_H
STRIP_H = 30                     # профиль с лентой
STRIP_Y = KEYS_Y - STRIP_H
HIT_Y = STRIP_Y                  # ноты «падают» на ленту
WHITE_W = 80
KEYBOARD_W = 15 * WHITE_W
X0 = (W - KEYBOARD_W) // 2
SLOT = 7 * WHITE_W / 12          # задняя зона: октава делится на 12 равных слотов
BLACK_H = round(KEYS_H * 95 / 150)  # чёрная видна на 95 мм из 150

BG = (15, 17, 21)
BG_LANE = (11, 12, 15)           # дорожки чёрных клавиш
PANEL = (26, 29, 35)
PANEL_HI = (38, 42, 50)
LINE = (44, 48, 57)
TEXT = (233, 235, 238)
MUTED = (140, 146, 156)
DIM = (90, 96, 106)
WHITE = (255, 255, 255)
DARK_TEXT = (16, 18, 22)

KEY_WHITE = (236, 237, 239)
KEY_BLACK = (24, 26, 30)


def mix(a, b, k: float):
    return tuple(round(a[i] + (b[i] - a[i]) * k) for i in range(3))


def luminance(c) -> float:
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def ink(fill):
    """Цвет подписи, читаемый на заливке."""
    return DARK_TEXT if luminance(fill) > 130 else WHITE


WHITES = [p for p in range(LOW, HIGH + 1) if not is_black(p)]
BLACKS = [p for p in range(LOW, HIGH + 1) if is_black(p)]
_WHITE_INDEX = {p: i for i, p in enumerate(WHITES)}


def slot_x(pitch: int) -> float:
    return X0 + (pitch - LOW) * SLOT


def lane_rect(pitch: int, top: float, bottom: float) -> pygame.FRect:
    """Дробные координаты: на Retina нота едет плавно, по пикселю экрана, а не по логической точке."""
    return pygame.FRect(slot_x(pitch) + 3, top, SLOT - 6, max(0.0, bottom - top))


def front_rect(pitch: int) -> pygame.Rect:
    return pygame.Rect(X0 + _WHITE_INDEX[pitch] * WHITE_W + 1, KEYS_Y + BLACK_H,
                       WHITE_W - 2, KEYS_H - BLACK_H - 1)


def back_rect(pitch: int) -> pygame.Rect:
    left = round(slot_x(pitch))
    right = X0 + KEYBOARD_W if pitch == HIGH else round(slot_x(pitch) + SLOT)
    inset = 2 if is_black(pitch) else 1
    return pygame.Rect(left + inset, KEYS_Y, right - left - 2 * inset, BLACK_H + (0 if is_black(pitch) else 2))


def key_at(pos) -> int | None:
    """Какая клавиша под мышью."""
    x, y = pos
    if not (KEYS_Y <= y < H and X0 <= x < X0 + KEYBOARD_W):
        return None
    if y < KEYS_Y + BLACK_H:
        return LOW + min(HIGH - LOW, int((x - X0) // SLOT))
    return WHITES[int((x - X0) // WHITE_W)]


# --- холст: логические координаты → пиксели экрана ---

_scale = 1.0   # масштаб текущего холста: по нему measure() считает ширину текста


class Canvas:
    """Кадр в пикселях экрана, в который рисуют в логических координатах 1280×720."""

    def __init__(self, surface: pygame.Surface, scale: float = 1.0):
        global _scale
        self.surface, self.scale = surface, scale
        _scale = scale

    def px(self, v: float) -> int:
        return round(v * self.scale)

    def point(self, p) -> tuple[int, int]:
        return round(p[0] * self.scale), round(p[1] * self.scale)

    def rect(self, r) -> pygame.Rect:
        """Края переводятся по отдельности — соседние прямоугольники стыкуются без щелей."""
        x, y, w, h = r
        s = self.scale
        x0, y0 = round(x * s), round(y * s)
        return pygame.Rect(x0, y0, round((x + w) * s) - x0, round((y + h) * s) - y0)

    def fill(self, color, r=None) -> None:
        self.surface.fill(color, None if r is None else self.rect(r))

    def set_clip(self, r) -> None:
        self.surface.set_clip(None if r is None else self.rect(r))

    def get_at(self, pos):
        return self.surface.get_at(self.point(pos))


def _width(c: Canvas, width: float) -> int:
    return 0 if width <= 0 else max(1, round(width * c.scale))


def rect(c: Canvas, color, r, width: float = 0, border_radius: float = 0, **corners) -> None:
    """Как pygame.draw.rect, но в логических координатах; corners — border_*_radius."""
    corners = {k: c.px(v) if v > 0 else v for k, v in corners.items()}
    pygame.draw.rect(c.surface, color, c.rect(r), _width(c, width), c.px(border_radius), **corners)


def line(c: Canvas, color, start, end, width: float = 1) -> None:
    pygame.draw.line(c.surface, color, c.point(start), c.point(end), _width(c, width))


def polygon(c: Canvas, color, points, width: float = 0) -> None:
    pygame.draw.polygon(c.surface, color, [c.point(p) for p in points], _width(c, width))


def circle(c: Canvas, color, center, radius: float, width: float = 0) -> None:
    pygame.draw.circle(c.surface, color, c.point(center), radius * c.scale, _width(c, width))


_shades: dict = {}


def shade(c: Canvas, r, rgba) -> None:
    """Полупрозрачная заливка поверх кадра: пауза, справка, прошлое под курсором."""
    box = c.rect(r)
    key = (box.size, tuple(rgba))
    veil = _shades.get(key)
    if veil is None:
        if len(_shades) > 16:
            _shades.clear()
        veil = _shades[key] = pygame.Surface(box.size, pygame.SRCALPHA)
        veil.fill(rgba)
    c.surface.blit(veil, box)


# --- текст ---

_FONT_NAMES = "arial,helveticaneue,dejavusans,liberationsans"
_fonts: dict = {}
_texts: dict = {}


def _font(px: int, bold: bool) -> pygame.font.Font:
    """Шрифт размером px настоящих пикселей: на Retina буквы рисуются вдвое крупнее, а не растягиваются."""
    key = (px, bold)
    if key not in _fonts:
        _fonts[key] = pygame.font.SysFont(_FONT_NAMES, px, bold=bold)
    return _fonts[key]


def measure(s: str, size: int, bold: bool = False) -> tuple[int, int]:
    """Ширина и высота надписи в логических точках."""
    w, h = _font(max(1, round(size * _scale)), bold).size(s)
    return round(w / _scale), round(h / _scale)


def _anchor(v, s: float):
    return (round(v[0] * s), round(v[1] * s)) if isinstance(v, (tuple, list)) else round(v * s)


def text(c: Canvas, s: str, size: int, color, bold: bool = False, alpha: int = 255, **anchor) -> pygame.Rect:
    """Надпись; anchor — как у Rect (center=…, topleft=…). Возвращает её место в логических точках."""
    px = max(1, round(size * c.scale))
    key = (s, px, color, bold)
    img = _texts.get(key)
    if img is None:
        if len(_texts) > 800:
            _texts.clear()
        img = _texts[key] = _font(px, bold).render(s, True, color)
    box = img.get_rect(**{k: _anchor(v, c.scale) for k, v in anchor.items()})
    if alpha < 255:
        img = img.copy()
        img.set_alpha(alpha)
    c.surface.blit(img, box)
    k = c.scale
    return pygame.Rect(round(box.x / k), round(box.y / k), round(box.w / k), round(box.h / k))


def plural(n: int, forms: tuple[str, str, str]) -> str:
    """«1 нота», «3 ноты», «5 нот»."""
    if n % 10 == 1 and n % 100 != 11:
        return f"{n} {forms[0]}"
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return f"{n} {forms[1]}"
    return f"{n} {forms[2]}"


def wrap(s: str, size: int, width: int) -> list[str]:
    lines, cur = [], ""
    for word in s.split():
        probe = f"{cur} {word}".strip()
        if measure(probe, size)[0] <= width:
            cur = probe
        else:
            lines.append(cur)
            cur = word
    return lines + [cur] if cur else lines


# --- лента подсветки ---

_glow_bases: dict = {}
_glow_cache: dict = {}


def _glow(color, level: float, scale: float) -> pygame.Surface:
    q = max(1, round(level * 12))
    r = max(4, round(40 * scale))
    key = (color, q, r)
    surf = _glow_cache.get(key)
    if surf is None:
        base = _glow_bases.get(r)
        if base is None:
            yy, xx = np.mgrid[-r:r, -r:r] + 0.5
            fall = np.clip(1.0 - np.hypot(xx, yy * 1.2) / r, 0.0, 1.0) ** 1.6 * 0.9 * 255
            base = _glow_bases[r] = pygame.surfarray.make_surface(np.dstack([fall.T] * 3).astype(np.uint8))
        surf = base.copy()
        surf.fill(tuple(round(c * q / 12) for c in color), special_flags=pygame.BLEND_RGB_MULT)
        if len(_glow_cache) > 600:
            _glow_cache.clear()
        _glow_cache[key] = surf
    return surf


def draw_strip(c: Canvas, leds: dict) -> None:
    """leds: высота → (цвет, яркость 0..1). Клавиша k горит светодиодами 2k и 2k + 1."""
    body = pygame.Rect(X0 - 10, STRIP_Y, KEYBOARD_W + 20, STRIP_H)
    rect(c, (31, 34, 40), body, border_radius=6)            # алюминиевый профиль
    line(c, (56, 61, 70), (body.left + 6, body.top), (body.right - 7, body.top))
    rect(c, (17, 19, 23), body.inflate(-12, -12), border_radius=4)  # рассеиватель
    cy = STRIP_Y + STRIP_H / 2
    glows = []
    for p in range(LOW, HIGH + 1):
        color, lvl = leds.get(p, (None, 0.0))
        for half in (0.25, 0.75):
            cx = slot_x(p) + SLOT * half
            led = (cx - 4, cy - 3, 8, 6)
            if lvl > 0.02:
                core = mix(color, WHITE, 0.45 * lvl)  # горящий светодиод светлее своего ореола
                rect(c, mix((17, 19, 23), core, min(1.0, 0.3 + lvl)), led, border_radius=2)
                glows.append((color, lvl, (cx, cy)))
            else:
                rect(c, (42, 46, 53), led, border_radius=2)
    for color, lvl, center in glows:
        g = _glow(color, lvl, c.scale)
        c.surface.blit(g, g.get_rect(center=c.point(center)), special_flags=pygame.BLEND_RGB_ADD)


# --- клавиатура ---

def draw_keyboard(c: Canvas, tint: dict) -> None:
    """tint: высота → цвет подкраски нажатой клавиши."""
    rect(c, (7, 8, 10), (X0 - 10, KEYS_Y, KEYBOARD_W + 20, KEYS_H))
    for p in WHITES:
        t = tint.get(p)
        fill = mix(KEY_WHITE, t, 0.6) if t else KEY_WHITE
        rect(c, fill, back_rect(p))
        front = front_rect(p)
        rect(c, fill, front, border_bottom_left_radius=6, border_bottom_right_radius=6)
        label = DARK_TEXT if t else (64, 69, 78)
        text(c, NOTE_LABEL[p], 18, label, bold=True, midbottom=(front.centerx, front.bottom - 9))
        text(c, note_name(p), 13, label if t else (128, 134, 144), midbottom=(front.centerx, front.bottom - 32))
    for p in BLACKS:
        t = tint.get(p)
        r = back_rect(p)
        body = mix(KEY_BLACK, t, 0.8) if t else KEY_BLACK
        rect(c, body, r, border_bottom_left_radius=5, border_bottom_right_radius=5)
        face = pygame.Rect(r.left + 4, r.top, r.width - 8, r.height - 12)  # чёрная выше белых — светлая грань
        rect(c, mix(body, WHITE, 0.12), face, border_bottom_left_radius=4, border_bottom_right_radius=4)
        text(c, NOTE_LABEL[p], 14, ink(body) if t else (176, 181, 190), bold=True,
             midbottom=(r.centerx, r.bottom - 18))


# --- падающие ноты ---

def draw_lanes(c: Canvas, top: int = TOP, bottom: int = HIT_Y) -> None:
    c.fill(BG, (0, top, W, bottom - top))
    for p in BLACKS:
        c.fill(BG_LANE, (slot_x(p), top, SLOT, bottom - top))
    for p in range(LOW, HIGH + 1, 12):  # граница октавы — перед каждым «до»
        x = slot_x(p)
        line(c, (35, 39, 47), (x, top), (x, bottom))


def draw_notes(c: Canvas, session, look: float, now: float) -> None:
    """Ноты урока. look — сколько секунд видно над линией."""
    pps = (HIT_Y - TOP) / look
    t = session.song_time
    c.set_clip(pygame.Rect(0, TOP, W, HIT_Y - TOP))
    first = math.ceil(t / session.spb)
    for k in range(first, math.floor((t + look) / session.spb) + 1):  # доли и такты
        y = HIT_Y - (k * session.spb - t) * pps
        color = (40, 44, 53) if k % 4 == 0 else (24, 27, 33)
        line(c, color, (X0 - 10, y), (X0 + KEYBOARD_W + 10, y))
    current = set(session.current_notes())
    cur_step = session.steps[session.cur].notes[0].step if current else None
    pulse = 0.5 + 0.5 * math.sin(now * 6)
    for n in session.notes:
        if n.start > t + look:
            break
        if n.start + n.dur < t:
            continue
        bottom = HIT_Y - (n.start - t) * pps
        top = bottom - n.dur * pps
        if bottom - top > 10:
            top += 3  # зазор между соседними нотами одной высоты
        box = lane_rect(n.pitch, top, bottom)
        base = note_color(n.pitch)
        if n.auto:
            fill, edge = mix(base, BG, 0.62), mix(base, BG, 0.3)
        elif n.result == "miss":
            fill, edge = (58, 62, 70), (84, 89, 98)
        elif n.result is not None:
            fill, edge = mix(base, WHITE, 0.4), WHITE
        else:
            fill, edge = base, mix(base, (0, 0, 0), 0.35)
        rect(c, fill, box, border_radius=6)
        if n.step == cur_step and n.pitch in current and not n.auto:
            rect(c, mix(fill, WHITE, 0.5 + 0.5 * pulse), box.inflate(4, 4), 2, border_radius=8)
        else:
            rect(c, edge, box, 1, border_radius=6)
        if not n.auto and box.height >= 26:
            text(c, NOTE_LABEL[n.pitch], 17, ink(fill), bold=True,
                 midbottom=(box.centerx, min(box.bottom, HIT_Y) - 6))
    c.set_clip(None)


def draw_trails(c: Canvas, trails, now: float, speed: float) -> None:
    """Свободная игра: от нажатой клавиши вверх уходит полоса её цвета."""
    c.set_clip(pygame.Rect(0, TOP, W, HIT_Y - TOP))
    for pitch, on, off in trails:
        top = HIT_Y - (now - on) * speed
        bottom = HIT_Y if off is None else HIT_Y - (now - off) * speed
        rect(c, note_color(pitch), lane_rect(pitch, top, max(bottom, top + 6)), border_radius=6)
    c.set_clip(None)
