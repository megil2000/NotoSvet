"""Главное меню: карточки режимов и цветовая легенда над живым пианино."""
from __future__ import annotations

import pygame

from . import draw as D
from .config import BLACK_PCS, COLORS, NOTE_NAMES

MINT = (151, 231, 203)
LILAC = (187, 175, 248)
PEACH = (245, 195, 143)
SURFACE = (23, 28, 35)
BORDER = (43, 51, 61)
SECONDARY = (161, 173, 184)


def _fit(label: str, size: int, width: int) -> str:
    if D.measure(label, size)[0] <= width:
        return label
    while label and D.measure(label + "…", size)[0] > width:
        label = label[:-1]
    return label + "…"


def _icon(s: D.Canvas, kind: int, x: int, y: int, color) -> None:
    """Небольшие векторные значки, чёткие и на Retina."""
    if kind == 0:
        for i, h in enumerate((12, 24, 18, 30, 14)):
            D.rect(s, color, (x + i * 6, y + (30 - h) / 2, 3, h), border_radius=2)
    elif kind == 1:
        D.rect(s, color, (x + 2, y + 2, 25, 27), 2, border_radius=5)
        D.line(s, color, (x + 10, y + 3), (x + 10, y + 28), 2)
        for dy in (10, 16, 22):
            D.line(s, color, (x + 15, y + dy), (x + 22, y + dy), 2)
    else:
        D.line(s, color, (x + 17, y + 4), (x + 17, y + 23), 3)
        D.line(s, color, (x + 17, y + 4), (x + 26, y + 7), 3)
        D.circle(s, color, (x + 12, y + 24), 6)
        D.line(s, color, (x + 3, y + 4), (x + 3, y + 12), 2)
        D.line(s, color, (x - 1, y + 8), (x + 7, y + 8), 2)


def draw_menu(app) -> None:
    s = app.screen
    items = app.menu_items()

    # Спокойный фон с едва заметным переходом к синему справа.
    for x in range(0, D.W, 8):
        D.rect(s, D.mix((15, 19, 25), (21, 28, 38), x / D.W), (x, 0, 8, D.HIT_Y))

    D.rect(s, (32, 49, 48), (40, 24, 38, 38), border_radius=12)
    _icon(s, 0, 46, 28, MINT)
    D.text(s, "НотоСвет", 24, D.TEXT, bold=True, midleft=(90, 42))
    D.line(s, BORDER, (239, 30), (239, 54))
    D.text(s, "ВАША МУЗЫКАЛЬНАЯ ПРАКТИКА", 11, SECONDARY, midleft=(259, 43))

    # Имя MIDI-устройства может быть длинным: статус не теснит кнопки окна.
    status = _fit(app.midi.label, 13, 230)
    status_w = D.measure(status, 13)[0] + 40
    status_x = 931 - status_w
    D.rect(s, SURFACE, (status_x, 25, status_w, 34), border_radius=17)
    D.circle(s, MINT if app.midi.name else SECONDARY, (status_x + 15, 42), 3)
    D.text(s, status, 13, SECONDARY, midleft=(status_x + 27, 42))

    D.text(s, "Играйте. Чувствуйте. Создавайте.", 39, D.TEXT, bold=True, topleft=(40, 98))
    D.text(s, "Следуйте за цветом — и ноты сложатся в музыку.", 19, SECONDARY, topleft=(42, 154))
    D.text(s, "Начните с любой клавиши или выберите свой режим.", 15, SECONDARY, topleft=(42, 185))

    # Абстрактная партитура повторяет форму падающих нот в самой игре.
    for i in range(8):
        x = 881 + i * 47
        D.line(s, (35, 44, 55), (x, 94), (x, 208))
    notes = ((0, 148, 43, MINT), (1, 120, 34, LILAC), (2, 166, 32, PEACH),
             (3, 99, 46, MINT), (4, 145, 38, (135, 195, 237)),
             (5, 112, 47, LILAC), (6, 163, 29, PEACH))
    for lane, y, height, color in notes:
        x = 889 + lane * 47
        D.rect(s, D.mix((21, 28, 38), color, 0.12), (x - 4, y - 4, 37, height + 8), border_radius=12)
        D.rect(s, D.mix((21, 28, 38), color, 0.7), (x, y, 29, height), border_radius=8)
        D.line(s, color, (x + 8, y + 6), (x + 21, y + 6), 2)

    descriptions = (("Знакомьтесь со звуком и импровизируйте.", "Без правил и спешки", MINT),
                    ("Разбирайте любимые мелодии по нотам.", "В своём темпе", LILAC),
                    ("Превращайте свои идеи в музыку.", "Редактор и импорт песен", PEACH))
    for i, (description, detail, accent) in enumerate(descriptions):
        r = pygame.Rect(40 + i * 406, 240, 388, 148)
        selected = app.menu_i == i
        D.rect(s, (10, 14, 19), r.move(0, 4), border_radius=18)
        D.rect(s, D.mix(SURFACE, accent, 0.09) if selected else SURFACE, r, border_radius=18)
        D.rect(s, accent if selected else BORDER, r, 2 if selected else 1, border_radius=18)
        D.rect(s, D.mix(SURFACE, accent, 0.13), (r.x + 20, r.y + 20, 44, 44), border_radius=13)
        _icon(s, i, r.x + 28, r.y + 27, accent)
        D.text(s, items[i][0], 23, D.TEXT, bold=True, midleft=(r.x + 78, r.y + 42))
        D.text(s, description, 15, SECONDARY, topleft=(r.x + 22, r.y + 80))
        D.text(s, detail, 13, accent, topleft=(r.x + 22, r.y + 112))
        if selected:
            D.rect(s, accent, (r.right - 53, r.bottom - 37, 32, 22), border_radius=7)
        arrow = D.DARK_TEXT if selected else SECONDARY
        cx, cy = r.right - 37, r.bottom - 26
        D.line(s, arrow, (cx - 7, cy), (cx + 6, cy), 2)
        D.line(s, arrow, (cx + 2, cy - 4), (cx + 6, cy), 2)
        D.line(s, arrow, (cx + 2, cy + 4), (cx + 6, cy), 2)
        app.zones.append((r, items[i][1]))

    # Порядок зон совпадает с menu_items: мышь и стрелки выбирают один пункт.
    for i, r in ((3, pygame.Rect(947, 24, 185, 36)), (4, pygame.Rect(1144, 24, 96, 36))):
        selected = app.menu_i == i
        D.rect(s, (37, 46, 56) if selected else SURFACE, r, border_radius=10)
        D.rect(s, MINT if selected else BORDER, r, 2 if selected else 1, border_radius=10)
        D.text(s, items[i][0], 14, D.TEXT if selected else SECONDARY, center=r.center)
        app.zones.append((r, items[i][1]))

    D.rect(s, SURFACE, (40, 406, 1200, 65), border_radius=14)
    D.text(s, "У каждой ноты свой цвет", 16, D.TEXT, bold=True, topleft=(60, 416))
    D.text(s, "Смотрите на подсветку клавиш", 13, SECONDARY, topleft=(60, 442))
    D.line(s, BORDER, (320, 420), (320, 457))
    for i in range(12):
        x = 345 + i * 73
        D.rect(s, COLORS[i], (x, 423, 47, 5), border_radius=3)
        D.text(s, NOTE_NAMES[i], 13, SECONDARY if i in BLACK_PCS else D.TEXT,
               midtop=(x + 23, 440))

    D.text(s, "Z…M / Q…I — играть   ·   Пробел — педаль", 13, SECONDARY, topleft=(42, 488))
    D.text(s, "← → ↑ ↓ — выбор   ·   Enter — открыть", 13, SECONDARY, midtop=(652, 488))
    D.text(s, "Своя мелодия? Перетащите .mid / .json", 13, SECONDARY, topright=(1238, 488))
