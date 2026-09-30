"""Общие константы: диапазон клавиатуры, цветовой код нот, раскладка клавиатуры ПК."""
from __future__ import annotations

import pygame

LOW, HIGH = 48, 72  # C3..C5 — 25 клавиш, как на пианино НотоСвет

NOTE_NAMES = ["до", "до#", "ре", "ре#", "ми", "фа", "фа#", "соль", "соль#", "ля", "ля#", "си"]

# Цветовой код из плана: семь цветов белых клавиш, чёрные — оттенок между соседями.
COLORS = [
    (255, 0, 0),      # C  — красный
    (255, 40, 0),     # C# — красно-оранжевый
    (255, 100, 0),    # D  — оранжевый
    (255, 160, 0),    # D# — янтарный
    (255, 230, 0),    # E  — жёлтый
    (0, 255, 0),      # F  — зелёный
    (0, 255, 140),    # F# — бирюзовый
    (0, 200, 255),    # G  — голубой
    (0, 90, 255),     # G# — светло-синий
    (0, 0, 255),      # A  — синий
    (80, 0, 255),     # A# — сине-фиолетовый
    (170, 0, 255),    # B  — фиолетовый
]

BLACK_PCS = {1, 3, 6, 8, 10}


def is_black(pitch: int) -> bool:
    return pitch % 12 in BLACK_PCS


def note_color(pitch: int) -> tuple[int, int, int]:
    return COLORS[pitch % 12]


def note_name(pitch: int) -> str:
    return NOTE_NAMES[pitch % 12]


# Раскладка «как в трекерах»: нижний ряд — малая октава, верхний — первая и «до» второй.
# Клавиши задаются физическим положением (scancode), поэтому работают и в русской раскладке.
_LOWER = [  # C3..B3
    ("Z", pygame.KSCAN_Z), ("S", pygame.KSCAN_S), ("X", pygame.KSCAN_X), ("D", pygame.KSCAN_D),
    ("C", pygame.KSCAN_C), ("V", pygame.KSCAN_V), ("G", pygame.KSCAN_G), ("B", pygame.KSCAN_B),
    ("H", pygame.KSCAN_H), ("N", pygame.KSCAN_N), ("J", pygame.KSCAN_J), ("M", pygame.KSCAN_M),
]
_UPPER = [  # C4..C5
    ("Q", pygame.KSCAN_Q), ("2", pygame.KSCAN_2), ("W", pygame.KSCAN_W), ("3", pygame.KSCAN_3),
    ("E", pygame.KSCAN_E), ("R", pygame.KSCAN_R), ("5", pygame.KSCAN_5), ("T", pygame.KSCAN_T),
    ("6", pygame.KSCAN_6), ("Y", pygame.KSCAN_Y), ("7", pygame.KSCAN_7), ("U", pygame.KSCAN_U),
    ("I", pygame.KSCAN_I),
]

SCAN_TO_NOTE = {scan: LOW + i for i, (_, scan) in enumerate(_LOWER + _UPPER)}
NOTE_LABEL = {LOW + i: label for i, (label, _) in enumerate(_LOWER + _UPPER)}
