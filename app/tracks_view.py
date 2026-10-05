"""Дорожки песни, как в Songsterr: нейросеть разделила фрагмент на инструменты — выберите,
какую дорожку играть. Остальные потом звучат вместе с вами («минус»).

В каждой строке — значок инструмента, сколько нот нашлось, мини-ролл нот цветами клавиш
(видно, где инструмент играет) и кнопки «Слушать» (только эта дорожка) и «Играть».
"""
from __future__ import annotations

import math

import pygame

from . import draw as D
from .config import note_color
from .neural import QUIET

COLORS = {"vocals": (240, 98, 146), "guitar": (255, 167, 38), "piano": (100, 181, 246),
          "bass": (186, 104, 200), "other": (77, 208, 160), "drums": (176, 184, 196), "melody": (233, 235, 238)}
KIND_LABEL = {"melody": "один голос", "bass": "бас", "chords": "аккорды"}
TOP, ROW_H = 84, 68
ROLL = pygame.Rect(440, 0, 560, 0)        # мини-ролл: по горизонтали — весь фрагмент


def fmt(sec: float) -> str:
    m, s = divmod(int(max(0.0, sec)), 60)
    return f"{m}:{s:02d}"


def octave_note(shift: int) -> str:
    if not shift:
        return ""
    n = abs(shift) // 12
    where = "выше" if shift > 0 else "ниже"
    return f"на октаву {where}" if n == 1 else f"на {n} октавы {where}"


def track_info(track) -> str:
    if track.playable:
        parts = [D.plural(len(track.notes), ("нота", "ноты", "нот")), KIND_LABEL.get(track.kind, ""),
                 octave_note(track.shift)]
        return " · ".join(p for p in parts if p)
    if track.kind == "drums":
        return "без нот — звучит в минусе"
    if track.level < QUIET:
        return "почти не слышно в этом куске"
    return "нот не нашлось"


def draw_icon(c: D.Canvas, key: str, center, size: float, color) -> None:
    """Значок инструмента в квадрате size×size: микрофон, гитара, клавиши, бас, барабан, волна."""
    x, y = center
    u = size / 28

    def p(dx, dy):
        return x + dx * u, y + dy * u

    w = max(1.0, 2 * u)
    if key == "vocals":
        D.rect(c, color, (x - 5 * u, y - 12 * u, 10 * u, 15 * u), border_radius=5 * u)
        cup = [p(-9, -4), p(-9, 0), p(-5, 5), p(5, 5), p(9, 0), p(9, -4)]
        for a, b in zip(cup, cup[1:]):
            D.line(c, color, a, b, w)
        D.line(c, color, p(0, 5), p(0, 11), w)
        D.line(c, color, p(-6, 11), p(6, 11), w)
    elif key in ("guitar", "bass"):
        long = key == "bass"
        D.line(c, color, p(-2, 2), p(12 if long else 9, -12 if long else -9), 3 * u)
        D.rect(c, color, (*p(9 if long else 6, -14 if long else -11), 5 * u, 5 * u), border_radius=u)
        D.circle(c, color, p(-6, 6), (6 if long else 8) * u)
        D.circle(c, color, p(-1, 1), (4 if long else 6) * u)
        if not long:
            D.circle(c, D.mix(color, (0, 0, 0), 0.6), p(-3, 3), 2 * u)
    elif key == "piano":
        D.rect(c, color, (*p(-12, -8), 24 * u, 16 * u), border_radius=2 * u)
        for k in range(1, 4):
            D.line(c, D.mix(color, (0, 0, 0), 0.5), p(-12 + 6 * k, -8), p(-12 + 6 * k, 8), max(1.0, u))
        for k in (0, 1, 2):
            D.rect(c, D.mix(color, (0, 0, 0), 0.75), (*p(-9 + 6 * k + (1 if k == 2 else 0), -8), 4 * u, 9 * u))
    elif key == "drums":
        D.rect(c, color, (*p(-11, -1), 22 * u, 10 * u), border_radius=2 * u)
        pygame.draw.ellipse(c.surface, color, c.rect((*p(-11, -5), 22 * u, 8 * u)))
        pygame.draw.ellipse(c.surface, D.mix(color, (0, 0, 0), 0.4), c.rect((*p(-9, -4), 18 * u, 5 * u)))
        D.line(c, color, p(-9, -13), p(-1, -5), w)
        D.line(c, color, p(9, -13), p(1, -5), w)
    elif key == "other":
        pts = [p(-12 + k, -7 * math.sin(k / 24 * 2 * math.pi)) for k in range(0, 25, 2)]
        for a, b in zip(pts, pts[1:]):
            D.line(c, color, a, b, w)
    else:  # мелодия: нота
        D.circle(c, color, p(-4, 7), 5 * u)
        D.line(c, color, p(0, 7), p(0, -11), w)
        D.line(c, color, p(0, -11), p(7, -6), w)


class TracksScreen:
    def __init__(self, app, rec, audio, start: float):
        self.app, self.rec, self.audio, self.start = app, rec, audio, start
        self.sel = rec.best()
        self.listen: tuple[int, float] | None = None   # (дорожка, время запуска)
        self.channel = None
        self.sound = None

    # --- действия ---

    def play(self, i: int | None = None) -> None:
        i = self.sel if i is None else i
        if self.rec.tracks[i].playable:
            self.close()
            self.app.play_track(i)
        else:
            self.sel = i
            self.app.notify(f"У дорожки «{self.rec.tracks[i].name}» нет нот для клавиш — выберите другую", error=True)

    def toggle_listen(self, i: int | None = None) -> None:
        i = self.sel if i is None else i
        was = self.listen[0] if self.listen else None
        self.stop_listen()
        if was == i or pygame.mixer.get_init() is None or self.rec.tracks[i].pcm is None:
            return
        self.sel = i
        self.sound = pygame.sndarray.make_sound(self.rec.tracks[i].pcm)
        self.channel = self.sound.play()
        self.listen = (i, self.app.now)

    def stop_listen(self) -> None:
        if self.channel is not None:
            self.channel.stop()
        self.channel = self.listen = None

    def back_out(self) -> None:
        self.close()
        self.app.close_tracks()

    def close(self) -> None:
        self.stop_listen()

    def _row(self, i: int) -> pygame.Rect:
        return pygame.Rect(60, TOP + i * ROW_H, D.W - 120, ROW_H - 6)

    # --- ввод ---

    def handle(self, e) -> bool:
        n = len(self.rec.tracks)
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE:
                self.back_out()
            elif e.key in (pygame.K_UP, pygame.K_DOWN):
                self.sel = (self.sel + (1 if e.key == pygame.K_DOWN else -1)) % n
            elif e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.play()
            elif e.key == pygame.K_SPACE:
                self.toggle_listen()
            else:
                return False
            return True
        if e.type == pygame.KEYUP and e.key == pygame.K_SPACE:
            return True  # пробел здесь — «слушать», а не педаль
        if e.type == pygame.MOUSEMOTION:
            for i in range(n):
                if self._row(i).collidepoint(e.pos):
                    self.sel = i
        return False

    def update(self, dt: float) -> None:
        if self.listen is not None and (self.channel is None or not self.channel.get_busy()):
            self.listen = self.channel = None

    # --- кадр ---

    def draw(self) -> None:
        s, rec = self.app.screen, self.rec
        D.rect(s, D.PANEL, (16, 12, 88, 36), border_radius=8)
        D.text(s, "‹ Назад", 15, D.MUTED, center=(60, 30))
        self.app.zones.append((pygame.Rect(16, 12, 88, 36), self.back_out))
        D.text(s, self.audio.title[:44], 24, D.TEXT, bold=True, midleft=(124, 30))
        a, b = self.start, self.start + rec.duration
        D.text(s, f"Нейросеть разделила фрагмент {fmt(a)} – {fmt(b)} на дорожки · темп {rec.bpm:g} уд/мин",
               14, D.MUTED, midleft=(124, 58))
        D.text(s, "Выберите, какую дорожку играть —", 14, D.TEXT, bold=True, topright=(D.W - 60, 18))
        D.text(s, "остальные зазвучат вместе с вами", 14, D.MUTED, topright=(D.W - 60, 38))
        for i, track in enumerate(rec.tracks):
            self.draw_row(s, i, track)
        D.text(s, "↑↓ — дорожка   ·   Пробел — слушать   ·   Enter — играть   ·   Esc — назад", 13, D.DIM,
               center=(D.W // 2, TOP + len(rec.tracks) * ROW_H + 6))

    def draw_row(self, s, i: int, track) -> None:
        row = self._row(i)
        sel = i == self.sel
        D.rect(s, D.PANEL_HI if sel else D.PANEL, row, border_radius=10)
        if sel:
            D.rect(s, D.TEXT, row, 1, border_radius=10)
        color = COLORS.get(track.key, D.TEXT)
        if not track.playable:
            color = D.mix(color, D.PANEL, 0.55)
        center = (row.left + 34, row.centery)
        D.circle(s, color, center, 22)
        draw_icon(s, track.key, center, 26, D.DARK_TEXT)
        D.text(s, track.name, 18, D.TEXT if track.playable else D.MUTED, bold=True, topleft=(row.left + 70, row.top + 9))
        D.text(s, track_info(track), 13, D.MUTED, topleft=(row.left + 70, row.top + 35))

        roll = pygame.Rect(ROLL.left, row.top + 8, ROLL.width, row.height - 16)
        D.rect(s, (18, 20, 25), roll, border_radius=6)
        self.draw_roll(s, roll, track)
        if self.listen is not None and self.listen[0] == i:
            x = roll.left + min(1.0, (self.app.now - self.listen[1]) / max(0.1, self.rec.duration)) * roll.width
            D.line(s, (255, 90, 90), (x, roll.top - 2), (x, roll.bottom + 2), 2)

        listen = pygame.Rect(roll.right + 16, row.centery - 18, 94, 36)
        playing = self.listen is not None and self.listen[0] == i
        D.rect(s, D.PANEL_HI if not sel else (52, 57, 67), listen, border_radius=8)
        D.text(s, "Стоп" if playing else "Слушать", 14, D.TEXT, bold=playing, center=listen.center)
        self.app.zones.append((listen, lambda i=i: self.toggle_listen(i)))
        go = pygame.Rect(listen.right + 8, row.centery - 18, row.right - listen.right - 18, 36)
        accent = sel and track.playable
        D.rect(s, D.TEXT if accent else D.PANEL_HI, go, border_radius=8)
        D.text(s, "Играть ›", 15, D.DARK_TEXT if accent else (D.TEXT if track.playable else D.DIM), bold=True,
               center=go.center)
        self.app.zones.append((go, lambda i=i: self.play(i)))
        self.app.zones.append((row, lambda i=i: setattr(self, "sel", i)))

    def draw_roll(self, s, roll: pygame.Rect, track) -> None:
        """Ноты дорожки по времени фрагмента; у дорожки без нот — её громкость."""
        dur = max(0.1, self.rec.duration)
        if not track.raw:
            env = track.env
            if env is None or not len(env):
                return
            box = s.rect(roll)
            mid = box.centery
            for px in range(box.width):  # по пикселю экрана: чётко и на Retina
                k = int(px / box.width * len(env))
                v = env[k:max(k + 1, int((px + 1) / box.width * len(env)))].max() * min(1.0, track.level * 3)
                h = round(v * (box.height / 2 - 2))
                if h:
                    pygame.draw.line(s.surface, (70, 76, 88), (box.left + px, mid - h), (box.left + px, mid + h))
            return
        pitches = sorted(p for _, _, p in track.raw)
        k = len(pitches) // 20  # редкие крайние ноты не сплющивают остальные: края — без 5 % нот
        lo, hi = pitches[k] - 1, pitches[-1 - k] + 1
        step = (roll.height - 6) / max(1, hi - lo)
        s.set_clip(roll)
        for t, d, p in track.raw:
            x = roll.left + t / dur * roll.width
            y = roll.bottom - 3 - (min(hi, max(lo, p)) - lo) * step
            D.rect(s, note_color(p), pygame.FRect(x, y - max(2.0, step) / 2, max(2.0, d / dur * roll.width), max(2.0, step)))
        s.set_clip(None)
