"""Экран редактора мелодии.

Бит-сетка как в rekordbox: такты пронумерованы, сильная доля — красная линия (каждые
4 такта — толще), доли и деления сетки — тонкие. Курсор стоит на месте, а сетка едет
сквозь него. Время идёт снизу вверх, как ноты в игре, поэтому каждая нота стоит точно
над своей клавишей. Если мелодия распознана из песни, слева видна волна песни (синяя,
как в rekordbox), а «Оригинал» играет песню в такт с нотами.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pygame

from . import draw as D
from .audio import Audio
from .config import HIGH, LOW, NOTE_LABEL, note_color
from .editor import BEATS_PER_BAR, EPS, FREE_STEP, GRIDS, HAND_MODES, LENGTHS, EditorModel

TOOL_H = 96
AREA_TOP, AREA_BOTTOM = TOOL_H, D.HIT_Y
PLAY_Y = D.HIT_Y - 72                     # курсор: ниже — прошлое, выше — то, что прозвучит
GRID_X1 = round(D.X0 + (HIGH - LOW + 1) * D.SLOT)
ZOOMS = (24, 32, 44, 60, 80, 110, 150)    # пикселей на долю
BAR_LINE = (196, 58, 68)                  # сильная доля — красная, как в rekordbox
BEAT_LINE = (60, 65, 76)
SUB_LINE = (31, 34, 41)
PLAYHEAD = (245, 245, 245)
REC = (235, 64, 64)
CTRL = pygame.KMOD_CTRL | pygame.KMOD_META
WAVE = (46, 118, 214)                     # волна песни — синяя, как в rekordbox
ORIGINAL_VOLUME = 0.7

HELP = [
    ("Мышь", ""),
    ("Щелчок по сетке", "поставить ноту; потянуть вверх — длиннее"),
    ("Перетащить ноту", "сдвинуть по времени и высоте; за верхний край — длина"),
    ("Правый щелчок", "удалить ноту (или Alt + щелчок)"),
    ("Shift + щелчок", "выбрать несколько нот"),
    ("Линейка слева", "щелчок — курсор сюда, тянуть — прокрутка"),
    ("Колесо / трекпад", "прокрутка; с Ctrl или Cmd — масштаб"),
    ("Клавиатура", ""),
    ("Клавиши нот", "ввод по шагам: нота у курсора, курсор дальше на «Длину»; аккорд — зажать вместе"),
    ("Tab", "ввод клавишами вкл/выкл"),
    ("Enter", "запись под метроном: такт отсчёта, ноты встают по сетке (Q)"),
    ("Кнопка «Q»", "привязка к сетке: новые ноты, перетаскивание, запись и курсор встают на линии сетки"),
    ("Выровнять, Ctrl/Cmd + U", "уже стоящие ноты — на сетку: выбранные, а если не выбрано — все"),
    ("Пробел", "играть / стоп"),
    ("↑ ↓   PgUp PgDn   Home", "курсор на шаг сетки / на такт / в начало"),
    ("← →", "выбранные ноты на полтона, с Shift — на октаву"),
    ("Backspace", "удалить выбранные; при вводе по шагам — шаг назад"),
    ("[ ]   − =", "темп   масштаб"),
    ("×2  ÷2", "темп вдвое быстрее / медленнее — ноты звучат так же, меняется запись"),
    ("Из песни", "ноты из своей песни: программа найдёт мелодию, темп и сетку"),
    ("Оригинал", "песня играет вместе с нотами — удобно проверять распознанное"),
    ("Ctrl/Cmd + Z, Shift + Z, S, A", "отменить, вернуть, сохранить, выбрать все"),
    ("Esc", "выйти из редактора"),
]


@dataclass
class Reference:
    """Песня, из которой распознана мелодия: доля 0 звучит на секунде start."""
    audio: Audio
    start: float


class EditorScreen:
    def __init__(self, app, model: EditorModel, back: str, reference: Reference | None = None):
        self.app, self.m, self.back = app, model, back
        self.ref = reference
        self.play_original = reference is not None
        self.ref_channel = None
        self.ref_sound = None
        if reference is not None and pygame.mixer.get_init():
            pygame.mixer.set_reserved(1)  # канал 0 — только для песни, синтезатор его не займёт
            self.ref_channel = pygame.mixer.Channel(0)
        self.zoom_i = 3
        self.drag: dict | None = None
        self.scrub: tuple[int, float] | None = None
        self.editing_title = False
        self.title_fresh = False           # название выделено целиком: первый ввод заменяет его
        self.title_rect = pygame.Rect(116, 10, 300, 36)
        self.help = False
        self.exit_armed: float | None = None
        self.wheel = self.zoom_wheel = 0.0
        self.audition: int | None = None
        self.cursor = pygame.SYSTEM_CURSOR_ARROW

    # --- координаты ---

    @property
    def ppb(self) -> int:
        return ZOOMS[self.zoom_i]

    def y_of(self, t: float) -> float:
        return PLAY_Y - (t - self.m.playhead) * self.ppb

    def t_of(self, y: float) -> float:
        return self.m.playhead + (PLAY_Y - y) / self.ppb

    @staticmethod
    def pitch_of(x: float) -> int | None:
        k = math.floor((x - D.X0) / D.SLOT)
        return LOW + k if 0 <= k <= HIGH - LOW else None

    @staticmethod
    def in_grid(pos) -> bool:
        return D.X0 <= pos[0] < GRID_X1 and AREA_TOP <= pos[1] < AREA_BOTTOM

    @staticmethod
    def in_ruler(pos) -> bool:
        return pos[0] < D.X0 and AREA_TOP <= pos[1] < AREA_BOTTOM

    # --- действия ---

    def move_cursor(self, beats: float) -> None:
        m = self.m
        base = m.playhead if m.playing else m.snap(m.playhead)
        m.seek(max(0.0, base + beats))

    def zoom(self, d: int) -> None:
        self.zoom_i = max(0, min(len(ZOOMS) - 1, self.zoom_i + d))

    def cycle(self, what: str) -> None:
        m = self.m
        if what == "grid":
            m.grid_i = (m.grid_i + 1) % len(GRIDS)
        elif what == "length":
            m.length_i = (m.length_i + 1) % len(LENGTHS)
        elif what == "hand":
            keys = list(HAND_MODES)
            m.hand = keys[(keys.index(m.hand) + 1) % len(keys)]
            if m.hand != "auto":
                m.set_hand(m.hand)  # выбранные ноты сразу получают эту руку
        elif what == "quantize":
            m.quantize = not m.quantize
        elif what == "metronome":
            m.metronome = not m.metronome
        elif what == "step":
            m.step_input = not m.step_input

    def scale(self, factor: float) -> None:
        if self.m.scale_tempo(factor):
            self._stop_original()
            self.app.notify(f"Темп {self.m.bpm:g} уд/мин — ноты звучат так же, поменялась только запись")

    def align(self) -> None:
        m = self.m
        grid = GRIDS[m.grid_i][1]
        if not m.notes:
            self.app.notify("Выравнивать пока нечего — поставьте ноты")
            return
        moved = m.align()
        which = "Выбранные ноты" if m.selected else "Все ноты"
        if moved:
            self.app.notify(f"Выровнено по сетке {grid}: {D.plural(moved, ('нота', 'ноты', 'нот'))}"
                            "  ·  Ctrl/Cmd+Z — вернуть как было")
        else:
            self.app.notify(f"{which} уже стоят на сетке {grid}")

    def delete_selected(self) -> None:
        m = self.m
        if m.selected:
            m.checkpoint()
            m.delete(list(m.selected))
        elif m.step_input and not m.playing:
            m.step_back()

    def save(self) -> None:
        self.end_title()
        was = self.m.path
        try:
            path = self.m.save()
        except (ValueError, OSError) as err:
            self.app.notify(f"Не сохранено: {err}", error=True)
            return
        self.app.reload_lessons(select=path)
        copy = " как своя копия" if was is not None and was != path else ""
        self.app.notify(f"Сохранено{copy}: «{self.m.title}» — её можно учить в «Учить мелодию»")

    def can_leave(self) -> bool:
        """С несохранёнными изменениями выходим со второго раза — первый только предупреждает."""
        if not self.m.dirty or (self.exit_armed is not None and self.app.now - self.exit_armed < 4):
            return True
        self.exit_armed = self.app.now
        self.app.notify("Есть несохранённые изменения: Ctrl/Cmd+S — сохранить, ещё раз — выйти без сохранения",
                        error=True)
        return False

    def exit(self) -> None:
        if self.can_leave():
            self.close()
            self.app.close_editor(self.back)

    def close(self) -> None:
        self.end_title()
        self.m.stop()
        self.app.play_events(self.m.pop_events())
        self._audition(None)
        self._set_cursor(pygame.SYSTEM_CURSOR_ARROW)
        self._stop_original()
        if self.ref_channel is not None:
            pygame.mixer.set_reserved(0)

    def from_song(self) -> None:
        """Ноты из своей песни. Текущая мелодия остаётся открытой, пока песню не распознают."""
        if self.can_leave():
            self.m.stop()
            self.app.play_events(self.m.pop_events())
            self._stop_original()
            self.app.open_library()

    # --- «Оригинал» ---

    def toggle_original(self) -> None:
        self.play_original = not self.play_original
        if not self.play_original:
            self._stop_original()
        elif self.m.playing:
            self._start_original(self.m.playhead)

    def _start_original(self, beat: float) -> None:
        if self.ref is None or self.ref_channel is None or not self.play_original:
            return
        spb = 60.0 / self.m.bpm
        start = self.ref.start + beat * spb
        end = self.ref.start + (self.m.length_beats() + 2 * BEATS_PER_BAR) * spb
        part = self.ref.audio.segment(start, max(end, start + 1))
        if len(part) == 0:
            return
        self.ref_sound = pygame.sndarray.make_sound(part)
        self.ref_channel.set_volume(ORIGINAL_VOLUME)
        self.ref_channel.play(self.ref_sound)

    def _stop_original(self) -> None:
        if self.ref_channel is not None:
            self.ref_channel.stop()

    def _set_cursor(self, cursor: int) -> None:
        if cursor != self.cursor:
            self.cursor = cursor
            try:
                pygame.mouse.set_cursor(cursor)
            except pygame.error:  # системные курсоры есть не везде — тогда остаётся обычный
                pass

    def _audition(self, pitch: int | None) -> None:
        """Прослушать ноту, пока её тянут мышью."""
        if self.audition is not None and self.audition not in self.app.held:
            self.app.synth.note_off(self.audition)
        if pitch is not None:
            self.app.synth.note_on(pitch)
        self.audition = pitch

    # --- название ---

    def begin_title(self) -> None:
        if not self.editing_title:
            self.app.release_keys()
            self.editing_title = self.title_fresh = True
            pygame.key.start_text_input()
            pygame.key.set_text_input_rect(self.app.to_window(self.title_rect))

    def end_title(self) -> None:
        if self.editing_title:
            self.editing_title = False
            pygame.key.stop_text_input()
            self.m.title = self.m.title.strip() or "Моя мелодия"

    def _title_event(self, e) -> bool:
        m = self.m
        if e.type == pygame.TEXTINPUT:
            if self.title_fresh:
                m.title, self.title_fresh = "", False
            if len(m.title) < 40:
                m.title += e.text
                m.dirty = True
            return True
        if e.type == pygame.KEYDOWN:
            if e.mod & CTRL:
                self.end_title()
                return False  # Ctrl+S и прочее — дальше
            if e.key == pygame.K_BACKSPACE:
                m.title = "" if self.title_fresh else m.title[:-1]
                self.title_fresh = False
                m.dirty = True
            elif e.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_ESCAPE, pygame.K_TAB):
                self.end_title()
            return True
        if e.type == pygame.KEYUP:
            return True
        if e.type == pygame.MOUSEBUTTONDOWN and not self.title_rect.collidepoint(e.pos):
            self.end_title()
        return False

    # --- ввод ---

    def handle(self, e) -> bool:
        """True — событие обработано редактором; иначе его обрабатывает программа (ноты, кнопки)."""
        if self.editing_title and self._title_event(e):
            return True
        if e.type == pygame.KEYDOWN:
            return self._key(e)
        if e.type == pygame.KEYUP:
            return e.key == pygame.K_SPACE
        if e.type == pygame.MOUSEWHEEL:
            return self._wheel(e)
        if e.type == pygame.MOUSEBUTTONDOWN:
            if self.help:
                self.help = False
                return True
            if self.in_ruler(e.pos) and e.button == 1:
                self.m.seek(self.m.snap(self.t_of(e.pos[1])))
                self.scrub = (e.pos[1], self.m.playhead)
                return True
            if self.in_grid(e.pos):
                self._press(e)
                return True
            return False
        if e.type == pygame.MOUSEMOTION:
            if self.scrub is not None:  # «взять» сетку и протянуть, как дорожку в rekordbox
                y0, t0 = self.scrub
                self.m.seek(max(0.0, self.m.snap(t0 + (e.pos[1] - y0) / self.ppb)))
                return True
            if self.drag is not None:
                self._drag_to(e.pos)
                return True
            self._hover(e.pos)
            return False
        if e.type == pygame.MOUSEBUTTONUP and e.button == 1:
            if self.scrub is not None:
                self.scrub = None
                return True
            if self.drag is not None:
                self._release()
                return True
        return False

    def _key(self, e) -> bool:
        m, k, sc = self.m, e.key, e.scancode
        shift = e.mod & pygame.KMOD_SHIFT
        if e.mod & CTRL:
            if sc == pygame.KSCAN_Z:
                m.redo() if shift else m.undo()
            elif sc == pygame.KSCAN_Y:
                m.redo()
            elif sc == pygame.KSCAN_S:
                self.save()
            elif sc == pygame.KSCAN_A:
                m.selected = set(m.notes)
            elif sc == pygame.KSCAN_U:
                self.align()
            else:
                return False
            return True
        if k == pygame.K_ESCAPE:
            if self.help:
                self.help = False
            else:
                self.exit()
        elif k == pygame.K_F1:
            self.help = not self.help
        elif k == pygame.K_SPACE:
            m.toggle_play()
        elif k in (pygame.K_RETURN, pygame.K_KP_ENTER):
            m.toggle_record()
        elif k == pygame.K_TAB:
            self.cycle("step")
        elif k in (pygame.K_BACKSPACE, pygame.K_DELETE):
            self.delete_selected()
        elif k in (pygame.K_UP, pygame.K_DOWN):
            self.move_cursor(m.grid if k == pygame.K_UP else -m.grid)
        elif k in (pygame.K_PAGEUP, pygame.K_PAGEDOWN):
            self.move_cursor(BEATS_PER_BAR if k == pygame.K_PAGEUP else -BEATS_PER_BAR)
        elif k == pygame.K_HOME:
            m.seek(0.0)
        elif k == pygame.K_END:
            m.seek(m.snap(m.length_beats()))
        elif k in (pygame.K_LEFT, pygame.K_RIGHT):
            m.transpose((12 if shift else 1) * (1 if k == pygame.K_RIGHT else -1))
        elif sc in (pygame.KSCAN_MINUS, pygame.KSCAN_EQUALS):
            self.zoom(1 if sc == pygame.KSCAN_EQUALS else -1)
        elif sc in (pygame.KSCAN_LEFTBRACKET, pygame.KSCAN_RIGHTBRACKET):
            m.set_bpm(m.bpm + (1 if sc == pygame.KSCAN_RIGHTBRACKET else -1) * (5 if shift else 1))
        else:
            return False
        return True

    def _wheel(self, e) -> bool:
        if self.app.mouse_pos()[1] >= AREA_BOTTOM:
            return False
        if pygame.key.get_mods() & CTRL:
            self.zoom_wheel += e.precise_y
            if abs(self.zoom_wheel) >= 1:
                self.zoom(1 if self.zoom_wheel > 0 else -1)
                self.zoom_wheel = 0.0
            return True
        self.wheel += e.precise_y
        steps = int(self.wheel)
        self.wheel -= steps
        if steps:
            self.move_cursor(steps * self.m.grid)
        return True

    def _press(self, e) -> None:
        m = self.m
        if m.recording:
            return
        if m.playing:
            m.stop()
        x, y = e.pos
        t, p = self.t_of(y), self.pitch_of(x)
        note = m.note_at(t, p)
        if e.button == 3 or (e.button == 1 and pygame.key.get_mods() & pygame.KMOD_ALT):
            if note is not None:
                m.checkpoint()
                m.delete([note])
            return
        if e.button != 1:
            return
        snap = m.snapshot()
        if note is not None:
            if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                m.selected ^= {note}
                return
            if note not in m.selected:
                m.selected = {note}
            near_top = y - self.y_of(note.end) < 8
            self.drag = {"mode": "resize" if near_top else "move", "note": note, "snap": snap,
                         "grab_t": t - note.t, "grab_p": p, "dn": 0, "changed": False,
                         "orig": {x: (x.t, x.n) for x in m.selected}}
        else:
            if not pygame.key.get_mods() & pygame.KMOD_SHIFT:
                m.selected.clear()
            note = m.add(m.snap_down(t), p)
            m.selected.add(note)
            self.drag = {"mode": "resize", "note": note, "snap": snap, "changed": True}
        self.drag.update(x0=x, y0=y, active=False)
        self._audition(note.n)

    def _drag_to(self, pos) -> None:
        d, m = self.drag, self.m
        x, y = pos
        if not d["active"]:  # мелкое дрожание мыши при щелчке — не перетаскивание
            if abs(y - d["y0"]) < 5 and abs(x - d["x0"]) < 5:
                return
            d["active"] = True
        t, note = self.t_of(y), d["note"]
        if d["mode"] == "resize":
            step = m.grid if m.quantize else FREE_STEP
            end = max(note.t + step, m.snap(t))
            if abs(end - note.end) > EPS:
                note.d = round(end - note.t, 4)
                d["changed"] = True
            return
        orig = d["orig"]
        dt = m.snap(t - d["grab_t"]) - orig[note][0]
        dt = max(dt, -min(o[0] for o in orig.values()))
        p = self.pitch_of(x)
        dn = d["dn"] if p is None else p - d["grab_p"]
        dn = max(LOW - min(o[1] for o in orig.values()), min(HIGH - max(o[1] for o in orig.values()), dn))
        for n, (t0, n0) in orig.items():
            n.t, n.n = round(t0 + dt, 4), n0 + dn
        if dn != d["dn"]:
            self._audition(orig[note][1] + dn)
        d["dn"] = dn
        d["changed"] = d["changed"] or abs(dt) > EPS or dn != 0

    def _release(self) -> None:
        d, self.drag = self.drag, None
        if d["changed"]:
            self.m.normalize()
            self.m.push_undo(d["snap"])
        self._audition(None)

    def _hover(self, pos) -> None:
        cursor = pygame.SYSTEM_CURSOR_ARROW
        if self.in_grid(pos):
            note = self.m.note_at(self.t_of(pos[1]), self.pitch_of(pos[0]))
            if note is not None:
                near_top = pos[1] - self.y_of(note.end) < 8
                cursor = pygame.SYSTEM_CURSOR_SIZENS if near_top else pygame.SYSTEM_CURSOR_HAND
        elif self.in_ruler(pos):
            cursor = pygame.SYSTEM_CURSOR_SIZENS
        self._set_cursor(cursor)

    # --- кадр ---

    def update(self, dt: float) -> None:
        self.m.update(dt)
        events = self.m.pop_events()
        for ev in events:
            if ev[0] == "start":
                self._start_original(ev[1])
            elif ev[0] == "stop":
                self._stop_original()
        self.app.play_events(events)

    def lit(self) -> set[int]:
        out = self.m.lit()
        if self.audition is not None:
            out.add(self.audition)
        return out

    def draw(self) -> None:
        s = self.app.screen
        self.draw_grid(s)
        self.draw_toolbar(s)
        if self.help:
            self.draw_help(s)

    def _button(self, rect, label, action, active=False, size=15, accent=False) -> pygame.Rect:
        s = self.app.screen
        fill = D.TEXT if accent else (D.PANEL_HI if active else D.PANEL)
        D.rect(s, fill, rect, border_radius=8)
        color = D.DARK_TEXT if accent else (D.TEXT if active else D.MUTED)
        if label:
            D.text(s, label, size, color, bold=active or accent, center=rect.center)
        self.app.zones.append((rect, action))
        return rect

    def _chip(self, x, label, action, active=True) -> int:
        w = D.measure(label, 14, True)[0] + 24
        self._button(pygame.Rect(x, 54, w, 32), label, action, active, 14)
        return x + w + 8

    def draw_toolbar(self, s) -> None:
        m = self.m
        D.rect(s, (20, 22, 27), (0, 0, D.W, TOOL_H))
        D.line(s, D.LINE, (0, TOOL_H - 1), (D.W, TOOL_H - 1))
        self._button(pygame.Rect(16, 10, 88, 36), "‹ Назад", self.exit)

        r = self.title_rect
        D.rect(s, D.PANEL_HI if self.editing_title else D.PANEL, r, border_radius=8)
        if self.editing_title:
            D.rect(s, D.TEXT, r, 2, border_radius=8)
        if self.editing_title and self.title_fresh:  # выделение всего названия
            sel = D.measure(m.title, 16, True)[0]
            D.rect(s, (52, 84, 140), (r.left + 10, r.top + 7, sel + 4, r.height - 14), border_radius=3)
        title = D.text(s, m.title or " ", 16, D.TEXT, bold=True, midleft=(r.left + 12, r.centery))
        if self.editing_title and not self.title_fresh and int(self.app.now * 2) % 2 == 0:
            D.line(s, D.TEXT, (title.right + 2, r.top + 9), (title.right + 2, r.bottom - 9), 2)
        elif not self.editing_title:
            D.text(s, "название", 12, D.DIM, midright=(r.right - 10, r.centery))
        self.app.zones.append((r, self.begin_title))

        # транспорт: в начало, играть/стоп, запись
        x = 436
        home = self._button(pygame.Rect(x, 10, 40, 36), "", lambda: m.seek(0.0))
        D.rect(s, D.TEXT, (home.x + 12, home.y + 11, 3, 14))
        D.polygon(s, D.TEXT, [(home.x + 28, home.y + 11), (home.x + 28, home.y + 25), (home.x + 16, home.y + 18)])
        play = self._button(pygame.Rect(x + 46, 10, 40, 36), "", m.toggle_play, active=m.playing and not m.recording)
        if m.playing:
            D.rect(s, D.TEXT, (play.x + 13, play.y + 11, 14, 14))
        else:
            D.polygon(s, D.TEXT, [(play.x + 14, play.y + 10), (play.x + 14, play.y + 26), (play.x + 28, play.y + 18)])
        rec = self._button(pygame.Rect(x + 92, 10, 40, 36), "", m.toggle_record, active=m.recording)
        on = not m.recording or int(self.app.now * 3) % 2 == 0
        D.circle(s, REC if on else (120, 40, 40), rec.center, 8)

        ph = max(0.0, m.playhead)
        bar, beat = int(ph // BEATS_PER_BAR) + 1, int(ph % BEATS_PER_BAR) + 1
        D.text(s, f"Такт {bar} · доля {beat}", 16, REC if m.recording else D.TEXT, bold=True, topleft=(590, 11))
        bars = math.ceil(m.length_beats() / BEATS_PER_BAR - EPS)
        status = "ЗАПИСЬ" if m.recording else "не сохранено" if m.dirty else "сохранено"
        line = (f"{D.plural(len(m.notes), ('нота', 'ноты', 'нот'))} · {D.plural(bars, ('такт', 'такта', 'тактов'))}"
                f" · {status}" + (f" · выбрано {len(m.selected)}" if m.selected else ""))
        D.text(s, line, 12, REC if m.recording else D.MUTED, topleft=(590, 32))

        self._button(pygame.Rect(790, 10, 110, 36), "Из песни…", self.from_song, active=True)
        right = D.W - 16
        save = pygame.Rect(right - 128, 10, 128, 36)
        self._button(save, "Сохранить", self.save, active=True, accent=m.dirty)
        self._button(pygame.Rect(save.left - 112, 10, 104, 36), "Отменить", m.undo)
        self._button(pygame.Rect(save.left - 224, 10, 104, 36), "Справка F1", lambda: setattr(self, "help", not self.help))

        # вторая строка: темп и настройки сетки
        D.text(s, "Темп", 14, D.MUTED, midleft=(16, 70))
        self._button(pygame.Rect(56, 54, 30, 32), "−", lambda: m.set_bpm(m.bpm - 1), size=18)
        box = pygame.Rect(90, 54, 100, 32)
        D.rect(s, D.PANEL, box, border_radius=8)
        D.text(s, f"{m.bpm:g} уд/мин", 14, D.TEXT, bold=True, center=box.center)
        self._button(pygame.Rect(194, 54, 30, 32), "+", lambda: m.set_bpm(m.bpm + 1), size=18)
        self._button(pygame.Rect(228, 54, 52, 32), "Тап", lambda: m.tap(self.app.now), active=True, size=14)
        self._button(pygame.Rect(284, 54, 40, 32), "×2", lambda: self.scale(2.0), active=True, size=14)
        self._button(pygame.Rect(328, 54, 40, 32), "÷2", lambda: self.scale(0.5), active=True, size=14)
        x = 384
        x = self._chip(x, f"Сетка {GRIDS[m.grid_i][1]}", lambda: self.cycle("grid"))
        x = self._chip(x, f"Длина {LENGTHS[m.length_i][1]}", lambda: self.cycle("length"))
        x = self._chip(x, f"Рука: {HAND_MODES[m.hand]}", lambda: self.cycle("hand"))
        x = self._chip(x, "Q", lambda: self.cycle("quantize"), m.quantize)
        x = self._chip(x, "Выровнять", self.align)
        x = self._chip(x, "Метроном", lambda: self.cycle("metronome"), m.metronome)
        x = self._chip(x, "Ввод клавишами", lambda: self.cycle("step"), m.step_input)
        if self.ref is not None:
            self._chip(x, "Оригинал", self.toggle_original, self.play_original)

    def draw_grid(self, s) -> None:
        m = self.m
        D.draw_lanes(s, AREA_TOP, AREA_BOTTOM)
        s.fill((20, 22, 27), (0, AREA_TOP, D.X0 - 2, AREA_BOTTOM - AREA_TOP))  # линейка
        if self.ref is not None:  # волна песни вдоль времени — видно, где мелодия и где пауза
            spb, env, k = 60.0 / m.bpm, self.ref.audio.env, s.scale
            right, room = s.px(D.X0 - 3), (D.X0 - 6) * k
            for py in range(s.px(AREA_TOP), s.px(AREA_BOTTOM)):  # по пикселю экрана — чёткая и на Retina
                i = int((self.ref.start + self.t_of(py / k) * spb) * 100)
                if 0 <= i < len(env) and env[i] > 0.02:
                    pygame.draw.line(s.surface, WAVE, (right - round(env[i] * room), py), (right, py))
        s.set_clip(pygame.Rect(0, AREA_TOP, D.W, AREA_BOTTOM - AREA_TOP))
        y0 = self.y_of(0.0)
        if y0 < AREA_BOTTOM:  # до начала мелодии
            s.fill((8, 9, 11), (D.X0, max(AREA_TOP, y0), GRID_X1 - D.X0, AREA_BOTTOM))

        t_lo, t_hi = max(0.0, self.t_of(AREA_BOTTOM)), self.t_of(AREA_TOP)
        step = m.grid
        for k in range(math.ceil(t_lo / step - EPS), math.floor(t_hi / step + EPS) + 1):
            t = k * step
            y = self.y_of(t)
            beat = abs(t - round(t)) < EPS
            if beat and round(t) % BEATS_PER_BAR == 0:
                phrase = round(t) % (4 * BEATS_PER_BAR) == 0
                D.line(s, BAR_LINE, (D.X0 - 8, y), (GRID_X1, y), 2 if phrase else 1)
                label = str(round(t) // BEATS_PER_BAR + 1)
                box = pygame.Rect((0, 0), D.measure(label, 13, True))
                box.midright = (D.X0 - 10, round(y))
                D.rect(s, (20, 22, 27), box.inflate(6, 2), border_radius=3)  # номер поверх волны
                D.text(s, label, 13, BAR_LINE, bold=True, midright=(D.X0 - 10, y))
            elif beat:
                D.line(s, BEAT_LINE, (D.X0 - 5, y), (GRID_X1, y))
            elif step * self.ppb >= 8:
                D.line(s, SUB_LINE, (D.X0, y), (GRID_X1, y))

        for x in m.notes:
            yb, yt = self.y_of(x.t), self.y_of(x.end)
            if yt > AREA_BOTTOM or yb < AREA_TOP:
                continue
            rect = D.lane_rect(x.n, yt + 1, yb - 1)
            fill = note_color(x.n)
            D.rect(s, fill, rect, border_radius=5)
            if x.h == "L":  # левая рука — тёмная полоска слева
                D.rect(s, D.mix(fill, (0, 0, 0), 0.5), (rect.x + 3, rect.y + 3, 4, max(0, rect.h - 6)),
                       border_radius=2)
            if x in m.selected:
                D.rect(s, D.WHITE, rect.inflate(4, 4), 2, border_radius=7)
            else:
                D.rect(s, D.mix(fill, (0, 0, 0), 0.35), rect, 1, border_radius=5)
            if rect.h >= 22:
                D.text(s, NOTE_LABEL[x.n], 15, D.ink(fill), bold=True, midbottom=(rect.centerx, rect.bottom - 4))

        for p, start in m.pending.items():  # записываемые сейчас ноты растут от нажатия
            rect = D.lane_rect(p, self.y_of(m.playhead), self.y_of(max(start, m.record_from)))
            D.rect(s, note_color(p), rect, border_radius=5)
            D.rect(s, REC, rect.inflate(4, 4), 2, border_radius=7)

        D.shade(s, (D.X0, PLAY_Y, GRID_X1 - D.X0, AREA_BOTTOM - PLAY_Y), (0, 0, 0, 110))  # прошлое притушено
        s.set_clip(None)

        color = REC if m.recording else PLAYHEAD
        D.line(s, color, (D.X0 - 8, PLAY_Y), (GRID_X1, PLAY_Y), 2)
        D.polygon(s, color, [(2, PLAY_Y - 7), (13, PLAY_Y), (2, PLAY_Y + 7)])

        mid = (D.W // 2, (AREA_TOP + PLAY_Y) // 2)
        if m.recording and m.playhead < m.record_from:
            left = m.record_from - m.playhead
            D.text(s, str(math.ceil(left)), 110, REC, bold=True, alpha=round(80 + 175 * (left % 1)), center=mid)
        elif not m.notes and not m.recording:
            lines = ["Щёлкните по сетке, чтобы поставить ноту,",
                     "или играйте на клавиатуре — ноты встанут по шагам от курсора.",
                     "«Из песни…» — ноты из своей музыки  ·  Enter — запись  ·  F1 — все клавиши"]
            for i, line in enumerate(lines):
                D.text(s, line, 18 if i < 2 else 15, D.TEXT if i < 2 else D.MUTED, center=(mid[0], mid[1] - 30 + i * 30))

    def draw_help(self, s) -> None:
        panel = pygame.Rect(0, 0, 940, 20 + 17 * len(HELP))
        panel.center = (D.W // 2, (AREA_TOP + AREA_BOTTOM) // 2)
        D.shade(s, (0, 0, D.W, AREA_BOTTOM), (6, 7, 9, 190))
        D.rect(s, D.PANEL, panel, border_radius=12)
        y = panel.top + 10
        for keys, desc in HELP:
            if not desc:
                D.text(s, keys, 13, D.MUTED, bold=True, topleft=(panel.left + 28, y))
            else:
                D.text(s, keys, 13, D.TEXT, bold=True, topleft=(panel.left + 28, y))
                D.text(s, desc, 13, D.TEXT, topleft=(panel.left + 260, y))
            y += 17
