"""Программа НотоСвет для ПК: меню, выбор мелодии, «Падающие ноты», итог, редактор мелодии.

Ввод — клавиатура компьютера, мышь по клавишам на экране или MIDI (пианино НотоСвет
либо любая MIDI-клавиатура). Все три источника равноправны.
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from pathlib import Path

import pygame

from . import draw as D
from .draw import plural
from . import paths, records, settings
from .audio import AUDIO_EXT
from .converter import import_file
from .display import Display
from .editor import EditorModel
from .editor_view import EditorScreen, Reference
from .import_view import FragmentScreen, LibraryScreen
from .config import BLACK_PCS, COLORS, HIGH, LOW, NOTE_LABEL, NOTE_NAMES, SCAN_TO_NOTE, note_color, note_name
from .icon import make_icon
from .lesson import Lesson, load_lessons
from .midi_io import MidiLink
from .synth import Synth
from .trainer import HANDS, MODES, TEMPOS, Session

MODE_KEYS = list(MODES)
HAND_KEYS = list(HANDS)
MODE_HELP = {
    "wait": "Подсказка ждёт правильную клавишу, темп задаёте вы. Для первых занятий с мелодией.",
    "rhythm": "Ноты идут в темпе под метроном. «Отлично» — ±60 мс, «Хорошо» — ±150 мс, "
              "иначе «Мимо». Очки, серия и итоговая оценка S/A/B/C.",
    "demo": "Программа сама играет мелодию и подсвечивает клавиши — чтобы услышать, как должно звучать.",
}
JUDGE = {"perfect": ("Отлично", D.WHITE), "good": ("Хорошо", (186, 192, 202)), "miss": ("Мимо", (255, 110, 110))}
FREE_SPEED = 170      # px/с — полосы свободной игры уходят вверх
RELEASE_FADE = 0.3    # отпущенная клавиша гаснет за 0,3 с
POPUP_LIFE = 0.7
TOAST_LIFE = 4.0
FULLSCREEN_MODS = pygame.KMOD_CTRL | pygame.KMOD_META


def lookahead(session: Session) -> float:
    """Сколько секунд видно над линией: около 4 долей при любом темпе."""
    return min(4.5, max(1.8, 4 * session.spb))


def fmt_time(sec: float) -> str:
    m, s = divmod(int(round(sec)), 60)
    return f"{m}:{s:02d}"


def run_line(r: dict) -> str:
    d = r.get("date", "")
    when = f"{d[8:10]}.{d[5:7]} {d[11:16]}" if len(d) >= 16 else d
    line = f"{when}   {MODES.get(r['mode'], r['mode'])}, {r['tempo']} %   точность {r['accuracy']} %   ошибок {r['errors']}"
    return line + (f"   оценка {r['grade']}" if "grade" in r else "")


class App:
    def __init__(self, screen: D.Canvas, display: Display | None = None):
        self.screen = screen
        self.display = display       # None — без окна (самопроверка)
        self.synth = Synth()
        self.midi = MidiLink()
        self.lessons, self.lesson_errors = load_lessons()
        self.records = records.load()
        self.state = "menu"          # menu / setup / play / results
        self.menu_i = 0
        self.row = 0                 # строка на экране выбора мелодии
        self.opt = {"lesson": 0, "mode": 0, "tempo": 2, "hand": 0}
        self.session: Session | None = None
        self.paused = False
        self.result: dict | None = None
        self.result_title = ""
        self.held: set[int] = set()
        self.release_at: dict[int, float] = {}
        self.error_at: dict[int, float] = {}
        self.trails: list[list] = []
        self.popups: list[tuple] = []
        self.last_note: int | None = None
        self.mouse_note: int | None = None
        self.zones: list[tuple[pygame.Rect, object]] = []
        self.now = 0.0
        self.toast: tuple[str, bool, float] | None = None
        self.editor: EditorScreen | None = None
        self.library: LibraryScreen | None = None
        self.fragment: FragmentScreen | None = None
        self.running = True

    @property
    def lesson(self) -> Lesson:
        return self.lessons[self.opt["lesson"]]

    def menu_items(self):
        full = self.display is not None and self.display.fullscreen
        return [("Свободная игра", self.start_free), ("Учить мелодию", self.open_setup),
                ("Создать мелодию", self.new_melody),
                ("Оконный режим" if full else "Полный экран", self.toggle_fullscreen), ("Выход", self.quit)]

    # --- цикл ---

    def loop(self) -> None:
        clock = pygame.time.Clock()
        last = time.perf_counter()
        while self.running:
            now = time.perf_counter()
            self.step(min(now - last, 0.1))
            last = now
            self.display.present()
            clock.tick(120)
        self.midi.close()

    def step(self, dt: float) -> None:
        self.now += dt
        if self.display is not None and self.display.layout():  # окно изменилось — новый холст
            self.screen = self.display.canvas
        self.update(dt)
        for e in pygame.event.get():
            self.handle(e)
        for kind, value in self.midi.poll():
            if kind == "on":
                self.note_down(value)
            elif kind == "off":
                self.note_up(value)
            else:
                self.set_pedal(value)
        self.draw()

    def update(self, dt: float) -> None:
        s = self.session
        if self.state == "play" and s is not None and not self.paused:
            s.update(dt)
            self.drain()
            if s.finished:
                self.finish()
        screen = self.active_screen()
        if screen is not None:
            screen.update(dt)
        self.popups = [x for x in self.popups if self.now - x[2] < POPUP_LIFE]
        self.trails = [tr for tr in self.trails
                       if tr[2] is None or D.HIT_Y - (self.now - tr[2]) * FREE_SPEED > D.TOP]
        self.release_at = {p: t for p, t in self.release_at.items() if self.now - t < RELEASE_FADE}
        self.error_at = {p: t for p, t in self.error_at.items() if self.now - t < 0.4}
        playing = self.state == "play" and self.session is not None
        self.midi.send_hints(self.session.hints() if playing else {})

    def drain(self) -> None:
        self.play_events(self.session.pop_events())

    def play_events(self, events: list[tuple]) -> None:
        """События урока или редактора → звук и эффекты."""
        for ev in events:
            kind = ev[0]
            if kind == "on":
                self.synth.note_on(ev[1])
            elif kind == "off" and ev[1] not in self.held:
                self.synth.note_off(ev[1])
            elif kind == "click":
                self.synth.click(ev[1])
            elif kind == "error":
                self.synth.error()
                self.error_at[ev[1]] = self.now
            elif kind == "judge":
                self.popups.append((ev[1], ev[2], self.now))

    # --- ввод ---

    def active_screen(self):
        return {"editor": self.editor, "library": self.library, "fragment": self.fragment}.get(self.state)

    def mouse_pos(self) -> tuple[int, int]:
        return self.display.mouse_pos() if self.display is not None else pygame.mouse.get_pos()

    def to_window(self, rect) -> pygame.Rect:
        return self.display.to_window(rect) if self.display is not None else pygame.Rect(rect)

    def toggle_fullscreen(self) -> None:
        if self.display is not None:
            self.display.set_fullscreen(not self.display.fullscreen)
            settings.save(fullscreen=self.display.fullscreen)

    def handle(self, e: pygame.event.Event) -> None:
        if self.display is not None:
            e = self.display.map_event(e)  # мышь — в координаты 1280×720
        if e.type == pygame.KEYDOWN and (e.key == pygame.K_F11 or
                                         e.scancode == pygame.KSCAN_F and e.mod & FULLSCREEN_MODS):
            self.toggle_fullscreen()  # на любом экране, даже в поиске песен
            return
        screen = self.active_screen()
        if screen is not None and screen.handle(e):
            return
        if e.type == pygame.QUIT:
            self.quit()
        elif e.type == pygame.DROPFILE:
            if Path(e.file).suffix.lower() in AUDIO_EXT | {".m4p"}:
                self.open_fragment(Path(e.file))
            else:
                self.import_lesson(e.file)
        elif e.type == pygame.KEYDOWN:
            pitch = SCAN_TO_NOTE.get(e.scancode)
            if pitch is not None and not e.mod & (pygame.KMOD_CTRL | pygame.KMOD_META):
                self.note_down(pitch)
            elif e.key == pygame.K_SPACE:
                self.set_pedal(True)
            else:
                self.on_key(e.key)
        elif e.type == pygame.KEYUP:
            pitch = SCAN_TO_NOTE.get(e.scancode)
            if pitch is not None:
                self.note_up(pitch)
            elif e.key == pygame.K_SPACE:
                self.set_pedal(False)
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            pitch = D.key_at(e.pos)
            if pitch is not None:
                self.mouse_note = pitch
                self.note_down(pitch)
                return
            for rect, action in self.zones:
                if rect.collidepoint(e.pos):
                    action()
                    break
        elif e.type == pygame.MOUSEBUTTONUP and e.button == 1 and self.mouse_note is not None:
            self.note_up(self.mouse_note)
            self.mouse_note = None
        elif e.type == pygame.MOUSEMOTION and self.mouse_note is not None:
            pitch = D.key_at(e.pos)  # глиссандо мышью
            if pitch != self.mouse_note:
                self.note_up(self.mouse_note)
                self.mouse_note = pitch
                if pitch is not None:
                    self.note_down(pitch)
        elif e.type == pygame.WINDOWFOCUSLOST:
            for p in list(self.held):  # иначе клавиши «залипнут»
                self.note_up(p)
            self.set_pedal(False)
            if self.state == "play" and self.session is not None:
                self.paused = True

    def on_key(self, key: int) -> None:
        enter = key in (pygame.K_RETURN, pygame.K_KP_ENTER)
        if self.state == "menu":
            items = self.menu_items()
            if key == pygame.K_UP:
                self.menu_i = (self.menu_i - 1) % len(items)
            elif key == pygame.K_DOWN:
                self.menu_i = (self.menu_i + 1) % len(items)
            elif enter:
                items[self.menu_i][1]()
        elif self.state == "setup":
            if key == pygame.K_ESCAPE:
                self.state = "menu"
            elif key == pygame.K_UP:
                self.row = (self.row - 1) % 5
            elif key == pygame.K_DOWN:
                self.row = (self.row + 1) % 5
            elif key in (pygame.K_LEFT, pygame.K_RIGHT):
                self.pick(self.row, -1 if key == pygame.K_LEFT else 1)
            elif enter:
                self.start_session()
        elif self.state == "play":
            if key == pygame.K_ESCAPE:
                self.stop_play()
            elif enter and self.session is not None:
                self.paused = not self.paused
            elif key == pygame.K_TAB and self.session is not None:
                self.session.metronome = not self.session.metronome
        elif self.state == "results":
            if enter:
                self.start_session()
            elif key == pygame.K_ESCAPE:
                self.state = "setup"

    def note_down(self, pitch: int) -> None:
        if pitch in self.held or not LOW <= pitch <= HIGH:
            return
        self.held.add(pitch)
        self.release_at.pop(pitch, None)
        self.synth.note_on(pitch)
        self.last_note = pitch
        if self.state == "editor":
            self.editor.m.key_down(pitch)
        if self.state != "play":
            return
        if self.session is None:
            self.trails.append([pitch, self.now, None])
        elif not self.paused:
            self.session.press(pitch)
            self.drain()

    def note_up(self, pitch: int) -> None:
        if pitch not in self.held:
            return
        self.held.discard(pitch)
        self.release_at[pitch] = self.now
        if self.state == "editor":
            self.editor.m.key_up(pitch)
        if not (self.session and pitch in self.session.sounding()):
            self.synth.note_off(pitch)
        for tr in self.trails:
            if tr[0] == pitch and tr[2] is None:
                tr[2] = self.now

    def set_pedal(self, on: bool) -> None:
        self.synth.set_pedal(on)

    # --- свои мелодии ---

    def notify(self, message: str, error: bool = False) -> None:
        self.toast = (message, error, self.now)

    def import_lesson(self, file: str) -> None:
        """Файл перетащили в окно: MIDI превращаем в урок, JSON проверяем и копируем."""
        try:
            target = import_file(Path(file))
        except ValueError as err:
            self.notify(f"{Path(file).name} не добавлен: {err}", error=True)
            return
        self.reload_lessons(select=target)
        if self.state == "editor":
            if not self.editor.m.dirty:  # пустой или сохранённый редактор — открываем мелодию в нём
                self.open_editor(self.lesson, self.editor.back)
        elif self.state != "play":
            self.state = "setup"
        self.notify(f"Мелодия «{self.lesson.title}» добавлена — {plural(len(self.lesson.notes), ('нота', 'ноты', 'нот'))}")

    def reload_lessons(self, select: Path | None = None) -> None:
        self.lessons, self.lesson_errors = load_lessons()
        if select is not None:
            self.opt["lesson"] = next((i for i, les in enumerate(self.lessons)
                                       if les.path and les.path.resolve() == Path(select).resolve()), 0)
            self.opt["hand"] = 0
            self.row = 0
        self.opt["lesson"] = min(self.opt["lesson"], max(0, len(self.lessons) - 1))

    def release_keys(self) -> None:
        for p in list(self.held):
            self.note_up(p)

    # --- редактор ---

    def open_editor(self, lesson: Lesson | None, back: str) -> None:
        if self.editor is not None:
            self.editor.close()
        self.release_keys()
        model = EditorModel.from_lesson(lesson) if lesson is not None else EditorModel()
        self.editor = EditorScreen(self, model, back)
        self.state = "editor"
        self.notify("Щелчок по сетке — нота  ·  клавиши — ввод по шагам  ·  Enter — запись  ·  F1 — справка")

    def new_melody(self) -> None:
        self.open_editor(None, "menu")

    def edit_lesson(self) -> None:
        if self.lessons:
            self.open_editor(self.lesson, "setup")

    def close_editor(self, back: str) -> None:
        self.editor = None
        self.state = back if back != "setup" or self.lessons else "menu"

    # --- мелодия из песни ---

    def open_library(self) -> None:
        self.release_keys()
        self.toast = None
        self.library = LibraryScreen(self)
        self.state = "library"

    def close_library(self) -> None:
        if self.library is not None:
            self.library.close()
        self.library = None
        self.state = "editor" if self.editor is not None else "menu"

    def open_fragment(self, path: Path) -> None:
        if self.state == "play":
            self.notify("Сначала выйдите из урока — Esc", error=True)
            return
        if self.state == "editor" and not self.editor.can_leave():
            return  # несохранённая мелодия: сначала предупреждаем, со второго раза — продолжаем
        if self.fragment is not None:
            self.fragment.close()
        if self.state == "library":
            pygame.key.stop_text_input()
        back = self.state if self.state in ("library", "editor", "menu", "setup") else "menu"
        self.toast = None
        self.fragment = FragmentScreen(self, path, back)
        self.state = "fragment"

    def close_fragment(self) -> None:
        back = self.fragment.back if self.fragment else "menu"
        self.fragment = None
        self.state = back
        if back == "library" and self.library is not None:
            self.library.resume()
        elif back == "editor" and self.editor is None:
            self.state = "menu"

    def finish_import(self, model: EditorModel, reference: Reference, result) -> None:
        """Мелодия распознана — открываем её в редакторе вместе с песней."""
        back = self.fragment.back if self.fragment else "menu"
        if self.library is not None:
            self.library.close()
            self.library = None
        self.fragment = None
        if self.editor is not None:
            back = self.editor.back
            self.editor.close()
        elif back not in ("menu", "setup"):
            back = "menu"
        self.release_keys()
        self.editor = EditorScreen(self, model, back, reference)
        self.state = "editor"
        hint = " Кажется вдвое быстрее? «÷2»." if result.bpm >= 120 else ""
        self.notify(f"Распознано {plural(len(model.notes), ('нота', 'ноты', 'нот'))}, темп {result.bpm:g} уд/мин. "
                    f"Послушайте с «Оригиналом» и поправьте ноты.{hint}")

    def open_folder(self) -> None:
        folder = paths.USER_LESSONS
        folder.mkdir(parents=True, exist_ok=True)
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
        elif os.name == "nt":
            os.startfile(folder)
        else:
            subprocess.Popen(["xdg-open", str(folder)])

    # --- переходы ---

    def quit(self) -> None:
        if self.editor is not None and not self.editor.can_leave():
            return
        self.running = False

    def start_free(self) -> None:
        self.session = None
        self.trails.clear()
        self.paused = False
        self.state = "play"

    def open_setup(self) -> None:
        self.state = "setup"

    def select_row(self, row: int) -> None:
        self.row = row

    def pick(self, row: int, d: int) -> None:
        self.row = row
        if not self.lessons:
            return
        if row == 0:
            self.opt["lesson"] = (self.opt["lesson"] + d) % len(self.lessons)
            if len(self.lesson.hands) < 2:
                self.opt["hand"] = 0
        elif row == 1:
            self.opt["mode"] = (self.opt["mode"] + d) % len(MODE_KEYS)
        elif row == 2:
            self.opt["tempo"] = max(0, min(len(TEMPOS) - 1, self.opt["tempo"] + d))
        elif row == 3 and len(self.lesson.hands) > 1:
            self.opt["hand"] = (self.opt["hand"] + d) % len(HAND_KEYS)

    def start_session(self) -> None:
        if not self.lessons:
            return
        self.session = Session(self.lesson, MODE_KEYS[self.opt["mode"]],
                               TEMPOS[self.opt["tempo"]], HAND_KEYS[self.opt["hand"]])
        self.popups.clear()
        self.error_at.clear()
        self.paused = False
        self.state = "play"

    def release_auto(self) -> None:
        if self.session is not None:
            for p in self.session.sounding():
                if p not in self.held:
                    self.synth.note_off(p)

    def stop_play(self) -> None:
        self.midi.clear()
        self.trails.clear()
        if self.session is None:
            self.state = "menu"
            return
        self.release_auto()
        self.session = None
        self.paused = False
        self.state = "setup"

    def finish(self) -> None:
        s = self.session
        self.session = None
        self.midi.clear()
        if s.mode == "demo":
            self.state = "setup"
            return
        self.result = s.result()
        self.result_title = s.lesson.title
        try:
            self.records[s.lesson.title] = records.add(s.lesson.title, self.result)
        except OSError:  # нет прав на запись — показываем итог без сохранения
            self.records.setdefault(s.lesson.title, []).append(self.result)
        self.state = "results"

    # --- кадр ---

    def leds(self) -> dict:
        """Кадр ленты слоями: подсказки урока → отклик на нажатия → вспышки ошибок."""
        s = self.session if self.state == "play" else None
        out = {} if s is None else {p: (note_color(p), lvl) for p, lvl in s.hints().items()}
        if self.state == "editor":
            out = {p: (note_color(p), 1.0) for p in self.editor.lit()}
        if s is None or s.mode == "demo":
            for p, t in self.release_at.items():
                out.setdefault(p, (note_color(p), 1.0 - (self.now - t) / RELEASE_FADE))
            for p in self.held:
                out[p] = (note_color(p), 1.0)
        for p, t in self.error_at.items():  # ошибка: клавиша дважды мигает белым
            phase = int((self.now - t) / 0.1)
            if phase < 4:
                out[p] = (D.WHITE, 1.0 if phase % 2 == 0 else 0.0)
        return out

    def tints(self) -> dict:
        out = {}
        if self.state == "play" and self.session is not None:
            for p in self.session.sounding():
                out[p] = D.mix(note_color(p), D.KEY_WHITE, 0.4)
        if self.state == "editor":
            for p in self.editor.lit():
                out[p] = D.mix(note_color(p), D.KEY_WHITE, 0.4)
        for p in self.held:
            out[p] = (150, 155, 165) if p in self.error_at else note_color(p)
        return out

    def draw(self) -> None:
        self.zones = []
        self.screen.fill(D.BG)
        getattr(self, f"draw_{self.state}")()
        self.draw_toast()
        D.draw_strip(self.screen, self.leds())
        D.draw_keyboard(self.screen, self.tints())

    def draw_toast(self) -> None:
        if self.toast is None or self.now - self.toast[2] > TOAST_LIFE:
            return
        if self.state == "editor" and self.editor.help:
            return
        message, error, t0 = self.toast
        w, h = D.measure(message, 15, True)
        box = pygame.Rect(0, 0, w + 36, h + 18)
        box.midbottom = (D.W // 2, D.HIT_Y - 14)
        D.rect(self.screen, D.PANEL_HI, box, border_radius=box.height // 2)
        D.text(self.screen, message, 15, (255, 120, 120) if error else D.TEXT, bold=True, center=box.center)

    def button(self, rect: pygame.Rect, label: str, selected: bool, action, size: int = 20) -> None:
        s = self.screen
        D.rect(s, D.PANEL_HI if selected else D.PANEL, rect, border_radius=10)
        if selected:
            D.rect(s, D.TEXT, rect, 2, border_radius=10)
        D.text(s, label, size, D.TEXT if selected else D.MUTED, bold=selected, center=rect.center)
        self.zones.append((rect, action))

    def draw_editor(self) -> None:
        self.editor.draw()

    def draw_library(self) -> None:
        self.library.draw()

    def draw_fragment(self) -> None:
        self.fragment.draw()

    def draw_menu(self) -> None:
        s, cx = self.screen, D.W // 2
        D.text(s, "НотоСвет", 56, D.TEXT, bold=True, midtop=(cx, 22))
        D.text(s, "Программа для обучающего пианино. Пока пианино не собрано, "
                  "играйте на клавиатуре компьютера.", 16, D.MUTED, midtop=(cx, 92))
        for i, (label, action) in enumerate(self.menu_items()):
            r = pygame.Rect(0, 0, 340, 42)
            r.midtop = (cx, 126 + i * 50)
            self.button(r, label, i == self.menu_i, action)
        D.text(s, "Цвет — это нота", 14, D.MUTED, midtop=(cx, 390))
        chip = 78
        for i in range(12):
            r = pygame.Rect(cx - chip * 6 + i * chip + 8, 414, chip - 16, 10)
            D.rect(s, COLORS[i], r, border_radius=5)
            D.text(s, NOTE_NAMES[i], 14, D.MUTED if i in BLACK_PCS else D.TEXT, midtop=(r.centerx, 430))
        D.text(s, "Нижний ряд Z…M — малая октава (C3–B3)  ·  верхний ряд Q…I — первая октава и «до» второй "
                  "(C4–C5)  ·  Пробел — педаль", 14, D.DIM, midtop=(cx, 466))
        D.text(s, "Свои мелодии — «Создать мелодию» или перетащите .mid/.json в окно  ·  "
                  "F11 или Ctrl/Cmd+F — полный экран", 14, D.DIM, midtop=(cx, 488))
        D.text(s, self.midi.label, 13, D.DIM, topright=(D.W - 20, 16))

    def draw_setup(self) -> None:
        s = self.screen
        D.text(s, "Учить мелодию", 30, D.TEXT, bold=True, topleft=(60, 32))
        D.text(s, "Esc — назад", 14, D.MUTED, topright=(D.W - 60, 44))
        if not self.lessons:
            D.text(s, "В папке lessons нет уроков (*.json).", 18, D.MUTED, topleft=(60, 110))
            return
        lesson = self.lesson
        one_hand = len(lesson.hands) < 2
        mode = MODE_KEYS[self.opt["mode"]]
        tempo = TEMPOS[self.opt["tempo"]]
        rows = [("Мелодия", lesson.title), ("Режим", MODES[mode]), ("Темп", f"{round(tempo * 100)} %"),
                ("Рука", "одна партия" if one_hand else HANDS[HAND_KEYS[self.opt["hand"]]].capitalize())]
        for i, (label, value) in enumerate(rows):
            box = pygame.Rect(180, 100 + i * 62, 420, 50)
            sel = self.row == i
            disabled = i == 3 and one_hand
            D.text(s, label, 16, D.TEXT if sel else D.MUTED, midleft=(60, box.centery))
            D.rect(s, D.PANEL_HI if sel else D.PANEL, box, border_radius=10)
            if sel:
                D.rect(s, D.TEXT, box, 2, border_radius=10)
            D.text(s, value, 18, D.DIM if disabled else D.TEXT, bold=sel, center=box.center)
            if not disabled:
                left, right = pygame.Rect(box.left, box.top, 50, 50), pygame.Rect(box.right - 50, box.top, 50, 50)
                D.text(s, "‹", 28, D.MUTED, center=left.center)
                D.text(s, "›", 28, D.MUTED, center=right.center)
                self.zones += [(left, lambda i=i: self.pick(i, -1)), (right, lambda i=i: self.pick(i, 1))]
            self.zones.append((box, lambda i=i: self.select_row(i)))
        start = pygame.Rect(180, 356, 420, 54)
        sel = self.row == 4
        D.rect(s, D.TEXT if sel else D.PANEL, start, border_radius=10)
        D.text(s, "Старт", 20, D.DARK_TEXT if sel else D.TEXT, bold=True, center=start.center)
        self.zones.append((start, self.start_session))
        D.text(s, "↑↓ — строка   ←→ — значение   Enter — старт", 14, D.DIM, topleft=(180, 424))
        D.text(s, "Своя мелодия: «Создать мелодию» в меню или перетащите .mid/.json в окно.", 14, D.DIM,
               topleft=(60, 456))
        for x, label, action in ((60, "Изменить в редакторе", self.edit_lesson),
                                 (260, "Открыть папку своих мелодий", self.open_folder)):
            link = D.text(s, label, 14, D.MUTED, topleft=(x, 480))
            D.line(s, D.DIM, (link.left, link.bottom), (link.right, link.bottom))
            self.zones.append((link.inflate(8, 8), action))

        panel = pygame.Rect(660, 100, 560, 370)
        D.rect(s, D.PANEL, panel, border_radius=12)
        x, y = panel.left + 28, panel.top + 24
        D.text(s, lesson.title, 22, D.TEXT, bold=True, topleft=(x, y))
        meta = (f"{plural(len(lesson.notes), ('нота', 'ноты', 'нот'))} · "
                f"{plural(lesson.steps, ('шаг', 'шага', 'шагов'))} · {lesson.bpm:g} уд/мин · "
                f"{fmt_time(lesson.seconds(tempo))} при темпе {round(tempo * 100)} %")
        D.text(s, meta, 14, D.MUTED, topleft=(x, y + 34))
        parts = ("Партии правой и левой руки" if not one_hand
                 else "Партия правой руки" if lesson.hands == ["R"] else "Партия левой руки")
        D.text(s, parts, 14, D.MUTED, topleft=(x, y + 56))
        y += 100
        D.text(s, MODES[mode], 16, D.TEXT, bold=True, topleft=(x, y))
        for line in D.wrap(MODE_HELP[mode], 15, panel.width - 56):
            y += 24
            D.text(s, line, 15, D.TEXT, topleft=(x, y))
        y += 48
        D.text(s, "Последние попытки", 14, D.MUTED, topleft=(x, y))
        runs = self.records.get(lesson.title, [])[-4:][::-1]
        if not runs:
            D.text(s, "Ещё не играли — результаты появятся здесь.", 14, D.DIM, topleft=(x, y + 26))
        for i, r in enumerate(runs):
            D.text(s, run_line(r), 14, D.TEXT, topleft=(x, y + 26 + i * 24))
        for i, err in enumerate(self.lesson_errors[:2]):
            D.text(s, f"Не загружен {err}"[:80], 13, (255, 110, 110), bottomleft=(x, panel.bottom - 16 - i * 20))

    def draw_play(self) -> None:
        s, ses = self.screen, self.session
        D.draw_lanes(s)
        if ses is None:
            D.draw_trails(s, self.trails, self.now, FREE_SPEED)
        else:
            D.draw_notes(s, ses, lookahead(ses), self.now)
            for kind, p, t0 in self.popups:
                age = self.now - t0
                label, color = JUDGE[kind]
                D.text(s, label, 16, color, bold=True, alpha=round(255 * (1 - age / POPUP_LIFE)),
                       midbottom=(round(D.slot_x(p) + D.SLOT / 2), round(D.HIT_Y - 12 - age * 60)))
            mid = (D.W // 2, (D.TOP + D.HIT_Y) // 2)
            if ses.mode == "rhythm" and ses.song_time < 0:
                left = -ses.song_time / ses.spb
                if 0 < left <= 4:
                    D.text(s, str(math.ceil(left)), 110, D.TEXT, bold=True,
                           alpha=round(60 + 195 * (left % 1)), center=mid)
            if (ses.mode == "wait" and ses.presses == 0 and not self.paused
                    and ses.cur < len(ses.steps) and ses.song_time >= ses.steps[ses.cur].time):
                D.text(s, "Нажмите подсвеченную клавишу — ноты подождут", 18, D.TEXT, center=(mid[0], D.TOP + 70))
        self.draw_topbar()
        if self.paused:
            D.shade(s, (0, D.TOP, D.W, D.HIT_Y - D.TOP), (8, 9, 12, 200))
            D.text(s, "Пауза", 44, D.TEXT, bold=True, center=(D.W // 2, 250))
            D.text(s, "Enter — продолжить   ·   Esc — к выбору мелодии", 16, D.MUTED, center=(D.W // 2, 300))

    def draw_topbar(self) -> None:
        s, ses, cx = self.screen, self.session, D.W // 2
        D.rect(s, D.PANEL, (0, 0, D.W, D.TOP))
        if ses is None:
            D.text(s, "Свободная игра", 20, D.TEXT, bold=True, topleft=(24, 11))
            D.text(s, "каждая клавиша горит своим цветом", 14, D.MUTED, topleft=(24, 38))
            if self.last_note is not None:
                D.text(s, note_name(self.last_note), 30, note_color(self.last_note), bold=True, center=(cx, D.TOP // 2))
            D.text(s, "Esc — в меню   ·   Пробел — педаль", 13, D.MUTED, topright=(D.W - 24, 14))
            D.text(s, self.midi.label, 13, D.DIM, topright=(D.W - 24, 38))
            return
        D.text(s, ses.lesson.title, 20, D.TEXT, bold=True, topleft=(24, 11))
        D.text(s, f"{MODES[ses.mode]} · темп {round(ses.tempo * 100)} % · {HANDS[ses.hand]}",
               14, D.MUTED, topleft=(24, 38))
        if ses.mode == "wait":
            now_notes = ses.current_notes()
            if now_notes:
                self.draw_note_names(now_notes, cx, 24)
            acc = f"{round(100 * ses.correct / ses.presses)} %" if ses.presses else "—"
            D.text(s, f"Ошибки {ses.errors}    Точность {acc}", 18, D.TEXT, bold=True, topright=(D.W - 24, 10))
        elif ses.mode == "rhythm":
            if ses.combo > 1:
                D.text(s, f"Серия {ses.combo}", 22, D.TEXT, bold=True, center=(cx, 22))
            D.text(s, f"Отлично {ses.perfect} · Хорошо {ses.good} · Мимо {ses.miss}", 13, D.MUTED, center=(cx, 48))
            D.text(s, f"Очки {ses.score:,}".replace(",", " "), 18, D.TEXT, bold=True, topright=(D.W - 24, 10))
        else:
            D.text(s, "Слушайте и смотрите на подсветку", 16, D.MUTED, center=(cx, D.TOP // 2))
        metro = "вкл" if ses.metronome else "выкл"
        D.text(s, f"Enter — пауза  ·  Tab — метроном ({metro})  ·  Esc — выход", 13, D.MUTED,
               topright=(D.W - 24, 40))
        D.rect(s, (48, 52, 61), (0, D.TOP - 3, D.W, 3))
        D.rect(s, (200, 205, 212), (0, D.TOP - 3, round(D.W * ses.progress()), 3))

    def draw_note_names(self, pitches: list[int], cx: int, cy: int) -> None:
        """«Сейчас: ми E + соль T» — имя ноты её цветом и клавиша ПК."""
        parts = [("Сейчас:", D.MUTED, False)]
        for i, p in enumerate(pitches):
            if i:
                parts.append(("+", D.MUTED, False))
            parts += [(note_name(p), note_color(p), True), (NOTE_LABEL[p], D.MUTED, False)]
        widths = [D.measure(t, 20, bold)[0] for t, _, bold in parts]
        x = cx - (sum(widths) + 8 * (len(parts) - 1)) // 2
        for (t, color, bold), w in zip(parts, widths):
            D.text(self.screen, t, 20, color, bold=bold, midleft=(x, cy))
            x += w + 8

    def draw_results(self) -> None:
        s, r = self.screen, self.result
        rhythm = r["mode"] == "rhythm"
        D.text(s, "Мелодия пройдена", 30, D.TEXT, bold=True, topleft=(60, 32))
        D.text(s, f"{self.result_title} · {MODES[r['mode']]} · темп {r['tempo']} % · {HANDS[r['hand']]}",
               15, D.MUTED, topleft=(60, 74))
        if rhythm:
            cards = [("Оценка", r["grade"]), ("Очки", f"{r['score']:,}".replace(",", " ")),
                     ("Точность ритма", f"{r['accuracy']} %"), ("Лучшая серия", str(r["max_combo"]))]
        else:
            cards = [("Время", fmt_time(r["time"])), ("Ошибки", str(r["errors"])),
                     ("Точность", f"{r['accuracy']} %")]
        for i, (label, value) in enumerate(cards):
            card = pygame.Rect(60 + i * 272, 116, 252, 118)
            D.rect(s, D.PANEL, card, border_radius=12)
            D.text(s, label, 14, D.MUTED, topleft=(card.x + 22, card.y + 18))
            D.text(s, value, 46, D.TEXT, bold=True, topleft=(card.x + 20, card.y + 44))
        runs = self.records.get(self.result_title, [])
        same = [x for x in runs[:-1] if x.get("mode") == r["mode"]]
        notes = []
        if rhythm:
            notes.append(f"Отлично {r['perfect']} · Хорошо {r['good']} · Мимо {r['miss']} · "
                         f"лишних нажатий {r['errors'] - r['miss']}")
        if same:
            diff = r["accuracy"] - same[-1]["accuracy"]
            notes.append("Точность как в прошлый раз" if diff == 0 else
                         f"Точность {'выросла' if diff > 0 else 'упала'} на {abs(diff)} % по сравнению с прошлой попыткой")
        for i, line in enumerate(notes):
            D.text(s, line, 15, D.TEXT, topleft=(60, 252 + i * 24))
        y = 318
        D.text(s, "Последние попытки", 14, D.MUTED, topleft=(60, y))
        for i, x in enumerate(runs[-5:][::-1]):
            D.text(s, run_line(x), 14, D.TEXT if i == 0 else D.MUTED, topleft=(60, y + 26 + i * 22))
        again = pygame.Rect(D.W - 60 - 260, 318, 260, 50)
        back = pygame.Rect(D.W - 60 - 260, 380, 260, 50)
        self.button(again, "Ещё раз — Enter", True, self.start_session, 18)
        self.button(back, "К выбору — Esc", False, self.open_setup, 18)


def run() -> None:
    # Windows: окно в точках, а кадр в настоящих пикселях — как на Retina
    os.environ.setdefault("SDL_WINDOWS_DPI_AWARENESS", "permonitorv2")
    os.environ.setdefault("SDL_WINDOWS_DPI_SCALING", "1")
    pygame.mixer.pre_init(44100, -16, 2, 512)  # маленький буфер — короткая задержка звука
    pygame.init()
    display = Display("НотоСвет", make_icon(256), fullscreen=bool(settings.load().get("fullscreen")))
    pygame.key.stop_text_input()  # иначе macOS ловит зажатые буквы для ввода текста
    App(display.canvas, display).loop()
    pygame.quit()
