"""Мелодия из своей песни: библиотека → фрагмент на волне → распознавание → дорожки → редактор.

Долгие дела (поиск песен, чтение файла, распознавание) идут в отдельном потоке —
окно не зависает, а распознавание можно отменить.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pygame

from . import draw as D
from . import library, neural
from .audio import Audio, AudioError, load
from .editor import GRIDS
from .editor_view import WAVE
from .transcribe import Cancelled

ROW_H = 46
LIST_TOP, LIST_BOTTOM = 136, D.HIT_Y - 12
WAVE_RECT = pygame.Rect(60, 128, 1160, 170)
DEFAULT_LEN = 40.0      # сек: столько берём по умолчанию — примерно куплет или припев
MIN_LEN = 3.0
TIPS = {
    "neural": ["Нейросеть разделит песню на дорожки: вокал, гитару, пианино, бас, барабаны.",
               "Потом выберите дорожку, как в Songsterr, — остальные зазвучат вместе с вами.",
               "40 секунд песни — около полуминуты работы; можно выделить и всю песню."],
    "classic": ["Лучше всего распознаётся чёткая мелодия — голос или соло-инструмент.",
                "Выделите на волне кусок, где мелодию слышно яснее всего, например припев.",
                "Потом ноты можно поправить в редакторе и послушать вместе с оригиналом."],
}
ENGINE_LABEL = {"neural": "Нейросеть: дорожки инструментов", "classic": "Без нейросети: только мелодия"}


def fmt(sec: float) -> str:
    m, s = divmod(int(max(0.0, sec)), 60)
    return f"{m}:{s:02d}"


class LibraryScreen:
    """Список песен из «Музыки» и «Загрузок» с поиском, плюс «Открыть файл…»."""

    def __init__(self, app):
        self.app = app
        self.tracks: list[library.Track] | None = None
        self.query = ""
        self.sel = 0
        self.scroll = 0
        self.wheel = 0.0
        self.picked: list[str | None] = []
        self.picking = False
        self.cancel = threading.Event()
        threading.Thread(target=self._scan, daemon=True).start()
        self.resume()

    def _scan(self) -> None:
        self.tracks = library.scan(cancel=self.cancel)

    def _pick(self) -> None:
        self.picked.append(library.pick_file())

    def resume(self) -> None:
        pygame.key.start_text_input()  # буквы — в поиск, а не в пианино

    def close(self) -> None:
        self.cancel.set()
        pygame.key.stop_text_input()

    @property
    def shown(self) -> list[library.Track]:
        return library.search(self.tracks or [], self.query)

    def open_dialog(self) -> None:
        if not self.picking:
            self.picking = True
            threading.Thread(target=self._pick, daemon=True).start()

    def choose(self, i: int) -> None:
        shown = self.shown
        if 0 <= i < len(shown):
            self.app.open_fragment(shown[i].path)

    def _visible_rows(self) -> int:
        return (LIST_BOTTOM - LIST_TOP) // ROW_H

    def _follow(self) -> None:
        rows = self._visible_rows()
        self.scroll = min(max(self.scroll, self.sel - rows + 1), self.sel)
        self.scroll = max(0, min(self.scroll, max(0, len(self.shown) - rows)))

    def handle(self, e) -> bool:
        if e.type == pygame.TEXTINPUT:
            self.query = (self.query + e.text)[:60]
            self.sel = self.scroll = 0
            return True
        if e.type == pygame.KEYDOWN:
            n = len(self.shown)
            if e.key == pygame.K_ESCAPE:
                if self.query:
                    self.query = ""
                else:
                    self.app.close_library()
            elif e.key == pygame.K_BACKSPACE:
                self.query = self.query[:-1]
                self.sel = self.scroll = 0
            elif e.key in (pygame.K_UP, pygame.K_DOWN) and n:
                self.sel = max(0, min(n - 1, self.sel + (1 if e.key == pygame.K_DOWN else -1)))
                self._follow()
            elif e.key in (pygame.K_PAGEUP, pygame.K_PAGEDOWN) and n:
                step = self._visible_rows() * (1 if e.key == pygame.K_PAGEDOWN else -1)
                self.sel = max(0, min(n - 1, self.sel + step))
                self._follow()
            elif e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.choose(self.sel)
            elif e.scancode == pygame.KSCAN_O and e.mod & (pygame.KMOD_CTRL | pygame.KMOD_META):
                self.open_dialog()
            return True  # остальные клавиши печатают в поиск, пианино молчит
        if e.type == pygame.KEYUP:
            return True
        if e.type == pygame.MOUSEWHEEL:
            self.wheel += e.precise_y
            steps = int(self.wheel)
            self.wheel -= steps
            self.scroll = max(0, min(self.scroll - steps, max(0, len(self.shown) - self._visible_rows())))
            return True
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and LIST_TOP <= e.pos[1] < LIST_BOTTOM \
                and 60 <= e.pos[0] < D.W - 60:
            i = self.scroll + (e.pos[1] - LIST_TOP) // ROW_H
            if i < len(self.shown):
                self.sel = i
                self.choose(i)
            return True
        return False

    def update(self, dt: float) -> None:
        if self.picked:
            path = self.picked.pop()
            self.picking = False
            if path:
                self.app.open_fragment(Path(path))

    def draw(self) -> None:
        s = self.app.screen
        D.rect(s, D.PANEL, (16, 12, 88, 36), border_radius=8)
        D.text(s, "‹ Назад", 15, D.MUTED, center=(60, 30))
        self.app.zones.append((pygame.Rect(16, 12, 88, 36), self.app.close_library))
        D.text(s, "Мелодия из песни", 24, D.TEXT, bold=True, midleft=(124, 30))
        D.text(s, "Нейросеть разделит песню на дорожки и найдёт ноты — выберите, какую играть"
               if neural.available() else "Программа найдёт мелодию, темп и сетку — ноты откроются в редакторе",
               14, D.MUTED, midleft=(360, 31))

        box = pygame.Rect(60, 70, 760, 44)
        D.rect(s, D.PANEL_HI, box, border_radius=10)
        D.rect(s, D.LINE, box, 1, border_radius=10)
        if self.query:
            r = D.text(s, self.query, 17, D.TEXT, midleft=(box.left + 16, box.centery))
            caret_x = r.right + 2
        else:
            D.text(s, "Поиск: название, исполнитель, альбом", 16, D.DIM, midleft=(box.left + 16, box.centery))
            caret_x = box.left + 15
        if int(self.app.now * 2) % 2 == 0:
            D.line(s, D.TEXT, (caret_x, box.top + 12), (caret_x, box.bottom - 12), 2)
        pick = pygame.Rect(836, 70, 190, 44)
        D.rect(s, D.TEXT, pick, border_radius=10)
        D.text(s, "Ожидаю выбора…" if self.picking else "Открыть файл…", 16, D.DARK_TEXT, bold=True, center=pick.center)
        self.app.zones.append((pick, self.open_dialog))
        D.text(s, "или перетащите", 13, D.MUTED, midleft=(1040, 84))
        D.text(s, "песню в окно", 13, D.MUTED, midleft=(1040, 101))

        if self.tracks is None:
            D.text(s, "Ищу песни в папках «Музыка» и «Загрузки»…", 17, D.MUTED, center=(D.W // 2, 300))
            return
        shown = self.shown
        if not shown:
            msg = ("Ничего не нашлось — измените запрос." if self.tracks else
                   "В «Музыке» и «Загрузках» песен не нашлось. Нажмите «Открыть файл…» или перетащите песню в окно.")
            D.text(s, msg, 17, D.MUTED, center=(D.W // 2, 300))
            return
        D.text(s, D.plural(len(shown), ("песня", "песни", "песен")), 13, D.DIM, bottomright=(D.W - 60, LIST_TOP - 6))
        rows = self._visible_rows()
        s.set_clip(pygame.Rect(0, LIST_TOP, D.W, LIST_BOTTOM - LIST_TOP))
        for k, t in enumerate(shown[self.scroll:self.scroll + rows]):
            i = self.scroll + k
            row = pygame.Rect(60, LIST_TOP + k * ROW_H, D.W - 120, ROW_H - 4)
            if i == self.sel:
                D.rect(s, D.PANEL_HI, row, border_radius=8)
                D.rect(s, D.TEXT, row, 1, border_radius=8)
            D.text(s, t.name[:70], 16, D.TEXT, bold=i == self.sel, topleft=(row.left + 16, row.top + 4))
            D.text(s, t.where[:90], 13, D.MUTED, topleft=(row.left + 16, row.top + 24))
            D.text(s, t.kind, 12, D.DIM, bold=True, midright=(row.right - 16, row.centery))
        s.set_clip(None)
        if len(shown) > rows:  # полоса прокрутки
            track_h = LIST_BOTTOM - LIST_TOP
            h = max(20, track_h * rows // len(shown))
            y = LIST_TOP + (track_h - h) * self.scroll // max(1, len(shown) - rows)
            D.rect(s, D.LINE, (D.W - 48, y, 4, h), border_radius=2)


class FragmentScreen:
    """Волна песни: выделить кусок, прослушать и распознать."""

    def __init__(self, app, path: Path, back: str):
        self.app, self.path, self.back = app, Path(path), back
        self.audio: Audio | None = None
        self.error = ""
        self.phase = "loading"          # loading / ready / analyzing
        self.sel = [0.0, 0.0]
        self.anchor: float | None = None
        self.grid_i = 1
        self.engine = "neural" if neural.available() else "classic"
        self.stage, self.frac = "Открываю песню", 0.0
        self.cancel = threading.Event()
        self.result = None
        self.channel = None
        self.preview: tuple[float, float] | None = None   # (секунда начала, время запуска)
        self.sound = None
        self.peaks = None
        threading.Thread(target=self._load, daemon=True).start()

    # --- фон ---

    def _load(self) -> None:
        try:
            self.audio = load(self.path)
        except AudioError as e:
            self.error, self.phase = str(e), "error"
            return
        except Exception as e:  # noqa: BLE001 — поток не должен умирать молча, иначе экран «зависнет»
            self.error, self.phase = f"не удалось открыть ({type(e).__name__}: {e})", "error"
            return
        a = self.audio
        loud = np.nonzero(a.env > 0.25)[0]
        start = max(0.0, loud[0] / 100 - 0.3) if len(loud) else 0.0
        self.sel = [0.0, a.duration] if a.duration <= 60 else [start, min(a.duration, start + DEFAULT_LEN)]
        self.phase = "ready"

    def _analyze(self, start: float, end: float, grid: float, engine: str) -> None:
        def progress(stage, frac):
            self.stage, self.frac = stage, frac
        try:
            self.result = (start, neural.recognize(self.audio, start, end, grid, engine, progress, self.cancel))
        except Cancelled:
            self.phase = "ready"
        except ValueError as e:
            self.error, self.phase = str(e), "error"
        except OSError as e:  # нет интернета, когда нейросеть скачивается в первый раз
            self.error, self.phase = f"не удалось скачать нейросеть ({e}) — проверьте интернет", "error"
        except Exception as e:  # noqa: BLE001 — поток не должен умирать молча, иначе экран «зависнет»
            self.error, self.phase = f"не удалось распознать ({type(e).__name__}: {e})", "error"

    # --- действия ---

    @property
    def editable(self) -> bool:
        return self.audio is not None and self.phase in ("ready", "error")

    def recognize(self) -> None:
        if not self.editable:
            return
        self.stop_preview()
        self.cancel.clear()
        self.phase, self.frac = "analyzing", 0.0
        self.stage = "Загружаю нейросети" if self.engine == "neural" else "Спектр"
        a, b = self.sel
        threading.Thread(target=self._analyze, args=(a, b, GRIDS[self.grid_i][0], self.engine), daemon=True).start()

    def toggle_engine(self) -> None:
        if not neural.available():
            self.app.notify("Нейросеть не установлена: pip install -r requirements-ai.txt (подробнее — в README)",
                            error=True)
            return
        if self.editable:
            self.engine = "classic" if self.engine == "neural" else "neural"

    def toggle_preview(self) -> None:
        if self.preview is not None:
            self.stop_preview()
            return
        if not self.editable or pygame.mixer.get_init() is None:
            return
        a, b = self.sel
        self.sound = pygame.sndarray.make_sound(self.audio.segment(a, b))
        self.channel = self.sound.play()
        self.preview = (a, self.app.now)

    def stop_preview(self) -> None:
        if self.channel is not None:
            self.channel.stop()
        self.channel = self.preview = None

    def back_out(self) -> None:
        if self.phase == "analyzing":
            self.cancel.set()
            return
        self.close()
        self.app.close_fragment()

    def close(self) -> None:
        self.cancel.set()
        self.stop_preview()

    def _time_at(self, x: float) -> float:
        r = WAVE_RECT
        return max(0.0, min(self.audio.duration, (x - r.left) / r.width * self.audio.duration))

    def handle(self, e) -> bool:
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE:
                self.back_out()
            elif e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.recognize()
            elif e.key == pygame.K_SPACE:
                self.toggle_preview()
            else:
                return False
            return True
        if e.type == pygame.KEYUP and e.key == pygame.K_SPACE:
            return True
        if not self.editable:
            return False
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and WAVE_RECT.inflate(0, 40).collidepoint(e.pos):
            self.stop_preview()
            self.anchor = self._time_at(e.pos[0])
            return True
        if e.type == pygame.MOUSEMOTION and self.anchor is not None:
            t = self._time_at(e.pos[0])
            self.sel = [min(self.anchor, t), max(self.anchor, t)]
            return True
        if e.type == pygame.MOUSEBUTTONUP and e.button == 1 and self.anchor is not None:
            t = self._time_at(e.pos[0])
            a, b = min(self.anchor, t), max(self.anchor, t)
            if b - a < MIN_LEN:  # просто щелчок — выделение того же размера начинается здесь
                length = max(MIN_LEN, self.sel[1] - self.sel[0])
                a = min(self.anchor, max(0.0, self.audio.duration - length))
                b = min(self.audio.duration, a + length)
            self.sel, self.anchor = [a, b], None
            return True
        return False

    def update(self, dt: float) -> None:
        if self.phase in ("loading", "analyzing"):
            # Python делит время между потоками грубо: пусть главный цикл чуть отдыхает,
            # тогда распознавание идёт на полной скорости (доли секунды вместо минут)
            pygame.time.wait(10)
        if self.preview is not None and (self.channel is None or not self.channel.get_busy()):
            self.preview = self.channel = None
        if self.result is not None:
            start, rec = self.result
            self.result = None
            self.phase = "ready"
            if not any(t.playable for t in rec.tracks):
                self.error, self.phase = (("нейросеть не нашла нот — выделите кусок, где инструменты слышно яснее"
                                           if rec.neural else "мелодию не удалось найти — выделите кусок, где лучше "
                                           "слышен голос или ведущий инструмент"), "error")
                return
            if rec.neural:  # дорожки — выбрать, какую играть; фрагмент остаётся, чтобы вернуться
                self.app.open_tracks(rec, self.audio, start)
            else:
                self.close()
                self.app.finish_import(rec, self.audio, start, rec.best())

    # --- кадр ---

    def draw(self) -> None:
        s = self.app.screen
        D.rect(s, D.PANEL, (16, 12, 88, 36), border_radius=8)
        D.text(s, "‹ Назад", 15, D.MUTED, center=(60, 30))
        self.app.zones.append((pygame.Rect(16, 12, 88, 36), self.back_out))
        title = self.audio.title if self.audio else self.path.stem
        D.text(s, title[:60], 24, D.TEXT, bold=True, midleft=(124, 30))
        if self.audio:
            D.text(s, f"{fmt(self.audio.duration)} · {self.path.suffix.lstrip('.').upper()}", 14, D.MUTED,
                   midleft=(124, 58))

        if self.phase == "loading":
            D.text(s, "Открываю песню…", 18, D.MUTED, center=(D.W // 2, 260))
            return
        if self.phase == "error" and self.audio is None:
            D.text(s, f"Не открылась: {self.error}", 17, (255, 120, 120), center=(D.W // 2, 250))
            D.text(s, "Esc — назад", 15, D.MUTED, center=(D.W // 2, 290))
            return
        self.draw_wave(s)
        a, b = self.sel
        y = WAVE_RECT.bottom + 34
        D.text(s, f"Фрагмент {fmt(a)} – {fmt(b)} · {round(b - a)} с", 18, D.TEXT, bold=True, topleft=(60, y))
        D.text(s, "Потяните по волне, чтобы выделить другой кусок", 14, D.MUTED, topleft=(60, y + 28))

        if self.phase == "analyzing":
            bar = pygame.Rect(60, y + 72, 700, 14)
            D.rect(s, D.PANEL_HI, bar, border_radius=7)
            D.rect(s, WAVE, (bar.x, bar.y, round(bar.width * self.frac), bar.height), border_radius=7)
            D.text(s, f"{self.stage}… {round(self.frac * 100)} %   ·   Esc — отменить", 15, D.TEXT,
                   topleft=(60, bar.bottom + 12))
            return
        if self.phase == "error":
            for i, line in enumerate(D.wrap(self.error, 15, 720)[:3]):
                D.text(s, line, 15, (255, 120, 120), topleft=(60, y + 60 + i * 22))
        else:
            for i, tip in enumerate(TIPS[self.engine]):
                D.text(s, tip, 14, D.MUTED, topleft=(60, y + 64 + i * 22))

        x = 820
        listen = pygame.Rect(x, y, 180, 44)
        D.rect(s, D.PANEL_HI, listen, border_radius=10)
        D.text(s, "Стоп — Пробел" if self.preview else "Слушать — Пробел", 15, D.TEXT, bold=True, center=listen.center)
        self.app.zones.append((listen, self.toggle_preview))
        grid = pygame.Rect(x + 190, y, 210, 44)
        D.rect(s, D.PANEL, grid, border_radius=10)
        D.text(s, f"Сетка нот: {GRIDS[self.grid_i][1]}", 15, D.TEXT, center=grid.center)
        self.app.zones.append((grid, lambda: setattr(self, "grid_i", 1 if self.grid_i == 2 else 2)))
        go = pygame.Rect(x, y + 56, 400, 52)
        D.rect(s, D.TEXT, go, border_radius=10)
        D.text(s, "Разделить на дорожки — Enter" if self.engine == "neural" else "Распознать мелодию — Enter",
               18, D.DARK_TEXT, bold=True, center=go.center)
        self.app.zones.append((go, self.recognize))
        engine = pygame.Rect(x, y + 118, 400, 36)
        D.rect(s, D.PANEL, engine, border_radius=10)
        label = ENGINE_LABEL[self.engine] if neural.available() else "Нейросеть не установлена — только мелодия"
        D.text(s, label, 14, D.TEXT if neural.available() else D.DIM, center=engine.center)
        self.app.zones.append((engine, self.toggle_engine))

    def draw_wave(self, s) -> None:
        r, a = WAVE_RECT, self.audio
        D.rect(s, (18, 20, 25), r.inflate(0, 20), border_radius=10)
        box = s.rect(r)  # волна — по пикселю экрана: на Retina вдвое подробнее
        if self.peaks is None or len(self.peaks) != box.width:
            self.peaks = a.peaks(box.width)
        x0 = r.left + round(self.sel[0] / a.duration * r.width)
        x1 = r.left + round(self.sel[1] / a.duration * r.width)
        D.rect(s, (30, 40, 58), (x0, r.top - 10, max(1, x1 - x0), r.height + 20))
        px0, px1, mid = s.px(x0), s.px(x1), box.centery
        for i, v in enumerate(self.peaks):
            x = box.left + i
            h = max(1, round(v * box.height / 2))
            pygame.draw.line(s.surface, WAVE if px0 <= x <= px1 else (66, 72, 84), (x, mid - h), (x, mid + h))
        for x in (x0, x1):
            D.line(s, D.TEXT, (x, r.top - 10), (x, r.bottom + 10), 2)
        step = next(v for v in (5, 10, 15, 30, 60, 120) if a.duration / v <= 12)
        for k in range(0, int(a.duration) + 1, step):
            x = r.left + round(k / a.duration * r.width)
            D.line(s, D.LINE, (x, r.bottom + 10), (x, r.bottom + 16))
            D.text(s, fmt(k), 12, D.DIM, midtop=(x, r.bottom + 18))
        if self.preview is not None:
            start, t0 = self.preview
            pos = start + (self.app.now - t0)
            x = r.left + round(pos / a.duration * r.width)
            D.line(s, (255, 90, 90), (x, r.top - 10), (x, r.bottom + 10), 2)
