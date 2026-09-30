"""Связь по USB-MIDI с пианино НотоСвет или любой MIDI-клавиатурой (протокол из плана).

Необязательна: без mido/python-rtmidi или без подключённого устройства программа
работает с клавиатуры компьютера.

  пианино → ПК: Note On/Off (клавиши), Control Change 64 (педаль)
  ПК → пианино: Note On канал 16 — зажечь подсказку, velocity = яркость;
                Note Off канал 16 — погасить; CC 123 канал 16 — погасить все
"""
from __future__ import annotations

PIANO = "notosvet"
HINT_CHANNEL = 15  # канал 16: в mido каналы считаются с нуля


class MidiLink:
    def __init__(self) -> None:
        self.inp = self.out = None
        self.name: str | None = None
        self.sent: dict[int, int] = {}
        try:
            import mido
            names = mido.get_input_names()
        except Exception:  # нет mido или бэкенда rtmidi — работаем с клавиатуры ПК
            return
        self.mido = mido
        piano = [n for n in names if PIANO in n.lower()]
        name = (piano or names or [None])[0]
        if name is None:
            return
        try:
            self.inp = mido.open_input(name)
            self.name = name
            if piano:  # подсказки шлём только своему пианино, не чужому синтезатору
                outs = [n for n in mido.get_output_names() if PIANO in n.lower()]
                if outs:
                    self.out = mido.open_output(outs[0])
                    self.clear()
        except Exception:  # порт занят другой программой или пропал
            self.inp = self.out = None
            self.name = None

    @property
    def label(self) -> str:
        return f"MIDI: {self.name}" if self.name else "Клавиатура ПК"

    def poll(self) -> list[tuple[str, int | bool]]:
        if self.inp is None:
            return []
        out: list[tuple[str, int | bool]] = []
        for msg in self.inp.iter_pending():
            if msg.type == "note_on" and msg.velocity > 0:
                out.append(("on", msg.note))
            elif msg.type in ("note_on", "note_off"):
                out.append(("off", msg.note))
            elif msg.type == "control_change" and msg.control == 64:
                out.append(("pedal", msg.value >= 64))
        return out

    def send_hints(self, levels: dict[int, float]) -> None:
        """Подсказки на ленту пианино: шлём только то, что изменилось."""
        if self.out is None:
            return
        want = {p: max(1, min(127, round(lvl * 127))) for p, lvl in levels.items() if lvl > 0.02}
        for p in [p for p in self.sent if p not in want]:
            self.out.send(self.mido.Message("note_off", channel=HINT_CHANNEL, note=p))
            del self.sent[p]
        for p, v in want.items():
            if self.sent.get(p) != v:
                self.out.send(self.mido.Message("note_on", channel=HINT_CHANNEL, note=p, velocity=v))
                self.sent[p] = v

    def clear(self) -> None:
        if self.out is not None:
            self.out.send(self.mido.Message("control_change", channel=HINT_CHANNEL, control=123, value=0))
        self.sent.clear()

    def close(self) -> None:
        self.clear()
        for port in (self.inp, self.out):
            if port is not None:
                port.close()
