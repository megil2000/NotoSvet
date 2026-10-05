"""Веса Basic Pitch для PyTorch: nmp.onnx из пакета basic-pitch → app/models/basic_pitch.npz.

Нужно один раз, готовый файл уже лежит в app/models. Повторить:

    pip download basic-pitch==0.4.0 --no-deps -d /tmp/bp && unzip /tmp/bp/*.whl -d /tmp/bp
    pip install onnx
    python tools/convert_basic_pitch.py /tmp/bp/basic_pitch/saved_models/icassp_2022/nmp.onnx

В ONNX-графе пакетная нормализация уже вписана в свёртки, поэтому берутся только веса
шести свёрток, ядра CQT, фильтр понижения частоты и множители. Порядок свёрток в графе:
onset1, contour1, contour2, note1, note2, onset2 — так их и называет app/basic_pitch.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "app" / "models" / "basic_pitch.npz"
CONVS = ("onset1", "contour1", "contour2", "note1", "note2", "onset2")


def convert(onnx_path: Path) -> dict[str, np.ndarray]:
    import onnx
    from onnx import numpy_helper

    graph = onnx.load(str(onnx_path)).graph
    consts = {x.name: numpy_helper.to_array(x) for x in graph.initializer}
    nodes = [n for n in graph.node if n.op_type != "Constant"]
    convs = [n for n in nodes if n.op_type == "Conv"]

    cqt = sorted({n.input[1] for n in convs if consts[n.input[1]].shape == (36, 1, 1, 256)})
    lowpass = {n.input[1] for n in convs if consts[n.input[1]].shape == (1, 1, 1, 256)}
    assert len(cqt) == 2 and len(lowpass) == 1, "граф не похож на Basic Pitch icassp_2022"
    out = {"cqt_a": consts[cqt[0]][:, 0, 0, :], "cqt_b": consts[cqt[1]][:, 0, 0, :],
           "lowpass": consts[lowpass.pop()][0, 0, 0, :]}

    scalars = []  # √длины ядер CQT, затем пакетная нормализация после CQT: ×scale + shift
    for n in nodes:
        if n.op_type in ("Mul", "Add"):
            for name in n.input:
                if name in consts and consts[name].shape in ((309, 1, 1), (1,)):
                    scalars.append((n.op_type, consts[name]))
    (_, sqrt_len), (_, scale), (_, shift) = scalars
    out["sqrt_len"] = sqrt_len[:, 0, 0]
    out["bn"] = np.array([scale[0], shift[0]], np.float32)

    for name, n in zip(CONVS, convs[-len(CONVS):]):
        out[f"{name}.weight"] = consts[n.input[1]]
        out[f"{name}.bias"] = consts[n.input[2]]
    return {k: np.ascontiguousarray(v, np.float32) for k, v in out.items()}


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    weights = convert(Path(sys.argv[1]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **weights)
    print(f"{OUT.relative_to(ROOT)}: {sum(v.size for v in weights.values())} чисел")
    return 0


if __name__ == "__main__":
    sys.exit(main())
