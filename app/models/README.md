# Веса нейросетей

`basic_pitch.npz` — веса Basic Pitch (Spotify, ICASSP 2022), перенесённые из `nmp.onnx`
пакета [basic-pitch](https://github.com/spotify/basic-pitch) 0.4.0 скриптом
`tools/convert_basic_pitch.py`. Лицензия Apache 2.0, © 2022 Spotify AB.
Архитектура для PyTorch — `app/basic_pitch.py`.

Demucs (htdemucs_6s, [Meta](https://github.com/facebookresearch/demucs), лицензия MIT)
сюда не кладётся: 53 МБ скачиваются при первом распознавании в `.model-cache/checkpoints/`
(в собранной программе — в папку пользователя) и проверяются по SHA-256.
