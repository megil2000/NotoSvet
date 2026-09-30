"""НотоСвет — «Падающие ноты». Запуск: python main.py (или готовая программа NotoSvet).

python main.py --selftest — проверка без окна: уроки, звук, MIDI, импорт мелодии, прохождение урока.
"""
import sys
import traceback


def main() -> None:
    if "--selftest" in sys.argv:
        from app.selftest import selftest
        sys.exit(selftest())
    from app.game import run
    try:
        run()
    except Exception:
        # у готовой программы нет консоли — сохраняем причину падения в файл
        from app import paths
        paths.DATA_DIR.mkdir(parents=True, exist_ok=True)
        paths.ERROR_LOG.write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
