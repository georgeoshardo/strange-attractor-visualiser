import sys


def main() -> int:
    from PySide6 import QtWidgets

    from .widgets import create_main_window

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    main_window = create_main_window()
    main_window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
