import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")


def test_desktop_main_window_schedules_solve_on_slider_change():
    from PySide6 import QtCore, QtWidgets

    from strange_attractor_visualiser.desktop.widgets import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    slider = window.parameter_sliders["$a$"]

    slider.set_value(12.34)
    window.density_toggle.setChecked(True)
    window.performance_toggle.setChecked(True)
    window.info_toggle.setChecked(True)
    window.reset_button.click()

    assert window.param_values["$a$"] == 10.0
    assert window.solve_timer.isActive()

    loop = QtCore.QEventLoop()
    poller = QtCore.QTimer()

    def stop_when_payload_arrives():
        if window.coordinator.last_payload is not None:
            loop.quit()

    poller.setInterval(20)
    poller.timeout.connect(stop_when_payload_arrives)
    poller.start()
    QtCore.QTimer.singleShot(2000, loop.quit)
    loop.exec()

    assert window.coordinator.last_payload is not None
    window.window.close()
    app.processEvents()
