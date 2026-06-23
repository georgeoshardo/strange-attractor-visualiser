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


def test_desktop_main_window_uses_preview_solves_during_slider_drag():
    from PySide6 import QtWidgets

    from strange_attractor_visualiser.desktop.widgets import (
        PREVIEW_DEBOUNCE_MS,
        PREVIEW_LINE_INTERPOLATION,
        SOLVE_MODE_FULL,
        SOLVE_MODE_PREVIEW,
        MainWindow,
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.solve_timer.stop()

    window.density_toggle.setChecked(True)
    window.solve_timer.stop()
    window.slider_drag_started()
    window.parameter_changed("$a$", 12.34)

    assert window.queued_solve_mode == SOLVE_MODE_PREVIEW
    assert window.solve_timer.interval() == PREVIEW_DEBOUNCE_MS
    assert window.current_settings(preview=True).use_density is True
    assert (
        window.current_settings(preview=True).line_interpolation
        == PREVIEW_LINE_INTERPOLATION
    )

    window.slider_drag_finished()

    assert window.queued_solve_mode == SOLVE_MODE_FULL
    window.window.close()
    app.processEvents()


def test_projection_view_accepts_density_coloured_lines():
    import numpy as np
    from PySide6 import QtWidgets

    from strange_attractor_visualiser.desktop.render import ProjectionView

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    projection = ProjectionView("x-y")
    x = np.linspace(0, 1, 8)
    y = np.sin(x)
    colors = np.ones((8, 4), dtype=np.float32)
    colors[:, 0] = np.linspace(0, 1, 8)
    colors[:, 1] = np.linspace(1, 0, 8)

    projection.set_data(
        x,
        y,
        show_points=False,
        show_lines=True,
        line_colors=colors,
    )

    assert projection.colored_line_item.picture is not None
    projection.widget.close()
    app.processEvents()


def test_desktop_main_window_exposes_solver_controls():
    from PySide6 import QtWidgets

    from strange_attractor_visualiser.core.solver import SOLVER_LSODA, SOLVER_RK4
    from strange_attractor_visualiser.desktop.widgets import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.solve_timer.stop()

    assert window.integrator_combo.currentText() == SOLVER_LSODA
    assert window.current_solver_settings().method == SOLVER_LSODA
    assert window.current_solver_settings().lsoda_rtol is not None

    window.integrator_combo.setCurrentText(SOLVER_RK4)

    assert window.current_solver_settings().method == SOLVER_RK4
    assert not window.lsoda_tolerance_combo.isEnabled()
    assert window.solve_timer.isActive()
    window.window.close()
    app.processEvents()
