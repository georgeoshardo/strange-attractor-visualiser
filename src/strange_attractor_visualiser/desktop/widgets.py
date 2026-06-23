from collections import OrderedDict
import random
import time

from ..attractors.registry import ATTRACTORS
from ..core.display import (
    DEFAULT_DISPLAY_MODE,
    DEFAULT_USE_DENSITY,
    DISPLAY_MODE_LINES,
    DISPLAY_MODE_LINES_POINTS,
    DISPLAY_MODE_POINTS,
)
from ..core.solver import (
    AdaptiveHorizonSettings,
    SOLVER_LSODA,
    SOLVER_RK4,
    SolverSettings,
    get_default_params,
    solve_attractor,
)
from .controller import ResultCoordinator
from .controls import FloatSliderSpec
from .equations import format_equation_text
from .render import AttractorView3D, ProjectionView
from .render_data import (
    DisplaySettings,
    build_render_payload,
    preview_display_settings,
)
from .state import parameter_cache_key

POINT_BUDGETS = {
    "Fast (10000)": 10_000,
    "Balanced (30000)": 30_000,
    "Full source": None,
}
LSODA_TOLERANCES = {
    "Strict (1e-8 / 1e-10)": (1e-8, 1e-10),
    "Balanced (1e-6 / 1e-8)": (1e-6, 1e-8),
    "Fast (1e-5 / 1e-7)": (1e-5, 1e-7),
    "Very fast (1e-4 / 1e-6)": (1e-4, 1e-6),
}
DEFAULT_LSODA_TOLERANCE = "Balanced (1e-6 / 1e-8)"
CACHE_MAX_ENTRIES = 128
FULL_DEBOUNCE_MS = 33
PREVIEW_DEBOUNCE_MS = 0
PREVIEW_POINT_BUDGET = 2_500
PREVIEW_SOLVE_STEPS = 2_500
PREVIEW_LINE_INTERPOLATION = 4
SOLVE_MODE_FULL = "full"
SOLVE_MODE_PREVIEW = "preview"


class ParameterSlider:
    def __init__(self, param):
        from PySide6 import QtCore, QtWidgets

        self.param = param
        self.spec = FloatSliderSpec(param.min_val, param.max_val, param.step)
        self.widget = QtWidgets.QWidget()

        layout = QtWidgets.QVBoxLayout(self.widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setDecimals(self.spec.decimal_places)
        self.spin.setRange(param.min_val, param.max_val)
        self.spin.setSingleStep(param.step)
        self.spin.setValue(param.default)
        self.spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Vertical)
        self.slider.setRange(self.spec.minimum_tick, self.spec.maximum_tick)
        self.slider.setValue(self.spec.value_to_tick(param.default))
        self.slider.setMinimumHeight(160)

        self.min_label = QtWidgets.QLabel(f"{param.min_val:g}")
        self.name_label = QtWidgets.QLabel(param.name.strip("$"))
        self.min_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.name_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)

        layout.addWidget(self.spin)
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.min_label)
        layout.addWidget(self.name_label)

        self._callbacks = []
        self._drag_started_callbacks = []
        self._drag_finished_callbacks = []
        self.slider.valueChanged.connect(self._slider_changed)
        self.slider.sliderPressed.connect(self._emit_drag_started)
        self.slider.sliderReleased.connect(self._emit_drag_finished)
        self.spin.valueChanged.connect(self._spin_changed)

    def connect(self, callback) -> None:
        self._callbacks.append(callback)

    def connect_drag_started(self, callback) -> None:
        self._drag_started_callbacks.append(callback)

    def connect_drag_finished(self, callback) -> None:
        self._drag_finished_callbacks.append(callback)

    def value(self) -> float:
        return float(self.spin.value())

    def set_value(self, value: float, emit: bool = True) -> None:
        from PySide6 import QtCore

        value = self.spec.tick_to_value(self.spec.value_to_tick(value))
        blocker_spin = QtCore.QSignalBlocker(self.spin)
        blocker_slider = QtCore.QSignalBlocker(self.slider)
        self.spin.setValue(value)
        self.slider.setValue(self.spec.value_to_tick(value))
        del blocker_spin
        del blocker_slider
        if emit:
            self._emit(value)

    def _slider_changed(self, tick: int) -> None:
        from PySide6 import QtCore

        value = self.spec.tick_to_value(tick)
        blocker = QtCore.QSignalBlocker(self.spin)
        self.spin.setValue(value)
        del blocker
        self._emit(value)

    def _spin_changed(self, value: float) -> None:
        from PySide6 import QtCore

        tick = self.spec.value_to_tick(value)
        rounded = self.spec.tick_to_value(tick)
        blocker = QtCore.QSignalBlocker(self.slider)
        self.slider.setValue(tick)
        del blocker
        self._emit(rounded)

    def _emit(self, value: float) -> None:
        for callback in self._callbacks:
            callback(self.param.name, value)

    def _emit_drag_started(self) -> None:
        for callback in self._drag_started_callbacks:
            callback()

    def _emit_drag_finished(self) -> None:
        for callback in self._drag_finished_callbacks:
            callback()


class SolveSignals:
    def __init__(self):
        from PySide6 import QtCore

        class _Signals(QtCore.QObject):
            finished = QtCore.Signal(int, object, object, dict, object, bool)
            failed = QtCore.Signal(int, object)

        self.object = _Signals()


class SolveWorker:
    def __init__(
        self,
        generation: int,
        selected_name: str,
        param_values: dict[str, float],
        settings: DisplaySettings,
        cached_solution,
        solve_steps: int | None,
        solver_settings: SolverSettings,
        sampling_settings: AdaptiveHorizonSettings,
        preview: bool,
    ):
        from PySide6 import QtCore

        class _Worker(QtCore.QRunnable):
            def __init__(inner_self):
                super().__init__()
                inner_self.signals = SolveSignals().object

            def run(inner_self):
                try:
                    config = ATTRACTORS[selected_name]
                    solve_start = time.perf_counter()
                    solution = cached_solution
                    cache_key = None
                    if not preview:
                        cache_key = parameter_cache_key(
                            selected_name,
                            config,
                            param_values,
                            solver_settings,
                            sampling_settings,
                        )
                    if solution is None:
                        solution = solve_attractor(
                            config,
                            param_values,
                            n_steps=solve_steps,
                            solver_settings=solver_settings,
                            adaptive_settings=sampling_settings,
                        )
                    solve_ms = (time.perf_counter() - solve_start) * 1000

                    render_start = time.perf_counter()
                    payload = build_render_payload(solution, settings)
                    render_ms = (time.perf_counter() - render_start) * 1000
                    inner_self.signals.finished.emit(
                        generation,
                        solution,
                        payload,
                        {"solve_ms": solve_ms, "render_ms": render_ms},
                        cache_key,
                        preview,
                    )
                except Exception as exc:
                    inner_self.signals.failed.emit(generation, exc)

        self.runnable = _Worker()
        self.signals = self.runnable.signals


class MainWindow:
    def __init__(self):
        from PySide6 import QtCore, QtWidgets

        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle("Strange Attractor Visualiser")
        self.window.resize(1500, 900)

        self.thread_pool = QtCore.QThreadPool.globalInstance()
        self.coordinator = ResultCoordinator()
        self.solution_cache = OrderedDict()
        self.parameter_sliders: dict[str, ParameterSlider] = {}
        self.saved_values: list[dict] = []
        self.current_solution = None
        self.animation_index = 0
        self.solve_in_progress = False
        self.pending_solve_mode = None
        self.queued_solve_mode = SOLVE_MODE_FULL
        self.slider_drag_depth = 0
        self.preview_render_active = False
        self.active_worker = None

        self.solve_timer = QtCore.QTimer()
        self.solve_timer.setSingleShot(True)
        self.solve_timer.setInterval(FULL_DEBOUNCE_MS)
        self.solve_timer.timeout.connect(self.start_solve)

        self.animation_timer = QtCore.QTimer()
        self.animation_timer.setInterval(35)
        self.animation_timer.timeout.connect(self.advance_animation)

        self.selected_name = "Lorenz"
        self.param_values = get_default_params(ATTRACTORS[self.selected_name])

        root = QtWidgets.QWidget()
        self.window.setCentralWidget(root)
        root_layout = QtWidgets.QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.left_panel = self._build_left_panel()
        self.renderer = AttractorView3D()
        self.right_panel = self._build_right_panel()
        root_layout.addWidget(self.left_panel, stretch=0)
        root_layout.addWidget(self.renderer.widget, stretch=1)
        root_layout.addWidget(self.right_panel, stretch=0)

        self._apply_style()
        self.rebuild_parameter_sliders()
        self.schedule_solve()

    def show(self):
        self.window.show()

    def _build_left_panel(self):
        from PySide6 import QtWidgets

        panel = QtWidgets.QWidget()
        panel.setFixedWidth(380)
        layout = QtWidgets.QVBoxLayout(panel)

        self.simple_mode = QtWidgets.QCheckBox("Simple UI")
        self.simple_mode.toggled.connect(self.apply_simple_mode)
        layout.addWidget(self.simple_mode)

        layout.addWidget(QtWidgets.QLabel("Attractor"))
        self.attractor_combo = QtWidgets.QComboBox()
        self.attractor_combo.addItems(list(ATTRACTORS.keys()))
        self.attractor_combo.currentTextChanged.connect(self.change_attractor)
        layout.addWidget(self.attractor_combo)

        self.info_toggle = QtWidgets.QCheckBox("Show attractor info")
        self.info_toggle.toggled.connect(self.update_info_text)
        layout.addWidget(self.info_toggle)
        self.info_text = QtWidgets.QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setVisible(False)
        self.info_text.setMaximumHeight(170)
        layout.addWidget(self.info_text)

        layout.addWidget(QtWidgets.QLabel("Parameters"))
        self.parameter_area = QtWidgets.QWidget()
        self.parameter_layout = QtWidgets.QHBoxLayout(self.parameter_area)
        layout.addWidget(self.parameter_area)

        buttons = QtWidgets.QHBoxLayout()
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.save_button = QtWidgets.QPushButton("Save")
        self.random_button = QtWidgets.QPushButton("Random")
        self.reset_button.clicked.connect(self.reset_parameters)
        self.save_button.clicked.connect(self.save_current_values)
        self.random_button.clicked.connect(self.randomise_parameters)
        buttons.addWidget(self.reset_button)
        buttons.addWidget(self.save_button)
        buttons.addWidget(self.random_button)
        layout.addLayout(buttons)

        layout.addWidget(QtWidgets.QLabel("Preset"))
        self.preset_combo = QtWidgets.QComboBox()
        self.apply_preset_button = QtWidgets.QPushButton("Apply preset")
        self.apply_preset_button.clicked.connect(self.apply_selected_preset)
        layout.addWidget(self.preset_combo)
        layout.addWidget(self.apply_preset_button)

        self.saved_list = QtWidgets.QListWidget()
        self.saved_list.itemDoubleClicked.connect(self.load_saved_item)
        layout.addWidget(self.saved_list, stretch=1)
        return panel

    def _build_right_panel(self):
        from PySide6 import QtWidgets

        panel = QtWidgets.QWidget()
        panel.setFixedWidth(300)
        layout = QtWidgets.QVBoxLayout(panel)

        layout.addWidget(QtWidgets.QLabel("Display"))
        self.density_toggle = QtWidgets.QCheckBox("Use density colouring")
        self.density_toggle.setChecked(DEFAULT_USE_DENSITY)
        self.density_toggle.toggled.connect(self.schedule_solve)
        layout.addWidget(self.density_toggle)

        self.display_mode_combo = QtWidgets.QComboBox()
        self.display_mode_combo.addItems(
            [DISPLAY_MODE_POINTS, DISPLAY_MODE_LINES, DISPLAY_MODE_LINES_POINTS]
        )
        self.display_mode_combo.setCurrentText(DEFAULT_DISPLAY_MODE)
        self.display_mode_combo.currentTextChanged.connect(self.schedule_solve)
        layout.addWidget(self.display_mode_combo)

        self.point_budget_combo = QtWidgets.QComboBox()
        self.point_budget_combo.addItems(list(POINT_BUDGETS.keys()))
        self.point_budget_combo.setCurrentText("Balanced (30000)")
        self.point_budget_combo.currentTextChanged.connect(self.schedule_solve)
        layout.addWidget(self.point_budget_combo)

        layout.addWidget(QtWidgets.QLabel("Integrator"))
        self.integrator_combo = QtWidgets.QComboBox()
        self.integrator_combo.addItems([SOLVER_LSODA, SOLVER_RK4])
        self.integrator_combo.currentTextChanged.connect(self.integrator_changed)
        layout.addWidget(self.integrator_combo)

        layout.addWidget(QtWidgets.QLabel("LSODA tolerance"))
        self.lsoda_tolerance_combo = QtWidgets.QComboBox()
        self.lsoda_tolerance_combo.addItems(list(LSODA_TOLERANCES.keys()))
        self.lsoda_tolerance_combo.setCurrentText(DEFAULT_LSODA_TOLERANCE)
        self.lsoda_tolerance_combo.currentTextChanged.connect(self.schedule_solve)
        layout.addWidget(self.lsoda_tolerance_combo)

        sampling_group = QtWidgets.QGroupBox("Sampling")
        sampling_layout = QtWidgets.QFormLayout(sampling_group)
        self.adaptive_horizon_toggle = QtWidgets.QCheckBox("Adaptive horizon")
        self.adaptive_horizon_toggle.setChecked(False)
        sampling_layout.addRow(self.adaptive_horizon_toggle)

        self.burn_in_spin = QtWidgets.QDoubleSpinBox()
        self.burn_in_spin.setRange(0.0, 80.0)
        self.burn_in_spin.setSingleStep(5.0)
        self.burn_in_spin.setDecimals(0)
        self.burn_in_spin.setSuffix(" %")
        self.burn_in_spin.setValue(0.0)
        sampling_layout.addRow("Burn-in", self.burn_in_spin)

        self.batch_steps_spin = QtWidgets.QSpinBox()
        self.batch_steps_spin.setRange(500, 50_000)
        self.batch_steps_spin.setSingleStep(500)
        self.batch_steps_spin.setValue(5_000)
        sampling_layout.addRow("Batch points", self.batch_steps_spin)

        self.max_points_spin = QtWidgets.QSpinBox()
        self.max_points_spin.setRange(1_000, 300_000)
        self.max_points_spin.setSingleStep(5_000)
        self.max_points_spin.setValue(60_000)
        sampling_layout.addRow("Max points", self.max_points_spin)

        self.bounds_tolerance_spin = QtWidgets.QDoubleSpinBox()
        self.bounds_tolerance_spin.setRange(0.1, 20.0)
        self.bounds_tolerance_spin.setSingleStep(0.5)
        self.bounds_tolerance_spin.setDecimals(1)
        self.bounds_tolerance_spin.setSuffix(" %")
        self.bounds_tolerance_spin.setValue(2.0)
        sampling_layout.addRow("Bounds tol", self.bounds_tolerance_spin)

        self.coverage_tolerance_spin = QtWidgets.QDoubleSpinBox()
        self.coverage_tolerance_spin.setRange(0.1, 20.0)
        self.coverage_tolerance_spin.setSingleStep(0.5)
        self.coverage_tolerance_spin.setDecimals(1)
        self.coverage_tolerance_spin.setSuffix(" %")
        self.coverage_tolerance_spin.setValue(1.0)
        sampling_layout.addRow("Coverage tol", self.coverage_tolerance_spin)

        self.stable_batches_spin = QtWidgets.QSpinBox()
        self.stable_batches_spin.setRange(1, 10)
        self.stable_batches_spin.setSingleStep(1)
        self.stable_batches_spin.setValue(2)
        sampling_layout.addRow("Stable batches", self.stable_batches_spin)

        self.coverage_bins_spin = QtWidgets.QSpinBox()
        self.coverage_bins_spin.setRange(8, 64)
        self.coverage_bins_spin.setSingleStep(8)
        self.coverage_bins_spin.setValue(32)
        sampling_layout.addRow("Coverage bins", self.coverage_bins_spin)

        self.adaptive_controls = [
            self.burn_in_spin,
            self.batch_steps_spin,
            self.max_points_spin,
            self.bounds_tolerance_spin,
            self.coverage_tolerance_spin,
            self.stable_batches_spin,
            self.coverage_bins_spin,
        ]
        for control in self.adaptive_controls:
            control.setEnabled(False)
        self.adaptive_horizon_toggle.toggled.connect(self.sampling_settings_changed)
        for control in self.adaptive_controls:
            control.valueChanged.connect(self.schedule_solve)
        layout.addWidget(sampling_group)

        self.performance_toggle = QtWidgets.QCheckBox("Show performance")
        self.performance_toggle.toggled.connect(self.update_status)
        layout.addWidget(self.performance_toggle)

        self.animate_toggle = QtWidgets.QCheckBox("Animate trajectory")
        self.animate_toggle.toggled.connect(self.set_animation_enabled)
        layout.addWidget(self.animate_toggle)

        self.system_label = QtWidgets.QLabel()
        self.system_label.setWordWrap(True)
        layout.addWidget(self.system_label)
        self.status_label = QtWidgets.QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addWidget(QtWidgets.QLabel("Projections"))
        self.projections = {
            "x-y": ProjectionView("x-y"),
            "x-z": ProjectionView("x-z"),
            "y-z": ProjectionView("y-z"),
        }
        for projection in self.projections.values():
            layout.addWidget(projection.widget)

        layout.addStretch(1)
        return panel

    def _apply_style(self):
        self.window.setStyleSheet(
            """
            QWidget { background: #0f2259; color: #eeeeee; font-family: Courier New; }
            QComboBox, QPushButton, QDoubleSpinBox, QListWidget, QTextEdit {
                border: 1px solid #eeeeee; padding: 4px; background: #0f2259;
            }
            QPushButton:hover { background: #eeeeee; color: #0f2259; }
            QSlider::groove:vertical { background: #eeeeee; width: 4px; }
            QSlider::handle:vertical { background: #da5700; height: 18px; margin: 0 -8px; }
            QLabel { font-weight: 600; }
            """
        )

    def current_settings(self, preview: bool = False) -> DisplaySettings:
        settings = DisplaySettings(
            display_mode=self.display_mode_combo.currentText(),
            point_budget=POINT_BUDGETS[self.point_budget_combo.currentText()],
            use_density=self.density_toggle.isChecked(),
        )
        if preview:
            return preview_display_settings(
                settings,
                PREVIEW_POINT_BUDGET,
                line_interpolation=PREVIEW_LINE_INTERPOLATION,
            )
        return settings

    def current_solver_settings(self) -> SolverSettings:
        method = self.integrator_combo.currentText()
        if method == SOLVER_RK4:
            return SolverSettings(method=SOLVER_RK4)
        rtol, atol = LSODA_TOLERANCES[self.lsoda_tolerance_combo.currentText()]
        return SolverSettings(
            method=SOLVER_LSODA,
            lsoda_rtol=rtol,
            lsoda_atol=atol,
        )

    def current_sampling_settings(self, preview: bool = False) -> AdaptiveHorizonSettings:
        if preview:
            return AdaptiveHorizonSettings(enabled=False)
        return AdaptiveHorizonSettings(
            enabled=self.adaptive_horizon_toggle.isChecked(),
            burn_in_fraction=self.burn_in_spin.value() / 100.0,
            batch_steps=self.batch_steps_spin.value(),
            max_points=self.max_points_spin.value(),
            stable_batches=self.stable_batches_spin.value(),
            bounds_tolerance=self.bounds_tolerance_spin.value() / 100.0,
            coverage_tolerance=self.coverage_tolerance_spin.value() / 100.0,
            coverage_bins=self.coverage_bins_spin.value(),
        )

    def integrator_changed(self, *_unused) -> None:
        self.lsoda_tolerance_combo.setEnabled(
            self.integrator_combo.currentText() == SOLVER_LSODA
        )
        self.schedule_solve()

    def sampling_settings_changed(self, enabled: bool) -> None:
        for control in self.adaptive_controls:
            control.setEnabled(enabled)
        self.schedule_solve()

    def rebuild_parameter_sliders(self):
        from PySide6 import QtWidgets

        while self.parameter_layout.count():
            item = self.parameter_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.parameter_sliders.clear()
        config = ATTRACTORS[self.selected_name]
        self.param_values = get_default_params(config)
        for param in config.params:
            slider = ParameterSlider(param)
            slider.connect(self.parameter_changed)
            slider.connect_drag_started(self.slider_drag_started)
            slider.connect_drag_finished(self.slider_drag_finished)
            self.parameter_sliders[param.name] = slider
            self.parameter_layout.addWidget(slider.widget)

        self.preset_combo.clear()
        self.preset_combo.addItems(list(config.presets.keys()))
        self.update_info_text()
        self.update_status()

    def parameter_changed(self, name: str, value: float) -> None:
        self.param_values[name] = value
        self.animation_timer.stop()
        self.animate_toggle.setChecked(False)
        self.update_status()
        self.schedule_solve(preview=self.is_slider_dragging())

    def slider_drag_started(self) -> None:
        self.slider_drag_depth += 1
        self.animation_timer.stop()
        self.animate_toggle.setChecked(False)
        self.update_status()

    def slider_drag_finished(self) -> None:
        self.slider_drag_depth = max(0, self.slider_drag_depth - 1)
        if not self.is_slider_dragging():
            self.schedule_solve(preview=False, delay_ms=0)
            self.update_status()

    def is_slider_dragging(self) -> bool:
        return self.slider_drag_depth > 0

    def schedule_solve(
        self,
        *_unused,
        preview: bool | None = None,
        delay_ms: int | None = None,
    ):
        if preview is None:
            preview = self.is_slider_dragging()
        requested_mode = SOLVE_MODE_PREVIEW if preview else SOLVE_MODE_FULL
        self.queued_solve_mode = requested_mode
        if delay_ms is None:
            delay_ms = (
                PREVIEW_DEBOUNCE_MS
                if self.queued_solve_mode == SOLVE_MODE_PREVIEW
                else FULL_DEBOUNCE_MS
            )
        self.solve_timer.start(delay_ms)

    def start_solve(self):
        solve_mode = self.queued_solve_mode
        self.queued_solve_mode = SOLVE_MODE_FULL
        if self.solve_in_progress:
            self.pending_solve_mode = solve_mode
            self.coordinator.next_generation()
            return

        preview = solve_mode == SOLVE_MODE_PREVIEW
        selected_name = self.selected_name
        config = ATTRACTORS[selected_name]
        param_values = dict(self.param_values)
        solver_settings = self.current_solver_settings()
        sampling_settings = self.current_sampling_settings(preview=preview)
        cached_solution = None
        if not preview:
            cache_key = parameter_cache_key(
                selected_name,
                config,
                param_values,
                solver_settings,
                sampling_settings,
            )
            cached_solution = self.solution_cache.get(cache_key)
            if cached_solution is not None:
                self.solution_cache.move_to_end(cache_key)
        generation = self.coordinator.next_generation()
        worker = SolveWorker(
            generation,
            selected_name,
            param_values,
            self.current_settings(preview=preview),
            cached_solution,
            PREVIEW_SOLVE_STEPS if preview else None,
            solver_settings,
            sampling_settings,
            preview,
        )
        worker.signals.finished.connect(self.solve_finished)
        worker.signals.failed.connect(self.solve_failed)
        self.active_worker = worker
        self.solve_in_progress = True
        self.thread_pool.start(worker.runnable)

    def solve_finished(self, generation, solution, payload, timings, cache_key, preview):
        self.solve_in_progress = False
        self.active_worker = None
        if cache_key is not None:
            self.solution_cache[cache_key] = solution
            self.solution_cache.move_to_end(cache_key)
            while len(self.solution_cache) > CACHE_MAX_ENTRIES:
                self.solution_cache.popitem(last=False)
        accepted = self.coordinator.accept_success(generation, payload, timings)
        if accepted:
            if not preview:
                self.current_solution = solution
            self.preview_render_active = preview
            self.apply_payload(payload)
            self.update_status()
        self._run_pending_solve()

    def solve_failed(self, generation, error):
        self.solve_in_progress = False
        self.active_worker = None
        if self.coordinator.accept_error(generation, error):
            self.update_status()
        self._run_pending_solve()

    def _run_pending_solve(self):
        if self.pending_solve_mode is not None:
            self.queued_solve_mode = self.pending_solve_mode
            self.pending_solve_mode = None
            self.solve_timer.start(0)

    def apply_payload(self, payload):
        self.renderer.set_payload(payload)
        for name, projection in payload.projections.items():
            self.projections[name].set_data(
                projection.x,
                projection.y,
                projection.show_points,
                projection.show_lines,
                projection.line_colors,
                projection.line_x,
                projection.line_y,
            )

    def update_status(self, *_unused):
        config = ATTRACTORS[self.selected_name]
        parts = [
            f"System: {self.selected_name}",
            " ".join(
                f"{name.strip('$')}: {value:.2f}"
                for name, value in self.param_values.items()
            ),
        ]
        if self.is_slider_dragging() or self.preview_render_active:
            parts.append("Preview: low resolution; full render pending")
        if self.coordinator.status.error:
            parts.append(f"Error: {self.coordinator.status.error}")
        elif self.performance_toggle.isChecked():
            timings = self.coordinator.status.timings
            parts.append(
                " ".join(f"{key}: {value:.1f} ms" for key, value in timings.items())
            )
        solver_settings = self.current_solver_settings()
        if solver_settings.method == SOLVER_RK4:
            parts.append("Integrator: RK4")
        else:
            parts.append(f"Integrator: LSODA {self.lsoda_tolerance_combo.currentText()}")
        sampling_settings = self.current_sampling_settings(preview=False)
        if sampling_settings.enabled:
            parts.append(
                "Sampling: adaptive "
                f"burn-in {self.burn_in_spin.value():.0f}%, "
                f"max {sampling_settings.max_points}"
            )
        else:
            parts.append("Sampling: fixed horizon")
        self.system_label.setText(format_equation_text(config.equation_text))
        self.status_label.setText("\n".join(parts))

    def update_info_text(self, *_unused):
        config = ATTRACTORS[self.selected_name]
        self.info_text.setPlainText(config.description)
        self.info_text.setVisible(self.info_toggle.isChecked())

    def change_attractor(self, selected_name: str):
        self.selected_name = selected_name
        self.rebuild_parameter_sliders()
        self.schedule_solve()

    def reset_parameters(self, *_unused):
        self.set_param_values(get_default_params(ATTRACTORS[self.selected_name]))

    def randomise_parameters(self, *_unused):
        config = ATTRACTORS[self.selected_name]
        values = {
            param.name: random.uniform(param.min_val, param.max_val)
            for param in config.params
        }
        self.set_param_values(values)

    def apply_selected_preset(self, *_unused):
        config = ATTRACTORS[self.selected_name]
        preset = config.presets.get(self.preset_combo.currentText())
        if preset:
            values = get_default_params(config)
            values.update(preset)
            self.set_param_values(values)

    def set_param_values(self, values: dict[str, float]):
        for name, value in values.items():
            slider = self.parameter_sliders.get(name)
            if slider is not None:
                slider.set_value(value, emit=False)
        self.param_values.update(values)
        self.schedule_solve()
        self.update_status()

    def save_current_values(self, *_unused):
        entry = {"attractor": self.selected_name, "params": dict(self.param_values)}
        self.saved_values.append(entry)
        label = f"{entry['attractor']}: " + " ".join(
            f"{k.strip('$')}={v:.2f}" for k, v in entry["params"].items()
        )
        self.saved_list.addItem(label)

    def load_saved_item(self, item):
        index = self.saved_list.row(item)
        entry = self.saved_values[index]
        if entry["attractor"] != self.selected_name:
            self.attractor_combo.setCurrentText(entry["attractor"])
        self.set_param_values(entry["params"])

    def apply_simple_mode(self, enabled: bool):
        self.info_toggle.setVisible(not enabled)
        self.info_text.setVisible(not enabled and self.info_toggle.isChecked())
        self.saved_list.setVisible(not enabled)
        self.save_button.setVisible(not enabled)
        self.right_panel.setVisible(not enabled)

    def set_animation_enabled(self, enabled: bool):
        if enabled and self.current_solution is not None:
            self.animation_index = 1
            self.animation_timer.start()
        else:
            self.animation_timer.stop()
            payload = self.coordinator.last_payload
            if payload is not None:
                self.apply_payload(payload)

    def advance_animation(self):
        if self.current_solution is None:
            self.animation_timer.stop()
            return
        step = max(1, len(self.current_solution) // 180)
        self.animation_index = min(len(self.current_solution), self.animation_index + step)
        settings = self.current_settings()
        payload = build_render_payload(self.current_solution[: self.animation_index], settings)
        self.apply_payload(payload)
        if self.animation_index >= len(self.current_solution):
            self.animation_timer.stop()
            self.animate_toggle.setChecked(False)


def create_main_window():
    return MainWindow()
