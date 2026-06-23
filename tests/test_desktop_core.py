import tomllib
from pathlib import Path

import numpy as np

from strange_attractor_visualiser.attractors.registry import ATTRACTORS
from strange_attractor_visualiser.desktop.controller import ResultCoordinator
from strange_attractor_visualiser.desktop.controls import FloatSliderSpec
from strange_attractor_visualiser.desktop.equations import format_equation_text
from strange_attractor_visualiser.desktop.render_data import (
    DisplaySettings,
    build_render_payload,
    downsample_solution,
    preview_display_settings,
)
from strange_attractor_visualiser.desktop.state import parameter_cache_key
from strange_attractor_visualiser.core.display import (
    DISPLAY_MODE_LINES,
    DISPLAY_MODE_LINES_POINTS,
    DISPLAY_MODE_POINTS,
)
from strange_attractor_visualiser.core.solver import get_default_params, solve_attractor


def test_pyproject_defines_desktop_extra_and_qt_entrypoint():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    desktop_deps = pyproject["project"]["optional-dependencies"]["desktop"]
    assert any(dep.startswith("PySide6") for dep in desktop_deps)
    assert any(dep.startswith("pyqtgraph") for dep in desktop_deps)
    assert any(dep.startswith("PyOpenGL") for dep in desktop_deps)
    assert (
        pyproject["project"]["scripts"]["strange-attractor-qt"]
        == "strange_attractor_visualiser.desktop.app:main"
    )


def test_float_slider_spec_maps_float_steps_to_integer_ticks():
    spec = FloatSliderSpec(min_value=-5.83, max_value=2.4, step=0.01)

    assert spec.minimum_tick == 0
    assert spec.maximum_tick == 823
    assert spec.tick_to_value(0) == -5.83
    assert spec.tick_to_value(823) == 2.4
    assert spec.value_to_tick(-5.825) == 0
    assert spec.value_to_tick(2.399) == 823


def test_float_slider_spec_supports_thousandth_steps():
    spec = FloatSliderSpec(min_value=0.47, max_value=2.0, step=0.001)

    assert spec.maximum_tick == 1530
    assert spec.tick_to_value(spec.value_to_tick(0.571)) == 0.571


def test_parameter_cache_key_uses_config_order_and_step_rounding():
    config = ATTRACTORS["Lorenz"]
    values = {"$c$": 2.674, "$a$": 10.004, "$b$": 28.006}

    key = parameter_cache_key("Lorenz", config, values)

    assert key == ("Lorenz", (("$a$", 10.0), ("$b$", 28.01), ("$c$", 2.67)))


def test_solve_attractor_accepts_step_count_override():
    config = ATTRACTORS["Lorenz"]
    params = get_default_params(config)

    preview_solution = solve_attractor(config, params, n_steps=1234)
    full_solution = solve_attractor(config, params)

    assert preview_solution.shape == (1234, 3)
    assert full_solution.shape == (config.time_defaults["n"], 3)


def test_preview_display_settings_caps_points_and_disables_density():
    settings = DisplaySettings(
        display_mode=DISPLAY_MODE_LINES_POINTS,
        point_budget=None,
        use_density=True,
    )

    preview = preview_display_settings(settings, point_budget=2500)

    assert preview.display_mode == DISPLAY_MODE_LINES_POINTS
    assert preview.point_budget == 2500
    assert preview.use_density is False


def test_preview_display_settings_keeps_smaller_existing_budget():
    settings = DisplaySettings(
        display_mode=DISPLAY_MODE_POINTS,
        point_budget=1000,
        use_density=True,
    )

    preview = preview_display_settings(settings, point_budget=2500)

    assert preview.point_budget == 1000
    assert preview.use_density is False


def test_format_equation_text_removes_streamlit_latex_for_qt_label():
    text = format_equation_text(ATTRACTORS["Lorenz"].equation_text)

    assert text.splitlines() == [
        "ẋ = a(y - x)",
        "ẏ = x(b - z) - y",
        "ż = xy - c z",
    ]
    assert "$" not in text
    assert "\\" not in text


def test_format_equation_text_handles_common_fraction_markup():
    text = format_equation_text(ATTRACTORS["Aizawa"].equation_text)

    assert "ż = c + az - (z³)/(3) - (x² + y²)(1 + ez) + fzx³" in text


def test_downsample_solution_returns_exact_budget():
    solution = np.column_stack([np.arange(10_000), np.arange(10_000), np.arange(10_000)])

    sampled = downsample_solution(solution, 3_000)

    assert sampled.shape == (3_000, 3)
    assert np.array_equal(sampled[0], solution[0])
    assert np.array_equal(sampled[-1], solution[-1])


def test_build_render_payload_for_points_only():
    solution = np.column_stack([np.arange(20), np.arange(20) + 1, np.arange(20) + 2])
    settings = DisplaySettings(
        display_mode=DISPLAY_MODE_POINTS,
        point_budget=8,
        use_density=False,
    )

    payload = build_render_payload(solution, settings)

    assert payload.positions.shape == (8, 3)
    assert payload.positions.dtype == np.float32
    assert payload.show_points is True
    assert payload.show_lines is False
    assert payload.point_colors.shape == (8, 4)
    assert payload.projections["x-y"].show_points is True
    assert payload.projections["x-y"].show_lines is False


def test_build_render_payload_for_lines_with_points():
    solution = np.column_stack([np.arange(20), np.arange(20) + 1, np.arange(20) + 2])
    settings = DisplaySettings(
        display_mode=DISPLAY_MODE_LINES_POINTS,
        point_budget=12,
        use_density=False,
    )

    payload = build_render_payload(solution, settings)

    assert payload.positions.shape == (12, 3)
    assert payload.show_points is True
    assert payload.show_lines is True
    assert payload.projections["x-z"].show_points is True
    assert payload.projections["x-z"].show_lines is True


def test_build_render_payload_disables_density_for_lines_only():
    rng = np.random.default_rng(0)
    solution = rng.normal(size=(200, 3))
    settings = DisplaySettings(
        display_mode=DISPLAY_MODE_LINES,
        point_budget=None,
        use_density=True,
    )

    payload = build_render_payload(solution, settings)

    assert payload.show_points is False
    assert payload.show_lines is True
    assert payload.point_colors is None


def test_build_render_payload_density_colours_are_rgba():
    rng = np.random.default_rng(1)
    solution = rng.normal(size=(250, 3))
    settings = DisplaySettings(
        display_mode=DISPLAY_MODE_POINTS,
        point_budget=120,
        use_density=True,
    )

    payload = build_render_payload(solution, settings)

    assert payload.point_colors.shape == (120, 4)
    assert np.isfinite(payload.point_colors).all()
    assert payload.point_colors.min() >= 0.0
    assert payload.point_colors.max() <= 1.0


def test_result_coordinator_ignores_stale_successes_and_keeps_last_good_on_error():
    coordinator = ResultCoordinator()
    old_generation = coordinator.next_generation()
    current_generation = coordinator.next_generation()
    solution = np.column_stack([np.arange(5), np.arange(5), np.arange(5)])
    settings = DisplaySettings(
        display_mode=DISPLAY_MODE_POINTS,
        point_budget=None,
        use_density=False,
    )
    payload = build_render_payload(solution, settings)

    assert coordinator.accept_success(old_generation, payload, {"solve_ms": 1.0}) is False
    assert coordinator.last_payload is None

    assert (
        coordinator.accept_success(current_generation, payload, {"solve_ms": 2.0})
        is True
    )
    assert coordinator.last_payload is payload
    assert coordinator.status.error is None

    error_generation = coordinator.next_generation()
    assert coordinator.accept_error(error_generation, RuntimeError("bad solve")) is True
    assert coordinator.last_payload is payload
    assert coordinator.status.error == "bad solve"
