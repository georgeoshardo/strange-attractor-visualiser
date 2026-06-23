import numpy as np
import streamlit as st
from pathlib import Path

import strange_attractor_visualiser.ui.plot_page as plot_page_module
import strange_attractor_visualiser.ui.sidebar as sidebar_module
import strange_attractor_visualiser.ui.theme as theme_module
from strange_attractor_visualiser.attractors.registry import ATTRACTORS
from strange_attractor_visualiser.components.live_vertical_slider import (
    _resolve_slider_default,
)
from strange_attractor_visualiser.ui.figure import build_figure, build_static_data
from strange_attractor_visualiser.ui.plane_figures import _draw_plane_points
from strange_attractor_visualiser.ui.plot_page import (
    POINT_BUDGETS,
    downsample_points,
    param_cache_items,
    render_interactive_attractor,
)
from strange_attractor_visualiser.ui.sidebar import _apply_preset, _reset_parameters


def test_reset_parameters_increments_version():
    st.session_state.clear()
    config = ATTRACTORS["Lorenz"]
    selected_name = "Lorenz"

    _reset_parameters(config, selected_name)
    v1 = st.session_state.get(f"{selected_name}_version", 0)

    _reset_parameters(config, selected_name)
    v2 = st.session_state.get(f"{selected_name}_version", 0)

    assert v2 == v1 + 1


def test_apply_preset_updates_session_state_and_unknown_is_noop():
    st.session_state.clear()
    config = ATTRACTORS["Rossler"]
    selected_name = "Rossler"

    _apply_preset(config, selected_name, "Classic")
    version = st.session_state.get(f"{selected_name}_version", 0)
    assert version > 0

    for k, v in config.presets["Classic"].items():
        assert st.session_state[f"{selected_name}_{k}_v{version}"] == v

    _apply_preset(config, selected_name, "__does_not_exist__")
    v2 = st.session_state.get(f"{selected_name}_version", 0)
    assert v2 > version

    for k, v in config.presets["Classic"].items():
        assert st.session_state[f"{selected_name}_{k}_v{version}"] == v
        assert st.session_state.get(f"{selected_name}_{k}_v{v2}") is None


def test_downsample_points_respects_display_cap():
    x = list(range(10_000))
    y = list(range(10_000))
    z = list(range(10_000))

    x_plot, y_plot, z_plot = downsample_points(x, y, z, max_points=8_000)

    assert len(x_plot) == 8_000
    assert len(y_plot) == len(x_plot)
    assert len(z_plot) == len(x_plot)


def test_point_budgets_downsample_to_selected_budget():
    x = list(range(10_000))
    y = list(range(10_000))
    z = list(range(10_000))

    x_fast, y_fast, z_fast = downsample_points(
        x, y, z, max_points=POINT_BUDGETS["Fast (3000)"]
    )
    x_full, y_full, z_full = downsample_points(
        x, y, z, max_points=POINT_BUDGETS["Full (8000)"]
    )

    assert len(x_fast) == 3_000
    assert len(y_fast) == len(x_fast)
    assert len(z_fast) == len(x_fast)
    assert len(x_full) == 8_000
    assert len(y_full) == len(x_full)
    assert len(z_full) == len(x_full)


def test_downsample_points_keeps_short_trajectory_unchanged():
    x = np.arange(100)
    y = np.arange(100) + 1
    z = np.arange(100) + 2

    x_plot, y_plot, z_plot = downsample_points(
        x, y, z, max_points=POINT_BUDGETS["Fast (3000)"]
    )

    assert x_plot is x
    assert y_plot is y
    assert z_plot is z


def test_param_cache_items_follow_config_order_and_round_to_step():
    config = ATTRACTORS["Lorenz"]
    values = {"$c$": 2.674, "$a$": 10.004, "$b$": 28.006}

    items = param_cache_items(config, values)

    assert items == (("$a$", 10.0), ("$b$", 28.01), ("$c$", 2.67))


def test_interactive_renderer_is_fragment_wrapped():
    assert hasattr(render_interactive_attractor, "__wrapped__")


def test_normal_fragment_controls_are_wrapped_in_dedicated_panel():
    source = Path(plot_page_module.__file__).read_text()

    assert 'key="normal-controls-panel"' in source


def test_normal_fragment_layout_reserves_central_plot_area():
    css = Path(theme_module.__file__).with_name("theme.css").read_text()

    assert ".st-key-normal-controls-panel" in css
    assert ".stApp:has(.st-key-normal-controls-panel) .st-key-plot-frame" in css
    assert (
        "calc(100vw - var(--normal-left-rail-width) - var(--normal-right-rail-width))"
        in css
    )


def test_parameter_controls_use_local_live_slider_component():
    source = Path(sidebar_module.__file__).read_text()

    assert "from ..components.live_vertical_slider import live_vertical_slider" in source
    assert "streamlit_vertical_slider" not in source


def test_live_vertical_slider_frontend_emits_during_drag():
    component_html = (
        Path(sidebar_module.__file__).parents[1]
        / "components"
        / "live_vertical_slider"
        / "frontend"
        / "index.html"
    )
    html = component_html.read_text()

    input_handler = html[html.index('addEventListener("input"') :]
    assert "Streamlit.setComponentValue" in input_handler
    assert "emitValue" in input_handler
    assert "requestAnimationFrame" in html
    assert 'dataType: "json"' in html


def test_live_vertical_slider_uses_default_when_session_value_is_none():
    st.session_state.clear()
    st.session_state["live-slider"] = None

    assert _resolve_slider_default("live-slider", 10.0) == 10.0


def test_build_static_data_supports_lines_with_points():
    values = np.arange(5)
    marker = {"size": 1.25, "color": np.linspace(0, 1, 5), "colorscale": "Viridis"}

    trace, _ = build_static_data(
        values, values + 1, values + 2, marker, display_mode="Lines + points"
    )

    assert trace["mode"] == "lines+markers"
    assert trace["marker"]["color"] == marker["color"].tolist()
    assert "line" in trace


def test_build_static_data_supports_lines_only():
    values = np.arange(5)

    trace, _ = build_static_data(
        values, values + 1, values + 2, {"size": 1.25}, display_mode="Lines"
    )

    assert trace["mode"] == "lines"
    assert "line" in trace
    assert "marker" not in trace


def test_build_figure_animation_supports_lines_only():
    values = np.arange(20)

    fig = build_figure(
        values,
        values + 1,
        values + 2,
        {"size": 1.25},
        animate=True,
        display_mode="Lines",
    )

    assert fig.data[0].mode == "lines"
    assert fig.frames[0].data[0].mode == "lines"


def test_plane_projection_lines_mode_uses_line_plot(mocker):
    ax = mocker.Mock()
    x = np.arange(5)
    y = np.arange(5)

    _draw_plane_points(ax, x, y, display_mode="Lines only")

    ax.plot.assert_called_once()
    ax.scatter.assert_not_called()


def test_plane_projection_lines_with_points_uses_both(mocker):
    ax = mocker.Mock()
    x = np.arange(5)
    y = np.arange(5)

    _draw_plane_points(ax, x, y, display_mode="Lines + points")

    ax.plot.assert_called_once()
    ax.scatter.assert_called_once()
