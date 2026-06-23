import time

import numpy as np
import plotly.express as px
import streamlit as st

from ..attractors.registry import ATTRACTORS
from ..components.plotly_fast import plotly_fast
from ..core.solver import solve_attractor
from ..ui.figure import (
    DISPLAY_MODE_LINES,
    DISPLAY_MODE_POINTS,
    DISPLAY_MODES,
    build_figure,
    build_static_data,
)
from ..ui.plane_figures import x_y_plane, x_z_plane, y_z_plane
from ..ui.sidebar import (
    compute_marker_style,
    render_horizontal_parameter_controls,
    render_info_panel,
    render_parameter_controls,
    render_saved_values_ui,
    select_attractor_ui,
)
from ..ui.theme import apply_theme

POINT_BUDGETS = {
    "Fast (3000)": 3000,
    "Balanced (6000)": 6000,
    "Full (8000)": 8000,
}


def init_page():
    st.set_page_config(layout="wide")
    apply_theme()


def downsample_points(x, y, z, max_points: int):
    if len(x) <= max_points:
        return x, y, z

    indices = np.linspace(0, len(x) - 1, max_points, dtype=int)

    if isinstance(x, list):
        return (
            [x[i] for i in indices],
            [y[i] for i in indices],
            [z[i] for i in indices],
        )

    return x[indices], y[indices], z[indices]


def _step_decimal_places(step: float) -> int:
    step_text = f"{step:.12f}".rstrip("0")
    if "." not in step_text:
        return 0

    return len(step_text.split(".", 1)[1])


def param_cache_items(
    config, param_values: dict[str, float]
) -> tuple[tuple[str, float], ...]:
    items = []
    for param in config.params:
        decimals = _step_decimal_places(param.step)
        items.append((param.name, round(float(param_values[param.name]), decimals)))

    return tuple(items)


@st.cache_data(max_entries=128, show_spinner=False)
def solve_attractor_cached(
    selected_name: str, param_items: tuple[tuple[str, float], ...]
):
    config = ATTRACTORS[selected_name]
    return solve_attractor(config, dict(param_items))


def render_plot_page():
    init_page()

    simple_mode = st.toggle("SIMPLE UI", key="simple-mode-toggle")

    render_interactive_attractor(simple_mode)


@st.fragment
def render_interactive_attractor(simple_mode: bool):
    if simple_mode:
        st.markdown(
            "<style>[data-testid='stSidebar'] { display: none !important; } </style>",
            unsafe_allow_html=True,
        )
        simple_panel = st.container(key="simple-panel")
        selected_name = simple_panel.radio(
            "ATTRACTOR",
            options=list(ATTRACTORS.keys()),
            label_visibility="collapsed",
        )
        config = ATTRACTORS[selected_name]
        param_values = render_horizontal_parameter_controls(
            config, simple_panel, selected_name
        )
        use_density = False
        colourscale = None
        animate = False
        display_mode = DISPLAY_MODE_POINTS
        point_budget = POINT_BUDGETS["Full (8000)"]
        show_performance = False
    else:
        controls_panel = st.container(key="normal-controls-panel")
        controls_section = controls_panel.container(key="fragment-section-controls")
        config, selected_name = select_attractor_ui(controls_section)
        show_info = controls_section.toggle(
            "SHOW ATTRACTOR INFO", value=False, key="toggle_attractor_info"
        )
        if config.description and show_info:
            render_info_panel(True, controls_section, config)

        if "saved_values" not in st.session_state:
            st.session_state.saved_values = []

        parameter_section = controls_panel.container(key="fragment-section-parameters")
        parameter_section.markdown("### Parameters")
        param_values = render_parameter_controls(
            config, parameter_section, selected_name
        )

        saved_section = controls_panel.container(key="fragment-section-saved")
        render_saved_values_ui(selected_name, saved_section, config, param_values)

    if simple_mode:
        st.container(key="simple-equation").markdown(config.equation_text)

    if not simple_mode:
        selected_point_budget = st.session_state.get(
            "point_budget_select", "Full (8000)"
        )
        point_budget = POINT_BUDGETS.get(
            selected_point_budget, POINT_BUDGETS["Full (8000)"]
        )

    solve_start = time.perf_counter()
    param_items = param_cache_items(config, param_values)
    solution = solve_attractor_cached(selected_name, param_items)
    solve_seconds = time.perf_counter() - solve_start
    x, y, z = solution.T

    x, y, z = downsample_points(x, y, z, point_budget)

    plot_shell = st.container(key="plot-shell")

    if not simple_mode:
        right_rail = plot_shell.container(key="rp-rail")

        display_section = right_rail.container(key="rp-section-display")
        display_section.markdown("### Display")
        use_density = display_section.toggle(
            "USE DENSITY COLOURING (SLOWER PERFORMANCE)", value=False
        )

        colourscale_list = px.colors.named_colorscales()
        colourscale = display_section.selectbox(
            "DENSITY COLOURSCALE",
            options=colourscale_list,
            label_visibility="collapsed",
        )
        display_mode = display_section.selectbox(
            "DISPLAY MODE",
            options=DISPLAY_MODES,
            label_visibility="collapsed",
        )
        display_section.selectbox(
            "POINT BUDGET",
            options=list(POINT_BUDGETS.keys()),
            index=2,
            key="point_budget_select",
            label_visibility="collapsed",
        )
        show_performance = display_section.toggle("SHOW PERFORMANCE", value=False)

        run_section = right_rail.container(key="rp-section-run")
        run_section.markdown("### Run")
        animate = run_section.toggle("ANIMATE TRAJECTORY", value=False)

        status_section = right_rail.container(key="rp-section-status")
        status_section.markdown(f"### System: {config.name}")
        status_section.markdown(config.equation_text)

        plane_plot = right_rail.container(key="rp-section-plot")
        plane_plot.markdown("### Projections")
        projection_start = time.perf_counter()
        for img in (
            x_y_plane(x, y, display_mode),
            x_z_plane(x, z, display_mode),
            y_z_plane(y, z, display_mode),
        ):
            plane_plot.image(img, use_container_width=False, width=180)
        projection_seconds = time.perf_counter() - projection_start
    else:
        projection_seconds = 0.0

    render_start = time.perf_counter()
    marker_dict = compute_marker_style(
        x, y, use_density and display_mode != DISPLAY_MODE_LINES, colourscale
    )

    plot_frame = plot_shell.container(key="plot-frame")

    if animate:
        fig = build_figure(x, y, z, marker_dict, animate, display_mode)
        plot_frame.plotly_chart(
            fig,
            width="stretch",
            height="stretch",
            config={"responsive": True},
            key="main-attractor-plot",
        )
    else:
        trace, layout = build_static_data(x, y, z, marker_dict, display_mode)
        with plot_frame:
            plotly_fast(trace, layout, key="main-attractor-plot")
    render_seconds = time.perf_counter() - render_start

    html = '<div class="status-bar">'
    html += '<span class="status-info">'
    html += "<span><strong>SYSTEM:</strong> " + str(selected_name) + "</span>"
    html += (
        f"<span><strong>INITIAL CONDITIONS:</strong> {config.initial_conditions}"
        + "</span>"
    )
    if param_values:
        params_parts = []
        for k, v in param_values.items():
            name = k.strip("$")
            params_parts.append("<strong>" + name + ":</strong> " + f"{v:.2f}")
        html += "<span>" + "  ".join(params_parts) + "</span>"
    if show_performance:
        html += (
            "<span><strong>POINTS:</strong> "
            + str(len(x))
            + f"  <strong>SOLVE:</strong> {solve_seconds * 1000:.1f} ms"
            + f"  <strong>PROJ:</strong> {projection_seconds * 1000:.1f} ms"
            + f"  <strong>RENDER PREP:</strong> {render_seconds * 1000:.1f} ms"
            + "</span>"
        )
    html += "</span></div>"
    plot_shell.markdown(html, unsafe_allow_html=True)
