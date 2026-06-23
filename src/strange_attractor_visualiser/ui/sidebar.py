import random
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import gaussian_kde
from streamlit.delta_generator import DeltaGenerator

from ..attractors.registry import (
    ATTRACTORS,
)
from ..components.live_vertical_slider import live_vertical_slider
from ..core.models import AttractorConfig


def _reset_parameters(config: AttractorConfig, selected_name: str):
    version = st.session_state.get(f"{selected_name}_version", 0) + 1
    st.session_state[f"{selected_name}_version"] = version


def _apply_preset(config: AttractorConfig, selected_name: str, preset_name: str):
    version = st.session_state.get(f"{selected_name}_version", 0) + 1
    st.session_state[f"{selected_name}_version"] = version
    preset = config.presets.get(preset_name, {})
    for param_name, value in preset.items():
        key = f"{selected_name}_{param_name}_v{version}"
        st.session_state[key] = value


def _random_param_values(config: AttractorConfig, selected_name: str):
    version = st.session_state.get(f"{selected_name}_version", 0) + 1
    st.session_state[f"{selected_name}_version"] = version
    for param in config.params:
        key = f"{selected_name}_{param.name}_v{version}"
        st.session_state[key] = random.uniform(param.min_val, param.max_val)


def select_attractor_ui(
    config_container: DeltaGenerator,
) -> tuple[AttractorConfig, str]:
    config_container.markdown("### Attractor")
    selected_name = config_container.selectbox(
        "ATTRACTOR", options=list(ATTRACTORS.keys()), label_visibility="collapsed"
    )
    config = ATTRACTORS[selected_name]

    return config, selected_name


def render_parameter_controls(
    config: AttractorConfig, config_container: DeltaGenerator, selected_name: str
) -> dict[str, float]:
    param_values = {}
    n = len(config.params)
    if n == 0:
        return param_values

    version = st.session_state.get(f"{selected_name}_version", 0)
    cols = config_container.columns(n)
    for i, param in enumerate(config.params):
        with cols[i]:
            value = live_vertical_slider(
                key=f"{selected_name}_{param.name}_v{version}",
                width=35,
                height=160,
                default_value=param.default,
                min_value=param.min_val,
                max_value=param.max_val,
                step=param.step,
                label=param.name,
                value_always_visible=True,
                show_marks=True,
                thumb_color="#DA5700",
                track_color="#0f2259",
                slider_color="#ffffff",
                slider_border_color="#cccccc",
                slider_border_width=3,
                value_font_size=12,
                mark_font_size=10,
            )
            param_values[param.name] = value

    return param_values


def render_info_panel(
    attractor_info: bool,
    config_container: DeltaGenerator,
    config: AttractorConfig,
):
    if attractor_info:
        config_container.subheader("Overview")
        config_container.write(config.description)
        config_container.markdown(
            f"**Equations**  {config.equation_text}",
        )
        if config.prompts:
            config_container.subheader("Parameters")
            for prompt in config.prompts:
                config_container.write(f"- {prompt}")


def filter_saved_values(show_all: bool, selected_name: str) -> list[dict[str, Any]]:
    filtered = (
        st.session_state.saved_values
        if show_all
        else [
            entry
            for entry in st.session_state.saved_values
            if entry.get("attractor") == selected_name
        ]
    )

    return filtered


def build_saved_rows(filtered: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for idx, entry in enumerate(filtered, start=1):
        row = {"SET": idx, "ATTRACTOR": entry.get("attractor")}
        row.update(entry.get("params", {}))
        rows.append(row)

    return rows


def render_saved_values_ui(
    selected_name: str,
    config_container: DeltaGenerator,
    config: AttractorConfig,
    param_values: dict,
):
    reset_button, save_button, randomise_button = config_container.columns(3)
    reset_button.button(
        "Reset",
        on_click=_reset_parameters,
        args=(config, selected_name),
        width="stretch",
        key=f"{selected_name}_reset",
    )

    if save_button.button("Save", width="stretch", key=f"{selected_name}_save"):
        st.session_state.saved_values.append({
            "attractor": selected_name,
            "params": {param.name: param_values[param.name] for param in config.params},
        })

    if st.session_state.saved_values:
        show_all = config_container.toggle(
            "SHOW ALL ATTRACTORS", value=False, key=f"{selected_name}_show_all"
        )
        filtered = filter_saved_values(show_all, selected_name)
        rows = build_saved_rows(filtered)
        config_container.caption(
            f"Showing: {len(filtered)} of {len(st.session_state.saved_values)}"
        )
        df = pd.DataFrame(rows)
        with config_container.expander(
            "SHOW SAVED VALUES", expanded=True, key=f"{selected_name}_saved_expander"
        ):
            st.table(df, hide_index=True)

    randomise_button.button(
        "Random",
        on_click=_random_param_values,
        args=(config, selected_name),
        width="stretch",
        key=f"{selected_name}_random",
    )

    preset_names = list(config.presets.keys())
    if preset_names:
        config_container.markdown("### Preset")
        selected_preset = config_container.selectbox(
            "PRESET",
            options=preset_names,
            label_visibility="collapsed",
            key=f"{selected_name}_preset_select",
        )
        config_container.button(
            "Apply preset",
            on_click=_apply_preset,
            args=(config, selected_name, selected_preset),
            width="stretch",
            key=f"{selected_name}_apply_preset",
        )


def render_horizontal_parameter_controls(
    config: AttractorConfig, config_container: DeltaGenerator, selected_name: str
) -> dict[str, float]:
    param_values = {}
    n = len(config.params)
    if n == 0:
        return param_values

    version = st.session_state.get(f"{selected_name}_version", 0)
    for param in config.params:
        value = config_container.slider(
            label=param.name,
            min_value=param.min_val,
            max_value=param.max_val,
            value=param.default,
            step=param.step,
            key=f"simple_{selected_name}_{param.name}_v{version}",
        )
        param_values[param.name] = value

    return param_values


def compute_marker_style(
    x: np.ndarray,
    y: np.ndarray,
    use_density: bool,
    colourscale: str | None,
) -> dict[str, Any]:
    if use_density:
        n = len(x)
        sample_size = min(1000, n)
        indices = np.random.choice(n, sample_size, replace=False)
        kde = gaussian_kde(np.vstack([x[indices], y[indices]]))
        density = kde(np.vstack([x, y]))
        marker_dict = dict(size=1, color=density, colorscale=colourscale)
    else:
        marker_dict = dict(size=1.25)

    return marker_dict
