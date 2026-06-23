import os
from typing import Literal

import streamlit as st
import streamlit.components.v1 as components

shape_types = Literal["circle", "square", "pill"]


def _resolve_slider_default(key: str | None, default_value: int | float):
    session_value = st.session_state.get(key)
    return default_value if session_value is None else session_value


def live_vertical_slider(
    label: str | None = None,
    key: str | None = None,
    height: int = 200,
    width: int = 20,
    thumb_shape: shape_types = "circle",
    step: int | float = 1,
    default_value: int | float = 0,
    min_value: int | float = 0,
    max_value: int | float = 10,
    track_color: str = "#E5E9F1",
    slider_color: str | tuple[str, ...] = "#FF4B4B",
    thumb_color: str = "#FF4B4B",
    value_always_visible: bool = False,
    show_marks: bool = False,
    slider_border_width: int = 0,
    slider_border_color: str = "#555555",
    slider_opacity: float = 1.0,
    value_font_size: int = 14,
    mark_font_size: int = 9,
):
    if thumb_shape not in {"circle", "square", "pill"}:
        raise ValueError("thumb_shape must be one of: circle, square, pill")

    if default_value < min_value:
        default_value = min_value

    resolved_default = _resolve_slider_default(key, default_value)
    if isinstance(slider_color, tuple):
        slider_color = ",".join(slider_color)

    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend")
    component_func = components.declare_component(
        "live_vertical_slider", path=build_dir
    )
    component_value = component_func(
        label=label,
        key=key,
        height=height,
        width=width,
        default_value=resolved_default,
        thumb_shape=thumb_shape,
        step=step,
        min_value=min_value,
        max_value=max_value,
        track_color=track_color,
        thumb_color=thumb_color,
        slider_color=slider_color,
        slider_opacity=slider_opacity,
        show_marks=show_marks,
        value_always_visible=value_always_visible,
        slider_border_width=slider_border_width,
        slider_border_color=slider_border_color,
        value_font_size=value_font_size,
        mark_font_size=mark_font_size,
        default=resolved_default,
    )

    return component_value if component_value is not None else resolved_default
