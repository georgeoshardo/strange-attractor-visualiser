from dataclasses import dataclass

import numpy as np

from ..core.density import binned_density
from ..core.display import (
    DISPLAY_MODE_LINES,
    DISPLAY_MODE_LINES_POINTS,
    DISPLAY_MODE_POINTS,
)


@dataclass(frozen=True)
class DisplaySettings:
    display_mode: str = DISPLAY_MODE_POINTS
    point_budget: int | None = 30_000
    use_density: bool = False
    line_interpolation: int = 1


@dataclass(frozen=True)
class ProjectionPayload:
    x: np.ndarray
    y: np.ndarray
    show_points: bool
    show_lines: bool
    line_colors: np.ndarray | None = None
    line_x: np.ndarray | None = None
    line_y: np.ndarray | None = None


@dataclass(frozen=True)
class RenderPayload:
    positions: np.ndarray
    line_positions: np.ndarray
    show_points: bool
    show_lines: bool
    point_colors: np.ndarray | None
    line_colors: np.ndarray | tuple[float, float, float, float]
    projections: dict[str, ProjectionPayload]


def downsample_solution(solution: np.ndarray, point_budget: int | None) -> np.ndarray:
    if point_budget is None or len(solution) <= point_budget:
        return solution
    indices = np.linspace(0, len(solution) - 1, point_budget, dtype=int)
    return solution[indices]


def preview_display_settings(
    settings: DisplaySettings, point_budget: int, line_interpolation: int = 1
) -> DisplaySettings:
    if settings.point_budget is None:
        preview_budget = point_budget
    else:
        preview_budget = min(settings.point_budget, point_budget)
    return DisplaySettings(
        display_mode=settings.display_mode,
        point_budget=preview_budget,
        use_density=settings.use_density,
        line_interpolation=line_interpolation,
    )


def _mode_visibility(display_mode: str) -> tuple[bool, bool]:
    if display_mode == DISPLAY_MODE_LINES:
        return False, True
    if display_mode == DISPLAY_MODE_LINES_POINTS:
        return True, True
    return True, False


def _constant_point_colours(n_points: int) -> np.ndarray:
    colours = np.ones((n_points, 4), dtype=np.float32)
    colours[:, :3] = 0.93
    colours[:, 3] = 0.74
    return colours


def _density_colours(x: np.ndarray, y: np.ndarray, alpha: float = 0.78) -> np.ndarray:
    if len(x) < 3:
        colours = _constant_point_colours(len(x))
        colours[:, 3] = alpha
        return colours

    density = binned_density(x, y)
    colours = np.empty((len(x), 4), dtype=np.float32)
    colours[:, 0] = density
    colours[:, 1] = 0.35 + 0.55 * (1.0 - density)
    colours[:, 2] = 1.0 - density
    colours[:, 3] = alpha
    return colours


def _density_position_colours(positions: np.ndarray, alpha: float = 0.78) -> np.ndarray:
    return _density_colours(positions[:, 0], positions[:, 1], alpha=alpha)


def _catmull_rom_interpolate(values: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1 or len(values) < 2:
        return values

    values = np.asarray(values, dtype=np.float32)
    segment_count = len(values) - 1
    indices = np.arange(segment_count)
    p0 = values[np.maximum(indices - 1, 0)]
    p1 = values[indices]
    p2 = values[indices + 1]
    p3 = values[np.minimum(indices + 2, len(values) - 1)]

    segments = []
    for step in range(factor):
        t = step / factor
        t2 = t * t
        t3 = t2 * t
        segments.append(
            0.5
            * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
        )

    interpolated = np.stack(segments, axis=1).reshape(-1, values.shape[1])
    return np.vstack([interpolated, values[-1:]])


def _linear_interpolate(values: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1 or len(values) < 2:
        return values

    values = np.asarray(values, dtype=np.float32)
    start = values[:-1]
    end = values[1:]
    segments = []
    for step in range(factor):
        t = step / factor
        segments.append(start * (1.0 - t) + end * t)

    interpolated = np.stack(segments, axis=1).reshape(-1, values.shape[1])
    return np.vstack([interpolated, values[-1:]])


def build_render_payload(
    solution: np.ndarray, settings: DisplaySettings
) -> RenderPayload:
    sampled = downsample_solution(solution, settings.point_budget)
    positions = np.asarray(sampled, dtype=np.float32)
    show_points, show_lines = _mode_visibility(settings.display_mode)

    density_colors = None
    if settings.use_density and (show_points or show_lines):
        density_colors = _density_position_colours(positions)

    point_colors = None
    if show_points:
        if density_colors is not None:
            point_colors = density_colors
        else:
            point_colors = _constant_point_colours(len(positions))

    if show_lines and density_colors is not None:
        line_colors = density_colors
    else:
        line_colors = (0.93, 0.93, 0.93, 0.38)

    line_positions = positions
    if show_lines and settings.line_interpolation > 1:
        line_positions = _catmull_rom_interpolate(
            positions,
            settings.line_interpolation,
        )
        if isinstance(line_colors, np.ndarray):
            line_colors = _linear_interpolate(
                line_colors,
                settings.line_interpolation,
            )

    projection_show_points = settings.display_mode != DISPLAY_MODE_LINES
    projection_show_lines = settings.display_mode in (
        DISPLAY_MODE_LINES,
        DISPLAY_MODE_LINES_POINTS,
    )
    projection_line_colors = {}
    if projection_show_lines and settings.use_density:
        projection_line_colors = {
            "x-y": density_colors,
            "x-z": _density_colours(positions[:, 0], positions[:, 2]),
            "y-z": _density_colours(positions[:, 1], positions[:, 2]),
        }
    projections = {
        "x-y": ProjectionPayload(
            positions[:, 0],
            positions[:, 1],
            projection_show_points,
            projection_show_lines,
            projection_line_colors.get("x-y"),
            positions[:, 0],
            positions[:, 1],
        ),
        "x-z": ProjectionPayload(
            positions[:, 0],
            positions[:, 2],
            projection_show_points,
            projection_show_lines,
            projection_line_colors.get("x-z"),
            positions[:, 0],
            positions[:, 2],
        ),
        "y-z": ProjectionPayload(
            positions[:, 1],
            positions[:, 2],
            projection_show_points,
            projection_show_lines,
            projection_line_colors.get("y-z"),
            positions[:, 1],
            positions[:, 2],
        ),
    }

    return RenderPayload(
        positions=positions,
        line_positions=line_positions,
        show_points=show_points,
        show_lines=show_lines,
        point_colors=point_colors,
        line_colors=line_colors,
        projections=projections,
    )
