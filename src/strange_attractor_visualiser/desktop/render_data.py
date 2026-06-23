from dataclasses import dataclass

import numpy as np
from scipy.stats import gaussian_kde

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


@dataclass(frozen=True)
class ProjectionPayload:
    x: np.ndarray
    y: np.ndarray
    show_points: bool
    show_lines: bool


@dataclass(frozen=True)
class RenderPayload:
    positions: np.ndarray
    show_points: bool
    show_lines: bool
    point_colors: np.ndarray | None
    line_color: tuple[float, float, float, float]
    projections: dict[str, ProjectionPayload]


def downsample_solution(solution: np.ndarray, point_budget: int | None) -> np.ndarray:
    if point_budget is None or len(solution) <= point_budget:
        return solution
    indices = np.linspace(0, len(solution) - 1, point_budget, dtype=int)
    return solution[indices]


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


def _density_point_colours(positions: np.ndarray) -> np.ndarray:
    if len(positions) < 3:
        return _constant_point_colours(len(positions))

    sample_size = min(1000, len(positions))
    indices = np.linspace(0, len(positions) - 1, sample_size, dtype=int)
    kde = gaussian_kde(np.vstack([positions[indices, 0], positions[indices, 1]]))
    density = kde(np.vstack([positions[:, 0], positions[:, 1]]))
    density = density - np.min(density)
    max_density = float(np.max(density))
    if max_density > 0:
        density = density / max_density

    colours = np.empty((len(positions), 4), dtype=np.float32)
    colours[:, 0] = density
    colours[:, 1] = 0.35 + 0.55 * (1.0 - density)
    colours[:, 2] = 1.0 - density
    colours[:, 3] = 0.78
    return colours


def build_render_payload(
    solution: np.ndarray, settings: DisplaySettings
) -> RenderPayload:
    sampled = downsample_solution(solution, settings.point_budget)
    positions = np.asarray(sampled, dtype=np.float32)
    show_points, show_lines = _mode_visibility(settings.display_mode)

    point_colors = None
    if show_points:
        if settings.use_density:
            point_colors = _density_point_colours(positions)
        else:
            point_colors = _constant_point_colours(len(positions))

    projection_show_points = settings.display_mode != DISPLAY_MODE_LINES
    projection_show_lines = settings.display_mode in (
        DISPLAY_MODE_LINES,
        DISPLAY_MODE_LINES_POINTS,
    )
    projections = {
        "x-y": ProjectionPayload(
            positions[:, 0], positions[:, 1], projection_show_points, projection_show_lines
        ),
        "x-z": ProjectionPayload(
            positions[:, 0], positions[:, 2], projection_show_points, projection_show_lines
        ),
        "y-z": ProjectionPayload(
            positions[:, 1], positions[:, 2], projection_show_points, projection_show_lines
        ),
    }

    return RenderPayload(
        positions=positions,
        show_points=show_points,
        show_lines=show_lines,
        point_colors=point_colors,
        line_color=(0.93, 0.93, 0.93, 0.38),
        projections=projections,
    )
