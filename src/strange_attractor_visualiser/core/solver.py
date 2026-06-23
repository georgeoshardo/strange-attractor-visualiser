from dataclasses import dataclass

import numpy as np
from scipy.integrate import odeint

from .models import AttractorConfig

SOLVER_LSODA = "LSODA"
SOLVER_RK4 = "RK4"


@dataclass(frozen=True)
class SolverSettings:
    method: str = SOLVER_LSODA
    lsoda_rtol: float | None = None
    lsoda_atol: float | None = None

    def cache_key(self) -> tuple[str, float | None, float | None]:
        if self.method == SOLVER_RK4:
            return self.method, None, None
        return self.method, self.lsoda_rtol, self.lsoda_atol


@dataclass(frozen=True)
class AdaptiveHorizonSettings:
    enabled: bool = False
    burn_in_fraction: float = 0.1
    batch_steps: int = 5_000
    max_points: int = 60_000
    min_batches: int = 3
    stable_batches: int = 2
    bounds_tolerance: float = 0.02
    coverage_tolerance: float = 0.01
    coverage_bins: int = 32

    def cache_key(self) -> tuple:
        return (
            self.enabled,
            round(self.burn_in_fraction, 4),
            self.batch_steps,
            self.max_points,
            self.min_batches,
            self.stable_batches,
            round(self.bounds_tolerance, 4),
            round(self.coverage_tolerance, 4),
            self.coverage_bins,
        )


def solve_attractor(
    config: AttractorConfig,
    param_values: dict[str, float],
    n_steps: int | None = None,
    solver_settings: SolverSettings | None = None,
    adaptive_settings: AdaptiveHorizonSettings | None = None,
) -> np.ndarray:
    solver_settings = solver_settings or SolverSettings()
    adaptive_settings = adaptive_settings or AdaptiveHorizonSettings()
    if adaptive_settings.enabled:
        return _solve_adaptive(
            config,
            param_values,
            n_steps,
            solver_settings,
            adaptive_settings,
        )

    if solver_settings.method == SOLVER_RK4:
        return _solve_rk4(config, param_values, n_steps)
    if solver_settings.method != SOLVER_LSODA:
        raise ValueError(f"Unknown solver: {solver_settings.method}")

    t_def = config.time_defaults
    t = np.linspace(t_def["t_min"], t_def["t_max"], n_steps or t_def["n"])

    args = tuple(param_values[p.name] for p in config.params)
    odeint_kwargs = {}
    if solver_settings.lsoda_rtol is not None:
        odeint_kwargs["rtol"] = solver_settings.lsoda_rtol
    if solver_settings.lsoda_atol is not None:
        odeint_kwargs["atol"] = solver_settings.lsoda_atol

    solution = odeint(
        config.equation,
        config.initial_conditions,
        t,
        args=args,
        **odeint_kwargs,
    )

    return solution


def get_default_params(config: AttractorConfig) -> dict[str, float]:
    return {p.name: p.default for p in config.params}


def _solve_rk4(
    config: AttractorConfig,
    param_values: dict[str, float],
    n_steps: int | None,
) -> np.ndarray:
    t_def = config.time_defaults
    n = n_steps or t_def["n"]
    if n < 2:
        raise ValueError("RK4 requires at least two time steps")
    t_min = t_def["t_min"]
    t_max = t_def["t_max"]
    h = (t_max - t_min) / (n - 1)

    solution = np.empty((n, 3), dtype=float)
    x, y, z = (float(value) for value in config.initial_conditions)
    solution[0] = (x, y, z)
    args = tuple(param_values[p.name] for p in config.params)
    equation = config.equation
    t = t_min

    for index in range(1, n):
        k1x, k1y, k1z = equation((x, y, z), t, *args)
        k2x, k2y, k2z = equation(
            (
                x + 0.5 * h * k1x,
                y + 0.5 * h * k1y,
                z + 0.5 * h * k1z,
            ),
            t + 0.5 * h,
            *args,
        )
        k3x, k3y, k3z = equation(
            (
                x + 0.5 * h * k2x,
                y + 0.5 * h * k2y,
                z + 0.5 * h * k2z,
            ),
            t + 0.5 * h,
            *args,
        )
        k4x, k4y, k4z = equation(
            (
                x + h * k3x,
                y + h * k3y,
                z + h * k3z,
            ),
            t + h,
            *args,
        )
        x += (h / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        y += (h / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
        z += (h / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z)
        t += h
        solution[index] = (x, y, z)

    return solution


def _solve_adaptive(
    config: AttractorConfig,
    param_values: dict[str, float],
    n_steps: int | None,
    solver_settings: SolverSettings,
    adaptive_settings: AdaptiveHorizonSettings,
) -> np.ndarray:
    if solver_settings.method not in (SOLVER_LSODA, SOLVER_RK4):
        raise ValueError(f"Unknown solver: {solver_settings.method}")

    t_def = config.time_defaults
    base_steps = n_steps or t_def["n"]
    if base_steps < 2:
        raise ValueError("Adaptive horizon requires at least two base time steps")

    h = (t_def["t_max"] - t_def["t_min"]) / (base_steps - 1)
    t = float(t_def["t_min"])
    state = np.asarray(config.initial_conditions, dtype=float)
    args = tuple(param_values[p.name] for p in config.params)

    burn_in_steps = max(0, int(base_steps * adaptive_settings.burn_in_fraction))
    if burn_in_steps:
        burn_in = _solve_segment(
            config,
            state,
            args,
            t,
            h,
            burn_in_steps,
            solver_settings,
        )
        state = burn_in[-1]
        t += h * burn_in_steps

    segments = []
    previous_bounds = None
    previous_coverage = None
    stable_count = 0
    batch_count = 0
    point_count = 0

    while point_count < adaptive_settings.max_points:
        remaining = adaptive_settings.max_points - point_count
        steps = min(adaptive_settings.batch_steps, remaining)
        if steps <= 0:
            break

        batch = _solve_segment(
            config,
            state,
            args,
            t,
            h,
            steps,
            solver_settings,
        )
        state = batch[-1]
        t += h * steps
        new_points = batch[1:]
        finite_new_points = new_points[np.isfinite(new_points).all(axis=1)]
        if len(finite_new_points) == 0:
            break
        segments.append(finite_new_points)
        point_count += len(finite_new_points)
        batch_count += 1

        current = np.vstack(segments)
        if len(current) < 3:
            continue

        bounds = _robust_bounds(current)
        coverage = _occupied_bins(current, bounds, adaptive_settings.coverage_bins)
        if previous_bounds is not None and previous_coverage is not None:
            bounds_change = _bounds_change(previous_bounds, bounds)
            coverage_growth = max(
                0.0,
                (coverage - previous_coverage) / max(float(previous_coverage), 1.0),
            )
            stable = (
                bounds_change <= adaptive_settings.bounds_tolerance
                and coverage_growth <= adaptive_settings.coverage_tolerance
            )
            stable_count = stable_count + 1 if stable else 0
            if (
                batch_count >= adaptive_settings.min_batches
                and stable_count >= adaptive_settings.stable_batches
            ):
                break

        previous_bounds = bounds
        previous_coverage = coverage
        if len(finite_new_points) < len(new_points):
            break

    if not segments:
        return state.reshape(1, 3)
    return np.vstack(segments)


def _solve_segment(
    config: AttractorConfig,
    initial_state: np.ndarray,
    args: tuple[float, ...],
    t_start: float,
    h: float,
    steps: int,
    solver_settings: SolverSettings,
) -> np.ndarray:
    if solver_settings.method == SOLVER_RK4:
        return _solve_rk4_segment(
            config.equation,
            initial_state,
            args,
            t_start,
            h,
            steps,
        )

    t = np.linspace(t_start, t_start + h * steps, steps + 1)
    odeint_kwargs = {}
    if solver_settings.lsoda_rtol is not None:
        odeint_kwargs["rtol"] = solver_settings.lsoda_rtol
    if solver_settings.lsoda_atol is not None:
        odeint_kwargs["atol"] = solver_settings.lsoda_atol
    return odeint(config.equation, initial_state, t, args=args, **odeint_kwargs)


def _solve_rk4_segment(equation, initial_state, args, t_start, h, steps):
    solution = np.empty((steps + 1, 3), dtype=float)
    x, y, z = (float(value) for value in initial_state)
    solution[0] = (x, y, z)
    t = t_start

    for index in range(1, steps + 1):
        k1x, k1y, k1z = equation((x, y, z), t, *args)
        k2x, k2y, k2z = equation(
            (
                x + 0.5 * h * k1x,
                y + 0.5 * h * k1y,
                z + 0.5 * h * k1z,
            ),
            t + 0.5 * h,
            *args,
        )
        k3x, k3y, k3z = equation(
            (
                x + 0.5 * h * k2x,
                y + 0.5 * h * k2y,
                z + 0.5 * h * k2z,
            ),
            t + 0.5 * h,
            *args,
        )
        k4x, k4y, k4z = equation(
            (
                x + h * k3x,
                y + h * k3y,
                z + h * k3z,
            ),
            t + h,
            *args,
        )
        x += (h / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        y += (h / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
        z += (h / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z)
        t += h
        solution[index] = (x, y, z)

    return solution


def _robust_bounds(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.percentile(points, 1.0, axis=0),
        np.percentile(points, 99.0, axis=0),
    )


def _bounds_change(
    previous: tuple[np.ndarray, np.ndarray],
    current: tuple[np.ndarray, np.ndarray],
) -> float:
    previous_min, previous_max = previous
    current_min, current_max = current
    previous_span = np.maximum(previous_max - previous_min, 1e-9)
    change = np.maximum(
        np.abs(current_min - previous_min),
        np.abs(current_max - previous_max),
    )
    return float(np.max(change / previous_span))


def _occupied_bins(
    points: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    bins: int,
) -> int:
    lower, upper = bounds
    span = np.maximum(upper - lower, 1e-9)
    normalised = np.clip((points - lower) / span, 0.0, 0.999999)
    indices = np.floor(normalised * bins).astype(np.int64)
    flat = np.ravel_multi_index(indices.T, (bins, bins, bins))
    return int(np.unique(flat).size)
