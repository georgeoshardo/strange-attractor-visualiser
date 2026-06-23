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


def solve_attractor(
    config: AttractorConfig,
    param_values: dict[str, float],
    n_steps: int | None = None,
    solver_settings: SolverSettings | None = None,
) -> np.ndarray:
    solver_settings = solver_settings or SolverSettings()
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
