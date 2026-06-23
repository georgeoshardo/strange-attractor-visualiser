import numpy as np
from scipy.integrate import odeint

from .models import AttractorConfig


def solve_attractor(
    config: AttractorConfig,
    param_values: dict[str, float],
    n_steps: int | None = None,
) -> np.ndarray:
    t_def = config.time_defaults
    t = np.linspace(t_def["t_min"], t_def["t_max"], n_steps or t_def["n"])

    args = tuple(param_values[p.name] for p in config.params)

    solution = odeint(config.equation, config.initial_conditions, t, args=args)

    return solution


def get_default_params(config: AttractorConfig) -> dict[str, float]:
    return {p.name: p.default for p in config.params}
