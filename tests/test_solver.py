import numpy as np
import pytest

from strange_attractor_visualiser.attractors.registry import ATTRACTORS
from strange_attractor_visualiser.core.solver import (
    SOLVER_LSODA,
    SOLVER_RK4,
    SolverSettings,
    get_default_params,
    solve_attractor,
)


@pytest.mark.parametrize("name,config", ATTRACTORS.items())
def test_solver_shape_and_finite_all_attractors(name, config):
    params = get_default_params(config)
    sol = solve_attractor(config, params)
    assert sol.shape == (config.time_defaults["n"], 3)
    assert np.isfinite(sol).all()


@pytest.mark.parametrize("name,config", ATTRACTORS.items())
def test_solver_starts_at_initial_conditions(name, config):
    params = get_default_params(config)
    sol = solve_attractor(config, params)
    assert np.allclose(sol[0], config.initial_conditions, atol=1e-6)


def test_solver_missing_param_raises_keyerror():
    config = ATTRACTORS["Lorenz"]
    params = get_default_params(config)
    params.pop("$b$")

    with pytest.raises(KeyError):
        solve_attractor(config, params)


def test_solver_accepts_lsoda_tolerances():
    config = ATTRACTORS["Lorenz"]
    params = get_default_params(config)

    sol = solve_attractor(
        config,
        params,
        solver_settings=SolverSettings(
            method=SOLVER_LSODA,
            lsoda_rtol=1e-5,
            lsoda_atol=1e-7,
        ),
    )

    assert sol.shape == (config.time_defaults["n"], 3)
    assert np.isfinite(sol).all()


def test_solver_supports_rk4_integrator():
    config = ATTRACTORS["Lorenz"]
    params = get_default_params(config)

    sol = solve_attractor(
        config,
        params,
        n_steps=1000,
        solver_settings=SolverSettings(method=SOLVER_RK4),
    )

    assert sol.shape == (1000, 3)
    assert np.isfinite(sol).all()
    assert np.allclose(sol[0], config.initial_conditions)


def test_solver_rejects_unknown_integrator():
    config = ATTRACTORS["Lorenz"]
    params = get_default_params(config)

    with pytest.raises(ValueError, match="Unknown solver"):
        solve_attractor(
            config,
            params,
            solver_settings=SolverSettings(method="bad"),
        )


def test_rk4_requires_at_least_two_steps():
    config = ATTRACTORS["Lorenz"]
    params = get_default_params(config)

    with pytest.raises(ValueError, match="at least two"):
        solve_attractor(
            config,
            params,
            n_steps=1,
            solver_settings=SolverSettings(method=SOLVER_RK4),
        )
