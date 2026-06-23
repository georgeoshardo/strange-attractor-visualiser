from dataclasses import dataclass, field

from ..core.display import DISPLAY_MODE_POINTS
from ..core.models import AttractorConfig
from ..core.solver import SolverSettings


def _decimal_places(step: float) -> int:
    step_text = f"{step:.12f}".rstrip("0")
    if "." not in step_text:
        return 0
    return len(step_text.split(".", 1)[1])


def parameter_cache_key(
    selected_name: str,
    config: AttractorConfig,
    param_values: dict[str, float],
    solver_settings: SolverSettings | None = None,
) -> tuple:
    items = []
    for param in config.params:
        decimals = _decimal_places(param.step)
        items.append((param.name, round(float(param_values[param.name]), decimals)))
    if solver_settings is None:
        return selected_name, tuple(items)
    return selected_name, tuple(items), ("solver", solver_settings.cache_key())


@dataclass
class DesktopState:
    selected_name: str
    param_values: dict[str, float]
    display_mode: str = DISPLAY_MODE_POINTS
    point_budget: int | None = 30_000
    use_density: bool = False
    animate: bool = False
    saved_values: list[dict] = field(default_factory=list)
