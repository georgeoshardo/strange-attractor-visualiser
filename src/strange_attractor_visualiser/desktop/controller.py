from dataclasses import dataclass, field

from .render_data import RenderPayload


@dataclass
class ControllerStatus:
    error: str | None = None
    timings: dict[str, float] = field(default_factory=dict)


class ResultCoordinator:
    def __init__(self):
        self.current_generation = 0
        self.last_payload: RenderPayload | None = None
        self.status = ControllerStatus()

    def next_generation(self) -> int:
        self.current_generation += 1
        return self.current_generation

    def accept_success(
        self,
        generation: int,
        payload: RenderPayload,
        timings: dict[str, float],
    ) -> bool:
        if generation != self.current_generation:
            return False

        self.last_payload = payload
        self.status = ControllerStatus(error=None, timings=dict(timings))
        return True

    def accept_error(self, generation: int, error: Exception) -> bool:
        if generation != self.current_generation:
            return False

        self.status = ControllerStatus(error=str(error), timings={})
        return True
