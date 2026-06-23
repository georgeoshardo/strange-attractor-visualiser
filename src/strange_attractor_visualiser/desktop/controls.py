from dataclasses import dataclass


@dataclass(frozen=True)
class FloatSliderSpec:
    min_value: float
    max_value: float
    step: float

    @property
    def minimum_tick(self) -> int:
        return 0

    @property
    def maximum_tick(self) -> int:
        return round((self.max_value - self.min_value) / self.step)

    def tick_to_value(self, tick: int) -> float:
        clamped = min(max(tick, self.minimum_tick), self.maximum_tick)
        value = self.min_value + clamped * self.step
        return round(value, self.decimal_places)

    def value_to_tick(self, value: float) -> int:
        tick = round((value - self.min_value) / self.step)
        return min(max(tick, self.minimum_tick), self.maximum_tick)

    @property
    def decimal_places(self) -> int:
        step_text = f"{self.step:.12f}".rstrip("0")
        if "." not in step_text:
            return 0
        return len(step_text.split(".", 1)[1])
