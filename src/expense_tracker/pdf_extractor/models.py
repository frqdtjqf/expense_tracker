from dataclasses import dataclass


@dataclass
class OcrElement:
    index: int
    text: str
    confidence: float
    polygon: list[tuple[int, int]]
    box: dict[str, int] | None = None

    @property
    def min_x(self) -> int:
        return min(x for x, _ in self.polygon)

    @property
    def max_x(self) -> int:
        return max(x for x, _ in self.polygon)

    @property
    def min_y(self) -> int:
        return min(y for _, y in self.polygon)

    @property
    def max_y(self) -> int:
        return max(y for _, y in self.polygon)

    @property
    def width(self) -> int:
        return self.max_x - self.min_x

    @property
    def height(self) -> int:
        return self.max_y - self.min_y

    @property
    def center_x(self) -> float:
        return (self.min_x + self.max_x) / 2

    @property
    def center_y(self) -> float:
        return (self.min_y + self.max_y) / 2
