import math
from array import array
from typing import Iterator, Tuple


class RaytracedOccupancyGrid:
    """Fixed-size log-odds grid with cropped OccupancyGrid output."""

    def __init__(
        self,
        resolution: float,
        width: int,
        height: int,
        origin_x: float,
        origin_y: float,
        hit_increment: int = 4,
        miss_decrement: int = 1,
        score_limit: int = 20,
        occupied_threshold: int = 3,
    ) -> None:
        if resolution <= 0.0:
            raise ValueError("resolution must be positive")
        if width <= 0 or height <= 0:
            raise ValueError("grid dimensions must be positive")
        if hit_increment <= 0 or miss_decrement <= 0:
            raise ValueError("hit and miss updates must be positive")
        if score_limit > 127 or occupied_threshold > score_limit:
            raise ValueError("occupancy scores exceed signed byte limits")

        self.resolution = resolution
        self.width = width
        self.height = height
        self.origin_x = origin_x
        self.origin_y = origin_y
        self._hit_increment = hit_increment
        self._miss_decrement = miss_decrement
        self._score_limit = score_limit
        self._occupied_threshold = occupied_threshold
        self._scores = array("b", [0]) * (width * height)
        self._observed = bytearray(width * height)
        self._min_x = width
        self._min_y = height
        self._max_x = -1
        self._max_y = -1

    @property
    def has_observations(self) -> bool:
        return self._max_x >= self._min_x and self._max_y >= self._min_y

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        return (
            math.floor((x - self.origin_x) / self.resolution),
            math.floor((y - self.origin_y) / self.resolution),
        )

    def update_ray(
        self,
        sensor_x: float,
        sensor_y: float,
        hit_x: float,
        hit_y: float,
    ) -> None:
        start_x, start_y = self.world_to_cell(sensor_x, sensor_y)
        end_x, end_y = self.world_to_cell(hit_x, hit_y)
        cells = self._bresenham(start_x, start_y, end_x, end_y)
        previous = None
        for cell in cells:
            if previous is not None:
                self._update_cell(previous[0], previous[1], -self._miss_decrement)
            previous = cell
        if previous is not None:
            self._update_cell(previous[0], previous[1], self._hit_increment)

    def cropped_data(self, padding_cells: int) -> Tuple[float, float, int, int, array]:
        if not self.has_observations:
            raise RuntimeError("cannot publish a grid without observations")
        padding = max(0, padding_cells)
        min_x = max(0, self._min_x - padding)
        min_y = max(0, self._min_y - padding)
        max_x = min(self.width - 1, self._max_x + padding)
        max_y = min(self.height - 1, self._max_y + padding)
        output_width = max_x - min_x + 1
        output_height = max_y - min_y + 1
        data = array("b")

        for y in range(min_y, max_y + 1):
            row_start = y * self.width
            for x in range(min_x, max_x + 1):
                index = row_start + x
                if not self._observed[index]:
                    data.append(-1)
                elif self._scores[index] >= self._occupied_threshold:
                    data.append(100)
                else:
                    data.append(0)

        return (
            self.origin_x + min_x * self.resolution,
            self.origin_y + min_y * self.resolution,
            output_width,
            output_height,
            data,
        )

    def _update_cell(self, x: int, y: int, delta: int) -> None:
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return
        index = y * self.width + x
        self._observed[index] = 1
        self._scores[index] = max(
            -self._score_limit,
            min(self._score_limit, self._scores[index] + delta),
        )
        self._min_x = min(self._min_x, x)
        self._min_y = min(self._min_y, y)
        self._max_x = max(self._max_x, x)
        self._max_y = max(self._max_y, y)

    @staticmethod
    def _bresenham(
        start_x: int, start_y: int, end_x: int, end_y: int
    ) -> Iterator[Tuple[int, int]]:
        x = start_x
        y = start_y
        delta_x = abs(end_x - start_x)
        step_x = 1 if start_x < end_x else -1
        delta_y = -abs(end_y - start_y)
        step_y = 1 if start_y < end_y else -1
        error = delta_x + delta_y

        while True:
            yield x, y
            if x == end_x and y == end_y:
                return
            doubled_error = 2 * error
            if doubled_error >= delta_y:
                error += delta_y
                x += step_x
            if doubled_error <= delta_x:
                error += delta_x
                y += step_y
