from collections import defaultdict


class Recorder:
    """Stores time-series data for every named variable a simulation emits"""

    def __init__(self, max_samples: int = 6000):
        self.max_samples = max_samples
        self._times: list[float] = []
        self._data: dict[str, list[float]] = defaultdict(list)

    def record(self, t: float, values: dict[str, float]) -> None:
        self._times.append(t)
        for key, val in values.items():
            self._data[key].append(float(val))

        if len(self._times) > self.max_samples:
            trim = len(self._times) - self.max_samples
            self._times = self._times[trim:]
            for key in self._data:
                self._data[key] = self._data[key][trim:]

    def times(self) -> list[float]:
        return list(self._times)

    def get(self, key: str) -> list[float]:
        return list(self._data.get(key, []))

    def keys(self) -> list[str]:
        return list(self._data.keys())

    def clear(self) -> None:
        self._times.clear()
        self._data.clear()

    @property
    def sample_count(self) -> int:
        return len(self._times)
