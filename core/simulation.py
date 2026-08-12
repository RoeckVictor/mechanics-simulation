from abc import ABC, abstractmethod

import numpy as np

from .integrators import INTEGRATORS
from .recorder import Recorder


class Simulation(ABC):

    name: str = "Unnamed"
    description: str = ""

    camera_scale: float = 40.0
    camera_center: tuple[float, float] = (0.0, 0.0)

    # Hard-coded named parameter snapshots. Each entry: (name, {param_key: value}).
    # The first preset should match the param defaults defined in _define_params().
    presets: list = []

    def __init__(self):
        self.t: float = 0.0
        self.paused: bool = True
        self.integrator_name: str = "RK4"
        self.recorder = Recorder()

        self.params: dict = {}
        self.overlays: dict = {}

        self._define_params()
        self.reset()

    @abstractmethod
    def _define_params(self) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        pass

    @abstractmethod
    def draw(self, draw_tag: str, cam) -> None:
        pass

    def get_record_values(self) -> dict[str, float]:
        return {}

    def update(self, dt: float) -> None:
        if self.paused or dt <= 0:
            return
        integrate = INTEGRATORS[self.integrator_name]
        self.state = integrate(self.state, self.t, dt, self.derivatives)
        self.t += dt
        vals = self.get_record_values()
        if vals:
            self.recorder.record(self.t, vals)

    def get_param(self, key: str) -> float:
        return self.params[key]["value"]

    def set_param(self, key: str, value: float) -> None:
        p = self.params[key]
        self.params[key]["value"] = max(p["min"], min(p["max"], float(value)))
