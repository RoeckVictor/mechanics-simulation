import numpy as np


def euler(state: np.ndarray, t: float, dt: float, f) -> np.ndarray:
    return state + f(state, t) * dt


def rk4(state: np.ndarray, t: float, dt: float, f) -> np.ndarray:
    k1 = f(state,                   t)
    k2 = f(state + k1 * (dt * 0.5), t + dt * 0.5)
    k3 = f(state + k2 * (dt * 0.5), t + dt * 0.5)
    k4 = f(state + k3 * dt,         t + dt)
    return state + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * (dt / 6.0)


INTEGRATORS = {
    "Euler": euler,
    "RK4":   rk4,
}
