import math

import dearpygui.dearpygui as dpg
import numpy as np

from core.simulation import Simulation


class ProjectileSimulation(Simulation):
    name = "Projectile Motion"
    description = "2D launch with optional quadratic air drag."

    camera_scale = 18.0
    camera_center = (20.0, 10.0)

    presets = [
        ("Default (45°, no drag)",  {"v0": 20.0, "angle": 45.0, "mass": 1.0, "cd": 0.0,  "g": 9.81}),
        ("Steep launch",            {"v0": 25.0, "angle": 75.0, "mass": 1.0, "cd": 0.0,  "g": 9.81}),
        ("Flat launch",             {"v0": 30.0, "angle": 15.0, "mass": 1.0, "cd": 0.0,  "g": 9.81}),
        ("Moderate drag",           {"v0": 30.0, "angle": 45.0, "mass": 1.0, "cd": 0.2,  "g": 9.81}),
        ("Heavy drag",              {"v0": 30.0, "angle": 45.0, "mass": 1.0, "cd": 0.8,  "g": 9.81}),
        ("Heavy projectile + drag", {"v0": 30.0, "angle": 45.0, "mass": 5.0, "cd": 0.5,  "g": 9.81}),
        ("Moon gravity",            {"v0": 20.0, "angle": 45.0, "mass": 1.0, "cd": 0.0,  "g": 1.62}),
    ]

    def _define_params(self):
        self.params = {
            "v0":    {"label": "Initial speed  (m/s)",  "value": 20.0, "min": 1.0,  "max": 120.0, "step": 0.5},
            "angle": {"label": "Launch angle   (deg)",  "value": 45.0, "min": 0.0,  "max": 90.0,  "step": 1.0},
            "mass":  {"label": "Mass           (kg)",   "value": 1.0,  "min": 0.1,  "max": 20.0,  "step": 0.1},
            "cd":    {"label": "Drag coeff     (kg/m)", "value": 0.0,  "min": 0.0,  "max": 2.0,   "step": 0.01},
            "g":     {"label": "Gravity        (m/s²)", "value": 9.81, "min": 0.1,  "max": 25.0,  "step": 0.1},
        }
        self.overlays = {
            "velocity":      {"label": "Velocity vector",      "enabled": True},
            "acceleration":  {"label": "Acceleration vector",  "enabled": False},
            "trail":         {"label": "Trajectory trail",     "enabled": True},
            "equations":     {"label": "Show equations",       "enabled": True},
        }
        self._trail: list[tuple[float, float]] = []

    def reset(self):
        v0    = self.get_param("v0")
        theta = math.radians(self.get_param("angle"))
        self.state = np.array([
            0.0,
            0.0,
            v0 * math.cos(theta),
            v0 * math.sin(theta),
        ])
        self.t = 0.0
        self._trail = []
        self.recorder.clear()

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y, vx, vy = state
        m  = self.get_param("mass")
        cd = self.get_param("cd")
        g  = self.get_param("g")

        v = math.sqrt(vx * vx + vy * vy)

        # a_drag = (cd/m) * v²/v = (cd/m)*v
        drag_factor = (cd / m) * v if v > 1e-10 else 0.0

        ax = -drag_factor * vx
        ay = -g - drag_factor * vy

        return np.array([vx, vy, ax, ay])

    def update(self, dt: float):
        if self.paused:
            return
        x, y = self.state[0], self.state[1]
        if self.overlays["trail"]["enabled"]:
            self._trail.append((x, y))
            if len(self._trail) > 3000:
                self._trail.pop(0)

        super().update(dt)

        # stop when projectile hits the ground
        if self.state[1] < 0.0 and self.state[3] < 0.0:
            self.state[1] = 0.0
            self.paused = True

    def get_record_values(self) -> dict:
        x, y, vx, vy = self.state
        m  = self.get_param("mass")
        cd = self.get_param("cd")
        g  = self.get_param("g")
        v  = math.sqrt(vx * vx + vy * vy)
        drag_factor = (cd / m) * v if v > 1e-10 else 0.0
        ax = -drag_factor * vx
        ay = -g - drag_factor * vy
        ke = 0.5 * m * v * v
        pe = m * g * y
        return {
            "x": x, "y": y,
            "vx": vx, "vy": vy,
            "speed": v,
            "ax": ax, "ay": ay,
            "KE": ke, "PE": pe, "E_total": ke + pe,
        }

    def draw(self, draw_tag: str, cam) -> None:
        x, y, vx, vy = self.state
        sx, sy = cam.w2s(x, y)

        gx0, gy0 = cam.w2s(-200.0, 0.0)
        gx1, gy1 = cam.w2s( 200.0, 0.0)
        dpg.draw_line((gx0, gy0), (gx1, gy1),
                      color=(160, 160, 160, 200), thickness=1, parent=draw_tag)

        if self.overlays["trail"]["enabled"] and len(self._trail) > 1:
            pts = [cam.w2s(px, py) for px, py in self._trail]
            n = len(pts)
            for i in range(n - 1):
                alpha = int(40 + 180 * i / n)
                dpg.draw_line(pts[i], pts[i + 1],
                              color=(100, 200, 255, alpha), thickness=1,
                              parent=draw_tag)

        if self.overlays["velocity"]["enabled"]:
            VSCALE = 3.0  # 1 m/s => 3 px
            tip = cam.arrow_tip(x, y, vx, vy, VSCALE)
            dpg.draw_arrow(tip, (sx, sy),
                           color=(60, 220, 60, 230), thickness=2, size=6,
                           parent=draw_tag)
            v = math.sqrt(vx * vx + vy * vy)
            dpg.draw_text((tip[0] + 5, tip[1] - 14),
                          f"v = {v:.1f} m/s",
                          color=(60, 220, 60, 230), size=13, parent=draw_tag)

        if self.overlays["acceleration"]["enabled"]:
            m  = self.get_param("mass")
            cd = self.get_param("cd")
            g  = self.get_param("g")
            v  = math.sqrt(vx * vx + vy * vy)
            drag_factor = (cd / m) * v if v > 1e-10 else 0.0
            ax = -drag_factor * vx
            ay = -g - drag_factor * vy
            ASCALE = 1.5
            tip = cam.arrow_tip(x, y, ax, ay, ASCALE)
            dpg.draw_arrow(tip, (sx, sy),
                           color=(255, 100, 50, 230), thickness=2, size=6,
                           parent=draw_tag)
            a_mag = math.sqrt(ax * ax + ay * ay)
            dpg.draw_text((tip[0] + 5, tip[1] - 14),
                          f"a = {a_mag:.1f} m/s²",
                          color=(255, 100, 50, 230), size=13, parent=draw_tag)

        dpg.draw_circle((sx, sy), 8,
                        color=(255, 220, 50, 255), fill=(255, 220, 50, 200),
                        parent=draw_tag)

        dpg.draw_text((10, 10),
                      f"x = {x:.2f} m    y = {y:.2f} m    t = {self.t:.2f} s",
                      color=(200, 200, 200, 200), size=13, parent=draw_tag)

        if self.overlays["equations"]["enabled"]:
            fx = cam.canvas_w - 320
            dpg.draw_text((fx, 10), "Equations of motion",
                          color=(140, 170, 180, 200), size=12, parent=draw_tag)
            dpg.draw_text((fx, 26), "dx/dt = vx       dy/dt = vy",
                          color=(180, 220, 230, 220), size=12, parent=draw_tag)
            dpg.draw_text((fx, 42), "dvx/dt = -(cd/m)*v*vx",
                          color=(180, 220, 230, 220), size=12, parent=draw_tag)
            dpg.draw_text((fx, 58), "dvy/dt = -g - (cd/m)*v*vy",
                          color=(180, 220, 230, 220), size=12, parent=draw_tag)
            dpg.draw_text((fx, 76), "v = sqrt(vx² + vy²)",
                          color=(180, 220, 230, 220), size=12, parent=draw_tag)
            dpg.draw_text((fx, 98), "Energy",
                          color=(140, 170, 180, 200), size=12, parent=draw_tag)
            dpg.draw_text((fx, 114), "KE = (1/2) m v²        PE = m g y",
                          color=(180, 220, 230, 220), size=12, parent=draw_tag)
