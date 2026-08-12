import math

import dearpygui.dearpygui as dpg
import numpy as np

from core.simulation import Simulation


class PendulumSimulation(Simulation):
    name = "Simple Pendulum"
    description = "Point mass on a massless rigid rod, with optional linear damping."

    camera_scale = 80.0
    camera_center = (0.0, 0.0)

    presets = [
        ("Default (45°)",            {"length": 1.5, "angle": 45.0,  "omega0": 0.0, "mass": 1.0, "g": 9.81, "damping": 0.0}),
        ("Small angle (linear-ish)", {"length": 1.5, "angle": 15.0,  "omega0": 0.0, "mass": 1.0, "g": 9.81, "damping": 0.0}),
        ("Large angle (nonlinear)",  {"length": 1.5, "angle": 160.0, "omega0": 0.0, "mass": 1.0, "g": 9.81, "damping": 0.0}),
        ("Released horizontal",      {"length": 1.5, "angle": 90.0,  "omega0": 0.0, "mass": 1.0, "g": 9.81, "damping": 0.0}),
        ("Underdamped",              {"length": 1.5, "angle": 60.0,  "omega0": 0.0, "mass": 1.0, "g": 9.81, "damping": 0.3}),
        ("Heavily damped",           {"length": 1.5, "angle": 60.0,  "omega0": 0.0, "mass": 1.0, "g": 9.81, "damping": 1.5}),
        ("Long pendulum",            {"length": 4.0, "angle": 30.0,  "omega0": 0.0, "mass": 1.0, "g": 9.81, "damping": 0.0}),
        ("Moon gravity",             {"length": 1.5, "angle": 45.0,  "omega0": 0.0, "mass": 1.0, "g": 1.62, "damping": 0.0}),
    ]

    def _define_params(self):
        self.params = {
            "length":  {"label": "Length        (m)",     "value": 1.5,  "min": 0.2,   "max": 5.0,   "step": 0.05},
            "angle":   {"label": "Init angle    (deg)",   "value": 45.0, "min": 0.0,   "max": 170.0, "step": 1.0},
            "omega0":  {"label": "Init omega    (rad/s)", "value": 0.0,  "min": -10.0, "max": 10.0,  "step": 0.05},
            "mass":    {"label": "Mass          (kg)",    "value": 1.0,  "min": 0.1,   "max": 20.0,  "step": 0.1},
            "g":       {"label": "Gravity       (m/s²)",  "value": 9.81, "min": 0.1,   "max": 25.0,  "step": 0.1},
            "damping": {"label": "Damping       (kg/s)",  "value": 0.0,  "min": 0.0,   "max": 2.0,   "step": 0.01},
        }
        self.overlays = {
            "velocity":     {"label": "Velocity vector",     "enabled": True},
            "acceleration": {"label": "Acceleration vector", "enabled": False},
            "forces":       {"label": "Gravity + tension",   "enabled": False},
            "trail":        {"label": "Bob trail",           "enabled": True},
            "equations":    {"label": "Show equations",      "enabled": True},
        }
        self._trail: list[tuple[float, float]] = []

    def reset(self):
        theta0 = math.radians(self.get_param("angle"))
        omega0 = self.get_param("omega0")
        self.state = np.array([theta0, omega0])
        self.t = 0.0
        self._trail = []
        self.recorder.clear()

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        theta, omega = state
        L = self.get_param("length")
        m = self.get_param("mass")
        g = self.get_param("g")
        c = self.get_param("damping")
        # I*theta_ddot = -m*g*L*sin(theta) - c*L^2*omega   with I = m*L^2
        # so   theta_ddot = -(g/L)*sin(theta) - (c/m)*omega
        domega = -(g / L) * math.sin(theta) - (c / m) * omega
        return np.array([omega, domega])

    def update(self, dt: float):
        if self.paused:
            return
        theta, _ = self.state
        L = self.get_param("length")
        if self.overlays["trail"]["enabled"]:
            self._trail.append((L * math.sin(theta), -L * math.cos(theta)))
            if len(self._trail) > 2000:
                self._trail.pop(0)
        super().update(dt)

    def get_record_values(self) -> dict:
        theta, omega = self.state
        L = self.get_param("length")
        m = self.get_param("mass")
        g = self.get_param("g")
        v = L * abs(omega)
        ke = 0.5 * m * (L * omega) ** 2
        pe = m * g * L * (1.0 - math.cos(theta))
        return {
            "theta_deg": math.degrees(theta),
            "omega": omega,
            "speed": v,
            "KE": ke, "PE": pe, "E_total": ke + pe,
        }

    def draw(self, draw_tag: str, cam) -> None:
        theta, omega = self.state
        L = self.get_param("length")
        m = self.get_param("mass")
        g = self.get_param("g")

        px_p, py_p = cam.w2s(0.0, 0.0)
        bx, by = L * math.sin(theta), -L * math.cos(theta)
        px_b, py_b = cam.w2s(bx, by)

        dpg.draw_circle((px_p, py_p), cam.length(L),
                        color=(80, 80, 80, 110), thickness=1, parent=draw_tag)

        if self.overlays["trail"]["enabled"] and len(self._trail) > 1:
            pts = [cam.w2s(tx, ty) for tx, ty in self._trail]
            n = len(pts)
            for i in range(n - 1):
                alpha = int(40 + 180 * i / n)
                dpg.draw_line(pts[i], pts[i + 1],
                              color=(100, 200, 255, alpha), thickness=1,
                              parent=draw_tag)

        dpg.draw_line((px_p, py_p), (px_b, py_b),
                      color=(200, 200, 200, 230), thickness=2, parent=draw_tag)

        dpg.draw_circle((px_p, py_p), 4,
                        color=(220, 220, 220, 255), fill=(60, 60, 60, 255),
                        parent=draw_tag)

        bob_r = max(6, int(4 + 3 * math.sqrt(m)))
        dpg.draw_circle((px_b, py_b), bob_r,
                        color=(255, 220, 50, 255), fill=(255, 220, 50, 200),
                        parent=draw_tag)

        if self.overlays["velocity"]["enabled"]:
            vx = L * omega * math.cos(theta)
            vy = L * omega * math.sin(theta)
            VSCALE = 3.0  # 1 m/s => 3 px
            tip = cam.arrow_tip(bx, by, vx, vy, VSCALE)
            dpg.draw_arrow(tip, (px_b, py_b),
                           color=(60, 220, 60, 230), thickness=2, size=6,
                           parent=draw_tag)
            v = math.sqrt(vx * vx + vy * vy)
            dpg.draw_text((tip[0] + 5, tip[1] - 14),
                          f"v = {v:.1f} m/s",
                          color=(60, 220, 60, 230), size=13, parent=draw_tag)

        if self.overlays["acceleration"]["enabled"]:
            # a_tangential = -g sin(theta) along tangent (cos theta, sin theta)
            # a_centripetal = L*omega^2 along radial (-sin theta, cos theta)
            ax = -math.sin(theta) * (g * math.cos(theta) + L * omega * omega)
            ay = L * omega * omega * math.cos(theta) - g * math.sin(theta) ** 2
            ASCALE = 2.0
            tip = cam.arrow_tip(bx, by, ax, ay, ASCALE)
            dpg.draw_arrow(tip, (px_b, py_b),
                           color=(255, 100, 50, 230), thickness=2, size=6,
                           parent=draw_tag)
            a_mag = math.sqrt(ax * ax + ay * ay)
            dpg.draw_text((tip[0] + 5, tip[1] - 14),
                          f"a = {a_mag:.1f} m/s²",
                          color=(255, 100, 50, 230), size=13, parent=draw_tag)

        if self.overlays["forces"]["enabled"]:
            FSCALE = 8.0  # 1 N => ~8 px

            tip = cam.arrow_tip(bx, by, 0.0, -m * g, FSCALE)
            dpg.draw_arrow(tip, (px_b, py_b),
                           color=(220, 80, 220, 230), thickness=2, size=6,
                           parent=draw_tag)
            dpg.draw_text((tip[0] + 5, tip[1] - 14),
                          f"mg = {m*g:.1f} N",
                          color=(220, 80, 220, 230), size=13, parent=draw_tag)

            # T = m*(g*cos(theta) + L*omega^2), along rod toward pivot
            T = m * (g * math.cos(theta) + L * omega * omega)
            tip = cam.arrow_tip(bx, by, -math.sin(theta) * T, math.cos(theta) * T, FSCALE)
            dpg.draw_arrow(tip, (px_b, py_b),
                           color=(80, 180, 220, 230), thickness=2, size=6,
                           parent=draw_tag)
            dpg.draw_text((tip[0] + 5, tip[1] - 14),
                          f"T = {T:.1f} N",
                          color=(80, 180, 220, 230), size=13, parent=draw_tag)

        ke = 0.5 * m * (L * omega) ** 2
        pe = m * g * L * (1.0 - math.cos(theta))
        dpg.draw_text((10, 10),
                      f"theta = {math.degrees(theta):.1f}°    omega = {omega:.2f} rad/s    t = {self.t:.2f} s",
                      color=(200, 200, 200, 200), size=13, parent=draw_tag)
        dpg.draw_text((10, 28),
                      f"E = {ke+pe:.3f} J   (KE = {ke:.3f},  PE = {pe:.3f})",
                      color=(180, 180, 180, 180), size=12, parent=draw_tag)

        if self.overlays["equations"]["enabled"]:
            fx = cam.canvas_w - 320
            T_small = 2.0 * math.pi * math.sqrt(L / g) if g > 0 else 0.0
            dpg.draw_text((fx, 10), "Equation of motion",
                          color=(140, 170, 180, 200), size=12, parent=draw_tag)
            dpg.draw_text((fx, 26), "d2theta/dt2 = -(g/L)*sin(theta) - (c/m)*omega",
                          color=(180, 220, 230, 220), size=12, parent=draw_tag)
            dpg.draw_text((fx, 48), "Energy",
                          color=(140, 170, 180, 200), size=12, parent=draw_tag)
            dpg.draw_text((fx, 64), "KE = (1/2) m (L omega)²",
                          color=(180, 220, 230, 220), size=12, parent=draw_tag)
            dpg.draw_text((fx, 80), "PE = m g L (1 - cos theta)",
                          color=(180, 220, 230, 220), size=12, parent=draw_tag)
            dpg.draw_text((fx, 102), "Small-angle period",
                          color=(140, 170, 180, 200), size=12, parent=draw_tag)
            dpg.draw_text((fx, 118), f"T = 2 pi sqrt(L/g) = {T_small:.3f} s",
                          color=(180, 220, 230, 220), size=12, parent=draw_tag)
