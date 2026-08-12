import math

import dearpygui.dearpygui as dpg
import numpy as np

from core.simulation import Simulation


class DoublePendulumSimulation(Simulation):
    name = "Double Pendulum"
    description = "Two coupled rigid rods with point masses. Chaotic for moderate energies."

    camera_scale = 70.0
    camera_center = (0.0, -0.8)

    presets = [
        ("Default chaos",         {"length1": 1.5, "length2": 1.5, "mass1": 1.0, "mass2": 1.0, "angle1":  90.0, "angle2":  90.000, "omega1": 0.0, "omega2": 0.0, "g": 9.81}),
        ("Chaos demo (+0.001°)",  {"length1": 1.5, "length2": 1.5, "mass1": 1.0, "mass2": 1.0, "angle1":  90.0, "angle2":  90.001, "omega1": 0.0, "omega2": 0.0, "g": 9.81}),
        ("Small angle (periodic)",{"length1": 1.5, "length2": 1.5, "mass1": 1.0, "mass2": 1.0, "angle1":  10.0, "angle2":  10.0,   "omega1": 0.0, "omega2": 0.0, "g": 9.81}),
        ("Heavy lower bob",       {"length1": 1.5, "length2": 1.5, "mass1": 1.0, "mass2": 5.0, "angle1":  90.0, "angle2":  90.0,   "omega1": 0.0, "omega2": 0.0, "g": 9.81}),
        ("Heavy upper bob",       {"length1": 1.5, "length2": 1.5, "mass1": 5.0, "mass2": 1.0, "angle1":  90.0, "angle2":  90.0,   "omega1": 0.0, "omega2": 0.0, "g": 9.81}),
        ("High-energy swing",     {"length1": 1.5, "length2": 1.5, "mass1": 1.0, "mass2": 1.0, "angle1": 180.0, "angle2":   0.0,   "omega1": 5.0, "omega2": 0.0, "g": 9.81}),
        ("Near-inverted release", {"length1": 1.5, "length2": 1.5, "mass1": 1.0, "mass2": 1.0, "angle1": 175.0, "angle2": 175.0,   "omega1": 0.0, "omega2": 0.0, "g": 9.81}),
    ]

    def _define_params(self):
        self.params = {
            "length1": {"label": "Length 1   (m)",     "value": 1.5,  "min": 0.3,    "max": 3.0,   "step": 0.05},
            "length2": {"label": "Length 2   (m)",     "value": 1.5,  "min": 0.3,    "max": 3.0,   "step": 0.05},
            "mass1":   {"label": "Mass 1     (kg)",    "value": 1.0,  "min": 0.1,    "max": 10.0,  "step": 0.1},
            "mass2":   {"label": "Mass 2     (kg)",    "value": 1.0,  "min": 0.1,    "max": 10.0,  "step": 0.1},
            "angle1":  {"label": "Init angle 1 (deg)", "value": 90.0, "min": -180.0, "max": 180.0, "step": 1.0},
            "angle2":  {"label": "Init angle 2 (deg)", "value": 90.0, "min": -180.0, "max": 180.0, "step": 1.0},
            "omega1":  {"label": "Init omega 1 (rad/s)", "value": 0.0, "min": -10.0, "max": 10.0,  "step": 0.05},
            "omega2":  {"label": "Init omega 2 (rad/s)", "value": 0.0, "min": -10.0, "max": 10.0,  "step": 0.05},
            "g":       {"label": "Gravity    (m/s²)",  "value": 9.81, "min": 0.1,    "max": 25.0,  "step": 0.1},
        }
        self.overlays = {
            "velocity":  {"label": "Velocity vectors", "enabled": False},
            "trail2":    {"label": "Bob-2 trail",      "enabled": True},
            "trail1":    {"label": "Bob-1 trail",      "enabled": False},
            "equations": {"label": "Show equations",   "enabled": True},
        }
        self._trail1: list[tuple[float, float]] = []
        self._trail2: list[tuple[float, float]] = []

    def reset(self):
        th1 = math.radians(self.get_param("angle1"))
        th2 = math.radians(self.get_param("angle2"))
        om1 = self.get_param("omega1")
        om2 = self.get_param("omega2")
        self.state = np.array([th1, th2, om1, om2])
        self.t = 0.0
        self._trail1 = []
        self._trail2 = []
        self.recorder.clear()

    def derivatives(self, state, t):
        th1, th2, om1, om2 = state
        L1 = self.get_param("length1")
        L2 = self.get_param("length2")
        m1 = self.get_param("mass1")
        m2 = self.get_param("mass2")
        g  = self.get_param("g")
        # Standard double-pendulum Lagrangian equations (point masses, massless rods).
        # delta = th1 - th2; denominator is common to both accelerations.
        delta = th1 - th2
        s_d = math.sin(delta)
        c_d = math.cos(delta)
        den = 2.0 * m1 + m2 - m2 * math.cos(2.0 * delta)

        a1 = (
            -g * (2.0 * m1 + m2) * math.sin(th1)
            - m2 * g * math.sin(th1 - 2.0 * th2)
            - 2.0 * s_d * m2 * (om2 * om2 * L2 + om1 * om1 * L1 * c_d)
        ) / (L1 * den)

        a2 = (
            2.0 * s_d * (
                om1 * om1 * L1 * (m1 + m2)
                + g * (m1 + m2) * math.cos(th1)
                + om2 * om2 * L2 * m2 * c_d
            )
        ) / (L2 * den)

        return np.array([om1, om2, a1, a2])

    def _positions(self):
        th1, th2 = self.state[0], self.state[1]
        L1 = self.get_param("length1")
        L2 = self.get_param("length2")
        x1 =  L1 * math.sin(th1)
        y1 = -L1 * math.cos(th1)
        x2 = x1 + L2 * math.sin(th2)
        y2 = y1 - L2 * math.cos(th2)
        return (x1, y1), (x2, y2)

    def _velocities(self):
        th1, th2, om1, om2 = self.state
        L1 = self.get_param("length1")
        L2 = self.get_param("length2")
        v1x = L1 * math.cos(th1) * om1
        v1y = L1 * math.sin(th1) * om1
        v2x = v1x + L2 * math.cos(th2) * om2
        v2y = v1y + L2 * math.sin(th2) * om2
        return (v1x, v1y), (v2x, v2y)

    def update(self, dt):
        if self.paused:
            return
        super().update(dt)
        (x1, y1), (x2, y2) = self._positions()
        if self.overlays["trail1"]["enabled"]:
            self._trail1.append((x1, y1))
            if len(self._trail1) > 3000:
                self._trail1.pop(0)
        if self.overlays["trail2"]["enabled"]:
            self._trail2.append((x2, y2))
            if len(self._trail2) > 3000:
                self._trail2.pop(0)

    def get_record_values(self) -> dict:
        th1, th2, om1, om2 = self.state
        L1 = self.get_param("length1")
        L2 = self.get_param("length2")
        m1 = self.get_param("mass1")
        m2 = self.get_param("mass2")
        g  = self.get_param("g")
        (x1, y1), (x2, y2) = self._positions()
        (v1x, v1y), (v2x, v2y) = self._velocities()
        v1_sq = v1x * v1x + v1y * v1y
        v2_sq = v2x * v2x + v2y * v2y
        ke = 0.5 * m1 * v1_sq + 0.5 * m2 * v2_sq
        # PE referenced so it's zero when both rods hang straight down
        pe = ((m1 + m2) * g * L1 * (1.0 - math.cos(th1))
              + m2 * g * L2 * (1.0 - math.cos(th2)))
        return {
            "theta1_deg": math.degrees(th1),
            "theta2_deg": math.degrees(th2),
            "delta_deg":  math.degrees(th1 - th2),
            "omega1": om1, "omega2": om2,
            "x2": x2, "y2": y2,
            "KE": ke, "PE": pe, "E_total": ke + pe,
        }

    def draw(self, draw_tag, cam):
        th1, th2, om1, om2 = self.state
        L1 = self.get_param("length1")
        L2 = self.get_param("length2")
        m1 = self.get_param("mass1")
        m2 = self.get_param("mass2")
        g  = self.get_param("g")

        (x1, y1), (x2, y2) = self._positions()
        p_pivot = cam.w2s(0.0, 0.0)
        p1 = cam.w2s(x1, y1)
        p2 = cam.w2s(x2, y2)

        if self.overlays["trail1"]["enabled"] and len(self._trail1) > 1:
            pts = [cam.w2s(*p) for p in self._trail1]
            n = len(pts)
            for i in range(n - 1):
                a_col = int(30 + 150 * i / n)
                dpg.draw_line(pts[i], pts[i + 1],
                              color=(255, 180, 80, a_col), thickness=1,
                              parent=draw_tag)

        if self.overlays["trail2"]["enabled"] and len(self._trail2) > 1:
            pts = [cam.w2s(*p) for p in self._trail2]
            n = len(pts)
            for i in range(n - 1):
                a_col = int(30 + 180 * i / n)
                dpg.draw_line(pts[i], pts[i + 1],
                              color=(100, 200, 255, a_col), thickness=1,
                              parent=draw_tag)

        dpg.draw_line(p_pivot, p1,
                      color=(200, 200, 200, 230), thickness=2, parent=draw_tag)
        dpg.draw_line(p1, p2,
                      color=(200, 200, 200, 230), thickness=2, parent=draw_tag)

        dpg.draw_circle(p_pivot, 4,
                        color=(220, 220, 220, 255), fill=(60, 60, 60, 255),
                        parent=draw_tag)

        r1 = max(6, int(4 + 3 * math.sqrt(m1)))
        r2 = max(6, int(4 + 3 * math.sqrt(m2)))
        dpg.draw_circle(p1, r1,
                        color=(255, 220, 50, 255), fill=(255, 220, 50, 200),
                        parent=draw_tag)
        dpg.draw_circle(p2, r2,
                        color=(255, 160, 50, 255), fill=(255, 160, 50, 200),
                        parent=draw_tag)

        if self.overlays["velocity"]["enabled"]:
            VSCALE = 3.0
            (v1x, v1y), (v2x, v2y) = self._velocities()
            t1 = cam.arrow_tip(x1, y1, v1x, v1y, VSCALE)
            dpg.draw_arrow(t1, p1,
                           color=(60, 220, 60, 230), thickness=2, size=6,
                           parent=draw_tag)
            v1m = math.sqrt(v1x * v1x + v1y * v1y)
            dpg.draw_text((t1[0] + 5, t1[1] - 14),
                          f"v1 = {v1m:.2f} m/s",
                          color=(60, 220, 60, 230), size=12, parent=draw_tag)
            t2 = cam.arrow_tip(x2, y2, v2x, v2y, VSCALE)
            dpg.draw_arrow(t2, p2,
                           color=(60, 220, 60, 230), thickness=2, size=6,
                           parent=draw_tag)
            v2m = math.sqrt(v2x * v2x + v2y * v2y)
            dpg.draw_text((t2[0] + 5, t2[1] - 14),
                          f"v2 = {v2m:.2f} m/s",
                          color=(60, 220, 60, 230), size=12, parent=draw_tag)

        # info text
        (v1x, v1y), (v2x, v2y) = self._velocities()
        v1_sq = v1x * v1x + v1y * v1y
        v2_sq = v2x * v2x + v2y * v2y
        ke = 0.5 * m1 * v1_sq + 0.5 * m2 * v2_sq
        pe = ((m1 + m2) * g * L1 * (1.0 - math.cos(th1))
              + m2 * g * L2 * (1.0 - math.cos(th2)))
        dpg.draw_text((10, 10),
                      f"theta1 = {math.degrees(th1):6.1f}°    theta2 = {math.degrees(th2):6.1f}°    t = {self.t:.2f} s",
                      color=(200, 200, 200, 200), size=13, parent=draw_tag)
        dpg.draw_text((10, 28),
                      f"omega1 = {om1:+.2f} rad/s    omega2 = {om2:+.2f} rad/s",
                      color=(180, 180, 180, 180), size=12, parent=draw_tag)
        dpg.draw_text((10, 44),
                      f"E = {ke+pe:.3f} J   (KE = {ke:.3f},  PE = {pe:.3f})",
                      color=(180, 180, 180, 180), size=12, parent=draw_tag)

        if self.overlays["equations"]["enabled"]:
            fx = cam.canvas_w - 360
            dpg.draw_text((fx, 10), "Positions",
                          color=(140, 170, 180, 200), size=12, parent=draw_tag)
            dpg.draw_text((fx, 26), "(x1, y1) = ( L1 sin th1, -L1 cos th1 )",
                          color=(180, 220, 230, 220), size=12, parent=draw_tag)
            dpg.draw_text((fx, 42), "(x2, y2) = (x1 + L2 sin th2, y1 - L2 cos th2)",
                          color=(180, 220, 230, 220), size=12, parent=draw_tag)
            dpg.draw_text((fx, 64), "Energy",
                          color=(140, 170, 180, 200), size=12, parent=draw_tag)
            dpg.draw_text((fx, 80), "T = (1/2)(m1 v1² + m2 v2²)",
                          color=(180, 220, 230, 220), size=12, parent=draw_tag)
            dpg.draw_text((fx, 96), "V = (m1+m2) g L1 (1-cos th1) + m2 g L2 (1-cos th2)",
                          color=(180, 220, 230, 220), size=12, parent=draw_tag)
            dpg.draw_text((fx, 118), "Equations of motion",
                          color=(140, 170, 180, 200), size=12, parent=draw_tag)
            dpg.draw_text((fx, 134), "Coupled Lagrangian EOM",
                          color=(180, 220, 230, 220), size=12, parent=draw_tag)
            dpg.draw_text((fx, 150), "(see derivatives() in sims/double_pendulum.py)",
                          color=(150, 180, 190, 180), size=11, parent=draw_tag)
            dpg.draw_text((fx, 172), "Chaos: tiny changes in initial",
                          color=(140, 170, 180, 200), size=12, parent=draw_tag)
            dpg.draw_text((fx, 188), "conditions blow up after seconds",
                          color=(140, 170, 180, 200), size=12, parent=draw_tag)
            dpg.draw_text((fx, 204), "(try theta1 = 90 vs 90.001 deg)",
                          color=(160, 200, 210, 200), size=11, parent=draw_tag)
