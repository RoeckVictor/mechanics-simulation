import math

import dearpygui.dearpygui as dpg
import numpy as np

from core.simulation import Simulation


L_NATURAL = 3.0     # visual natural length of the spring (m)
MASS_SIZE = 0.4
SPRING_COILS = 12
SPRING_AMP = 0.15


class SpringMassSimulation(Simulation):
    name = "Spring-Mass Oscillator"
    description = "1-D mass on a spring with optional damping and sinusoidal driving."

    camera_scale = 100.0
    camera_center = (3.0, 0.0)

    # Critical damping for default k=30, m=1: c_crit = 2*sqrt(k*m) ≈ 10.95
    presets = [
        ("Default (pure SHM)",   {"mass": 1.0, "k": 30.0,  "c": 0.0,   "x0": 1.0, "v0": 0.0, "F0": 0.0, "omega_drive": 0.00}),
        ("Underdamped",          {"mass": 1.0, "k": 30.0,  "c": 2.0,   "x0": 1.0, "v0": 0.0, "F0": 0.0, "omega_drive": 0.00}),
        ("Critically damped",    {"mass": 1.0, "k": 30.0,  "c": 10.95, "x0": 1.0, "v0": 0.0, "F0": 0.0, "omega_drive": 0.00}),
        ("Overdamped",           {"mass": 1.0, "k": 30.0,  "c": 20.0,  "x0": 1.0, "v0": 0.0, "F0": 0.0, "omega_drive": 0.00}),
        ("Forced resonance",     {"mass": 1.0, "k": 30.0,  "c": 0.5,   "x0": 0.0, "v0": 0.0, "F0": 2.0, "omega_drive": 5.48}),
        ("Beats (slight detune)",{"mass": 1.0, "k": 30.0,  "c": 0.0,   "x0": 0.0, "v0": 0.0, "F0": 2.0, "omega_drive": 5.00}),
        ("Off-resonance forcing",{"mass": 1.0, "k": 30.0,  "c": 1.0,   "x0": 0.0, "v0": 0.0, "F0": 5.0, "omega_drive": 10.0}),
        ("Stiff spring",         {"mass": 1.0, "k": 100.0, "c": 0.0,   "x0": 1.0, "v0": 0.0, "F0": 0.0, "omega_drive": 0.00}),
    ]

    def _define_params(self):
        self.params = {
            "mass":        {"label": "Mass               (kg)",    "value": 1.0,  "min": 0.1,  "max": 20.0,  "step": 0.1},
            "k":           {"label": "Spring constant    (N/m)",   "value": 30.0, "min": 1.0,  "max": 200.0, "step": 1.0},
            "c":           {"label": "Damping            (kg/s)",  "value": 0.0,  "min": 0.0,  "max": 10.0,  "step": 0.05},
            "x0":          {"label": "Init displacement  (m)",     "value": 1.0,  "min": -2.0, "max": 2.0,   "step": 0.05},
            "v0":          {"label": "Init velocity      (m/s)",   "value": 0.0,  "min": -5.0, "max": 5.0,   "step": 0.05},
            "F0":          {"label": "Drive amplitude    (N)",     "value": 0.0,  "min": 0.0,  "max": 30.0,  "step": 0.1},
            "omega_drive": {"label": "Drive frequency    (rad/s)", "value": 0.0,  "min": 0.0,  "max": 20.0,  "step": 0.05},
        }
        self.overlays = {
            "forces":       {"label": "Force vectors",       "enabled": True},
            "velocity":     {"label": "Velocity vector",     "enabled": True},
            "acceleration": {"label": "Acceleration vector", "enabled": False},
            "trail":        {"label": "Mass trail",          "enabled": False},
            "equations":    {"label": "Show equations",      "enabled": True},
        }
        self._trail: list[tuple[float, float]] = []

    def reset(self):
        x0 = self.get_param("x0")
        v0 = self.get_param("v0")
        self.state = np.array([x0, v0])
        self.t = 0.0
        self._trail = []
        self.recorder.clear()

    def derivatives(self, state, t):
        x, v = state
        m  = self.get_param("mass")
        k  = self.get_param("k")
        c  = self.get_param("c")
        F0 = self.get_param("F0")
        omega_d = self.get_param("omega_drive")
        # m x'' + c x' + k x = F0 cos(omega_d t)
        # => x'' = -(k/m) x - (c/m) v + (F0/m) cos(omega_d t)
        a = -(k / m) * x - (c / m) * v + (F0 / m) * math.cos(omega_d * t)
        return np.array([v, a])

    def update(self, dt):
        if self.paused:
            return
        super().update(dt)
        if self.overlays["trail"]["enabled"]:
            self._trail.append((L_NATURAL + self.state[0], 0.0))
            if len(self._trail) > 1500:
                self._trail.pop(0)

    def get_record_values(self):
        x, v = self.state
        m  = self.get_param("mass")
        k  = self.get_param("k")
        c  = self.get_param("c")
        F0 = self.get_param("F0")
        omega_d = self.get_param("omega_drive")
        F_spring = -k * x
        F_damp   = -c * v
        F_drive  = F0 * math.cos(omega_d * self.t)
        a = (F_spring + F_damp + F_drive) / m
        ke = 0.5 * m * v * v
        pe = 0.5 * k * x * x
        return {
            "x": x, "v": v, "a": a,
            "F_spring": F_spring,
            "F_damp": F_damp,
            "F_drive": F_drive,
            "KE": ke, "PE": pe, "E_total": ke + pe,
        }

    def _draw_spring(self, cam, draw_tag, x0, y0, x1, y1):
        dx, dy = x1 - x0, y1 - y0
        length = math.sqrt(dx * dx + dy * dy)
        if length < 0.05:
            return
        ux, uy = dx / length, dy / length
        # perpendicular unit (90 deg CCW)
        px, py = -uy, ux
        n_seg = SPRING_COILS * 2 + 1
        pts = []
        for i in range(n_seg + 1):
            t = i / n_seg
            wx = x0 + t * dx
            wy = y0 + t * dy
            if 0 < i < n_seg:
                sign = 1 if i % 2 == 1 else -1
                wx += sign * SPRING_AMP * px
                wy += sign * SPRING_AMP * py
            pts.append(cam.w2s(wx, wy))
        for i in range(len(pts) - 1):
            dpg.draw_line(pts[i], pts[i + 1],
                          color=(200, 200, 220, 255), thickness=2,
                          parent=draw_tag)

    def draw(self, draw_tag, cam):
        x, v = self.state
        m  = self.get_param("mass")
        k  = self.get_param("k")
        c  = self.get_param("c")
        F0 = self.get_param("F0")
        omega_d = self.get_param("omega_drive")

        # wall at x = 0
        dpg.draw_line(cam.w2s(0.0, 0.7), cam.w2s(0.0, -0.7),
                      color=(180, 180, 180, 255), thickness=3, parent=draw_tag)
        for i in range(-4, 5):
            yh = i * 0.13
            dpg.draw_line(cam.w2s(0.0, yh), cam.w2s(-0.15, yh + 0.10),
                          color=(160, 160, 160, 220), thickness=1, parent=draw_tag)

        # equilibrium reference (dashed-ish dotted line via short segments)
        for i in range(-4, 5):
            y0 = i * 0.13
            y1 = y0 + 0.06
            dpg.draw_line(cam.w2s(L_NATURAL, y0), cam.w2s(L_NATURAL, y1),
                          color=(110, 110, 120, 180), thickness=1, parent=draw_tag)
        dpg.draw_text(cam.w2s(L_NATURAL - 0.10, -0.65),
                      "equilibrium", color=(140, 140, 150, 200), size=11,
                      parent=draw_tag)

        mass_x = L_NATURAL + x
        ms2 = MASS_SIZE * 0.5

        if self.overlays["trail"]["enabled"] and len(self._trail) > 1:
            pts = [cam.w2s(*p) for p in self._trail]
            n = len(pts)
            for i in range(n - 1):
                a_col = int(20 + 100 * i / n)
                dpg.draw_line(pts[i], pts[i + 1],
                              color=(100, 200, 255, a_col), thickness=1,
                              parent=draw_tag)

        self._draw_spring(cam, draw_tag, 0.0, 0.0, mass_x - ms2, 0.0)

        corners = [
            cam.w2s(mass_x - ms2, -ms2),
            cam.w2s(mass_x + ms2, -ms2),
            cam.w2s(mass_x + ms2,  ms2),
            cam.w2s(mass_x - ms2,  ms2),
        ]
        dpg.draw_polygon(corners,
                         color=(255, 220, 50, 255),
                         fill=(255, 220, 50, 200),
                         thickness=2, parent=draw_tag)

        F_spring = -k * x
        F_damp   = -c * v
        F_drive  = F0 * math.cos(omega_d * self.t)

        if self.overlays["forces"]["enabled"]:
            F_SCALE = 25.0  # N per m of arrow in world space
            # F_spring at the mass centerline
            sp_base = cam.w2s(mass_x, 0.0)
            sp_tip  = cam.arrow_tip(mass_x, 0.0, F_spring, 0.0, F_SCALE)
            if abs(F_spring) > 0.01:
                dpg.draw_arrow(sp_tip, sp_base,
                               color=(80, 200, 220, 230), thickness=2, size=6,
                               parent=draw_tag)
                dpg.draw_text((sp_tip[0] + 5, sp_tip[1] - 16),
                              f"F_s = {F_spring:.1f} N",
                              color=(80, 200, 220, 230), size=12, parent=draw_tag)

            if abs(F_damp) > 0.01:
                dp_base = cam.w2s(mass_x, 0.30)
                dp_tip  = cam.arrow_tip(mass_x, 0.30, F_damp, 0.0, F_SCALE)
                dpg.draw_arrow(dp_tip, dp_base,
                               color=(220, 120, 80, 230), thickness=2, size=6,
                               parent=draw_tag)
                dpg.draw_text((dp_tip[0] + 5, dp_tip[1] - 16),
                              f"F_d = {F_damp:.1f} N",
                              color=(220, 120, 80, 230), size=12, parent=draw_tag)

            if F0 > 0.0:
                dr_base = cam.w2s(mass_x, -0.30)
                dr_tip  = cam.arrow_tip(mass_x, -0.30, F_drive, 0.0, F_SCALE)
                dpg.draw_arrow(dr_tip, dr_base,
                               color=(220, 80, 220, 230), thickness=2, size=6,
                               parent=draw_tag)
                dpg.draw_text((dr_tip[0] + 5, dr_tip[1] + 4),
                              f"F_drv = {F_drive:.1f} N",
                              color=(220, 80, 220, 230), size=12, parent=draw_tag)

        if self.overlays["velocity"]["enabled"] and abs(v) > 1e-3:
            VSCALE = 3.0  # m/s per m of arrow
            vbase = cam.w2s(mass_x,  0.70)
            vtip  = cam.arrow_tip(mass_x, 0.70, v, 0.0, VSCALE)
            dpg.draw_arrow(vtip, vbase,
                           color=(60, 220, 60, 230), thickness=2, size=6,
                           parent=draw_tag)
            dpg.draw_text((vtip[0] + 5, vtip[1] - 16),
                          f"v = {v:.2f} m/s",
                          color=(60, 220, 60, 230), size=12, parent=draw_tag)

        if self.overlays["acceleration"]["enabled"]:
            a_now = self.derivatives(self.state, self.t)[1]
            if abs(a_now) > 1e-3:
                ASCALE = 30.0  # m/s^2 per m of arrow
                abase = cam.w2s(mass_x, -0.70)
                atip  = cam.arrow_tip(mass_x, -0.70, a_now, 0.0, ASCALE)
                dpg.draw_arrow(atip, abase,
                               color=(255, 100, 50, 230), thickness=2, size=6,
                               parent=draw_tag)
                dpg.draw_text((atip[0] + 5, atip[1] + 4),
                              f"a = {a_now:.2f} m/s²",
                              color=(255, 100, 50, 230), size=12, parent=draw_tag)

        dpg.draw_text((10, 10),
                      f"x = {x:.3f} m    v = {v:.3f} m/s    t = {self.t:.2f} s",
                      color=(200, 200, 200, 200), size=13, parent=draw_tag)
        ke = 0.5 * m * v * v
        pe = 0.5 * k * x * x
        dpg.draw_text((10, 28),
                      f"KE = {ke:.3f} J   PE = {pe:.3f} J   E = {ke+pe:.3f} J",
                      color=(180, 180, 180, 180), size=12, parent=draw_tag)

        if self.overlays["equations"]["enabled"]:
            fx = cam.canvas_w - 360
            omega0 = math.sqrt(k / m) if m > 0 else 0.0
            T0 = (2.0 * math.pi / omega0) if omega0 > 0 else 0.0
            zeta = c / (2.0 * math.sqrt(k * m)) if (k > 0 and m > 0) else 0.0
            if zeta < 1.0 - 1e-3:
                regime = "underdamped"
            elif zeta > 1.0 + 1e-3:
                regime = "overdamped"
            else:
                regime = "critically damped"

            dpg.draw_text((fx, 10), "Forces",
                          color=(140, 170, 180, 200), size=12, parent=draw_tag)
            dpg.draw_text((fx, 26), "F_spring = -k x",
                          color=(180, 220, 230, 220), size=12, parent=draw_tag)
            dpg.draw_text((fx, 42), "F_damp   = -c v",
                          color=(180, 220, 230, 220), size=12, parent=draw_tag)
            dpg.draw_text((fx, 58), "F_drive  = F0 cos(omega_d t)",
                          color=(180, 220, 230, 220), size=12, parent=draw_tag)

            dpg.draw_text((fx, 80), "Equation of motion",
                          color=(140, 170, 180, 200), size=12, parent=draw_tag)
            dpg.draw_text((fx, 96), "m x'' + c x' + k x = F0 cos(omega_d t)",
                          color=(180, 220, 230, 220), size=12, parent=draw_tag)

            dpg.draw_text((fx, 118), "Natural frequency",
                          color=(140, 170, 180, 200), size=12, parent=draw_tag)
            dpg.draw_text((fx, 134), f"omega_0 = sqrt(k/m) = {omega0:.3f} rad/s",
                          color=(180, 220, 230, 220), size=12, parent=draw_tag)
            dpg.draw_text((fx, 150), f"T_0     = 2 pi / omega_0 = {T0:.3f} s",
                          color=(180, 220, 230, 220), size=12, parent=draw_tag)

            dpg.draw_text((fx, 172), "Damping ratio",
                          color=(140, 170, 180, 200), size=12, parent=draw_tag)
            dpg.draw_text((fx, 188), f"zeta = c / (2 sqrt(k m)) = {zeta:.3f}",
                          color=(180, 220, 230, 220), size=12, parent=draw_tag)
            dpg.draw_text((fx, 204), f"({regime})",
                          color=(160, 200, 210, 200), size=12, parent=draw_tag)
