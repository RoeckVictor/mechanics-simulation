import math

import dearpygui.dearpygui as dpg
import numpy as np

from core.simulation import Simulation


BLOCK_SIZE = 0.35  # side length of the (square) block, in metres


class InclineSimulation(Simulation):
    name = "Inclined Plane"
    description = "Block on a slope with static and kinetic friction."

    camera_scale = 100.0
    camera_center = (1.7, 1.0)

    presets = [
        ("Default (sliding)",      {"angle": 30.0, "length": 4.0, "start_s": 0.0, "start_v":  0.0, "mass": 2.0, "mu_s": 0.30, "mu_k": 0.20, "g": 9.81}),
        ("Stuck (below critical)", {"angle": 15.0, "length": 4.0, "start_s": 0.0, "start_v":  0.0, "mass": 2.0, "mu_s": 0.30, "mu_k": 0.20, "g": 9.81}),
        ("Frictionless slide",     {"angle": 30.0, "length": 4.0, "start_s": 0.0, "start_v":  0.0, "mass": 2.0, "mu_s": 0.0,  "mu_k": 0.0,  "g": 9.81}),
        ("Sticky steep slope",     {"angle": 45.0, "length": 4.0, "start_s": 0.0, "start_v":  0.0, "mass": 2.0, "mu_s": 1.0,  "mu_k": 0.8,  "g": 9.81}),
        ("Initial uphill push",    {"angle": 20.0, "length": 4.0, "start_s": 2.0, "start_v": -2.5, "mass": 2.0, "mu_s": 0.30, "mu_k": 0.20, "g": 9.81}),
        ("Steep slide",            {"angle": 60.0, "length": 5.0, "start_s": 0.0, "start_v":  0.0, "mass": 2.0, "mu_s": 0.20, "mu_k": 0.15, "g": 9.81}),
    ]

    def _define_params(self):
        self.params = {
            "angle":   {"label": "Angle         (deg)",   "value": 30.0, "min": 0.0,  "max": 80.0,  "step": 1.0},
            "length":  {"label": "Slope length  (m)",     "value": 4.0,  "min": 1.0,  "max": 10.0,  "step": 0.1},
            "start_s": {"label": "Start pos     (m from top)", "value": 0.0, "min": 0.0, "max": 10.0, "step": 0.05},
            "start_v": {"label": "Init speed    (m/s)",   "value": 0.0,  "min": -5.0, "max": 5.0,   "step": 0.05},
            "mass":    {"label": "Mass          (kg)",    "value": 2.0,  "min": 0.1,  "max": 50.0,  "step": 0.1},
            "mu_s":    {"label": "mu_static",             "value": 0.30, "min": 0.0,  "max": 1.5,   "step": 0.01},
            "mu_k":    {"label": "mu_kinetic",            "value": 0.20, "min": 0.0,  "max": 1.5,   "step": 0.01},
            "g":       {"label": "Gravity       (m/s²)",  "value": 9.81, "min": 0.1,  "max": 25.0,  "step": 0.1},
        }
        self.overlays = {
            "forces":        {"label": "Forces (g, N, friction)", "enabled": True},
            "decomposition": {"label": "Gravity decomposition",   "enabled": False},
            "velocity":      {"label": "Velocity vector",         "enabled": True},
            "acceleration":  {"label": "Acceleration vector",     "enabled": False},
            "trail":         {"label": "Block trail",             "enabled": True},
            "equations":     {"label": "Show equations",          "enabled": True},
        }
        self._trail: list[tuple[float, float]] = []

    def reset(self):
        L = self.get_param("length")
        s0 = max(0.0, min(self.get_param("start_s"), L))
        v0 = self.get_param("start_v")
        self.state = np.array([s0, v0])
        self.t = 0.0
        self._trail = []
        self.recorder.clear()

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        s, v = state
        alpha = math.radians(self.get_param("angle"))
        g = self.get_param("g")
        mu_s = self.get_param("mu_s")
        mu_k = self.get_param("mu_k")
        # a_par: gravity component down the slope (+down)
        # f_k:   kinetic-friction acceleration magnitude (= mu_k * g * cos α)
        # f_smax: max static-friction acceleration magnitude (= mu_s * g * cos α)
        a_par = g * math.sin(alpha)
        f_k = mu_k * g * math.cos(alpha)
        f_smax = mu_s * g * math.cos(alpha)
        REST = 1e-3
        if abs(v) < REST:
            if abs(a_par) <= f_smax:
                return np.array([0.0, 0.0])  # static friction holds; block at rest
            a = a_par - math.copysign(f_k, a_par)
        else:
            a = a_par - math.copysign(f_k, v)
        return np.array([v, a])

    def update(self, dt: float):
        if self.paused:
            return
        super().update(dt)
        L = self.get_param("length")
        if self.state[0] >= L:
            self.state[0] = L
            self.state[1] = 0.0
        elif self.state[0] <= 0.0 and self.state[1] < 0.0:
            self.state[0] = 0.0
            self.state[1] = 0.0
        if self.overlays["trail"]["enabled"]:
            cx, cy = self._block_center()
            self._trail.append((cx, cy))
            if len(self._trail) > 2000:
                self._trail.pop(0)

    def _block_center(self) -> tuple[float, float]:
        s = self.state[0]
        alpha = math.radians(self.get_param("angle"))
        L = self.get_param("length")
        h_top = L * math.sin(alpha)
        r = BLOCK_SIZE * 0.5
        dx_slope, dy_slope = math.cos(alpha), -math.sin(alpha)
        nx, ny = math.sin(alpha), math.cos(alpha)
        cx = 0.0 + s * dx_slope + r * nx
        cy = h_top + s * dy_slope + r * ny
        return cx, cy

    def get_record_values(self) -> dict:
        s, v = self.state
        m = self.get_param("mass")
        g = self.get_param("g")
        a = self.derivatives(self.state, self.t)[1]
        _, y_block = self._block_center()
        ke = 0.5 * m * v * v
        pe = m * g * y_block
        return {
            "s": s, "v": v, "a": a,
            "height": y_block,
            "KE": ke, "PE": pe, "E_total": ke + pe,
        }

    def draw(self, draw_tag: str, cam) -> None:
        s, v = self.state
        alpha = math.radians(self.get_param("angle"))
        L = self.get_param("length")
        m = self.get_param("mass")
        g = self.get_param("g")
        mu_s = self.get_param("mu_s")
        mu_k = self.get_param("mu_k")

        h_top = L * math.sin(alpha)
        x_bot = L * math.cos(alpha)

        p_origin = cam.w2s(0.0, 0.0)
        p_top    = cam.w2s(0.0, h_top)
        p_bot    = cam.w2s(x_bot, 0.0)

        dpg.draw_triangle(p_origin, p_top, p_bot,
                          color=(150, 150, 160, 220),
                          fill=(60, 60, 70, 100),
                          thickness=2, parent=draw_tag)

        # ground extension to the left and right of the triangle
        gx0 = cam.w2s(-2.0, 0.0)
        gx1 = cam.w2s(x_bot + 2.0, 0.0)
        dpg.draw_line(gx0, gx1, color=(150, 150, 160, 160),
                      thickness=1, parent=draw_tag)

        r = BLOCK_SIZE * 0.5
        dx_slope, dy_slope = math.cos(alpha), -math.sin(alpha)
        nx, ny = math.sin(alpha), math.cos(alpha)
        cx, cy = self._block_center()

        corners = []
        for xl, yl in ((-r, -r), (r, -r), (r, r), (-r, r)):
            wx = cx + xl * dx_slope + yl * nx
            wy = cy + xl * dy_slope + yl * ny
            corners.append(cam.w2s(wx, wy))
        dpg.draw_polygon(corners,
                         color=(255, 220, 50, 255),
                         fill=(255, 220, 50, 180),
                         thickness=2, parent=draw_tag)

        cs = cam.w2s(cx, cy)

        if self.overlays["trail"]["enabled"] and len(self._trail) > 1:
            pts = [cam.w2s(*p) for p in self._trail]
            n = len(pts)
            for i in range(n - 1):
                a_col = int(40 + 180 * i / n)
                dpg.draw_line(pts[i], pts[i + 1],
                              color=(100, 200, 255, a_col), thickness=1,
                              parent=draw_tag)

        if self.overlays["forces"]["enabled"]:
            F_SCALE = 16.0  # 1 N => 16 px
            mg = m * g

            # gravity (world-down)
            gtip = cam.arrow_tip(cx, cy, 0.0, -mg, F_SCALE)
            dpg.draw_arrow(gtip, cs,
                           color=(220, 80, 220, 230), thickness=2, size=6,
                           parent=draw_tag)
            dpg.draw_text((gtip[0] + 5, gtip[1] - 14),
                          f"mg = {mg:.1f} N",
                          color=(220, 80, 220, 230), size=12, parent=draw_tag)

            # normal force = mg cos α, along +normal
            N_mag = mg * math.cos(alpha)
            ntip = cam.arrow_tip(cx, cy, nx * N_mag, ny * N_mag, F_SCALE)
            dpg.draw_arrow(ntip, cs,
                           color=(80, 200, 220, 230), thickness=2, size=6,
                           parent=draw_tag)
            dpg.draw_text((ntip[0] + 5, ntip[1] - 14),
                          f"N = {N_mag:.1f} N",
                          color=(80, 200, 220, 230), size=12, parent=draw_tag)

            # friction
            a_par = g * math.sin(alpha)
            F_smax = mu_s * N_mag
            if abs(v) > 1e-3:
                f_mag = mu_k * N_mag
                f_sign = -math.copysign(1.0, v)
            else:
                net_par = m * a_par
                if abs(net_par) <= F_smax:
                    f_mag = abs(net_par)
                    f_sign = -math.copysign(1.0, a_par) if a_par != 0 else 0.0
                else:
                    f_mag = mu_k * N_mag
                    f_sign = -math.copysign(1.0, a_par)
            if f_mag > 0.01:
                ftip = cam.arrow_tip(cx, cy,
                                     f_sign * dx_slope * f_mag,
                                     f_sign * dy_slope * f_mag, F_SCALE)
                dpg.draw_arrow(ftip, cs,
                               color=(220, 120, 80, 230), thickness=2, size=6,
                               parent=draw_tag)
                dpg.draw_text((ftip[0] + 5, ftip[1] - 14),
                              f"f = {f_mag:.1f} N",
                              color=(220, 120, 80, 230), size=12, parent=draw_tag)

        if self.overlays["decomposition"]["enabled"]:
            F_SCALE = 16.0
            mg = m * g
            F_par = mg * math.sin(alpha)
            F_perp = mg * math.cos(alpha)
            # parallel down-slope
            pp = cam.arrow_tip(cx, cy, dx_slope * F_par, dy_slope * F_par, F_SCALE)
            dpg.draw_arrow(pp, cs,
                           color=(150, 220, 80, 200), thickness=2, size=5,
                           parent=draw_tag)
            dpg.draw_text((pp[0] + 5, pp[1] - 14),
                          f"mg sin = {F_par:.1f} N",
                          color=(150, 220, 80, 200), size=12, parent=draw_tag)
            # perpendicular into slope (opposite normal)
            pe = cam.arrow_tip(cx, cy, -nx * F_perp, -ny * F_perp, F_SCALE)
            dpg.draw_arrow(pe, cs,
                           color=(150, 80, 220, 200), thickness=2, size=5,
                           parent=draw_tag)
            dpg.draw_text((pe[0] + 5, pe[1] + 4),
                          f"mg cos = {F_perp:.1f} N",
                          color=(150, 80, 220, 200), size=12, parent=draw_tag)

        if self.overlays["velocity"]["enabled"]:
            VSCALE = 3.0  # 1 m/s => 3 px
            vtip = cam.arrow_tip(cx, cy, v * dx_slope, v * dy_slope, VSCALE)
            dpg.draw_arrow(vtip, cs,
                           color=(60, 220, 60, 230), thickness=2, size=6,
                           parent=draw_tag)
            dpg.draw_text((vtip[0] + 5, vtip[1] - 14),
                          f"v = {v:.2f} m/s",
                          color=(60, 220, 60, 230), size=12, parent=draw_tag)

        if self.overlays["acceleration"]["enabled"]:
            a = self.derivatives(self.state, self.t)[1]
            ASCALE = 2.0
            atip = cam.arrow_tip(cx, cy, a * dx_slope, a * dy_slope, ASCALE)
            dpg.draw_arrow(atip, cs,
                           color=(255, 100, 50, 230), thickness=2, size=6,
                           parent=draw_tag)
            dpg.draw_text((atip[0] + 5, atip[1] - 14),
                          f"a = {a:.2f} m/s²",
                          color=(255, 100, 50, 230), size=12, parent=draw_tag)

        dpg.draw_text((10, 10),
                      f"s = {s:.2f} m    v = {v:.2f} m/s    t = {self.t:.2f} s",
                      color=(200, 200, 200, 200), size=13, parent=draw_tag)

        ke = 0.5 * m * v * v
        pe = m * g * cy
        dpg.draw_text((10, 28),
                      f"KE = {ke:.2f} J   PE = {pe:.2f} J   E = {ke+pe:.2f} J",
                      color=(180, 180, 180, 180), size=12, parent=draw_tag)

        if self.overlays["equations"]["enabled"]:
            fx = cam.canvas_w - 340
            theta_c = math.degrees(math.atan(mu_s)) if mu_s > 0 else 90.0
            dpg.draw_text((fx, 10), "Forces",
                          color=(140, 170, 180, 200), size=12, parent=draw_tag)
            dpg.draw_text((fx, 26), "N      = m g cos(theta)",
                          color=(180, 220, 230, 220), size=12, parent=draw_tag)
            dpg.draw_text((fx, 42), "F_par  = m g sin(theta)   (down-slope)",
                          color=(180, 220, 230, 220), size=12, parent=draw_tag)
            dpg.draw_text((fx, 58), "f_k    = mu_k * N         (opposes v)",
                          color=(180, 220, 230, 220), size=12, parent=draw_tag)
            dpg.draw_text((fx, 80), "Equation of motion (sliding)",
                          color=(140, 170, 180, 200), size=12, parent=draw_tag)
            dpg.draw_text((fx, 96), "a = g sin(theta) - mu_k g cos(theta) sgn(v)",
                          color=(180, 220, 230, 220), size=12, parent=draw_tag)
            dpg.draw_text((fx, 118), "Critical angle",
                          color=(140, 170, 180, 200), size=12, parent=draw_tag)
            dpg.draw_text((fx, 134), f"theta_c = atan(mu_s) = {theta_c:.1f}°",
                          color=(180, 220, 230, 220), size=12, parent=draw_tag)
