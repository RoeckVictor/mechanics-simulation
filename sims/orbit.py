import math

import dearpygui.dearpygui as dpg
import numpy as np

from core.simulation import Simulation


SOFTENING = 0.05   # Plummer softening to avoid singularity at r = 0


class OrbitSimulation(Simulation):
    name = "Two-Body Orbit"
    description = "Two point masses interacting via Newtonian gravity. Initial conditions set up a circular orbit about the centre of mass."

    camera_scale = 50.0
    camera_center = (0.0, 0.0)

    presets = [
        ("Circular orbit",           {"G": 1.0, "m1": 100.0, "m2":  1.0, "r0": 5.0, "v_mult": 1.0}),
        ("Elliptical (start apo)",   {"G": 1.0, "m1": 100.0, "m2":  1.0, "r0": 5.0, "v_mult": 0.7}),
        ("Elliptical (start peri)",  {"G": 1.0, "m1": 100.0, "m2":  1.0, "r0": 5.0, "v_mult": 1.3}),
        ("Hyperbolic escape",        {"G": 1.0, "m1": 100.0, "m2":  1.0, "r0": 5.0, "v_mult": 1.7}),
        ("Equal-mass binary",        {"G": 1.0, "m1":  50.0, "m2": 50.0, "r0": 5.0, "v_mult": 1.0}),
        ("Earth-Moon-ish ratio",     {"G": 1.0, "m1":  80.0, "m2":  1.0, "r0": 5.0, "v_mult": 1.0}),
        ("Slow flyby",               {"G": 1.0, "m1": 100.0, "m2":  1.0, "r0": 8.0, "v_mult": 0.5}),
    ]

    def _define_params(self):
        self.params = {
            "G":      {"label": "G          (grav. const.)",       "value": 1.0,   "min": 0.01, "max": 10.0,   "step": 0.05},
            "m1":     {"label": "Mass 1     (kg)",                 "value": 100.0, "min": 0.1,  "max": 1000.0, "step": 1.0},
            "m2":     {"label": "Mass 2     (kg)",                 "value": 1.0,   "min": 0.1,  "max": 1000.0, "step": 1.0},
            "r0":     {"label": "Init separation (m)",             "value": 5.0,   "min": 0.5,  "max": 20.0,   "step": 0.1},
            "v_mult": {"label": "Speed x v_circular",              "value": 1.0,   "min": 0.1,  "max": 2.0,    "step": 0.01},
        }
        self.overlays = {
            "trail1":    {"label": "Body 1 trail",       "enabled": True},
            "trail2":    {"label": "Body 2 trail",       "enabled": True},
            "velocity":  {"label": "Velocity vectors",   "enabled": False},
            "forces":    {"label": "Force vectors",      "enabled": False},
            "com":       {"label": "Centre of mass",     "enabled": True},
            "equations": {"label": "Show equations",     "enabled": True},
        }
        self._trail1: list[tuple[float, float]] = []
        self._trail2: list[tuple[float, float]] = []

    def reset(self):
        G = self.get_param("G")
        m1 = self.get_param("m1")
        m2 = self.get_param("m2")
        r  = self.get_param("r0")
        mult = self.get_param("v_mult")

        M = m1 + m2
        # Each body's distance from COM (so COM is at origin)
        r1 = m2 / M * r
        r2 = m1 / M * r

        # Circular-orbit speeds about the COM (each body)
        v1c = math.sqrt(G * m2 * m2 / (M * r)) if r > 0 else 0.0
        v2c = math.sqrt(G * m1 * m1 / (M * r)) if r > 0 else 0.0

        x1, y1 = -r1, 0.0
        x2, y2 =  r2, 0.0
        vx1, vy1 = 0.0, -mult * v1c
        vx2, vy2 = 0.0,  mult * v2c

        self.state = np.array([x1, y1, x2, y2, vx1, vy1, vx2, vy2])
        self.t = 0.0
        self._trail1 = []
        self._trail2 = []
        self.recorder.clear()

    def derivatives(self, state, t):
        x1, y1, x2, y2, vx1, vy1, vx2, vy2 = state
        G  = self.get_param("G")
        m1 = self.get_param("m1")
        m2 = self.get_param("m2")
        dx = x2 - x1
        dy = y2 - y1
        # softened |r|^3 keeps acceleration finite at close approach
        r_soft = math.sqrt(dx * dx + dy * dy + SOFTENING * SOFTENING)
        inv_r3 = G / (r_soft * r_soft * r_soft)
        ax1 =  inv_r3 * m2 * dx
        ay1 =  inv_r3 * m2 * dy
        ax2 = -inv_r3 * m1 * dx
        ay2 = -inv_r3 * m1 * dy
        return np.array([vx1, vy1, vx2, vy2, ax1, ay1, ax2, ay2])

    def update(self, dt):
        if self.paused:
            return
        super().update(dt)
        x1, y1, x2, y2 = self.state[0], self.state[1], self.state[2], self.state[3]
        if self.overlays["trail1"]["enabled"]:
            self._trail1.append((x1, y1))
            if len(self._trail1) > 4000:
                self._trail1.pop(0)
        if self.overlays["trail2"]["enabled"]:
            self._trail2.append((x2, y2))
            if len(self._trail2) > 4000:
                self._trail2.pop(0)

    def get_record_values(self) -> dict:
        x1, y1, x2, y2, vx1, vy1, vx2, vy2 = self.state
        G  = self.get_param("G")
        m1 = self.get_param("m1")
        m2 = self.get_param("m2")
        dx, dy = x2 - x1, y2 - y1
        r = math.sqrt(dx * dx + dy * dy)
        v1 = math.sqrt(vx1 * vx1 + vy1 * vy1)
        v2 = math.sqrt(vx2 * vx2 + vy2 * vy2)
        ke = 0.5 * m1 * v1 * v1 + 0.5 * m2 * v2 * v2
        pe = -G * m1 * m2 / r if r > 1e-9 else 0.0
        # Angular momentum about origin (z-component)
        L_z = m1 * (x1 * vy1 - y1 * vx1) + m2 * (x2 * vy2 - y2 * vx2)
        return {
            "r": r, "v1": v1, "v2": v2,
            "KE": ke, "PE": pe, "E_total": ke + pe,
            "L_z": L_z,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        }

    def draw(self, draw_tag, cam):
        x1, y1, x2, y2, vx1, vy1, vx2, vy2 = self.state
        G  = self.get_param("G")
        m1 = self.get_param("m1")
        m2 = self.get_param("m2")

        if self.overlays["trail1"]["enabled"] and len(self._trail1) > 1:
            pts = [cam.w2s(*p) for p in self._trail1]
            n = len(pts)
            for i in range(n - 1):
                a = int(30 + 150 * i / n)
                dpg.draw_line(pts[i], pts[i + 1],
                              color=(255, 180, 80, a), thickness=1,
                              parent=draw_tag)
        if self.overlays["trail2"]["enabled"] and len(self._trail2) > 1:
            pts = [cam.w2s(*p) for p in self._trail2]
            n = len(pts)
            for i in range(n - 1):
                a = int(30 + 180 * i / n)
                dpg.draw_line(pts[i], pts[i + 1],
                              color=(100, 200, 255, a), thickness=1,
                              parent=draw_tag)

        if self.overlays["com"]["enabled"]:
            M = m1 + m2
            cx = (m1 * x1 + m2 * x2) / M
            cy = (m1 * y1 + m2 * y2) / M
            cs = cam.w2s(cx, cy)
            dpg.draw_circle(cs, 3,
                            color=(180, 180, 180, 220),
                            fill=(180, 180, 180, 180),
                            parent=draw_tag)
            # small crosshair
            dpg.draw_line((cs[0] - 6, cs[1]), (cs[0] + 6, cs[1]),
                          color=(180, 180, 180, 160), thickness=1, parent=draw_tag)
            dpg.draw_line((cs[0], cs[1] - 6), (cs[0], cs[1] + 6),
                          color=(180, 180, 180, 160), thickness=1, parent=draw_tag)

        # Body radii scale gently with mass^(1/3) so volume ~ mass
        r1_px = max(5, int(3 + 4 * m1 ** (1.0 / 3.0)))
        r2_px = max(5, int(3 + 4 * m2 ** (1.0 / 3.0)))
        p1 = cam.w2s(x1, y1)
        p2 = cam.w2s(x2, y2)
        dpg.draw_circle(p1, r1_px,
                        color=(255, 200, 80, 255),
                        fill=(255, 200, 80, 220),
                        parent=draw_tag)
        dpg.draw_circle(p2, r2_px,
                        color=(100, 180, 255, 255),
                        fill=(100, 180, 255, 220),
                        parent=draw_tag)

        if self.overlays["velocity"]["enabled"]:
            VSCALE = 1.5
            tip1 = cam.arrow_tip(x1, y1, vx1, vy1, VSCALE)
            tip2 = cam.arrow_tip(x2, y2, vx2, vy2, VSCALE)
            dpg.draw_arrow(tip1, p1, color=(60, 220, 60, 230), thickness=2, size=6, parent=draw_tag)
            dpg.draw_arrow(tip2, p2, color=(60, 220, 60, 230), thickness=2, size=6, parent=draw_tag)
            v1m = math.sqrt(vx1 * vx1 + vy1 * vy1)
            v2m = math.sqrt(vx2 * vx2 + vy2 * vy2)
            dpg.draw_text((tip1[0] + 5, tip1[1] - 14),
                          f"v1 = {v1m:.2f}", color=(60, 220, 60, 230), size=12, parent=draw_tag)
            dpg.draw_text((tip2[0] + 5, tip2[1] - 14),
                          f"v2 = {v2m:.2f}", color=(60, 220, 60, 230), size=12, parent=draw_tag)

        if self.overlays["forces"]["enabled"]:
            dx, dy = x2 - x1, y2 - y1
            r_soft = math.sqrt(dx * dx + dy * dy + SOFTENING * SOFTENING)
            F_mag = G * m1 * m2 / (r_soft * r_soft)
            FSCALE = 30.0  # N per m of arrow
            # F on body 1 points toward body 2
            ux, uy = dx / r_soft, dy / r_soft
            t1 = cam.arrow_tip(x1, y1,  ux * F_mag,  uy * F_mag, FSCALE)
            t2 = cam.arrow_tip(x2, y2, -ux * F_mag, -uy * F_mag, FSCALE)
            dpg.draw_arrow(t1, p1, color=(220, 80, 220, 230), thickness=2, size=6, parent=draw_tag)
            dpg.draw_arrow(t2, p2, color=(220, 80, 220, 230), thickness=2, size=6, parent=draw_tag)
            dpg.draw_text((t1[0] + 5, t1[1] - 14),
                          f"F = {F_mag:.3f}", color=(220, 80, 220, 230), size=12, parent=draw_tag)

        # info text
        dx, dy = x2 - x1, y2 - y1
        r = math.sqrt(dx * dx + dy * dy)
        v1 = math.sqrt(vx1 * vx1 + vy1 * vy1)
        v2 = math.sqrt(vx2 * vx2 + vy2 * vy2)
        ke = 0.5 * m1 * v1 * v1 + 0.5 * m2 * v2 * v2
        pe = -G * m1 * m2 / r if r > 1e-9 else 0.0
        e_total = ke + pe
        L_z = m1 * (x1 * vy1 - y1 * vx1) + m2 * (x2 * vy2 - y2 * vx2)

        dpg.draw_text((10, 10),
                      f"r = {r:.3f}    v1 = {v1:.3f}    v2 = {v2:.3f}    t = {self.t:.2f} s",
                      color=(200, 200, 200, 200), size=13, parent=draw_tag)
        dpg.draw_text((10, 28),
                      f"E = {e_total:+.4f}    L_z = {L_z:+.4f}",
                      color=(180, 180, 180, 180), size=12, parent=draw_tag)

        if self.overlays["equations"]["enabled"]:
            fx = cam.canvas_w - 360
            M = m1 + m2
            r0 = self.get_param("r0")
            v_c_m2 = math.sqrt(G * m1 * m1 / (M * r0)) if r0 > 0 else 0.0
            T_orbit = 2.0 * math.pi * math.sqrt(r0 ** 3 / (G * M)) if (G * M) > 0 else 0.0
            dpg.draw_text((fx, 10), "Newton's law of gravitation",
                          color=(140, 170, 180, 200), size=12, parent=draw_tag)
            dpg.draw_text((fx, 26), "F = G m1 m2 / r²",
                          color=(180, 220, 230, 220), size=12, parent=draw_tag)
            dpg.draw_text((fx, 48), "Equations of motion",
                          color=(140, 170, 180, 200), size=12, parent=draw_tag)
            dpg.draw_text((fx, 64), "r1'' = +G m2 (r2 - r1) / r³",
                          color=(180, 220, 230, 220), size=12, parent=draw_tag)
            dpg.draw_text((fx, 80), "r2'' = -G m1 (r2 - r1) / r³",
                          color=(180, 220, 230, 220), size=12, parent=draw_tag)
            dpg.draw_text((fx, 102), "Conserved (central forces, no torque)",
                          color=(140, 170, 180, 200), size=12, parent=draw_tag)
            dpg.draw_text((fx, 118), "E = T - G m1 m2 / r",
                          color=(180, 220, 230, 220), size=12, parent=draw_tag)
            dpg.draw_text((fx, 134), "L_z = m1 (x1 vy1 - y1 vx1)",
                          color=(180, 220, 230, 220), size=12, parent=draw_tag)
            dpg.draw_text((fx, 150), "      + m2 (x2 vy2 - y2 vx2)",
                          color=(180, 220, 230, 220), size=12, parent=draw_tag)
            dpg.draw_text((fx, 172), "Circular orbit speed (body 2)",
                          color=(140, 170, 180, 200), size=12, parent=draw_tag)
            dpg.draw_text((fx, 188), f"v_c = sqrt(G m1² / (M r)) = {v_c_m2:.3f}",
                          color=(180, 220, 230, 220), size=12, parent=draw_tag)
            dpg.draw_text((fx, 204), f"Period: T = 2 pi sqrt(r³/GM) = {T_orbit:.3f} s",
                          color=(180, 220, 230, 220), size=12, parent=draw_tag)
