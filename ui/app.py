import os
import shutil
import time

import dearpygui.dearpygui as dpg

from core.camera import Camera

TOOLBAR_H  = 30
DEFAULT_W  = 1400
DEFAULT_H  = 900
NAV_W      = 220
PARAM_W    = 310

# Fixed integer tags for the four panel windows
_T_NAV  = 10001
_T_CANV = 10002
_T_PARA = 10003
_T_ANA  = 10004


class App:
    def __init__(self, simulations: list):
        self.sims = {s.name: s for s in simulations}
        self.current = None
        self.cam: Camera | None = None
        self._last_frame_t: float | None = None
        self._last_plot_samples: int = -1
        self._plot_key: str | None = None
        self._canvas_w: int = 1
        self._canvas_h: int = 1
        self._panning: bool = False
        self._drag_last: tuple[float, float] = (0.0, 0.0)
        self._applying_preset: bool = False

    def _build(self):
        dpg.create_viewport(
            title="Mechanics Simulator",
            width=DEFAULT_W,
            height=DEFAULT_H,
            resizable=True,
        )

        # Tall mvAll FramePadding (8.5) makes BeginMainMenuBar size the bar at
        # 30 px. Per-widget overrides keep buttons/combos/drag_floats short, so
        # they sit inside the bar with visible padding below them (DPG doesn't
        # expose DisplaySafeAreaPadding, so we can't push items down for matching
        # top padding — items remain top-aligned within the bar).
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 6, 8.5,
                                    category=dpg.mvThemeCat_Core)
            for wt in (dpg.mvButton, dpg.mvCombo, dpg.mvDragFloat,
                       dpg.mvCheckbox, dpg.mvSelectable, dpg.mvTab):
                with dpg.theme_component(wt):
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 6, 8,
                                        category=dpg.mvThemeCat_Core)
        dpg.bind_theme(global_theme)

        # imgui adjusts WorkPos/WorkSize to exclude this bar, so
        # DockSpaceOverViewport starts exactly below it — no overlap.
        with dpg.viewport_menu_bar():
            dpg.add_text("no simulation loaded", tag="sim_title")
            dpg.add_spacer(width=10)
            dpg.add_button(label="Play",  width=52, callback=lambda: self._on_play())
            dpg.add_button(label="Pause", width=52, callback=lambda: self._on_pause())
            dpg.add_button(label="Reset", width=52, callback=lambda: self._on_reset())
            dpg.add_spacer(width=10)
            dpg.add_text("Integrator:")
            dpg.add_combo(["RK4", "Euler"], default_value="RK4",
                          tag="integrator_combo", width=76,
                          callback=lambda s, v: self._on_integrator(v))
            dpg.add_spacer(width=6)
            dpg.add_text("Speed:")
            dpg.add_drag_float(tag="speed_slider", default_value=1.0,
                                min_value=0.05, max_value=4.0,
                                speed=0.05, clamped=True,
                                width=100, format="%.2fx")
            dpg.add_spacer(width=10)
            dpg.add_checkbox(label="Norm. vec.", tag="normalize_vectors",
                             default_value=False)
            dpg.add_spacer(width=10)
            dpg.add_text("t = 0.000 s", tag="time_label")

        # Fixed integer tags (_T_*) make the ###NNNNN ids in the ini stable.
        # Initial positions are hints only — the ini overrides them on load.
        _panel_h = DEFAULT_H - TOOLBAR_H
        _center_w = DEFAULT_W - NAV_W - PARAM_W
        _canvas_h = _panel_h - 250

        with dpg.window(label="Simulations", tag=_T_NAV, no_close=True,
                        pos=(0, TOOLBAR_H), width=NAV_W, height=_panel_h):
            for name in self.sims:
                dpg.add_selectable(label=name, tag=f"nav__{name}",
                                   callback=self._make_nav_cb(name))

        with dpg.window(label="Viewport", tag=_T_CANV, no_close=True,
                        no_scrollbar=True,
                        pos=(NAV_W, TOOLBAR_H),
                        width=_center_w, height=_canvas_h):
            dpg.add_drawlist(width=10, height=10, tag="canvas")

        with dpg.window(label="Parameters", tag=_T_PARA, no_close=True,
                        pos=(DEFAULT_W - PARAM_W, TOOLBAR_H),
                        width=PARAM_W, height=_panel_h):
            with dpg.group(tag="param_group"):
                dpg.add_text("select a simulation")

        with dpg.window(label="Analysis", tag=_T_ANA, no_close=True,
                        pos=(NAV_W, TOOLBAR_H + _canvas_h),
                        width=_center_w, height=250):
            with dpg.group(horizontal=True):
                dpg.add_text("Plot:")
                dpg.add_combo([], tag="plot_key_combo", width=140,
                              callback=lambda s, v: self._on_plot_key(v))
                dpg.add_text(" vs time")
                dpg.add_spacer(width=20)
                dpg.add_button(label="Clear data",
                               callback=lambda: self._on_clear_analysis())
            with dpg.plot(label="", height=-1, width=-1, tag="analysis_plot"):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", tag="x_axis")
                with dpg.plot_axis(dpg.mvYAxis, label="", tag="y_axis"):
                    dpg.add_line_series([], [], label="", tag="plot_series")

        with dpg.handler_registry():
            dpg.add_mouse_wheel_handler(callback=self._on_mouse_wheel)

    def _is_mouse_on_canvas(self) -> bool:
        if not dpg.does_item_exist("canvas"):
            return False
        state = dpg.get_item_state("canvas")
        rmin = state.get("rect_min")
        rmax = state.get("rect_max")
        if not rmin or not rmax:
            return False
        mx, my = dpg.get_mouse_pos(local=False)
        return rmin[0] <= mx <= rmax[0] and rmin[1] <= my <= rmax[1]

    def _canvas_mouse_pos(self):
        state = dpg.get_item_state("canvas")
        rmin = state.get("rect_min")
        if not rmin:
            return None
        mx, my = dpg.get_mouse_pos(local=False)
        return (mx - rmin[0], my - rmin[1])

    def _process_pan(self):
        # Polled every frame: start panning when left button is pressed over the
        # canvas, accumulate incremental deltas while held, stop on release.
        if not self.cam:
            return
        if dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
            if not self._panning:
                if self._is_mouse_on_canvas():
                    self._panning = True
                    self._drag_last = dpg.get_mouse_pos(local=False)
            else:
                mx, my = dpg.get_mouse_pos(local=False)
                dx = mx - self._drag_last[0]
                dy = my - self._drag_last[1]
                if dx or dy:
                    self.cam.pan_screen(dx, dy)
                    self._drag_last = (mx, my)
        else:
            self._panning = False

    def _on_mouse_wheel(self, sender, app_data):
        if not self.cam or not self._is_mouse_on_canvas():
            return
        mp = self._canvas_mouse_pos()
        if mp is None:
            return
        factor = 1.1 if app_data > 0 else 1.0 / 1.1
        self.cam.zoom_at(mp[0], mp[1], factor)

    def _make_nav_cb(self, name: str):
        def cb(sender, selected):
            if selected:
                self._load_sim(name)
            else:
                dpg.set_value(f"nav__{name}", True)
        return cb

    def _load_sim(self, name: str):
        for n in self.sims:
            dpg.set_value(f"nav__{n}", n == name)

        self.current = self.sims[name]
        self.current.reset()

        self.cam = Camera(
            self._canvas_w, self._canvas_h,
            scale=self.current.camera_scale,
            center_world=self.current.camera_center,
        )

        dpg.set_value("sim_title", name)
        dpg.set_value("integrator_combo", self.current.integrator_name)

        self._rebuild_param_panel()
        self._refresh_plot_keys()
        self._last_plot_samples = -1

    def _rebuild_param_panel(self):
        sim = self.current
        dpg.delete_item("param_group", children_only=True)

        if sim.presets:
            dpg.add_text("Preset", parent="param_group")
            preset_names = [name for name, _ in sim.presets]
            dpg.add_combo(items=preset_names, default_value="Custom",
                          tag="preset_combo", width=-1,
                          callback=lambda s, v: self._apply_preset(v),
                          parent="param_group")
            dpg.add_spacer(height=8, parent="param_group")

        dpg.add_text("Parameters", parent="param_group")
        dpg.add_separator(parent="param_group")

        for key, p in sim.params.items():
            dpg.add_text(p["label"], parent="param_group")
            dpg.add_drag_float(
                tag=f"p__{key}",
                label=f"##{key}",
                default_value=p["value"],
                min_value=p["min"],
                max_value=p["max"],
                speed=p["step"],
                clamped=True,
                format="%.3f",
                width=-1,
                callback=self._make_param_cb(key),
                parent="param_group",
            )

        if sim.overlays:
            dpg.add_spacer(height=10, parent="param_group")
            dpg.add_text("Overlays", parent="param_group")
            dpg.add_separator(parent="param_group")
            for key, ov in sim.overlays.items():
                dpg.add_checkbox(
                    label=ov["label"],
                    default_value=ov["enabled"],
                    callback=self._make_overlay_cb(key),
                    parent="param_group",
                )

        # set the dropdown to the matching preset (or "Custom") for the
        # current parameter values
        self._check_preset_match()

    def _apply_preset(self, name: str):
        sim = self.current
        if not sim or not sim.presets:
            return
        for preset_name, values in sim.presets:
            if preset_name != name:
                continue
            self._applying_preset = True
            try:
                for key, val in values.items():
                    if key in sim.params:
                        sim.params[key]["value"] = val
                        if dpg.does_item_exist(f"p__{key}"):
                            dpg.set_value(f"p__{key}", val)
                sim.paused = True
                sim.reset()
                self._last_plot_samples = -1
            finally:
                self._applying_preset = False
            return

    def _check_preset_match(self):
        sim = self.current
        if not sim or not sim.presets or not dpg.does_item_exist("preset_combo"):
            return
        matched = None
        for preset_name, values in sim.presets:
            if all(k in sim.params and abs(sim.params[k]["value"] - v) < 1e-4
                   for k, v in values.items()):
                matched = preset_name
                break
        dpg.set_value("preset_combo", matched if matched else "Custom")

    def _make_param_cb(self, key: str):
        def cb(sender, val):
            if not self.current or self._applying_preset:
                return
            self.current.params[key]["value"] = val
            if self.current.paused and self.current.t == 0.0:
                self.current.reset()
            self._check_preset_match()
        return cb

    def _make_overlay_cb(self, key: str):
        def cb(sender, val):
            if self.current:
                self.current.overlays[key]["enabled"] = val
        return cb

    def _on_play(self):
        if self.current:
            self.current.paused = False

    def _on_pause(self):
        if self.current:
            self.current.paused = True

    def _on_reset(self):
        if self.current:
            self.current.paused = True
            self.current.reset()
            self._last_plot_samples = -1

    def _on_integrator(self, val: str):
        if self.current:
            self.current.integrator_name = val

    def _refresh_plot_keys(self):
        if not self.current:
            return
        keys = list(self.current.get_record_values().keys())
        dpg.configure_item("plot_key_combo", items=keys)
        if keys:
            self._plot_key = keys[0]
            dpg.set_value("plot_key_combo", keys[0])

    def _on_plot_key(self, val: str):
        self._plot_key = val
        self._last_plot_samples = -1

    def _on_clear_analysis(self):
        if self.current:
            self.current.recorder.clear()
            dpg.set_value("plot_series", [[], []])
            self._last_plot_samples = -1

    def _update_plot(self):
        if not self.current or not self._plot_key:
            return
        n = self.current.recorder.sample_count
        if n == self._last_plot_samples:
            return
        self._last_plot_samples = n
        if self._plot_key not in self.current.recorder.keys():
            return
        times = self.current.recorder.times()
        vals  = self.current.recorder.get(self._plot_key)
        if not times:
            return
        dpg.set_value("plot_series", [times, vals])
        dpg.set_item_label("plot_series", self._plot_key)
        dpg.set_item_label("y_axis", self._plot_key)
        dpg.fit_axis_data("x_axis")
        dpg.fit_axis_data("y_axis")

    def _sync_canvas_size(self):
        if not dpg.does_item_exist(_T_CANV):
            return
        state = dpg.get_item_state(_T_CANV)
        if "rect_size" not in state:
            return
        rect = state["rect_size"]
        new_w = max(10, int(rect[0]) - 16)
        new_h = max(10, int(rect[1]) - 36)   # subtract title bar
        if new_w != self._canvas_w or new_h != self._canvas_h:
            self._canvas_w = new_w
            self._canvas_h = new_h
            dpg.configure_item("canvas", width=new_w, height=new_h)
            if self.cam:
                self.cam.resize(new_w, new_h)

    def _draw_idle(self):
        dpg.delete_item("canvas", children_only=True)
        cx = self._canvas_w / 2
        cy = self._canvas_h / 2
        dpg.draw_text((cx - 130, cy - 8),
                      "Select a simulation from the left panel",
                      color=(130, 130, 130, 180), size=16, parent="canvas")

    def run(self):
        dpg.create_context()

        # load default layout on first run; user customisations persist in layout.ini
        if not os.path.exists("layout.ini") and os.path.exists("default_layout.ini"):
            shutil.copy("default_layout.ini", "layout.ini")
        ini = "layout.ini" if os.path.exists("layout.ini") else None
        dpg.configure_app(docking=True, docking_space=True,
                          **({"init_file": ini} if ini else {}))

        self._build()
        dpg.setup_dearpygui()
        dpg.show_viewport()

        while dpg.is_dearpygui_running():
            now = time.perf_counter()
            self._sync_canvas_size()
            self._process_pan()

            if self.current is not None and self.cam is not None:
                dt = 0.0
                if self._last_frame_t is not None:
                    speed = dpg.get_value("speed_slider")
                    dt = min(now - self._last_frame_t, 1.0 / 30.0) * speed

                self.current.update(dt)
                dpg.delete_item("canvas", children_only=True)
                self.cam.normalize_vectors = dpg.get_value("normalize_vectors")
                self.current.draw("canvas", self.cam)

                dpg.set_value("time_label", f"t = {self.current.t:.3f} s")
                self._update_plot()
            else:
                self._draw_idle()

            self._last_frame_t = now
            dpg.render_dearpygui_frame()

        dpg.save_init_file("layout.ini")
        dpg.destroy_context()
