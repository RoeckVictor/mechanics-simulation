"""Run once to discover DPG's internal DockSpace ID, then exit."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import dearpygui.dearpygui as dpg

dpg.create_context()
dpg.configure_app(docking=True, docking_space=True)
dpg.create_viewport(title="_", width=1400, height=900, resizable=True)

with dpg.window(tag="dockspace_win", no_title_bar=True, no_resize=True,
                no_move=True, no_scrollbar=True,
                pos=(0, 36), width=1400, height=864):
    pass

with dpg.window(label="Simulations", tag="nav_win",   no_close=True, pos=(0,   36), width=220, height=864): dpg.add_text(".")
with dpg.window(label="Viewport",    tag="canvas_win", no_close=True, pos=(220, 36), width=870, height=614): dpg.add_text(".")
with dpg.window(label="Parameters",  tag="param_win",  no_close=True, pos=(1090,36), width=310, height=864): dpg.add_text(".")
with dpg.window(label="Analysis",    tag="analysis_win",no_close=True,pos=(220,650), width=870, height=250): dpg.add_text(".")

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.render_dearpygui_frame()
dpg.save_init_file("_probe_layout.ini")
dpg.stop_dearpygui()
dpg.destroy_context()
print("saved _probe_layout.ini")
