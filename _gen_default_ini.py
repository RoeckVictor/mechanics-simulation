"""
Generates default_layout.ini with a pre-docked layout.
Run once: python _gen_default_ini.py
Re-run whenever panels are added or removed.
"""
import sys, os, re, shutil
sys.path.insert(0, os.path.dirname(__file__))

import dearpygui.dearpygui as dpg
from ui.app import App
from sims.projectile import ProjectileSimulation

app = App([ProjectileSimulation()])
dpg.create_context()
dpg.configure_app(docking=True, docking_space=True)
app._build()
dpg.setup_dearpygui()
dpg.show_viewport()
for _ in range(5):
    dpg.render_dearpygui_frame()
dpg.save_init_file("_probe_full.ini")
dpg.stop_dearpygui()
dpg.destroy_context()

probe = open("_probe_full.ini").read()
sections = re.findall(r'\[Window\]\[([^\]]+)\]\n((?:[^\[]+)*)', probe)

wins = {}
for name, body in sections:
    pm = re.search(r'Pos=(-?\d+),(-?\d+)', body)
    sm = re.search(r'Size=(\d+),(\d+)', body)
    if pm and sm:
        wins[name] = (int(pm[1]), int(pm[2]), int(sm[1]), int(sm[2]))

def find(tag_int):
    """Fixed tags are unique — just look up directly."""
    key = f"###{tag_int}"
    if key in wins:
        px, py, pw, ph = wins[key]
        return key, px, py, pw, ph
    return None, None, None, None, None

NAV,  nx, ny, nw, nh = find(10001)
CANV, cx, cy, cw, ch = find(10002)
PARA, px, py, pw, ph = find(10003)
ANA,  ax, ay, aw, ah = find(10004)

dsm  = re.search(r'DockSpace ID=(0x[0-9A-Fa-f]+) Window=(0x[0-9A-Fa-f]+)'
                 r' Pos=(-?\d+),(-?\d+) Size=(\d+),(\d+)', probe)
assert dsm, "DockSpace line not found in probe ini"
DSID, DSWN = dsm[1], dsm[2]
DS_POS  = f"{dsm[3]},{dsm[4]}"
DS_SIZE = f"{dsm[5]},{dsm[6]}"

assert all([NAV, CANV, PARA, ANA]), \
    f"Could not find all panels!\nwins={wins}"

print(f"NAV={NAV}  CANV={CANV}  PARA={PARA}  ANA={ANA}")
print(f"DockSpace ID={DSID}  Window={DSWN}  Pos={DS_POS}  Size={DS_SIZE}")

N_TOP = "0x00000002"
N_NAV = "0x00000003"
N_MID = "0x00000004"
N_VP  = "0x00000005"
N_PAR = "0x00000006"
N_ANA = "0x00000007"

ini = f"""\
[Window][WindowOverViewport_11111111]
Pos=0,0
Size=1384,861
Collapsed=0

[Window][{NAV}]
Pos={nx},{ny}
Size=220,593
Collapsed=0
DockId={N_NAV},0

[Window][{CANV}]
Pos={cx},{cy}
Size=854,593
Collapsed=0
DockId={N_VP},0

[Window][{PARA}]
Pos={px},{py}
Size=310,593
Collapsed=0
DockId={N_PAR},0

[Window][{ANA}]
Pos={ax},{ay}
Size=1384,243
Collapsed=0
DockId={N_ANA},0

[Docking][Data]
DockSpace ID={DSID} Window={DSWN} Pos={DS_POS} Size={DS_SIZE} Split=Y
  DockNode ID={N_TOP} Parent={DSID} SizeRef=1384,593 Split=X
    DockNode ID={N_NAV} Parent={N_TOP} SizeRef=220,593
    DockNode ID={N_MID} Parent={N_TOP} SizeRef=1164,593 Split=X
      DockNode ID={N_VP} Parent={N_MID} SizeRef=854,593
      DockNode ID={N_PAR} Parent={N_MID} SizeRef=310,593
  DockNode ID={N_ANA} Parent={DSID} SizeRef=1384,243

"""

with open("default_layout.ini", "w") as f:
    f.write(ini)
shutil.copy("default_layout.ini", "layout.ini")
print("Written default_layout.ini  +  layout.ini")
