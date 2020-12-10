from pathlib import Path
import importlib.resources

with importlib.resources.path("data", "top.pdb") as p:
    datapath = p.parent

TOP = str(datapath / "top.pdb")
TRAJ = str(datapath / "traj.xtc")