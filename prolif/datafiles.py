from pathlib import Path
import importlib.resources

with importlib.resources.path("data", "") as p:
    datapath = p

TOP = str(datapath / "top.pdb")
TRAJ = str(datapath / "traj.xtc")