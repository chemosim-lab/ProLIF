from pathlib import Path

from pkg_resources import resource_filename

datapath = Path(resource_filename("prolif", "data/"))

TOP = str(datapath / "top.pdb")
TRAJ = str(datapath / "traj.xtc")
