from pathlib import Path

datapath = Path(__file__).parents[1] / "tests" / "data"

TOP = str(datapath / "top.pdb")
TRAJ = str(datapath / "traj.xtc")