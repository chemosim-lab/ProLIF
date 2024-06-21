import atexit
from contextlib import ExitStack
from importlib import resources

_file_manager = ExitStack()
atexit.register(_file_manager.close)
_data_resource = resources.files("prolif") / "data/"
datapath = _file_manager.enter_context(resources.as_file(_data_resource))

TOP = str(datapath / "top.pdb")
TRAJ = str(datapath / "traj.xtc")
