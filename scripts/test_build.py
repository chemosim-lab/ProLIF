from contextlib import suppress
from pathlib import Path

import prolif
from prolif.plotting.network import LigNetwork

print(prolif.__version__)

assert Path(prolif.datafiles.TOP).is_file()

with suppress(ImportError, ModuleNotFoundError):
    import tests

    assert next(Path(tests.__file__).parent.glob("test_fingerprint.py"), None) is None
