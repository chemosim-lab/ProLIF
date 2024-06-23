from contextlib import suppress
from pathlib import Path

import prolif

print(prolif.__version__)  # noqa: T201

assert Path(prolif.datafiles.TOP).is_file()

with suppress(ImportError, ModuleNotFoundError):
    import tests

    assert next(Path(tests.__file__).parent.glob("test_fingerprint.py"), None) is None
