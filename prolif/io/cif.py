"""
I/O-related helper functions --- :mod:`prolif.io.cif`
=====================================================
This module provides a wrapper around `gemmi` for reading
Crystallographic Information File (CIF) format.

.. versionchanged:: 2.2.0
    Replaced the custom CIF parser with a thin wrapper around ``gemmi``.
"""

from pathlib import Path

import gemmi


def cif_template_reader(cif_filepath: Path | str) -> gemmi.cif.Document:
    """Reads a CIF file and returns a gemmi CIF Document.

    .. versionadded:: 2.2.0

    Parameters
    ----------
    cif_filepath : Path | str
        The path to the CIF file to read.

    Returns
    -------
    gemmi.cif.Document
        A gemmi CIF Document containing the parsed data blocks.

    """
    return gemmi.cif.read(str(cif_filepath))
