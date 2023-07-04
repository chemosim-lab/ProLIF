"""
Pickling support --- :mod:`prolif.pickling`
===========================================

This module provides the :class:`RDKitPickleHandler` class and a default instance
named `PICKLE_HANDLER` preconfigured to pickle all properties set on a molecule.

Note that on Windows the preconfigured handler will modify RDKit's default pickling
behavior upon importing the `prolif` package to ensure that the parallelization code
works as on other platforms.
"""
import sys
from typing import Optional

from rdkit import Chem


class RDKitPickleHandler:
    def __init__(self, pickle_options: int) -> None:
        self.pickle_options = pickle_options
        self.default_pickle = self.get()

    def get(self) -> int:
        """Get RDKit's current pickle properties option"""
        return Chem.GetDefaultPickleProperties()

    def set(self, pickle_options: Optional[int] = None) -> None:
        """Set RDKit to the specified pickle options (or the one specified
        upon initializing if ``None``).
        """
        if pickle_options is None:
            pickle_options = self.pickle_options
        Chem.SetDefaultPickleProperties(pickle_options)

    def reset(self) -> None:
        """Reset RDKit's pickle options to their default (before instantiating this
        class).
        """
        self.set(self.default_pickle)

    def __enter__(self):
        self.set()
        return self

    def __exit__(self, *exc):
        self.reset()


PICKLE_HANDLER = RDKitPickleHandler(Chem.PropertyPickleOptions.AllProps)

if sys.platform == "win32":
    PICKLE_HANDLER.set()
