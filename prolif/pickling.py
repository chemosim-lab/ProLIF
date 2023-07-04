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

PROLIF_PICKLE_OPTIONS = (
    Chem.PropertyPickleOptions.AtomProps ^ Chem.PropertyPickleOptions.MolProps
)


class RDKitPickleHandler:
    def __init__(self, pickle_options: int) -> None:
        self.pickle_options = pickle_options
        self.default_pickle = self.get()
        self.is_patched = False

    @staticmethod
    def get() -> int:
        """Get RDKit's current pickle properties option"""
        return Chem.GetDefaultPickleProperties()

    def set(self, pickle_options: Optional[int] = None) -> None:
        """Set RDKit to the specified pickle options (or the one specified
        upon initializing if ``None``).
        """
        if pickle_options is None:
            pickle_options = self.pickle_options
        Chem.SetDefaultPickleProperties(pickle_options)

    def reset(self, force: bool = False) -> None:
        """Reset RDKit's pickle options to their default (before instantiating this
        class).
        """
        if force or not self.is_patched:
            self.set(self.default_pickle)

    def patch(self) -> None:
        """Patches property pickling on Windows.

        Notes
        -----
        For some reason RDKit properties pickling need to be set early (before the
        context manager of the parallel classes) to work as expected on Windows.
        """
        if sys.platform == "win32":
            self.set()
            self.is_patched = True


PICKLE_HANDLER = RDKitPickleHandler(PROLIF_PICKLE_OPTIONS)
PICKLE_HANDLER.patch()
