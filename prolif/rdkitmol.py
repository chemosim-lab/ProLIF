"""
Reading RDKit molecules --- :mod:`prolif.rdkitmol`
==================================================
"""

from typing import TYPE_CHECKING

from rdkit import Chem
from rdkit.Chem.rdMolTransforms import ComputeCentroid

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from rdkit.Geometry import Point3D


class BaseRDKitMol(Chem.Mol):
    """Base molecular class that behaves like an RDKit :class:`~rdkit.Chem.rdchem.Mol`
    with extra attributes (see below).
    The sole purpose of this class is to define the common API between the
    :class:`~prolif.molecule.Molecule` and :class:`~prolif.residue.Residue` classes.
    This class should not be instantiated by users.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        A molecule (protein, ligand, or residue) with a single conformer

    Attributes
    ----------
    centroid : rdkit.Geometry.rdGeometry.Point3D
        XYZ coordinates of the centroid of the molecule
    xyz : numpy.ndarray
        XYZ coordinates of all atoms in the molecule
    """

    @property
    def centroid(self) -> "Point3D":
        return ComputeCentroid(self.GetConformer())  # type: ignore[no-any-return]

    @property
    def xyz(self) -> "NDArray[np.float64]":
        return self.GetConformer().GetPositions()  # type: ignore[no-any-return]
