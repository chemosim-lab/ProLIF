"""
Reading RDKit molecules --- :mod:`prolif.rdkitmol`
==================================================
"""
from rdkit import Chem
from rdkit.Chem.rdMolTransforms import ComputeCentroid

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
    centroid : numpy.ndarray
        XYZ coordinates of the centroid of the molecule
    xyz : numpy.ndarray
        XYZ coordinates of all atoms in the molecule
    """
    @property
    def centroid(self):
        return ComputeCentroid(self.GetConformer())

    @property
    def xyz(self):
        return self.GetConformer().GetPositions()
