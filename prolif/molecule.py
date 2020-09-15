"""
prolif.molecule
===============
"""
import copy
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from .residue import Residue, ResidueId, ResidueGroup
from .utils import split_mol_in_residues

class Molecule(Chem.Mol):
    """Main molecule class that behaves like an RDKit :class:`~rdkit.Chem.rdchem.Mol`
    with extra attributes (see below)

    Attributes
    ----------
    residues : prolif.residue.ResidueGroup
        A dictionnary storing one/many :class:`~prolif.residue.Residue` indexed
        by :class:`~prolif.residue.ResidueId`. The residue list is sorted.
    n_residues : int
        Number of residues
    centroid : numpy.ndarray
        XYZ coordinates of the centroid of the molecule
    xyz : numpy.ndarray
        XYZ coordinates of all atoms in the molecule
    
    Notes
    -----
    Residues can be accessed easily in different manners::
        TODO
        >>> mol["TYR51"]
        >>> mol[prolif.ResidueId("ALA")]
        >>> mol[42]

    """
    def __init__(self, mol):
        """
        RDKit-like molecule that is splitted in residues for a more convenient
        usage.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            A ligand or protein with a single conformer
        """
        super().__init__(mol)
        # set mapping of atoms
        for atom in self.GetAtoms():
            atom.SetUnsignedProp("mapindex", atom.GetIdx())
        # split in residues
        residues = split_mol_in_residues(self)
        residues = {ResidueId.from_atom(mol.GetAtomWithIdx(0)): Residue(mol)
                    for mol in residues}
        self.residues = ResidueGroup(sorted([(resid, res)
                                     for resid, res in residues.items()],
                                     key=lambda x: (x[0].chain, x[0].number)))

    def __iter__(self):
        for residue in self.residues.values():
            yield residue

    def __getitem__(self, key):
        return self.residues[key]

    def __repr__(self):
        name = ".".join([self.__class__.__module__, self.__class__.__name__])
        params = f"{self.n_residues} residues and {self.GetNumAtoms()} atoms"
        return f"<{name} with {params} at {id(self):#x}>"

    @property
    def n_residues(self):
        return len(self.residues)

    @property
    def centroid(self):
        return rdMolTransforms.ComputeCentroid(self.GetConformer())

    @property
    def xyz(self):
        return self.GetConformer().GetPositions()
