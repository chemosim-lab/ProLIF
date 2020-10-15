"""
Reading RDKit molecules --- :mod:`prolif.molecule`
==================================================
"""
import copy
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from .residue import Residue, ResidueId, ResidueGroup
from .utils import split_mol_in_residues

class Molecule(Chem.Mol):
    """Main molecule class that behaves like an RDKit :class:`~rdkit.Chem.rdchem.Mol`
    with extra attributes (see below). The main purpose of this class is to
    access residues as fragments of the molecule.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        A ligand or protein with a single conformer

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
    
    Example
    -------

    .. ipython:: python
        :okwarning:

        import MDAnalysis as mda
        import prolif
        u = mda.Universe(prolif.datafiles.TOP, prolif.datafiles.TRAJ)
        mol = u.select_atoms("protein").convert_to("RDKIT")
        mol = prolif.Molecule(mol)
        mol
    
    You can also create a Molecule directly from a
    :class:`~MDAnalysis.core.universe.Universe`:

    .. ipython:: python
        :okwarning:

        mol = prolif.Molecule.from_mda(u, "protein")
        mol

    
    Notes
    -----
    Residues can be accessed easily in different ways:
        
    .. ipython:: python

        mol["TYR38.0"] # by resid string (residue name + number + chain)
        mol[42] # by index
        mol[::10] # by slice (start:stop:step)
        mol[[0, 10, 100]] # by list of indices
        mol[["TRP125.0", "ASP129.0"]] # by list of resid
        mol[prolif.ResidueId("ALA")] # by ResidueId
    
    See :mod:`prolif.residue` for more information on residues

    When used on a ligand, this class will split it on peptide bonds (SMARTS
    pattern ``N-C-C(=O)-N``) and disulfide bridges (``[SX2v2]-[SX2v2]``) so use
    indexing on ligands with care: ``mol[0]`` isn't always the same as ``mol``
    """
    def __init__(self, mol):
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
    
    @classmethod
    def from_mda(cls, obj, selection=None):
        """Create a Molecule from an MDAnalysis object
        
        Parameters
        ----------
        obj : MDAnalysis.core.universe.Universe or MDAnalysis.core.groups.AtomGroup
            The MDAnalysis object to convert
        selection : None or str
            Apply a selection to `obj` to create an AtomGroup. Uses all atoms
            in `obj` if ``selection=None``

        Example
        -------
        .. ipython:: python
            :okwarning:

            mol = prolif.Molecule.from_mda(u, "protein")
            mol

        Which is equivalent to:

        .. ipython:: python
            :okwarning:

            protein = u.select_atoms("protein")
            mol = prolif.Molecule.from_mda(protein)
            mol

        """
        ag = obj.select_atoms(selection) if selection else obj.atoms
        return cls(ag.convert_to("RDKIT"))

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
