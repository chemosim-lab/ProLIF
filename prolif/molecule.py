"""
Reading proteins and ligands --- :mod:`prolif.molecule`
=======================================================
"""
from operator import attrgetter
from MDAnalysis import _CONVERTERS
from .rdkitmol import BaseRDKitMol
from .residue import Residue, ResidueGroup
from .utils import split_mol_by_residues


mda_to_rdkit = _CONVERTERS["RDKIT"]().convert


class Molecule(BaseRDKitMol):
    """Main molecule class that behaves like an RDKit :class:`~rdkit.Chem.rdchem.Mol`
    with extra attributes (see examples below). The main purpose of this class
    is to access residues as fragments of the molecule.

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
    
    Examples
    --------

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

        mol["TYR38.A"] # by resid string (residue name + number + chain)
        mol[42] # by index (from 0 to n_residues-1)
        mol[prolif.ResidueId("TYR", 38, "A")] # by ResidueId
    
    See :mod:`prolif.residue` for more information on residues
    """
    def __init__(self, mol):
        super().__init__(mol)
        # set mapping of atoms
        for atom in self.GetAtoms():
            atom.SetUnsignedProp("mapindex", atom.GetIdx())
        # split in residues
        residues = split_mol_by_residues(self)
        residues = [Residue(mol) for mol in residues]
        residues.sort(key=attrgetter("resid"))
        self.residues = ResidueGroup(residues)
    
    @classmethod
    def from_mda(cls, obj, selection=None, **kwargs):
        """Create a Molecule from an MDAnalysis object
        
        Parameters
        ----------
        obj : MDAnalysis.core.universe.Universe or MDAnalysis.core.groups.AtomGroup
            The MDAnalysis object to convert
        selection : None or str
            Apply a selection to `obj` to create an AtomGroup. Uses all atoms
            in `obj` if ``selection=None``
        **kwargs : object
            Other arguments passed to the RDKitConverter of MDAnalysis

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
        mol = mda_to_rdkit(ag, **kwargs)
        return cls(mol)

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
