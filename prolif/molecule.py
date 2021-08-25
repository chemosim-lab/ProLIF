"""
Reading proteins and ligands --- :mod:`prolif.molecule`
=======================================================
"""
import copy
from operator import attrgetter
import MDAnalysis as mda
from rdkit import Chem
from rdkit.Chem.AllChem import AssignBondOrdersFromTemplate
from .rdkitmol import BaseRDKitMol
from .residue import Residue, ResidueGroup
from .utils import split_mol_by_residues


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
        """Creates a Molecule from an MDAnalysis object
        
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
    
    @classmethod
    def from_rdkit(cls, mol, resname="UNL", resnumber=1, chain=""):
        """Creates a Molecule from an RDKit molecule
        
        While directly instantiating a molecule with ``prolif.Molecule(mol)``
        would also work, this method insures that every atom is linked to an
        AtomPDBResidueInfo which is required by ProLIF
        
        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            The input RDKit molecule
        resname : str
            The default residue name that is used if none was found
        resnumber : int
            The default residue number that is used if none was found
        chain : str
            The default chain Id that is used if none was found
        
        Notes
        -----
        This method only checks for an existing AtomPDBResidueInfo in the first
        atom. If none was found, it will patch all atoms with the one created
        from the method's arguments (resname, resnumber, chain).
        """
        if mol.GetAtomWithIdx(0).GetMonomerInfo():
            return cls(mol)
        mol = copy.deepcopy(mol)
        for atom in mol.GetAtoms():
            mi = Chem.AtomPDBResidueInfo(f" {atom.GetSymbol():<3.3}",
                                         residueName=resname,
                                         residueNumber=resnumber,
                                         chainId=chain)
            atom.SetMonomerInfo(mi)
        return cls(mol)

    def __iter__(self):
        for residue in self.residues.values():
            yield residue

    def __getitem__(self, key):
        return self.residues[key]

    def __repr__(self): # pragma: no cover
        name = ".".join([self.__class__.__module__, self.__class__.__name__])
        params = f"{self.n_residues} residues and {self.GetNumAtoms()} atoms"
        return f"<{name} with {params} at {id(self):#x}>"

    @property
    def n_residues(self):
        return len(self.residues)


def pdbqt_supplier(paths, template, **kwargs):
    """Supplies molecules, given a path to PDBQT files

    Parameters
    ----------
    paths : list
        A list (or any iterable) of PDBQT files
    template : rdkit.Chem.rdchem.Mol
        A template molecule with the correct bond orders and charges. It must
        match exactly the molecule inside the PDBQT file.
    kwargs : object
        Keyword arguments passed to the RDKitConverter of MDAnalysis

    Returns
    -------
    suppl : generator
        A generator object that will provide the :class:`Molecule` object as
        you iterate over it.

    Example
    -------
    The supplier is typically used like this::

        >>> import glob
        >>> pdbqts = glob.glob("docking/ligand1/*.pdbqt")
        >>> lig_suppl = pdbqt_supplier(pdbqts, template)
        >>> for lig in lig_suppl:
        ...     # do something with each ligand

    """
    for pdbqt_path in paths:
        pdbqt = mda.Universe(pdbqt_path)
        # set attributes needed by the converter
        elements = [mda.topology.guessers.guess_atom_element(x)
                    for x in pdbqt.atoms.names]
        pdbqt.add_TopologyAttr("elements", elements)
        pdbqt.add_TopologyAttr("chainIDs", pdbqt.atoms.segids)
        pdbqt.atoms.types = pdbqt.atoms.elements
        # convert without infering bond orders and charges
        kwargs.pop("NoImplicit", None)
        mol = pdbqt.atoms.convert_to.rdkit(NoImplicit=False, **kwargs)
        # assign BO from template then add hydrogens
        mol = Chem.RemoveHs(mol, sanitize=False)
        mol = AssignBondOrdersFromTemplate(template, mol)
        mol = Chem.AddHs(mol, addCoords=True, addResidueInfo=True)
        yield Molecule(mol)


def sdf_supplier(path, **kwargs):
    """Supplies molecules, given a path to an SDFile
    
    Parameters
    ----------
    path : str
        A path to the .sdf file
    resname : str
        Residue name for every ligand
    resnumber : int
        Residue number for every ligand
    chain : str
        Chain ID for every ligand

    Returns
    -------
    suppl : generator
        A generator object that will provide the :class:`Molecule` object as
        you iterate over it.

    Example
    -------
    The supplier is typically used like this::

        >>> lig_suppl = sdf_supplier("docking/output.sdf")
        >>> for lig in lig_suppl:
        ...     # do something with each ligand

    """
    suppl = Chem.SDMolSupplier(path, removeHs=False)
    for mol in suppl:
        yield Molecule.from_rdkit(mol, **kwargs)


def mol2_supplier(path, **kwargs):
    """Generates prolif.Molecule objects from a MOL2 file
    
    Parameters
    ----------
    path : str
        A path to the .mol2 file
    resname : str
        Residue name for every ligand
    resnumber : int
        Residue number for every ligand
    chain : str
        Chain ID for every ligand

    Returns
    -------
    suppl : generator
        A generator object that will provide the :class:`Molecule` object as
        you iterate over it.

    Example
    -------
    The supplier is typically used like this::

        >>> lig_suppl = mol2_supplier("docking/output.mol2")
        >>> for lig in lig_suppl:
        ...     # do something with each ligand

    """
    block = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            if block and line.startswith("@<TRIPOS>MOLECULE"):
                mol = Chem.MolFromMol2Block("".join(block), removeHs=False)
                yield Molecule.from_rdkit(mol, **kwargs)
                block = []
            block.append(line)
        mol = Chem.MolFromMol2Block("".join(block), removeHs=False)
        yield Molecule.from_rdkit(mol, **kwargs)
                
