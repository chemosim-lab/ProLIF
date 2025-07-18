"""
Reading proteins and ligands --- :mod:`prolif.molecule`
=======================================================
"""

import copy
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from operator import attrgetter
from typing import TYPE_CHECKING, Any, TypeVar, Union, overload

import MDAnalysis as mda
from rdkit import Chem
from rdkit.Chem.AllChem import AssignBondOrdersFromTemplate

from prolif.rdkitmol import BaseRDKitMol
from prolif.residue import Residue, ResidueGroup, ResidueId
from prolif.utils import catch_rdkit_logs, catch_warning, split_mol_by_residues

if TYPE_CHECKING:
    from pathlib import Path

    from prolif.typeshed import MDAObject, ResidueKey

    Self = TypeVar("Self", bound=BaseRDKitMol)


class Molecule(BaseRDKitMol):
    """Main molecule class that behaves like an RDKit :class:`~rdkit.Chem.rdchem.Mol`
    with extra attributes (see examples below). The main purpose of this class
    is to access residues as fragments of the molecule.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        A ligand or protein with a single conformer
    use_segid: bool, default = False
        Use the segment number rather than the chain identifier as a chain.

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

    .. versionchanged:: 2.1.0
        Added `use_segid`.
    """

    def __init__(self, mol: Chem.Mol, *, use_segid: bool = False) -> None:
        super().__init__(mol)
        # set mapping of atoms
        for atom in self.GetAtoms():
            atom.SetUnsignedProp("mapindex", atom.GetIdx())
        # split in residues
        residues = split_mol_by_residues(self)
        residues = [Residue(mol, use_segid=use_segid) for mol in residues]
        residues.sort(key=attrgetter("resid"))
        self.residues = ResidueGroup(residues)

    @classmethod
    def from_mda(
        cls,
        obj: "MDAObject",
        selection: str | None = None,
        *,
        use_segid: bool | None = None,
        **kwargs: Any,
    ) -> "Molecule":
        """Creates a Molecule from an MDAnalysis object

        Parameters
        ----------
        obj : MDAnalysis.core.universe.Universe or MDAnalysis.core.groups.AtomGroup
            The MDAnalysis object to convert
        selection : None or str
            Apply a selection to `obj` to create an AtomGroup. Uses all atoms
            in `obj` if ``selection=None``
        use_segid: bool | None, default = None
            Use the segment number rather than the chain identifier as a chain.
        **kwargs : object
            Other arguments passed to the
            :class:`~MDAnalysis.converters.RDKit.RDKitConverter` of MDAnalysis

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

        .. versionchanged:: 2.1.0
            Added `use_segid`.
        """
        ag = obj.select_atoms(selection) if selection else obj.atoms
        if ag.n_atoms == 0:
            raise mda.SelectionError("AtomGroup is empty, please check your selection")
        mol = ag.convert_to.rdkit(**kwargs)
        use_segid = cls._use_segid(ag, use_segid)
        return cls(mol, use_segid=use_segid)

    @classmethod
    def _use_segid(cls, obj: "MDAObject", use_segid: bool | None = None) -> bool:
        """Whether to use the segment index or the chainID as a chain."""
        if use_segid is not None:
            return use_segid
        ag = obj.atoms
        try:
            # chainID not always defined
            n_chains = len(set(ag.chainIDs))
        except Exception:
            return True
        # segindices is always defined
        n_segids = len(set(ag.residues.segindices))
        return n_segids > n_chains

    @classmethod
    def from_rdkit(
        cls,
        mol: Chem.Mol,
        resname: str = "UNL",
        resnumber: int = 1,
        chain: str = "",
        use_segid: bool = False,
    ) -> "Molecule":
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
        use_segid: bool, default = False
            Use the segment number rather than the chain identifier as a chain

        Notes
        -----
        This method only checks for an existing AtomPDBResidueInfo in the first
        atom. If none was found, it will patch all atoms with the one created
        from the method's arguments (resname, resnumber, chain).

        .. versionchanged:: 2.1.0
            Added `use_segid`.
        """
        if mol.GetAtomWithIdx(0).GetMonomerInfo():
            return cls(mol, use_segid=use_segid)
        mol = copy.deepcopy(mol)
        for atom in mol.GetAtoms():
            mi = Chem.AtomPDBResidueInfo(
                f" {atom.GetSymbol():<3.3}",
                residueName=resname,
                residueNumber=resnumber,
                chainId=chain,
            )
            atom.SetMonomerInfo(mi)
        return cls(mol)

    def __iter__(self) -> Iterator[Residue]:
        yield from self.residues.values()

    def __getitem__(self, key: "ResidueKey") -> Residue:
        return self.residues[key]

    def __repr__(self) -> str:  # pragma: no cover
        name = ".".join([self.__class__.__module__, self.__class__.__name__])
        params = f"{self.n_residues} residues and {self.GetNumAtoms()} atoms"
        return f"<{name} with {params} at {id(self):#x}>"

    @property
    def n_residues(self) -> int:
        return len(self.residues)


class pdbqt_supplier(Sequence[Molecule]):
    """Supplies molecules, given paths to PDBQT files

    Parameters
    ----------
    paths : list
        A list (or any iterable) of PDBQT files
    template : rdkit.Chem.rdchem.Mol
        A template molecule with the correct bond orders and charges. It must
        match exactly the molecule inside the PDBQT file.
    converter_kwargs : dict | None
        Keyword arguments passed to the RDKitConverter of MDAnalysis
    resname : str
        Residue name for every ligand
    resnumber : int
        Residue number for every ligand
    chain : str
        Chain ID for every ligand


    Returns
    -------
    suppl : Sequence
        A sequence that provides :class:`Molecule` objects

    Example
    -------
    The supplier is typically used like this::

        >>> import glob
        >>> pdbqts = glob.glob("docking/ligand1/*.pdbqt")
        >>> lig_suppl = pdbqt_supplier(pdbqts, template)
        >>> for lig in lig_suppl:
        ...     # do something with each ligand

    .. versionchanged:: 1.0.0
        Molecule suppliers are now sequences that can be reused, indexed,
        and can return their length, instead of single-use generators.

    .. versionchanged:: 1.1.0
        Because the PDBQT supplier needs to strip hydrogen atoms before
        assigning bond orders from the template, it used to replace them
        entirely with hydrogens containing new coordinates. It now directly
        uses the hydrogen atoms present in the file and won't add explicit
        ones anymore, to prevent the fingerprint from detecting hydrogen bonds
        with "random" hydrogen atoms.
        A lot of irrelevant warnings and logs have been disabled as well.

    """

    def __init__(
        self,
        paths: Iterable[Union[str, "Path"]],
        template: Chem.Mol,
        converter_kwargs: dict | None = None,
        **kwargs: Any,
    ) -> None:
        self.paths = list(paths)
        self.template = template
        converter_kwargs = converter_kwargs or {}
        converter_kwargs.pop("NoImplicit", None)
        self.converter_kwargs = converter_kwargs
        self._kwargs = kwargs

    def __iter__(self) -> Iterator[Molecule]:
        for pdbqt_path in self.paths:
            yield self.pdbqt_to_mol(pdbqt_path)

    @overload
    def __getitem__(self, index: int) -> Molecule: ...
    @overload
    def __getitem__(self, index: slice) -> "pdbqt_supplier": ...
    def __getitem__(self, index: int | slice) -> Union[Molecule, "pdbqt_supplier"]:
        if isinstance(index, slice):
            return pdbqt_supplier(
                self.paths[index],
                self.template,
                converter_kwargs=self.converter_kwargs,
                **self._kwargs,
            )

        pdbqt_path = self.paths[index]
        return self.pdbqt_to_mol(pdbqt_path)

    def pdbqt_to_mol(self, pdbqt_path: Union[str, "Path"]) -> Molecule:
        with catch_warning(message=r"^Failed to guess the mass"):
            pdbqt = mda.Universe(pdbqt_path)
        # set attributes needed by the converter
        elements = [
            mda.topology.guessers.guess_atom_element(x) for x in pdbqt.atoms.names
        ]
        pdbqt.add_TopologyAttr("elements", elements)
        pdbqt.add_TopologyAttr("chainIDs", pdbqt.atoms.segids)
        pdbqt.atoms.types = pdbqt.atoms.elements
        # convert without infering bond orders and charges
        with (
            catch_rdkit_logs(),
            catch_warning(
                message=r"^(Could not sanitize molecule)|"
                r"(No `bonds` attribute in this AtomGroup)",
            ),
        ):
            pdbqt_mol = pdbqt.atoms.convert_to.rdkit(
                NoImplicit=False,
                **self.converter_kwargs,
            )
        mol = self._adjust_hydrogens(self.template, pdbqt_mol)
        return Molecule.from_rdkit(mol, **self._kwargs)

    @staticmethod
    def _adjust_hydrogens(template: Chem.Mol, pdbqt_mol: Chem.Mol) -> Chem.Mol:
        # remove explicit hydrogens and assign BO from template
        pdbqt_noH = Chem.RemoveAllHs(pdbqt_mol, sanitize=False)
        with catch_rdkit_logs():
            mol: Chem.Mol = AssignBondOrdersFromTemplate(template, pdbqt_noH)
        # mapping between pdbindex of atom bearing H --> H atom(s)
        atoms_with_hydrogens: defaultdict[int, list[Chem.Atom]] = defaultdict(list)
        for atom in pdbqt_mol.GetAtoms():
            if atom.GetAtomicNum() == 1:
                atoms_with_hydrogens[
                    atom.GetNeighbors()[0].GetIntProp("_MDAnalysis_index")
                ].append(atom)
        # mapping between atom that should be bearing a H in RWMol and
        # corresponding hydrogens
        reverse_mapping: dict[int, list[Chem.Atom]] = {}
        for atom in mol.GetAtoms():
            if (idx := atom.GetIntProp("_MDAnalysis_index")) in atoms_with_hydrogens:
                reverse_mapping[atom.GetIdx()] = atoms_with_hydrogens[idx]
                atom.SetNumExplicitHs(0)
        # add missing Hs
        pdb_conf = pdbqt_mol.GetConformer()
        rwmol = Chem.RWMol(mol)
        mol_conf = rwmol.GetConformer()
        for atom_idx, hydrogens in reverse_mapping.items():
            for hydrogen in hydrogens:
                h_idx = rwmol.AddAtom(hydrogen)
                xyz = pdb_conf.GetAtomPosition(hydrogen.GetIdx())
                mol_conf.SetAtomPosition(h_idx, xyz)
                rwmol.AddBond(atom_idx, h_idx, Chem.BondType.SINGLE)
        mol = rwmol.GetMol()
        # sanitize
        mol.UpdatePropertyCache()
        Chem.SanitizeMol(mol)
        return mol

    def __len__(self) -> int:
        return len(self.paths)


class sdf_supplier(Sequence[Molecule]):
    """Supplies molecules, given a path to an SDFile

    Parameters
    ----------
    path : str
        A path to the .sdf file
    sanitize : bool
        Whether to sanitize each molecule or not.
    resname : str
        Residue name for every ligand
    resnumber : int
        Residue number for every ligand
    chain : str
        Chain ID for every ligand

    Returns
    -------
    suppl : Sequence
        A sequence that provides :class:`Molecule` objects. Can be indexed

    Example
    -------
    The supplier is typically used like this::

        >>> lig_suppl = sdf_supplier("docking/output.sdf")
        >>> for lig in lig_suppl:
        ...     # do something with each ligand

    .. versionchanged:: 1.0.0
        Molecule suppliers are now sequences that can be reused, indexed,
        and can return their length, instead of single-use generators.

    .. versionchanged:: 2.1.0
        Added ``sanitize`` parameter (defaults to ``True``, same behavior as before).

    """

    def __init__(self, path: str, sanitize: bool = True, **kwargs: Any) -> None:
        self.path = path
        self._suppl = Chem.SDMolSupplier(path, removeHs=False, sanitize=sanitize)
        self._kwargs = kwargs

    def __iter__(self) -> Iterator[Molecule]:
        for mol in self._suppl:
            yield Molecule.from_rdkit(mol, **self._kwargs)

    @overload
    def __getitem__(self, index: int) -> Molecule: ...
    @overload
    def __getitem__(self, index: slice) -> "sdf_supplier": ...
    def __getitem__(self, index: int | slice) -> Union[Molecule, "sdf_supplier"]:
        if isinstance(index, slice):
            suppl = sdf_supplier(self.path, **self._kwargs)
            suppl._suppl = [  # type: ignore[assignment]
                self[i] for i in range(*index.indices(len(self)))
            ]
            return suppl

        mol = self._suppl[index]
        return Molecule.from_rdkit(mol, **self._kwargs)

    def __len__(self) -> int:
        return len(self._suppl)


class mol2_supplier(Sequence[Molecule]):
    """Supplies molecules, given a path to a MOL2 file

    Parameters
    ----------
    path : str
        A path to the .mol2 file
    sanitize : bool
        Whether to sanitize each molecule or not.
    cleanup_substructures : bool
        Toggles standardizing some substructures found in mol2 files, based on atom
        types.
    resname : str
        Residue name for every ligand
    resnumber : int
        Residue number for every ligand
    chain : str
        Chain ID for every ligand

    Returns
    -------
    suppl : Sequence
        A sequence that provides :class:`Molecule` objects

    Example
    -------
    The supplier is typically used like this::

        >>> lig_suppl = mol2_supplier("docking/output.mol2")
        >>> for lig in lig_suppl:
        ...     # do something with each ligand

    .. versionchanged:: 1.0.0
        Molecule suppliers are now sequences that can be reused, indexed,
        and can return their length, instead of single-use generators.

    .. versionchanged:: 2.1.0
        Added ``cleanup_substructures`` and ``sanitize`` parameters
        (default to ``True``, same behavior as before).

    """

    def __init__(
        self,
        path: Union[str, "Path"],
        cleanup_substructures: bool = True,
        sanitize: bool = True,
        **kwargs: Any,
    ) -> None:
        self.path = path
        self.cleanup_substructures = cleanup_substructures
        self.sanitize = sanitize
        self._kwargs = kwargs

    def __iter__(self) -> Iterator[Molecule]:
        block: list[str] = []
        with open(self.path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                if block and line.startswith("@<TRIPOS>MOLECULE"):
                    yield self.block_to_mol(block)
                    block = []
                block.append(line)
            yield self.block_to_mol(block)

    def block_to_mol(self, block: list[str]) -> Molecule:
        mol = Chem.MolFromMol2Block(
            "".join(block),
            removeHs=False,
            cleanupSubstructures=self.cleanup_substructures,
            sanitize=self.sanitize,
        )
        return Molecule.from_rdkit(mol, **self._kwargs)

    @overload
    def __getitem__(self, index: int) -> Molecule: ...
    @overload
    def __getitem__(self, index: slice) -> "mol2_supplier": ...
    def __getitem__(self, index: int | slice) -> Union[Molecule, "mol2_supplier"]:
        if isinstance(index, slice):
            raise NotImplementedError("Slicing not available for mol2_supplier.")

        if index < 0:
            index %= len(self)
        mol_index = -1
        molblock_started = False
        block: list[str] = []
        with open(self.path) as f:
            for line in f:
                if line.startswith("@<TRIPOS>MOLECULE"):
                    mol_index += 1
                    if mol_index > index:
                        return self.block_to_mol(block)
                    if mol_index == index:
                        molblock_started = True
                if molblock_started:
                    block.append(line)
        if block:
            return self.block_to_mol(block)
        raise ValueError(f"Could not parse molecule with index {index}")

    def __len__(self) -> int:
        with open(self.path) as f:
            return sum(line.startswith("@<TRIPOS>MOLECULE") for line in f)


def split_molecule(
    mol: Molecule, predicate: Callable[[ResidueId], bool]
) -> tuple[Molecule, Molecule]:
    """
    Splits a molecule into two based on a predicate function. The first molecule
    returned contains all residues of the input mol for which the predicate function was
    true, the second molecule contains the rest.
    This function is typically used to extract a ligand or water molecules from a file
    containing a solvated complex::

        >>> solvated_system = Molecule(...)
        >>> water_mol, protein_mol = split_molecule(
        ...     solvated_system, lambda x: x.name == "WAT"
        ... )

    .. versionadded:: 2.1.0

    """
    with Chem.RWMol(mol) as lhs, Chem.RWMol(mol) as rhs:
        for residue in mol:
            target = rhs if predicate(residue.resid) else lhs
            for atom in residue.GetAtoms():
                target.RemoveAtom(atom.GetUnsignedProp("mapindex"))
    return Molecule(lhs.GetMol()), Molecule(rhs.GetMol())
