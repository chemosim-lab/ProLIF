import re, copy
from rdkit import Chem
from .chemistry import Atom, Bond
from .residue import Residues
from .utils import update_bonds_and_charges


class Topology(Chem.Mol):
    """
    The Topology class, which inherits from RDKit `Chem.Mol` class.

    In addition to the `Chem.Mol` attributes and methods, the `Topology` class
    has different methods to instantiate the object from different popular
    packages such as mdtraj and pytraj.
    """

    def __init__(self, atoms, bonds):
        residues_list = []
        # RDKit editable molecule
        mol = Chem.RWMol()
        # add atoms to the molecule
        for atom in atoms:
            # create RDKit atom
            a = atom.to_rdkit()
            # add atom to molecule
            mol.AddAtom(a)
            # update residues list
            if atom.residue not in residues_list:
                residues_list.append(atom.residue)
        residues_list.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
        # add bonds
        for bond in bonds:
            mol.AddBond(*bond.indices, bond.bond_type)
        mol.UpdatePropertyCache(strict=False)
        # update bondtype and charges
        update_bonds_and_charges(mol)
        super().__init__(mol)
        self.residues_list = residues_list
        self.residues = Residues(self)

    @classmethod
    def from_pytraj(cls, topology):
        """Create a topology from a pytraj `Topology`"""
        atoms = [Atom.from_pytraj(a) for a in topology.atoms]
        bonds = [Bond.from_pytraj(b) for b in topology.bonds]
        return cls(atoms, bonds)

    @classmethod
    def from_mdtraj(cls, topology):
        """Create a topology from a mdtraj `Topology`"""
        atoms = [Atom.from_mdtraj(a) for a in topology.atoms]
        bonds = [Bond.from_mdtraj(b) for b in topology.bonds]
        return cls(atoms, bonds)

    @classmethod
    def from_rdkit(cls, mol):
        """Create a topology from a RDKit `Chem.Mol`"""
        topology = copy.deepcopy(mol)
        residues_list = []
        for atom in topology.GetAtoms():
            mi = atom.GetMonomerInfo()
            if mi:
                resname = "%s%d" % (t.GetResidueName(), t.GetResidueNumber())
            else:
                mi = Chem.AtomPDBResidueInfo()
                mi.SetResidueName("UNL")
                mi.SetResidueNumber(0)
                atom.SetMonomerInfo(mi)
                resname = "UNL0"
                atom.SetProp("residue_name", resname)
            if resname not in residues_list:
                residues_list.append(resname)
        topology.residues_list = residues_list
        topology.residues = Residues(topology)
        return topology

    def __repr__(self):
        name = ".".join([self.__class__.__module__, self.__class__.__name__])
        params = f"{len(self.residues_list)} residues, {self.GetNumAtoms()} atoms, {self.GetNumBonds()} bonds, {Chem.GetSSSR(self)} rings"
        return f"<{name}: {params} at 0x{id(self):02x}>"
