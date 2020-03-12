from rdkit import Chem
from .utils import PERIODIC_TABLE, BONDTYPE_TO_RDKIT, BONDORDER_TO_RDKIT


class Atom:
    """
    The Atom class.

    Each atom is defined by an atomic number.
    Optionnal information include the symbol, name, residue name, and charge.
    """

    def __init__(self, atomic_number, symbol=None, name=None, charge=0,
        resname="UNL", resid=0):
        self.atomic_number = atomic_number
        self.resname = resname
        self.resid = resid
        self.residue = "%s%d" % (self.resname, self.resid)
        self.charge = charge
        if symbol:
            self.symbol = symbol
        else:
            self.symbol = PERIODIC_TABLE.GetElementSymbol(self.atomic_number)
        if name:
            self.name = name
        else:
            self.name = self.symbol

    @classmethod
    def from_pytraj(cls, atom):
        """Create an atom from a pytraj `Atom`"""
        return cls(atom.atomic_number, name=atom.name,
            resname=atom.resname, resid=atom.resid)

    @classmethod
    def from_mdtraj(cls, atom):
        """Create an atom from a mdtraj `Atom`"""
        number, name, symbol, mass, radius = atom.element
        res = atom.residue
        return cls(number, symbol=symbol, name=atom.name,
            resname=res.name, resid=res.resSeq)

    @classmethod
    def from_openbabel(cls, atom):
        """Create an atom from an openbabel `OBAtom`"""
        residue = atom.GetResidue()
        return cls(atom.GetAtomicNum(), name=atom.GetType(), charge=atom.GetFormalCharge(),
            resname=residue.GetName(), resid=residue.GetNum())

    def to_rdkit(self):
        a = Chem.Atom(self.atomic_number)
        # add atom name
        a.SetProp("_Name", self.name)
        # add residue name
        mi = Chem.AtomPDBResidueInfo()
        mi.SetResidueName(self.resname)
        mi.SetResidueNumber(self.resid)
        a.SetMonomerInfo(mi)
        a.SetProp("residue_name", self.residue)
        # set partial charge
        a.SetFormalCharge(self.charge)
        # disable adding H to the molecule
        a.SetNoImplicit(True)
        return a

    def __repr__(self):
        name = ".".join([self.__class__.__module__, self.__class__.__name__])
        params = f"{self.symbol}[{self.atomic_number}], name={self.name}, residue={self.residue}, charge={self.charge}"
        return f"<{name}({params}) at 0x{id(self):02x}>"


class Bond:
    """
    The Bond class.

    Each bond is defined by a tuple of indices, and a type from `Chem.BondType`.
    """

    def __init__(self, indices, bond_type=None):
        self.indices = indices
        if bond_type:
            bond_type = str(bond_type).upper()
            bond_type = BONDTYPE_TO_RDKIT.get(bond_type, Chem.BondType.SINGLE)
        else:
            bond_type = Chem.BondType.SINGLE
        self.bond_type = bond_type

    @classmethod
    def from_pytraj(cls, bond):
        """Create a bond from a pytraj `Bond`"""
        return cls([int(i) for i in bond.indices])

    @classmethod
    def from_mdtraj(cls, bond):
        """Create a bond from a mdtraj `Bond`"""
        a1, a2, bond_type = bond.atom1, bond.atom2, bond.type
        return cls([a1.index, a2.index], bond_type)

    @classmethod
    def from_openbabel(cls, bond):
        """Create a bond from an openbabel `OBBond`"""
        return cls([bond.GetBeginAtomIdx()-1, bond.GetEndAtomIdx()-1],
            BONDORDER_TO_RDKIT.get(bond.GetBondOrder(), Chem.BondType.SINGLE))

    def __repr__(self):
        name = ".".join([self.__class__.__module__, self.__class__.__name__])
        params = f"{self.indices}, bond_type={self.bond_type}"
        return f"<{name}({params}) at 0x{id(self):02x}>"
