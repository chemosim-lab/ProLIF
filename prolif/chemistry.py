from rdkit import Chem
from .utils import periodic_table


class Atom:
    """
    The Atom class.

    Each atom is defined by an atomic number.
    Optionnal information include the symbol, name, residue name, and charge.
    """

    def __init__(self, atomic_number, symbol=None, name=None, resname="UNL",
        charge=0.0):
        self.atomic_number = atomic_number
        self.resname = resname
        self.charge = charge
        if symbol:
            self.symbol = symbol
        else:
            self.symbol = periodic_table.GetElementSymbol(self.atomic_number)
        if name:
            self.name = name
        else:
            self.name = self.symbol

    @classmethod
    def from_pytraj(cls, atom):
        """Create an atom from a pytraj `Atom`"""
        kwargs = dict((name, getattr(atom, name))
            for name in ["atomic_number", "name", "resname", "charge"])
        return cls(**kwargs)


class Bond:
    """
    The Bond class.

    Each bond is defined by a tuple of indices, and a type from `Chem.BondType`.
    """

    def __init__(self, indices, type=Chem.BondType.UNSPECIFIED):
        self.indices = indices
        self.type = type

    @classmethod
    def from_pytraj(cls, bond):
        """Create a bond from a pytraj `Bond`"""
        return cls([int(i) for i in bond.indices])
