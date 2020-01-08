import pytraj as pt
from rdkit import Chem
from .chemistry import Atom, Bond

class Topology(Chem.Mol):
    """
    The Topology class, which inherits from RDKit `Chem.Mol` class.

    In addition to the `Chem.Mol` attributes and methods, the `Topology` class
    has different methods to instantiate the object from different popular
    packages such as pytraj.
    """

    def __init__(self, atoms, bonds):
        # RDKit editable molecule
        mol = Chem.RWMol()
        # add atoms to the molecule
        for atom in atoms:
            # create atom
            a = Chem.Atom(atom.atomic_number)
            # add atom name
            a.SetProp("_name", atom.name)
            # add residue name
            a.SetMonomerInfo(Chem.AtomPDBResidueInfo(atom.resname))
            # set charge
            a.SetFormalCharge(int(atom.charge))
            a.SetDoubleProp("_charge", atom.charge)
            # disable adding H to the molecule
            a.SetNoImplicit(True)
            # add atom to molecule
            mol.AddAtom(a)
        # add bonds
        for bond in bonds:
            mol.AddBond(*bond.indices, bond.type)
        mol.UpdatePropertyCache()
        super().__init__(mol)

    @classmethod
    def from_pytraj(cls, topology):
        """Create a topology from a pytraj `Topology`"""
        atoms = [Atom.from_pytraj(a) for a in topology.atoms]
        bonds = [Bond.from_pytraj(b) for b in topology.bonds]
        return cls(atoms, bonds)
