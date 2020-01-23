import copy
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from .logger import logger

class Residues(OrderedDict):
    """The Residues class, which inherits `collections.OrderedDict`, an ordered
    dictionnary of Residues"""

    def __init__(self, topology):
        super().__init__()
        atoms = {}
        atom_map = {}
        for resname in topology.residues_list:
            atom_map[resname] = {}
            self[resname] = Chem.RWMol()
        for atom in topology.GetAtoms():
            resname = atom.GetMonomerInfo().GetName()
            index = self.get(resname).AddAtom(atom)
            atoms[atom.GetIdx()] = index
            atom_map[resname][index] = atom.GetIdx()
        self.atom_map = atom_map
        for bond in topology.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            res1 = topology.GetAtomWithIdx(a1).GetMonomerInfo().GetName()
            res2 = topology.GetAtomWithIdx(a2).GetMonomerInfo().GetName()
            if res1 == res2:
                self.get(res1).AddBond(*[atoms[i] for i in (a1,a2)], bond.GetBondType())
        for resname in topology.residues_list:
            self.get(resname).SetProp("resname", resname)
            self.get(resname).UpdatePropertyCache(strict=False)

    def __repr__(self):
        name = ".".join([self.__class__.__module__, self.__class__.__name__])
        params = " ".join(list(self.keys()))
        return f"<{name}: [{params}] at 0x{id(self):02x}>"


class ResidueFrame(Chem.Mol):
    """The ResidueFrame class, which inherits the `rdkit.Chem.Mol` class.

    Contains the 3D information from a particular residue and frame"""

    def __init__(self, residue, xyz, atom_map):
        super().__init__(residue)
        conformer = Chem.Conformer()
        for atom in self.GetAtoms():
            atom_index = atom_map.get(atom.GetIdx())
            position = Chem.rdGeometry.Point3D(*xyz[atom_index])
            conformer.SetAtomPosition(atom_index, position)
        self.AddConformer(conformer)

    def __repr__(self):
        name = ".".join([self.__class__.__module__, self.__class__.__name__])
        params = f"{self.resname} with centroid {list(self.centroid)}"
        return f"<{name}: {params} at 0x{id(self):02x}>"

    @property
    def resname(self):
        return self.GetProp("resname")

    @property
    def xyz(self):
        return self.GetConformer().GetPositions()

    @property
    def centroid(self):
        return rdMolTransforms.ComputeCentroid(self.GetConformer())
