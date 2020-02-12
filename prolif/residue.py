import copy
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import rdMolTransforms


class Residues(OrderedDict):
    """The Residues class, which inherits `collections.OrderedDict`, an ordered
    dictionnary of Residues"""

    def __init__(self, topology):
        super().__init__()
        # map topology atoms to residue atoms
        atom_map_top_to_res = {}
        # map residue atoms to the the full topology
        atom_map = {}
        # init each residue
        for resname in topology.residues_list:
            atom_map[resname] = {}
            self[resname] = Chem.RWMol()
        # atoms
        for atom in topology.GetAtoms():
            resname = atom.GetProp("residue_name")
            top_index = atom.GetIdx()
            res_index = self[resname].AddAtom(atom)
            atom_map_top_to_res[top_index] = res_index
            atom_map[resname][res_index] = top_index
        # store the atom mapping for creating a ResidueFrame from a Frame
        self.atom_map = atom_map
        # bonds
        for bond in topology.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            res1 = topology.GetAtomWithIdx(a1).GetProp("residue_name")
            res2 = topology.GetAtomWithIdx(a2).GetProp("residue_name")
            if res1 == res2:
                self[res1].AddBond(*[atom_map_top_to_res[i] for i in (a1,a2)], bond.GetBondType())
        for resname in topology.residues_list:
            self[resname].SetProp("resname", resname)
            self[resname].UpdatePropertyCache(strict=False)

    def __repr__(self):
        name = ".".join([self.__class__.__module__, self.__class__.__name__])
        params = " ".join(list(self.keys()))
        return f"<{name}: [{params}] at 0x{id(self):02x}>"


class ResidueFrame(Chem.Mol):
    """The ResidueFrame class, which inherits the `rdkit.Chem.Mol` class.

    Contains the 3D information from a particular residue and frame"""

    def __init__(self, residue, xyz, n_frame=0):
        super().__init__(residue)
        conformer = Chem.Conformer()
        for i in range(self.GetNumAtoms()):
            position = Chem.rdGeometry.Point3D(*xyz[i])
            conformer.SetAtomPosition(i, position)
        self.AddConformer(conformer)
        self.n_frame = n_frame

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
